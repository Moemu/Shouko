"""Recompute screen evidence from checkpoints, raw episodes and saved physics poses."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path

import numpy as np
import torch

from .evaluate_goal import file_sha256
from .free_body import FreeBody, ROOT
from .ppo_goal import PPOSettings


def verify(output: Path) -> dict:
    summary = json.loads((output / 'summary.json').read_text(encoding='utf-8'))
    prereg = json.loads((output / 'preregistration.json').read_text(encoding='utf-8'))
    assert summary['phase'] == 'complete' and len(summary['rows']) == 9
    deck_hash = file_sha256(output / 'deck.json')
    assert deck_hash == prereg['deck_sha256']
    body = FreeBody()
    poses_checked = 0
    max_com_error = max_reward_error = 0.
    independent_passes = {group: 0 for group in prereg['groups']}
    reference_policies = {}
    shared_contract = None
    results = []
    for row in summary['rows']:
        group, seed = row['group'], row['seed']
        folder = output / f'{group}_{seed}'
        training = json.loads((folder / 'summary.json').read_text(encoding='utf-8'))
        evaluation = json.loads((output / f'eval_{group}_{seed}/summary.json').read_text(encoding='utf-8'))
        initial_eval = json.loads((output / f'eval_initial_{seed}/summary.json').read_text(encoding='utf-8'))
        config = prereg['groups'][group]
        expected_settings = asdict(PPOSettings(seed=seed, initial_std=config['std']))
        if config['work_cost'] != .001:
            expected_settings['training_work_cost'] = config['work_cost']
        assert training['settings'] == expected_settings
        assert training['initial_env_steps'] == 0 and training['final_env_steps'] == 262144
        assert training['launch_iterations'] == 256
        assert sum(h['transitions'] for h in training['history']) == 262144
        assert training['training_reward']['work_cost'] == config['work_cost']
        assert row['checkpoint_sha256'] == training['checkpoint_sha256'] == evaluation['checkpoint_sha256'] == file_sha256(folder / 'last.pt')
        assert evaluation['deck_sha256'] == initial_eval['deck_sha256'] == deck_hash
        assert training['initial_checkpoint_sha256'] == file_sha256(folder / 'initial.pt')
        assert initial_eval['checkpoint_sha256'] == file_sha256(output / f'baseline_{seed}/initial.pt')
        if shared_contract is None:
            shared_contract = training['contract']
        assert evaluation['contract'] == training['contract'] == shared_contract
        assert evaluation['episode_count'] == 16
        initial = torch.load(folder / 'initial.pt', map_location='cpu', weights_only=True)
        final = torch.load(folder / 'last.pt', map_location='cpu', weights_only=True)
        if seed not in reference_policies:
            reference_policies[seed] = initial['policy']
        for name, value in reference_policies[seed].items():
            torch.testing.assert_close(initial['policy'][name], value, rtol=0, atol=0)
        assert torch.allclose(initial['log_std'].exp(), torch.full((18,), config['std']), atol=1e-7, rtol=0)
        for report in (training, evaluation, initial_eval):
            for source, digest in report['source_sha256'].items():
                assert file_sha256(ROOT / source) == digest, source
        for history in training['history']:
            reward = history['reward_mean'] * history['transitions']
            expected = history['progress_sum_m'] + history['arrivals'] - config['work_cost'] * history['absolute_work_j']
            max_reward_error = max(max_reward_error, abs(reward - expected))
        trajectories = np.load(output / f'eval_{group}_{seed}/trajectories.npz')
        progress, late, initial_progress = [], [], []
        for episode, original in zip(evaluation['episodes'], initial_eval['episodes']):
            assert episode['id'] == original['id']
            first_target = episode['targets'][0]
            distance_change = first_target['initial_distance_m'] - first_target['final_distance_m']
            assert abs(distance_change - first_target['net_progress_m']) < 1e-9
            progress.append(distance_change)
            initial_progress.append(original['targets'][0]['net_progress_m'])
            trace = trajectories[f"episode_{episode['id']}"]
            assert np.isfinite(trace).all()
            first_trace = trace[trace[:, 1] == 0]
            if first_trace[-1, 0] <= 3.:
                late.append(0.)
            else:
                at_three = first_trace[np.argmin(abs(first_trace[:, 0] - 3.))]
                assert abs(at_three[0] - 3.) < 1e-8
                late.append(float(np.linalg.norm(at_three[8:10] - at_three[2:4]) - first_target['final_distance_m']))
            for index in (0, len(trace) // 2, len(trace) - 1):
                body.data.qpos[:] = trace[index, 12:12 + body.model.nq]
                body.refresh()
                max_com_error = max(max_com_error, float(np.max(abs(body.center - trace[index, 2:5]))))
                poses_checked += 1
        mean_progress = float(np.mean(progress))
        improvement = mean_progress - float(np.mean(initial_progress))
        mean_late = float(np.mean(late))
        assert abs(mean_progress - row['mean_first_target_progress_m']) < 1e-9
        assert abs(mean_late - row['mean_after_3s_first_target_progress_m']) < 1e-9
        passed = (mean_progress > 0 and improvement >= .1 - 1e-9 and mean_late >= .1 - 1e-9
                  and evaluation['mean_first_target_min_distance_m'] <= 1.5 + 1e-9
                  and evaluation['max_penetration_m'] <= .030
                  and max(e['max_joint_limit_error_rad'] for e in evaluation['episodes']) <= .15)
        assert passed == row['gate_passed']
        independent_passes[group] += int(passed)
        results.append(dict(group=group, seed=seed, mean_progress_m=mean_progress,
                            mean_after_3s_progress_m=mean_late, gate_passed=passed,
                            final_exploration_std_mean=float(final['log_std'].exp().clamp(.05, 1.5).mean()),
                            final_exploration_std_min=float(final['log_std'].exp().clamp(.05, 1.5).min()),
                            final_exploration_std_max=float(final['log_std'].exp().clamp(.05, 1.5).max())))
    assert max_reward_error < 1e-5 and max_com_error < 1e-9
    assert {item['group']: item['passed_seeds'] for item in summary['candidates']} == independent_passes
    result = dict(verified=True, runs=9, total_transitions=9 * 262144, poses_checked=poses_checked,
                  max_recomputed_com_error_m=max_com_error, max_training_reward_identity_error=max_reward_error,
                  initial_policies_equal=True, contracts_and_hashes_equal=True,
                  independently_recomputed_passed_seeds=independent_passes, rows=results,
                  verifier_sha256=file_sha256(Path(__file__)))
    (output / 'verification.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='Completed screen directory')
    verify(parser.parse_args().output)
