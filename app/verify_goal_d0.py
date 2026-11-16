"""Recompute D0 evidence from saved states and trajectories, without training."""
import argparse
from dataclasses import asdict
from pathlib import Path
import json

import mujoco
import numpy as np
import torch

from .evaluate_goal import file_sha256
from .free_body import FreeBody, ROOT
from .goal_checkpoint import goal_contract, read_goal_payload
from .mlp_policy import MLPPolicy
from .ppo_yumi import Value


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def verify(output: Path, checkpoint_root: Path) -> dict:
    controls = output / 'controls'
    distribution = output / 'distribution'
    report, menu = read(controls / 'summary.json'), read(controls / 'menu.json')
    top = read(output / 'summary.json')
    assert file_sha256(controls / 'summary.json') == top['controls_summary_sha256']
    assert file_sha256(distribution / 'summary.json') == top['distribution_summary_sha256']
    assert file_sha256(controls / 'menu.json') == report['menu_sha256']
    assert file_sha256(controls / 'states.json') == report['states_sha256']
    body = FreeBody()
    assert read(controls / 'contract.json') == json.loads(json.dumps(asdict(goal_contract(body))))
    for name, digest in report['source_sha256'].items():
        assert file_sha256(ROOT / name) == digest, name
    actual_rows = {row['id']: row for row in report['trials']}
    assert len(actual_rows) == len(menu['trials']) == 100
    state_meta = {row['id']: row for row in report['states']}
    for name, row in state_meta.items():
        assert file_sha256(controls / f'state_{name}.npz') == row['state_sha256']
        with np.load(controls / f'state_{name}.npz') as state:
            body.data.qpos[:] = state['qpos'][-1]
            body.refresh()
            up = body.rotation[2]
            axis = int(np.argmax(np.abs(up[:2])))
            category = ('upright' if up[2] >= .7 else 'inverted' if up[2] <= -.7 else
                        ('front' if axis == 0 else 'left') if up[axis] > 0 else ('back' if axis == 0 else 'right'))
            assert category == row['category']
            assert row['low_posture'] == (float(body.center[2]) < .35 and category != 'upright')
    references = {}
    recomputed = {}
    max_com_error = max_replay_error = max_work_error = 0.
    poses = replay_steps = 0
    for trial in menu['trials']:
        row = actual_rows[trial['id']]
        if row['reason'] == 'invalid_initial_state':
            assert not state_meta[trial['state']]['complete']
            continue
        path = controls / (trial['id'] + '.npz')
        assert file_sha256(path) == row['trace_sha256']
        with np.load(path) as z:
            times, centers = z['time'], z['center']
            target = centers[0, :2] + [2., 0.]
            total = np.linalg.norm(target - centers[0, :2]) - np.linalg.norm(target - centers[-1, :2])
            assert abs(total - row['total_progress_m']) < 1e-10
            assert abs(z['work_j'].sum() - row['work_j']) < 1e-7
            assert np.isfinite(z['qpos']).all() and np.isfinite(z['qvel']).all()
            assert np.all(np.diff(times) > 0) and np.all(z['substeps'] <= 10)
            for qpos, expected in zip(z['qpos'], centers):
                body.data.qpos[:] = qpos
                mujoco.mj_kinematics(body.model, body.data)
                mujoco.mj_comPos(body.model, body.data)
                max_com_error = max(max_com_error, float(np.max(np.abs(body.center - expected))))
                poses += 1
            if row['complete']:
                assert len(times) == 201 and abs(times[-1] - 4.) < 1e-9
                i = int(np.argmin(np.abs(times - 1.)))
                late = float(np.linalg.norm(target - centers[i, :2]) - np.linalg.norm(target - centers[-1, :2]))
                assert abs(late - row['late_progress_m']) < 1e-10
                recomputed[trial['id']] = late
                if trial['template'] == 'zero':
                    references[trial['state']] = late
                assert row['max_penetration_m'] <= .030 and row['max_limit_error_rad'] <= .15
            # Replay all zero references and the same fixed large template per state.
            if trial['template'] == 'zero' or (trial['template'] == 'fold_extend' and trial['amplitude'] == 4. and trial['hold_seconds'] == .4):
                body.reset()
                with np.load(controls / f"state_{trial['state']}.npz") as state:
                    mujoco.mj_setState(body.model, body.data, state['integration_state'], mujoco.mjtState.mjSTATE_INTEGRATION)
                    body.action[:] = state['previous_action']
                body.refresh()
                n = min(10, len(z['command']) - (0 if row['complete'] else 1))
                for i in range(n):
                    work = body.step(z['command'][i])
                    max_replay_error = max(max_replay_error, float(np.max(np.abs(body.data.qpos - z['qpos'][i+1]))))
                    max_work_error = max(max_work_error, abs(work - z['work_j'][i]))
                    replay_steps += 1
    assert max_com_error < 1e-9 and max_replay_error < 1e-8 and max_work_error < 1e-7
    categories = set()
    positive_ids = []
    for trial in menu['trials']:
        row = actual_rows[trial['id']]
        meta = state_meta[trial['state']]
        if trial['id'] in recomputed and trial['state'] in references:
            extra = recomputed[trial['id']] - references[trial['state']]
            assert abs(extra - row['late_extra_progress_m']) < 1e-10
            positive = meta['low_posture'] and extra >= .1
            assert row['positive'] == positive
            if positive:
                categories.add(meta['category']); positive_ids.append(trial['id'])
    assert sorted(categories) == report['positive_categories']
    dist_report = read(distribution / 'summary.json')
    for name, digest in dist_report['source_sha256'].items():
        assert file_sha256(ROOT / name) == digest, name
    max_advantage_error = max_reward_error = max_value_metric_error = 0.
    max_policy_error = max_critic_error = max_logprob_error = 0.
    torch.set_num_threads(1)
    transitions = 0
    for run in dist_report['runs']:
        seed, mode = run['seed'], run['mode']
        checkpoint = checkpoint_root / f'baseline_{seed}/last.pt'
        assert file_sha256(checkpoint) == run['checkpoint_sha256']
        payload = read_goal_payload(checkpoint, goal_contract(body), 'cpu')
        actor = MLPPolicy(69, 18, device='cpu').eval()
        actor.load_state_dict(payload['policy'])
        critic = Value(69).eval()
        critic.load_state_dict(payload['value'])
        path = distribution / f'{seed}_{mode}.npz'
        assert file_sha256(path) == run['trace_sha256']
        with np.load(path) as z:
            reward, value, next_value = z['reward'].astype(float), z['value'].astype(float), z['next_value'].astype(float)
            alive = 1 - z['done'].astype(float)
            bootstrap_reward = reward + .99 * next_value * z['truncated']
            assert np.max(np.abs(bootstrap_reward - z['bootstrap_reward'])) < 1e-6
            delta = bootstrap_reward + .99 * next_value * alive - value
            expected = np.zeros_like(delta)
            for start in range(0, len(reward), 64):
                carry = np.zeros(reward.shape[1])
                for t in range(min(start+64, len(reward))-1, start-1, -1):
                    carry = delta[t] + .99 * .95 * alive[t] * carry
                    expected[t] = carry
            error = float(np.max(np.abs(expected - z['advantage'])))
            max_advantage_error = max(max_advantage_error, error)
            assert error < 1e-5
            assert np.max(np.abs(z['returns'] - (z['advantage'] + z['value']))) < 1e-6
            bonus = reward - z['progress'].astype(float) + .001 * z['work_j'].astype(float)
            error = float(np.max(np.abs(bonus - np.rint(bonus))))
            assert np.all((np.rint(bonus) == 0) | (np.rint(bonus) == 1)) and error < 1e-6
            max_reward_error = max(max_reward_error, error)
            obs = z['observation'].reshape(-1, 69)
            with torch.inference_mode():
                mu, _ = actor(torch.from_numpy(obs))
                predicted = critic(torch.from_numpy(obs)).numpy().reshape(value.shape)
                density = torch.distributions.Normal(mu, payload['log_std'].exp().clamp(.05, 1.5))
                logp = density.log_prob(torch.from_numpy(z['action'].reshape(-1, 18))).sum(-1).numpy()
            max_policy_error = max(max_policy_error, float(np.max(np.abs(mu.numpy() - z['mean'].reshape(-1, 18)))))
            max_critic_error = max(max_critic_error, float(np.max(np.abs(predicted - value))))
            max_logprob_error = max(max_logprob_error, float(np.max(np.abs(logp - z['log_prob'].reshape(-1)))))
            assert max_policy_error < 2e-5 and max_critic_error < 2e-5 and max_logprob_error < 1e-4
            clip = np.mean(np.abs((obs - z['obs_mean']) / z['obs_std']) > 10, axis=0)
            assert np.allclose(clip, [r['actor_clipped_fraction'] for r in run['observations']], atol=1e-12)
            returns = z['returns'].astype(float)
            ev = 1 - np.var(returns - value) / np.var(returns)
            max_value_metric_error = max(max_value_metric_error, abs(ev - run['value_metrics']['explained_variance']))
            assert max_value_metric_error < 1e-9
            assert len(reward)*reward.shape[1] == run['transitions'] == 2048
            if mode == 'mean':
                assert np.array_equal(z['action'], z['mean'])
            else:
                assert not np.array_equal(z['action'], z['mean'])
            transitions += len(obs)
    for seed in (404, 505, 606):
        with np.load(distribution / f'{seed}_sampled.npz') as a, np.load(distribution / f'{seed}_mean.npz') as b:
            assert np.array_equal(a['initial_qpos'], b['initial_qpos'])
            assert np.array_equal(a['initial_target'], b['initial_target'])
    assert transitions == dist_report['transitions'] == top['distribution_transitions'] == 12288
    result = dict(verified=True, native_trials=len(actual_rows), native_poses=poses,
                  native_replay_steps=replay_steps, max_com_error_m=max_com_error,
                  max_replay_qpos_error=max_replay_error, max_replay_work_error_j=max_work_error,
                  positive_categories=sorted(categories), positive_trial_ids=positive_ids,
                  distribution_transitions=transitions, max_advantage_error=max_advantage_error,
                  max_reward_identity_error=max_reward_error, max_value_metric_error=max_value_metric_error,
                  max_policy_prediction_error=max_policy_error, max_critic_prediction_error=max_critic_error,
                  max_log_probability_error=max_logprob_error,
                  verifier_sha256=file_sha256(Path(__file__)),
                  run_summary_sha256=file_sha256(output / 'summary.json'))
    (output / 'verification.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, indent=2, allow_nan=False))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint-root', type=Path, default=ROOT / 'runs/goal_local_campaign_20261011/screen')
    args = parser.parse_args()
    verify(args.output, args.checkpoint_root)


if __name__ == '__main__':
    main()
