"""Independent numerical and provenance audit of the paired holding experiment."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .diagnose_goal_d0 import write_json
from .evaluate_goal import file_sha256
from .free_body import FreeBody, ROOT
from .goal_checkpoint import HoldingGoalContext, CoverageGoalContext, goal_contract, read_goal_payload
from .mlp_policy import MLPPolicy
from .verify_local_goal import verify_rollout


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


@torch.inference_mode()
def verify_evaluation(folder: Path, checkpoint: Path, state: Path, *, source_state: Path | None = None) -> dict:
    report = read(folder/'summary.json')
    assert report['checkpoint_sha256'] == file_sha256(checkpoint)
    assert report['state_sha256'] == file_sha256(state)
    assert report['trace_sha256'] == file_sha256(folder/'trace.npz')
    if report['schema'] == 'coverage_goal_evaluation':
        context = CoverageGoalContext(**report['source_context'])
        assert source_state is not None
        assert context.pool_sha256 == report['pool_sha256'] == file_sha256(source_state)
    else:
        context = HoldingGoalContext(**(report['source_context'] if source_state is not None else report['local_context']))
        assert context.reset_sha256 == file_sha256(source_state or state)
        if source_state is not None:
            assert report['source_state_sha256'] == file_sha256(source_state)
    if source_state is not None:
        assert report['transfer_contract']['protocol'] == 'fallen_finite_goal_v1'
        assert report['transfer_contract']['timeout_is_terminal']
    payload = read_goal_payload(checkpoint, goal_contract(FreeBody()), local_context=context)
    actor = MLPPolicy(69, 18, device='cpu').eval()
    actor.load_state_dict(payload['policy'])
    body = FreeBody()
    with np.load(folder/'trace.npz') as z, np.load(state) as initial:
        tol = 2e-6
        np.testing.assert_allclose(z['qpos'][0], initial['qpos'], atol=tol, rtol=0)
        np.testing.assert_allclose(z['qvel'][0], initial['qvel'], atol=tol, rtol=0)
        np.testing.assert_allclose(z['observation'][0, 48:66], initial['previous_action'], atol=tol, rtol=0)
        np.testing.assert_array_equal(z['target'], initial['target'])
        np.testing.assert_array_equal(z['origin'], initial['origin'])
        assert abs(z['time'][0]-float(initial['elapsed'])) < 1e-12
        assert z['time'][-1] <= 5.+1e-8
        complete = z['control_step_complete'] if 'control_step_complete' in z else np.ones(len(z['work_j']), dtype=bool)
        assert complete[:-1].all()
        np.testing.assert_allclose(np.diff(z['time'])[complete], .02, atol=1e-12, rtol=0)
        end = len(z['time']) if complete[-1] else len(z['time'])-1
        if not complete[-1]:
            assert 0 < np.diff(z['time'])[-1] <= .02+1e-10
        np.testing.assert_allclose(z['observation'][:end, 68], 1-z['time'][:end]/5., atol=tol, rtol=0)
        com_error = velocity_error = 0.
        for q, v, c, velocity in zip(z['qpos'], z['qvel'], z['center'], z['velocity']):
            body.data.qpos[:], body.data.qvel[:] = q, v
            body.refresh()
            com_error = max(com_error, float(np.abs(body.center-c).max()))
            velocity_error = max(velocity_error, float(np.abs(body.velocity-velocity).max()))
        assert com_error < tol and velocity_error < tol
        distance = np.linalg.norm(z['center'][:, :2]-z['target'], axis=1)
        speed = np.linalg.norm(z['velocity'][:, :2], axis=1)
        dwell = float(initial['dwell'])
        expected = []
        for i, (d, v) in enumerate(zip(distance[1:], speed[1:])):
            if complete[i]:
                dwell = dwell+.02 if d <= .3 and v <= .2 else 0.
            expected.append(dwell)
        np.testing.assert_allclose(z['dwell'], expected, atol=1e-8, rtol=0)
        np.testing.assert_allclose(z['observation'][1:, 67], expected, atol=tol, rtol=0)
        np.testing.assert_allclose(z['progress'], -np.diff(distance), atol=tol, rtol=0)
        assert abs(report['final_distance_m']-distance[-1]) < tol
        assert abs(report['final_speed_m_s']-speed[-1]) < tol
        reached = bool(complete[-1] and max(expected) >= 1.-1e-9)
        safe = bool(report['reason'] in ('timeout', 'complete') and np.isfinite(z['qpos']).all()
                    and np.isfinite(z['qvel']).all() and z['penetration'].max() <= .030
                    and z['joint_limit_error'].max() <= .15)
        assert safe == report['safe'] and reached == report['reached']
        assert bool(safe and reached) == report['passed']
        if safe:
            assert z['terminated'][-1] and not z['truncated'].any()
            assert reached or abs(z['time'][-1]-5.) < 1e-8
        bonus = np.zeros(len(expected))
        if reached:
            bonus[-1] = 1.
        np.testing.assert_allclose(z['reward'], -np.diff(distance)+bonus-.001*z['work_j'], atol=tol, rtol=0)
        mu, _ = actor(torch.from_numpy(z['observation'][:-1]))
        action_error = float(np.abs(mu.numpy()-z['raw_action']).max())
        assert action_error < 2e-5
        np.testing.assert_allclose(z['executed_action'], np.clip(z['raw_action'], -8, 8), atol=tol, rtol=0)
    return dict(folder=folder.name, max_com_error_m=com_error, max_velocity_error=velocity_error,
                max_action_error=action_error, passed=bool(safe and reached))


def verify(output: Path) -> dict:
    torch.set_num_threads(1)
    prereg, report = read(output/'preregistration.json'), read(output/'summary.json')
    assert file_sha256(output/'preregistration.json') == report['preregistration_sha256']
    assert file_sha256(output/'reset.npz') == prereg['state_sha256'] == report['state_sha256']
    for name, digest in prereg['source_sha256'].items():
        assert file_sha256(output/'source_snapshot'/name) == digest
    paired, audits, evaluations = {}, [], []
    transitions = attempted = 0
    counts = dict(baseline=0, potential=0)
    for row in report['runs']:
        folder = output/f"{row['group']}_{row['seed']}"
        status = read(folder/'training.json')
        context = HoldingGoalContext(**status['local_context'])
        assert context.reset_sha256 == prereg['state_sha256']
        assert context.potential_scale == row['scale']
        initial = read_goal_payload(folder/'initial.pt', goal_contract(FreeBody()), local_context=context)
        if row['seed'] not in paired:
            paired[row['seed']] = initial
        for section in ('policy', 'value'):
            assert all(torch.equal(v, paired[row['seed']][section][k]) for k, v in initial[section].items())
        assert torch.equal(initial['log_std'], paired[row['seed']]['log_std'])
        assert initial['settings'] == {**prereg['settings'], 'seed': row['seed']}
        assert torch.allclose(initial['log_std'].exp(), torch.full((18,), .15))
        count = sum(h['transitions'] for h in status['history'])
        assert count == status['env_steps'] == row['actual_transitions']
        transitions += count
        attempted += status['attempted_local_transitions']
        labels = [p.stem for p in sorted(folder.glob('rollout_*.json'))]
        checks = {label: verify_rollout(folder, label, context) for label in labels}
        # Use two consecutive saved rollouts to cover at least one actual deadline or arrival.
        terminal_count = 0
        if (folder/'rollout_2.npz').exists():
            with np.load(folder/'rollout_1.npz') as a, np.load(folder/'rollout_2.npz') as b:
                done = np.concatenate((a['done'], b['done'])).astype(bool)
                terminated = np.concatenate((a['terminated'], b['terminated']))
                truncated = np.concatenate((a['truncated'], b['truncated']))
                assert np.all(done.any(0)) and not truncated.any() and np.array_equal(done, terminated)
                terminal_count = int(done.sum())
                phi = np.concatenate((a['potential_reward'], b['potential_reward'])).astype(float)
                residuals = []
                for world in range(done.shape[1]):
                    total = 0.; power = 1.
                    for t in range(len(done)):
                        total += power*phi[t, world]
                        power *= .99
                        if done[t, world]:
                            residuals.append(abs(total)); total = 0.; power = 1.
                assert max(residuals) < 2e-7
        if row['complete']:
            train = read(folder/'summary.json')
            assert count == 262144 and len(status['history']) == 256
            assert file_sha256(folder/'last.pt') == train['checkpoint_sha256'] == row['checkpoint_sha256']
            final = read_goal_payload(folder/'last.pt', goal_contract(FreeBody()), local_context=context)
            assert final['env_steps'] == count and any(not torch.equal(v, initial['policy'][k]) for k, v in final['policy'].items())
            for name, digest in train['source_sha256'].items():
                assert prereg['source_sha256'][name] == digest
            for h in status['history']:
                expected = (h['progress_sum_m']+h['arrivals']-.001*h['absolute_work_j']+h['potential_reward_sum'])/h['transitions']
                assert abs(expected-h['reward_mean']) < 1e-6
                assert h['max_penetration_m'] <= .030 and h['max_joint_limit_error_rad'] <= .15
            if 'final' in row:
                for phase, checkpoint in (('initial', 'initial.pt'), ('final', 'last.pt')):
                    for backend in ('native', 'cuda'):
                        folder_eval = output/f"eval_{row['group']}_{row['seed']}_{phase}_{backend}"
                        for name, digest in read(folder_eval/'summary.json')['source_sha256'].items():
                            assert prereg['source_sha256'][name] == digest
                        evaluations.append(verify_evaluation(folder_eval, folder/checkpoint, output/'reset.npz'))
                passed = all(row['final'][b]['passed'] for b in ('native', 'cuda'))
                assert row['passed'] == passed
                counts[row['group']] += int(passed)
        elif status.get('physical_guard'):
            g = status['physical_guard']
            assert g['penetration_m'] > .030 or g['limit_error_rad'] > .15
            interrupted = read_goal_payload(folder/'interrupted.pt', goal_contract(FreeBody()), local_context=context)
            assert interrupted['env_steps'] == count
        audits.append(dict(group=row['group'], seed=row['seed'], complete=row['complete'],
                           verified_transitions=count, first_two_rollout_terminals=terminal_count, rollout_checks=checks))
    assert transitions == report['actual_transitions'] and counts == report['passed_seeds']
    assert sum(r['training_process_seconds'] for r in report['runs']) <= 2700.
    expected_branch = report['branch']
    if len(report['runs']) == 6 and all(r['complete'] for r in report['runs']):
        disagreement = any(r['final']['native']['passed'] != r['final']['cuda']['passed'] for r in report['runs'])
        unsafe = any(not x['safe'] for r in report['runs'] for phase in ('initial', 'final') for x in r[phase].values())
        expected_branch = ('route_uncertainty_stop' if disagreement or unsafe else 'baseline_preferred' if counts['baseline'] >= 2
                           else 'potential_course_positive' if counts['potential'] >= 2 else 'route_uncertainty_stop')
    assert expected_branch == report['branch']
    result = dict(verified=True, branch=report['branch'], verified_transitions=transitions,
                  attempted_transitions=attempted, passed_seeds=counts, runs=audits, evaluations=evaluations,
                  summary_sha256=file_sha256(output/'summary.json'), verifier_sha256=file_sha256(Path(__file__)))
    write_json(output/'verification.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('runs', 'evaluations')}))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    verify(args.output)


if __name__ == '__main__':
    main()
