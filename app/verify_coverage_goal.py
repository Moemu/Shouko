"""Independent audit of actor initialization, reset allocation, finite PPO targets and joint gates."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .diagnose_goal_d0 import write_json
from .evaluate_goal import file_sha256
from .free_body import FreeBody
from .goal_checkpoint import CoverageGoalContext, goal_contract, read_goal_payload
from .ppo_yumi import Value
from .verify_holding_goal import verify_evaluation
from .verify_local_goal import verify_rollout


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def audit_campaign_contract(prereg: dict, report: dict) -> None:
    expected = {(r['source_seed'], r['arm']): r['sampling_seed'] for r in prereg['order']}
    keys = [(r['source_seed'], r['arm']) for r in report['runs']]
    assert len(keys) == len(set(keys)) and set(keys) <= set(expected)
    if report['complete_comparison']:
        assert set(keys) == set(expected) and all(r['complete'] for r in report['runs'])
    assert abs(sum(r['training_process_seconds'] for r in report['runs'])-report['training_process_seconds']) < 1e-6
    for row in report['runs']:
        assert row['sampling_seed'] == expected[(row['source_seed'], row['arm'])]
        if row['complete']:
            assert row['training_complete']
            for phase in ('initial', 'final'):
                assert set(row[phase]) == {'original', 'near'}
                for backends in row[phase].values():
                    assert set(backends) == {'native', 'cuda'}
                    assert all(v['safe'] for v in backends.values())


def count_joint_success(report: dict, checks: list[dict]) -> dict[str, int]:
    by_name = {c['folder']: c for c in checks}
    assert len(by_name) == len(checks)
    expected_names = set()
    counts = dict(near=0, mixed=0)
    for row in report['runs']:
        final = []
        for phase in ('initial', 'final'):
            for start, backends in row.get(phase, {}).items():
                for backend, value in backends.items():
                    name = f"eval_{row['arm']}_{row['source_seed']}_{phase}_{start}_{backend}"
                    expected_names.add(name)
                    assert value['passed'] == by_name[name]['passed']
                    if phase == 'final':
                        final.append(by_name[name]['passed'])
        if row['complete']:
            assert len(final) == 4
            passed = all(final)
            assert row['passed'] == passed
            counts[row['arm']] += int(passed)
    assert set(by_name) == expected_names
    return counts


@torch.inference_mode()
def optimizer_categories(folder: Path, label: str, context: CoverageGoalContext, history: list[dict]) -> dict:
    """Replay saved RNG permutations, without gradients, updates or physics."""
    metadata = read(folder/(label+'.json'))
    payload = read_goal_payload(folder/metadata['policy_checkpoint'], goal_contract(FreeBody()), local_context=context)
    generator = torch.Generator(device='cuda:0')
    generator.set_state(payload['cuda_rng'][0].cpu())
    with np.load(folder/(label+'.npz')) as z:
        if label == 'rollout_1':
            # The initial checkpoint precedes sampling; other snapshots follow it.
            std = torch.from_numpy(z['log_std']).cuda().exp().clamp(.05, 1.5)
            for t in range(len(z['action'])):
                mean = torch.from_numpy(z['mean'][t]).cuda()
                sampled = torch.normal(mean, std.expand_as(mean), generator=generator)
                np.testing.assert_array_equal(sampled.cpu().numpy(), z['action'][t])
        classes = z['category'].reshape(-1)
        count = len(classes)
        remaining = history[int(label.split('_')[1])-1]['actor_optimizer_steps']
        uses = np.zeros(3, dtype=int)
        for _ in range(payload['settings']['epochs']):
            permutation = torch.randperm(count, device='cuda:0', generator=generator).cpu().numpy()
            for start in range(0, count, payload['settings']['minibatch']):
                if remaining == 0:
                    break
                indices = permutation[start:start+payload['settings']['minibatch']]
                uses += np.bincount(classes[indices], minlength=3)
                remaining -= 1
            if remaining == 0:
                break
        assert remaining == 0
        if context.arm == 'mixed':
            assert uses[0] > 0
    return dict(optimizer_sample_uses=dict(zip(('original', 'middle', 'near'), uses.tolist())),
                initial_action_rng_exact=label == 'rollout_1')


@torch.inference_mode()
def audit_rollout(folder: Path, label: str, context: CoverageGoalContext, pool: Path, history: list[dict]) -> dict:
    checks = verify_rollout(folder, label, context)
    metadata = read(folder/(label+'.json'))
    payload = read_goal_payload(folder/metadata['policy_checkpoint'], goal_contract(FreeBody()), local_context=context)
    critic = Value(69).eval()
    critic.load_state_dict(payload['value'])
    manifest = read(pool/'manifest.json')
    states = []
    for s in manifest['states']:
        with np.load(pool/s['filename']) as z:
            states.append({k: z[k].copy() for k in ('qpos', 'qvel', 'previous_action', 'elapsed', 'dwell')})
    iteration = int(label.split('_')[1])-1
    row = history[iteration]
    with np.load(folder/(label+'.npz')) as z:
        obs, nxt, after = z['observation'], z['next_observation'], z['after_reset_observation']
        steps, worlds = obs.shape[:2]
        categories = np.full(worlds, 2, dtype=int)
        if context.arm == 'mixed':
            categories[:worlds//2] = 0
            categories[worlds//2:3*worlds//4] = 1
        np.testing.assert_array_equal(z['category'], np.broadcast_to(categories, (steps, worlds)))
        expected_indices = np.where(categories == 0, 0, np.where(categories == 1,
                                    1+(np.arange(worlds)-worlds//2+z['reset_round'])%6, 7))
        np.testing.assert_array_equal(z['state_index'], expected_indices)
        done = z['done'].astype(bool)
        reached = z['reached'].astype(bool)
        truncated = z['truncated'].astype(bool)
        np.testing.assert_array_equal(z['terminated'], reached | ((nxt[..., 68] <= 1e-7) & ~truncated))
        np.testing.assert_array_equal(done, z['terminated'] | truncated)
        np.testing.assert_allclose(nxt[..., 68], np.maximum(obs[..., 68]-.004, 0), atol=1e-7, rtol=0)
        np.testing.assert_allclose(z['progress'], np.linalg.norm(obs[..., :3], axis=-1)-np.linalg.norm(nxt[..., :3], axis=-1),
                                   atol=3e-6, rtol=0)
        np.testing.assert_allclose(nxt[..., 48:66], np.clip(z['action'], -8, 8), atol=1e-7, rtol=0)
        next_values = critic(torch.from_numpy(nxt.reshape(-1, 69))).numpy().reshape(steps, worlds)
        np.testing.assert_allclose(next_values, z['next_value'], atol=2e-5, rtol=0)
        bootstrap = critic(torch.from_numpy(after[-1])).numpy()
        np.testing.assert_allclose(bootstrap, z['bootstrap_value'], atol=2e-5, rtol=0)
        if steps > 1:
            np.testing.assert_array_equal(obs[1:], after[:-1])
            np.testing.assert_array_equal(z['reset_round'][1:], z['reset_round'][:-1]+done[:-1])
        np.testing.assert_array_equal(after[~done], nxt[~done])
        reset_checks = 0
        for t in range(steps):
            for world in range(worlds):
                state = states[int(expected_indices[t, world])]
                initial_fraction = 1-float(state['elapsed'])/5.
                if abs(float(obs[t, world, 68])-initial_fraction) <= 1e-7:
                    np.testing.assert_array_equal(z['pre_qpos'][t, world], state['qpos'].astype(np.float32))
                    np.testing.assert_array_equal(z['pre_qvel'][t, world], state['qvel'].astype(np.float32))
                    np.testing.assert_array_equal(obs[t, world, 48:66], state['previous_action'].astype(np.float32))
                    assert abs(float(obs[t, world, 67])-float(state['dwell'])) < 1e-7
                    reset_checks += 1
        for category, name in enumerate(('original', 'middle', 'near')):
            mask = categories == category
            logged = row['state_categories'][name]
            values = dict(transitions=int(mask.sum())*steps, resets=int(done[:, mask].sum()),
                          arrivals=int(reached[:, mask].sum()), timeouts=int(((nxt[..., 68] <= 1e-7) & ~reached & ~truncated)[:, mask].sum()),
                          arena=int(truncated[:, mask].sum()), progress_sum_m=float(z['progress'][:, mask].astype(float).sum()),
                          backtracking_m=float(np.maximum(-z['progress'][:, mask].astype(float), 0).sum()),
                          work_j=float(z['work'][:, mask].astype(float).sum()))
            for k, v in values.items():
                assert abs(v-logged[k]) <= max(1e-6, abs(v)*1e-12), (label, name, k, v, logged[k])
            prior_resets = sum(h['state_categories'][name]['resets'] for h in history[:iteration])
            assert int(z['reset_round'][0, mask].sum()) == prior_resets
    return dict(**checks, reset_state_checks=reset_checks)


def verify(output: Path, pool: Path, screen: Path, contracts: Path) -> dict:
    torch.set_num_threads(1)
    prereg, report = read(output/'preregistration.json'), read(output/'summary.json')
    audit_campaign_contract(prereg, report)
    assert file_sha256(output/'preregistration.json') == report['preregistration_sha256']
    assert prereg['pool_sha256'] == file_sha256(pool/'manifest.json')
    assert prereg['contracts_sha256'] == file_sha256(contracts/'checks.json')
    engineering = read(contracts/'checks.json')
    assert engineering['passed'] and engineering['engineering_attempted_transitions'] <= 2048
    assert len(engineering['runs']) == 6
    for item in engineering['runs']:
        raw = torch.load(contracts/f"{item['arm']}_{item['source_seed']}"/'initial.pt', map_location='cpu', weights_only=True)
        assert raw['local_context']['pool_sha256'] == prereg['pool_sha256']
        assert raw['local_context']['source_actor_sha256'] == prereg['source_actors'][str(item['source_seed'])]
    manifest = read(pool/'manifest.json')
    for state in manifest['states']:
        assert file_sha256(pool/state['filename']) == state['sha256']
    for name, digest in prereg['source_sha256'].items():
        assert file_sha256(output/'source_snapshot'/name) == digest
    paired, audits, evaluation_checks = {}, [], []
    completed = attempted = 0
    for row in report['runs']:
        folder = output/f"{row['arm']}_{row['source_seed']}"
        status = read(folder/'training.json')
        context = CoverageGoalContext(**status['local_context'])
        assert context.arm == row['arm'] and context.pool_sha256 == prereg['pool_sha256']
        source = screen/f"iid_zero_{row['source_seed']}/last.pt"
        assert file_sha256(source) == context.source_actor_sha256 == prereg['source_actors'][str(row['source_seed'])]
        assert context.source_reset_sha256 == file_sha256(screen/'reset.npz') == prereg['source_reset_sha256']
        original = torch.load(source, map_location='cpu', weights_only=True)
        initial = read_goal_payload(folder/'initial.pt', goal_contract(FreeBody()), local_context=context)
        assert initial['iteration'] == 0 and initial['env_steps'] == 0
        assert status['resume_kind'] == 'actor_initialization_fresh_value_adam'
        assert status['source_checkpoint_sha256'] == context.source_actor_sha256
        assert all(torch.equal(v, initial['policy'][k]) for k, v in original['policy'].items())
        assert any(not torch.equal(v, initial['value'][k]) for k, v in original['value'].items())
        assert not initial['optimizer']['state'] and not initial['value_optimizer']['state']
        torch.testing.assert_close(initial['log_std'].exp(), torch.full((18,), .15))
        expected_settings = {**prereg['settings'], 'seed': row['sampling_seed'], 'training_work_cost': 0.}
        assert initial['settings'] == expected_settings
        if row['source_seed'] in paired:
            left = paired[row['source_seed']]
            assert all(torch.equal(v, left['value'][k]) for k, v in initial['value'].items())
            assert torch.equal(initial['log_std'], left['log_std'])
        else:
            paired[row['source_seed']] = initial
        history = status['history']
        transitions = sum(h['transitions'] for h in history)
        assert transitions == row['actual_transitions'] == status['env_steps']
        assert row['attempted_transitions'] == status['attempted_local_transitions']
        completed += transitions
        attempted += row['attempted_transitions']
        for h in history:
            category = h['state_categories']
            fractions = (0., 0., 1.) if context.arm == 'near' else (.5, .25, .25)
            for name, fraction in zip(('original', 'middle', 'near'), fractions):
                assert category[name]['transitions'] == fraction*h['transitions']
            assert sum(c['resets'] for c in category.values()) == h['arrivals']+h['timeout_episodes']+h['arena_episodes']
            assert abs(sum(c['progress_sum_m'] for c in category.values())-h['progress_sum_m']) < 2e-5
            assert abs((h['progress_sum_m']+h['arrivals'])/h['transitions']-h['reward_mean']) < 1e-6
            np.testing.assert_allclose(sum(c['work_j'] for c in category.values()), h['absolute_work_j'], rtol=2e-7, atol=1e-3)
            assert h['max_penetration_m'] <= .030 and h['max_joint_limit_error_rad'] <= .15
        labels = [p.stem for p in sorted(folder.glob('rollout_*.json'))]
        rollout_checks = {label: dict(**audit_rollout(folder, label, context, pool, history),
                                     **optimizer_categories(folder, label, context, history)) for label in labels}
        terminal_coverage = {}
        if row['training_complete']:
            assert transitions == 262144 and len(history) == 256
            assert set(labels) == {'rollout_1', 'rollout_2', 'rollout_3', 'rollout_4', 'rollout_256'}
            done, categories = [], []
            for label in ('rollout_1', 'rollout_2', 'rollout_3', 'rollout_4'):
                with np.load(folder/(label+'.npz')) as z:
                    done.append(z['done'].astype(bool)); categories.append(z['category'])
            endings, classes = np.concatenate(done), np.concatenate(categories)
            for c, name in enumerate(('original', 'middle', 'near')):
                worlds = classes[0] == c
                terminal_coverage[name] = int(endings[:, worlds].sum())
                if worlds.any():
                    assert endings[:, worlds].any(0).all()
            summary = read(folder/'summary.json')
            assert file_sha256(folder/'last.pt') == summary['checkpoint_sha256'] == row['checkpoint_sha256']
            for name, digest in summary['source_sha256'].items():
                assert prereg['source_sha256'][name] == digest
        else:
            assert 0 <= row['attempted_transitions']-transitions <= 1024
            if status.get('physical_guard'):
                g = status['physical_guard']
                assert g['penetration_m'] > .030 or g['limit_error_rad'] > .15
                with np.load(folder/'failure_state.npz') as z:
                    body = FreeBody()
                    q = z['qpos'][:, body.qadr]
                    limit = np.maximum(body.ranges[:, 0]-q, q-body.ranges[:, 1]).max()
                    assert z['penetration'].max() > .030 or limit > .15
                interrupted = read_goal_payload(folder/'interrupted.pt', goal_contract(FreeBody()), local_context=context)
                assert interrupted['env_steps'] == transitions
        for phase, filename in (('initial', 'initial.pt'), ('final', 'last.pt')):
            for start, backends in row.get(phase, {}).items():
                state = pool/('natural.npz' if start == 'original' else '1402_145.npz')
                for backend in backends:
                    dest = output/f"eval_{row['arm']}_{row['source_seed']}_{phase}_{start}_{backend}"
                    check = verify_evaluation(dest, folder/filename, state, source_state=pool/'manifest.json')
                    evaluation_checks.append(check)
                    for name, digest in read(dest/'summary.json')['source_sha256'].items():
                        assert prereg['source_sha256'][name] == digest
        audits.append(dict(arm=row['arm'], source_seed=row['source_seed'], verified_transitions=transitions,
                           attempted_transitions=row['attempted_transitions'], rollout_checks=rollout_checks,
                           first_four_rollout_endings=terminal_coverage))
    counts = count_joint_success(report, evaluation_checks)
    assert completed == report['actual_transitions'] and attempted == report['attempted_transitions']
    assert counts == report['passed_sources']
    assert completed <= 1572864 and report['training_process_seconds'] <= 2700.
    if report['complete_comparison']:
        assert len(report['runs']) == 6 and all(r['complete'] for r in report['runs'])
        expected = 'near_preferred' if counts['near'] >= 2 else 'mixed_coverage_positive' if counts['mixed'] >= 2 else 'route_uncertainty_stop'
        assert report['branch'] == expected
    result = dict(verified=True, branch=report['branch'], verified_transitions=completed, attempted_transitions=attempted,
                  passed_sources=counts, runs=audits, evaluation_checks=evaluation_checks,
                  summary_sha256=file_sha256(output/'summary.json'), verifier_sha256=file_sha256(Path(__file__)))
    write_json(output/'verification.json', result)
    print(json.dumps({k: v for k, v in result.items() if k not in ('runs', 'evaluation_checks')}))
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--source-screen', type=Path, required=True)
    parser.add_argument('--contracts', type=Path, required=True)
    args = parser.parse_args()
    verify(args.output, args.pool, args.source_screen, args.contracts)


if __name__ == '__main__':
    main()
