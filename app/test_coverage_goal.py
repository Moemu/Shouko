"""Owner contracts for mixed-state reset lifecycle and actor-only PPO initialization."""
import argparse
import json
from pathlib import Path
import shutil

import numpy as np
import torch
import warp as wp

from .coverage_goal import CoverageGoalEnv, CoverageGoalSetup, load_pool
from .diagnose_goal_d0 import write_json
from .goal_checkpoint import CoverageGoalContext, goal_contract, read_goal_payload
from .free_body import FreeBody
from .ppo_goal import PPOSettings, train
from .verify_local_goal import verify_rollout
from .evaluate_goal import file_sha256


def check_pool_identity(pool: Path, screen: Path, output: Path) -> None:
    """Checksums alone must not allow an extended deadline or a different original body state."""
    for name, index, field in (('wrong_deadline', 7, 'elapsed'), ('wrong_original', 0, 'qvel'), ('wrong_goal', 0, 'origin')):
        fixture = output/name
        shutil.copytree(pool, fixture)
        manifest = json.loads((fixture/'manifest.json').read_text(encoding='utf-8'))
        changed = manifest['states'] if field == 'origin' else [manifest['states'][index]]
        for entry in changed:
            path = fixture/entry['filename']
            with np.load(path) as z:
                arrays = {key: z[key].copy() for key in z.files}
            if field == 'elapsed':
                arrays[field] = np.asarray(0.)
            elif field == 'qvel':
                arrays[field][0] += .001
            else:
                arrays['origin'][0] -= 1.7
                arrays['target'][0] -= 1.7
            np.savez_compressed(path, **arrays)
            entry['sha256'] = file_sha256(path)
        write_json(fixture/'manifest.json', manifest)
        try:
            if field != 'qvel':
                load_pool(fixture)
            else:
                CoverageGoalSetup(fixture, screen/'iid_zero_1401/last.pt', screen/'reset.npz', 'mixed').context()
        except ValueError as error:
            expected = {'elapsed': 'frozen schedule', 'qvel': 'natural physics', 'origin': 'original centre'}[field]
            assert expected in str(error)
        else:
            raise AssertionError('Self-consistent hashes accepted a changed finite-task identity')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--source-screen', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    check_pool_identity(args.pool, args.source_screen, args.output)
    manifest = json.loads((args.pool/'manifest.json').read_text(encoding='utf-8'))
    source = args.source_screen/'iid_zero_1401/last.pt'
    setup = CoverageGoalSetup(args.pool, source, args.source_screen/'reset.npz', 'mixed')
    env = CoverageGoalEnv(16, 1681, setup)
    expected = [0]*8+[1, 2, 3, 4]+[7]*4

    def matches(world: int, index: int) -> None:
        with np.load(args.pool/manifest['states'][index]['filename']) as z:
            np.testing.assert_array_equal(env.qpos[world].cpu().numpy(), z['qpos'].astype(np.float32))
            np.testing.assert_array_equal(env.qvel[world].cpu().numpy(), z['qvel'].astype(np.float32))
            np.testing.assert_array_equal(env.actions[world].cpu().numpy(), z['previous_action'].astype(np.float32))
            np.testing.assert_array_equal(env.target[world].cpu().numpy(), z['target'].astype(np.float32))
            np.testing.assert_array_equal(env.origin[world].cpu().numpy(), z['origin'].astype(np.float32))
            np.testing.assert_array_equal(wp.to_torch(env.data.qacc_warmstart)[world].cpu().numpy(), z['qacc_warmstart'].astype(np.float32))
            assert abs(float(env.observation()[world, 68])-(1-float(z['elapsed'])/5.)) < 1e-7
    for world, index in enumerate(expected):
        matches(world, index)
    untouched = (env.qpos[9:].clone(), env.qvel[9:].clone(), env.actions[9:].clone(), env.target_steps[9:].clone())
    mask = torch.zeros(16, device='cuda', dtype=torch.bool)
    mask[8] = True
    for index in (2, 3, 4, 5, 6, 1):
        env.reset(mask)
        matches(8, index)
        for current, prior in zip((env.qpos[9:], env.qvel[9:], env.actions[9:], env.target_steps[9:]), untouched):
            assert torch.equal(current, prior)
    env.target_steps[:] = 249
    _, result = env.step(torch.zeros(16, 18, device='cuda'))
    assert bool(result.timeout.all() and result.terminated.all() and ~result.truncated.any())
    reports, paired = [], {}
    try:
        for source_seed, sampling_seed in ((1401, 1681), (1402, 1682), (1403, 1683)):
            source_path = args.source_screen/f'iid_zero_{source_seed}/last.pt'
            original = torch.load(source_path, map_location='cpu', weights_only=True)
            for arm in ('near', 'mixed'):
                folder = args.output/f'{arm}_{source_seed}'
                report = train(folder, PPOSettings(seed=sampling_seed, worlds=4, initial_std=.15), 1, 30., work_cost=0.,
                               local=CoverageGoalSetup(args.pool, source_path, args.source_screen/'reset.npz', arm))
                context = CoverageGoalContext(**report['local_context'])
                initial = read_goal_payload(folder/'initial.pt', goal_contract(FreeBody()), local_context=context)
                assert all(torch.equal(v, initial['policy'][k]) for k, v in original['policy'].items())
                assert any(not torch.equal(v, initial['value'][k]) for k, v in original['value'].items())
                assert not initial['optimizer']['state'] and not initial['value_optimizer']['state']
                torch.testing.assert_close(initial['log_std'].exp(), torch.full((18,), .15))
                if source_seed in paired:
                    assert all(torch.equal(v, paired[source_seed]['value'][k]) for k, v in initial['value'].items())
                else:
                    paired[source_seed] = initial
                row = report['history'][0]['state_categories']
                expected_counts = (0, 0, 256) if arm == 'near' else (128, 64, 64)
                assert tuple(row[k]['transitions'] for k in ('original', 'middle', 'near')) == expected_counts
                checks = verify_rollout(folder, 'rollout_1', context)
                try:
                    read_goal_payload(folder/'last.pt', goal_contract(FreeBody()))
                except ValueError:
                    pass
                else:
                    raise AssertionError('Global loader accepted coverage checkpoint')
                reports.append(dict(source_seed=source_seed, arm=arm, transitions=report['launch_transitions'], checks=checks))
    except Exception:
        attempted = sum(json.loads(p.read_text())['attempted_local_transitions'] for p in args.output.glob('*/training.json'))
        write_json(args.output/'checks.json', dict(passed=False, engineering_attempted_transitions=attempted, runs=reports))
        raise
    write_json(args.output/'checks.json', dict(passed=True, engineering_attempted_transitions=1536, runs=reports,
                                              mixed_full_and_partial_reset=True, actor_only_initialization=True,
                                              true_deadline_step=True, fresh_paired_critic_adam_noise=True))
    print('PASS: mixed reset lifecycle, finite deadline, actor initialization, category counts and PPO targets')


if __name__ == '__main__':
    main()
