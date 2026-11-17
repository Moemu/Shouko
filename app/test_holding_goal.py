"""Contract and optimizer-boundary checks for the finite holding course."""
import argparse
from pathlib import Path

import numpy as np
import torch

from .braking_task import HoldingGoalTask, HoldingGoalEnv, HoldingGoalSetup, dwell_potential_reward
from .diagnose_goal_d0 import write_json
from .free_body import FreeBody
from .goal_checkpoint import HoldingGoalContext, goal_contract, read_goal_payload
from .ppo_goal import PPOSettings, train
from .verify_local_goal import verify_rollout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    task = HoldingGoalTask(FreeBody(), args.state)
    env = HoldingGoalEnv(4, 1599, HoldingGoalSetup(args.state, 0.))
    with np.load(args.state) as state:
        np.testing.assert_array_equal(task.body.data.qvel, state['qvel'])
        np.testing.assert_array_equal(task.body.action, state['previous_action'])
        np.testing.assert_allclose(env.qvel.cpu().numpy(), np.tile(state['qvel'], (4, 1)), atol=1e-7)
        np.testing.assert_allclose(env.actions.cpu().numpy(), np.tile(state['previous_action'], (4, 1)), atol=1e-7)
        np.testing.assert_allclose(env.target.cpu().numpy(), np.tile(state['target'], (4, 1)), atol=1e-7)
        assert task.target_elapsed == float(state['elapsed'])
        np.testing.assert_allclose(env.observation()[:, 68].cpu(), 1-float(state['elapsed'])/5., atol=1e-7)
    # Partial reset must not move another world's target or restore its clock.
    env.target_steps[1] += 1
    before = env.qpos[1].clone()
    env.reset(torch.tensor([True, False, False, False], device='cuda'))
    assert int(env.target_steps[1]) == round(task.target_elapsed/.02)+1
    assert torch.equal(env.qpos[1], before)
    task.target_elapsed = 4.98
    env.target_steps[:] = 249
    _, scalar = task.step(np.zeros(18))
    _, batch = env.step(torch.zeros(4, 18, device='cuda'))
    assert scalar.reason == 'timeout' and scalar.terminated and not scalar.truncated
    assert bool(batch.timeout.all() and batch.terminated.all() and ~batch.truncated.any())
    # An interrupted streak and the terminal clearing telescope to zero discounted gain.
    dwell = torch.tensor([0., .2, .5, 0., .3, .7, 1.])
    obs, nxt = torch.zeros(6, 69), torch.zeros(6, 69)
    obs[:, 67], nxt[:, 67] = dwell[:-1], dwell[1:]
    terminated = torch.tensor([False]*5+[True])
    shaped = dwell_potential_reward(obs, nxt, terminated, .99, .5)
    assert abs(float((shaped*.99**torch.arange(6)).sum())) < 1e-7
    reports = []
    for scale in (0., .5):
        folder = args.output/('baseline' if not scale else 'potential')
        report = train(folder, PPOSettings(seed=1599, worlds=4, initial_std=.15), 2, 30.,
                       local=HoldingGoalSetup(args.state, scale))
        context = HoldingGoalContext(**report['local_context'])
        checks = {label: verify_rollout(folder, label, context) for label in ('rollout_1', 'rollout_2')}
        with np.load(folder/'rollout_2.npz') as z:
            assert z['done'].any() and not z['truncated'].any()
        try:
            read_goal_payload(folder/'last.pt', goal_contract(FreeBody()))
        except ValueError:
            pass
        else:
            raise AssertionError('Global loader accepted a holding checkpoint')
        reports.append(dict(scale=scale, transitions=report['launch_transitions'], rollouts=checks))
    write_json(args.output/'checks.json', dict(passed=True, engineering_transitions=1024, runs=reports))
    print('PASS: full near-field reset, original deadline, potential clearing, and PPO terminal evidence')


if __name__ == '__main__':
    main()
