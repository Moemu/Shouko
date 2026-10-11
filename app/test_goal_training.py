"""Local native/CUDA parity, goal boundaries, checkpoint restore, and GAE checks."""
import argparse
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import warp as wp

from .free_body import FreeBody, ROOT
from .goal_task import GoalTask, OBSERVATION_SIZE
from .goal_checkpoint import goal_contract, save_goal_checkpoint, load_goal_checkpoint, load_goal_policy
from .mlp_policy import MLPPolicy
from .ppo_yumi import Value
from .recovery_curriculum import bootstrap_timeouts
from .value_diagnostics import gae_targets


def training_parts(device: str = 'cpu') -> tuple:
    policy = MLPPolicy(OBSERVATION_SIZE, 18, device=device)
    value = Value(OBSERVATION_SIZE).to(device)
    log_std = torch.nn.Parameter(torch.full((18,), -1., device=device))
    optimizer = torch.optim.Adam([*policy.parameters(), log_std], lr=3e-4, foreach=False)
    value_optimizer = torch.optim.Adam(value.parameters(), lr=1e-3, foreach=False)
    return policy, value, log_std, optimizer, value_optimizer


def check_checkpoint() -> dict:
    torch.manual_seed(11)
    body = FreeBody()
    contract = goal_contract(body)
    policy, value, log_std, optimizer, value_optimizer = training_parts()
    observations = torch.randn(5, OBSERVATION_SIZE)

    def update(parts):
        actor, critic, noise, opt, vopt = parts
        opt.zero_grad(set_to_none=True)
        (actor(observations)[0].square().mean() + noise.square().mean()).backward()
        opt.step()
        vopt.zero_grad(set_to_none=True)
        critic(observations).square().mean().backward()
        vopt.step()

    original = policy, value, log_std, optimizer, value_optimizer
    update(original)
    settings = dict(gamma=.99)
    with TemporaryDirectory() as directory:
        path = Path(directory) / 'goal.pt'
        digest = save_goal_checkpoint(path, contract=contract, policy=policy, value=value, log_std=log_std,
                                      optimizer=optimizer, value_optimizer=value_optimizer,
                                      iteration=1, env_steps=80, settings=settings)
        restored = training_parts()
        evaluation_policy = load_goal_policy(path, body)
        torch.testing.assert_close(policy(observations)[0], evaluation_policy(observations)[0], rtol=0, atol=0)
        counters = load_goal_checkpoint(path, contract=contract, policy=restored[0], value=restored[1],
                                       log_std=restored[2], optimizer=restored[3], value_optimizer=restored[4], settings=settings)
        assert counters == (1, 80)
        torch.testing.assert_close(policy(observations)[0], restored[0](observations)[0], rtol=0, atol=0)
        update(original)
        update(restored)
        for left, right in zip(policy.parameters(), restored[0].parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        for left, right in zip(value.parameters(), restored[1].parameters()):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        torch.testing.assert_close(log_std, restored[2], rtol=0, atol=0)
        mismatches = dict(body=replace(contract, body_sha256='0' * 64),
                          graph=replace(contract, graph_sha256='0' * 64),
                          joints=replace(contract, joint_names=tuple(reversed(contract.joint_names))),
                          observation=replace(contract, observation_fields=tuple(reversed(contract.observation_fields))),
                          control=replace(contract, action_scale=.5),
                          task=replace(contract, task=dict(contract.task, radius=.4)))
        for name, candidate in mismatches.items():
            try:
                load_goal_checkpoint(path, contract=candidate, policy=policy, value=value, log_std=log_std,
                                     optimizer=optimizer, value_optimizer=value_optimizer, settings=settings)
            except ValueError:
                pass
            else:
                raise AssertionError(f'Mismatched {name} accepted')
        damaged_body = FreeBody()
        damaged_body.action_scale = .5
        try:
            load_goal_policy(path, damaged_body)
        except ValueError:
            pass
        else:
            raise AssertionError('Evaluation accepted mismatched control contract')
        damaged = torch.load(path, weights_only=True)
        damaged['optimizer_names'][0] = list(reversed(damaged['optimizer_names'][0]))
        changed = Path(directory) / 'reordered.pt'
        torch.save(damaged, changed)
        try:
            load_goal_checkpoint(changed, contract=contract, policy=policy, value=value, log_std=log_std,
                                 optimizer=optimizer, value_optimizer=value_optimizer, settings=settings)
        except ValueError:
            pass
        else:
            raise AssertionError('Reordered optimizer parameters accepted')
        legacy = Path(directory) / 'legacy.pt'
        policy.save(legacy)
        try:
            load_goal_checkpoint(legacy, contract=contract, policy=policy, value=value, log_std=log_std,
                                 optimizer=optimizer, value_optimizer=value_optimizer, settings=settings)
        except ValueError:
            pass
        else:
            raise AssertionError('Legacy checkpoint accepted as goal-PPO')
    return dict(exact_adam_next_update=True, restored_counters=list(counters),
                mismatches_rejected=list(mismatches) + ['optimizer_order', 'legacy_format'], checkpoint_sha256=digest)


def check_gae() -> dict:
    # The next reset state's value (99) must not enter either terminal target.
    rewards = torch.tensor([[1., 1.], [2., 2.]])
    truncated = torch.tensor([True, False])
    rewards[0] = bootstrap_timeouts(rewards[0], torch.tensor([5., 5.]), truncated, .9)
    advantages, returns = gae_targets(rewards, torch.tensor([[3., 3.], [99., 99.], [4., 4.]]),
                                     torch.tensor([[1., 1.], [0., 0.]]), .9, .95)
    torch.testing.assert_close(returns[0], torch.tensor([5.5, 1.]))
    assert torch.isfinite(advantages).all()
    return dict(timeout_return=float(returns[0, 0]), terminal_return=float(returns[0, 1]), reset_value=99.)


def check_gpu() -> dict:
    from .gpu_goal import GPUGoalEnv
    from .test_free_movement import POSES, place_pose
    env = GPUGoalEnv(4, seed=11)
    natives = [GoalTask(FreeBody(), seed=11) for _ in range(4)]
    rng = np.random.default_rng(11)
    observation_error = 0.
    for index, (task, quaternion) in enumerate(zip(natives, POSES.values())):
        place_pose(task.body, quaternion)
        task.body.data.qvel[:] = rng.uniform(-.2, .2, task.body.model.nv)
        task.body.action[:] = rng.uniform(-.2, .2, 18)
        task.body.refresh()
        task.target = task.body.center[:2] + [1.5, -.6]
        task.dwell, task.target_elapsed = .3, 4.
        env.qpos[index] = torch.tensor(task.body.data.qpos, device='cuda')
        env.qvel[index] = torch.tensor(task.body.data.qvel, device='cuda')
        env.actions[index] = torch.tensor(task.body.action, device='cuda')
        env.target[index] = torch.tensor(task.target, device='cuda')
        env.dwell_steps[index], env.target_steps[index] = 15, 200
    env.refresh()
    for index, task in enumerate(natives):
        error = float(np.max(np.abs(task.observation() - env.observation()[index].cpu().numpy())))
        observation_error = max(observation_error, error)
    assert observation_error < 4e-6, observation_error

    # Identical initial states and fixed actions: compare one control step, including applied work.
    env.reset(torch.ones(4, device='cuda', dtype=torch.bool))
    for index, (task, quaternion) in enumerate(zip(natives, POSES.values())):
        task.reset()
        place_pose(task.body, quaternion)
        env.qpos[index] = torch.tensor(task.body.data.qpos, device='cuda')
    env.refresh()
    env.origin.copy_(env.center[:, :2])
    env.target.copy_(env.origin + torch.tensor([2., 0.], device='cuda'))
    env.previous_distance.fill_(2.)
    actions = rng.uniform(-8., 8., (4, 18)).astype(np.float32)
    actions[:, [14, 17]] = -8.  # Elbow targets must clamp to zero rather than below the lower limit.
    _, result = env.step(torch.tensor(actions, device='cuda'))
    position_error = velocity_error = work_error = 0.
    for index, task in enumerate(natives):
        work = task.body.step(actions[index])
        position_error = max(position_error, float(np.max(np.abs(task.body.data.qpos - env.qpos[index].cpu().numpy()))))
        velocity_error = max(velocity_error, float(np.max(np.abs(task.body.data.qvel - env.qvel[index].cpu().numpy()))))
        work_error = max(work_error, abs(work - float(result.work_j[index])))
    assert position_error < 1e-3 and velocity_error < .02 and work_error < .02, (position_error, velocity_error, work_error)

    # Reset one world and preserve the others' live trajectory and task state.
    saved = [tensor[1:].clone() for tensor in (env.qpos, env.qvel, env.target, env.target_steps, env.previous_distance)]
    mask = torch.tensor([True, False, False, False], device='cuda')
    env.reset(mask)
    for before, after in zip(saved, (env.qpos, env.qvel, env.target, env.target_steps, env.previous_distance)):
        torch.testing.assert_close(before, after[1:], rtol=0, atol=0)
    assert not env.qvel[0].any() and not env.actions[0].any()
    assert not wp.to_torch(env.data.qacc_warmstart)[0].any()

    all_worlds = torch.ones(4, device='cuda', dtype=torch.bool)
    env.reset(all_worlds)
    center = env.center.clone()
    env.target.copy_(center[:, :2] + torch.tensor([2., 0.], device='cuda'))
    env.previous_distance.fill_(2.)
    toward = center + torch.tensor([.1, 0., 0.], device='cuda')
    forward = env.evaluate(toward, torch.zeros_like(center), torch.zeros(4, device='cuda'))
    reverse = env.evaluate(center, torch.zeros_like(center), torch.zeros(4, device='cuda'))
    torch.testing.assert_close(forward.progress, -reverse.progress)
    assert torch.allclose(forward.progress, torch.full((4,), .1, device='cuda'), atol=1e-6)
    env.reset(all_worlds)
    center = env.center.clone()
    env.target.copy_(center[:, :2])
    env.previous_distance.zero_()
    slow = torch.zeros_like(center)
    zero_work = torch.zeros(4, device='cuda')
    for _ in range(49):
        assert not env.evaluate(center, slow, zero_work).reached.any()
    arrived = env.evaluate(center, slow, zero_work)
    assert arrived.reached.all() and (env.completed == 1).all() and not env.done.any()
    switched = env.evaluate(center, slow, zero_work)
    assert switched.progress.abs().max() < 1e-6 and not switched.reached.any()
    # Interrupt a partial dwell by exceeding the speed limit; fifty fresh eligible steps are required.
    env.target.copy_(center[:, :2])
    env.previous_distance.zero_()
    env.dwell_steps.fill_(25)
    fast = slow.clone()
    fast[:, 0] = .21
    env.evaluate(center, fast, zero_work)
    assert not env.dwell_steps.any()
    for _ in range(49):
        assert not env.evaluate(center, slow, zero_work).reached.any()
    assert env.evaluate(center, slow, zero_work).reached.all()
    env.target.copy_(center[:, :2])
    env.previous_distance.zero_()
    env.completed.fill_(2)
    env.dwell_steps.fill_(49)
    final = env.evaluate(center, slow, zero_work)
    assert final.complete.all() and final.terminated.all() and not final.truncated.any()

    env.reset(all_worlds)
    center = env.center.clone()
    env.target_steps[0] = env.target_limit - 1
    center[1, 0] += 10.1
    center[2, 2] = float('nan')
    center[3, 2] = .05
    boundary = env.evaluate(center, slow, zero_work)
    assert boundary.timeout[0] and boundary.arena[1] and boundary.numerical[2] and not env.done[3]
    assert boundary.truncated[:2].all() and boundary.terminated[2] and boundary.reward[2] == 0
    # Compare the scalar native rules to the batched rules on the same measured inputs.
    env.reset(all_worlds)
    center = env.center.clone()
    env.target.copy_(center[:, :2] + torch.tensor([2., 0.], device='cuda'))
    env.previous_distance.fill_(2.)
    env.target_steps[3] = env.target_limit - 1
    measured = center.clone()
    measured[0, 0] += .1
    measured[1, 0] -= .1
    measured[2, 2] = .05
    batch = env.evaluate(measured, slow, torch.full((4,), 5., device='cuda'))
    reward_error = 0.
    for index in range(4):
        task = GoalTask(FreeBody())
        task.origin = env.origin[index].cpu().numpy().astype(float)
        task.target = env.target[index].cpu().numpy().astype(float)
        task.previous_distance = 2.
        task.target_elapsed = GoalTask.target_seconds - env.dt if index == 3 else 0.
        scalar = task.evaluate(measured[index].cpu().numpy().astype(float), np.zeros(3), 5.)
        reward_error = max(reward_error, abs(scalar.reward - float(batch.reward[index])))
        assert scalar.terminated == bool(batch.terminated[index]) and scalar.truncated == bool(batch.truncated[index])
    assert reward_error < 1e-6, reward_error
    for bad in (torch.zeros(4, 12, device='cuda'), torch.full((4, 18), float('nan'), device='cuda')):
        try:
            env.step(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid CUDA action accepted')
    return dict(worlds=4, max_observation_error=observation_error, max_position_error=position_error,
                max_velocity_error=velocity_error, max_work_error_j=work_error,
                named_pd_target_limits=True, partial_reset_isolation=True,
                arrival_steps=50, switch_progress_max=float(switched.progress.abs().max()),
                native_batched_reward_error=reward_error, interrupted_dwell_restarts=True,
                timeout_arena_numerical_low_pose=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu', action='store_true')
    args = parser.parse_args()
    results = dict(checkpoint=check_checkpoint(), gae=check_gae())
    if args.gpu:
        results['gpu'] = check_gpu()
    sources = ('app/free_body.py', 'app/goal_task.py', 'app/gpu_goal.py', 'app/goal_checkpoint.py', 'app/test_goal_training.py')
    results['source_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'contracts.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(results, indent=2))
    print('PASS: goal training contracts')


if __name__ == '__main__':
    main()
