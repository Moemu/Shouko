"""Deterministic native/CUDA evaluation for the separate fallen-local task."""
import argparse
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .diagnose_goal_d0 import write_json
from .evaluate_goal import file_sha256
from .free_body import FreeBody, ROOT
from .local_goal import LocalGoalTask, LocalGoalEnv, LocalGoalSetup, load_local_policy, action_distribution


@torch.inference_mode()
def evaluate(checkpoint: Path, state_path: Path, output: Path, backend: str) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    policy, context = load_local_policy(checkpoint, state_path, 'cpu' if backend == 'native' else 'cuda')
    body = FreeBody()
    if backend == 'native':
        task = LocalGoalTask(body, state_path)
        observation = torch.from_numpy(task.observation())[None]
        origin, target = task.origin.copy(), task.target.copy()
        centers, positions, velocities = [body.center], [body.data.qpos.copy()], [body.data.qvel.copy()]
    else:
        env = LocalGoalEnv(1, 0, LocalGoalSetup(state_path, context.rho))
        observation = env.observation()
        origin, target = env.origin[0].cpu().numpy().copy(), env.target[0].cpu().numpy().copy()
        centers, positions, velocities = [env.center[0].cpu().numpy().copy()], [env.qpos[0].cpu().numpy().copy()], [env.qvel[0].cpu().numpy().copy()]
    observations = [observation[0].cpu().numpy().copy()]
    actions, means, executed = [], [], []
    works, progresses, rewards, penetrations, limits = [], [], [], [], []
    ground_bodies = set()
    reason, reached = '', False
    zero_log_std = torch.zeros(18, device=observation.device)
    for _ in range(250):
        mu, _ = policy(observation)
        raw = action_distribution(mu, zero_log_std, observation, context.rho).mean
        actions.append(raw[0].cpu().numpy().copy()); means.append(mu[0].cpu().numpy().copy())
        if backend == 'native':
            next_obs, result = task.step(raw[0].numpy())
            observation = torch.from_numpy(next_obs)[None]
            centers.append(body.center); positions.append(body.data.qpos.copy()); velocities.append(body.data.qvel.copy())
            executed.append(body.action.copy())
            work, progress, reward = result.work_j, result.progress, result.reward
            penetration, limit = body.last_contacts.penetration, body.last_contacts.limit_error
            ground_bodies.update(body.last_contacts.ground_bodies)
            done, reached = task.done, result.reached
            reason = result.reason
        else:
            observation, result = env.step(raw)
            centers.append(env.center[0].cpu().numpy().copy()); positions.append(env.qpos[0].cpu().numpy().copy()); velocities.append(env.qvel[0].cpu().numpy().copy())
            executed.append(env.actions[0].cpu().numpy().copy())
            work, progress, reward = float(result.work_j[0]), float(result.progress[0]), float(result.reward[0])
            q = env.qpos[0, env.qadr].cpu().numpy()
            penetration = float(env.peak_penetration[0])
            limit = max(0., float(np.maximum(body.ranges[:, 0]-q, q-body.ranges[:, 1]).max()))
            done, reached = bool(env.done[0]), bool(result.reached[0])
            reason = ('numerical' if bool(result.numerical[0]) else 'complete' if reached else
                      'timeout' if bool(result.timeout[0]) else 'arena' if bool(result.arena[0]) else '')
        observations.append(observation[0].cpu().numpy().copy())
        works.append(work); progresses.append(progress); rewards.append(reward)
        penetrations.append(penetration); limits.append(limit)
        if penetration > .030 or limit > .15:
            reason = 'physical_guard'; break
        if done:
            break
    trace = dict(time=np.arange(len(centers)) * .02, observation=np.asarray(observations), mean=np.asarray(means),
                 raw_action=np.asarray(actions), executed_action=np.asarray(executed), center=np.asarray(centers),
                 qpos=np.asarray(positions), qvel=np.asarray(velocities), work_j=np.asarray(works),
                 progress=np.asarray(progresses), reward=np.asarray(rewards), penetration=np.asarray(penetrations),
                 joint_limit_error=np.asarray(limits), target=target, origin=origin)
    np.savez_compressed(output / 'trace.npz', **trace)
    distance = np.linalg.norm(trace['center'][:, :2]-target, axis=1)
    def remaining_progress(seconds: float) -> float:
        i = min(round(seconds / .02), len(distance)-1)
        return float(distance[i]-distance[-1])
    report = dict(schema='local_goal_evaluation', backend=backend, local_context=asdict(context),
                  checkpoint_sha256=file_sha256(checkpoint), state_sha256=file_sha256(state_path),
                  trace_sha256=file_sha256(output / 'trace.npz'), duration_s=float(trace['time'][-1]),
                  reason=reason, reached=reached, safe=reason in ('timeout', 'complete'),
                  net_progress_m=float(distance[0]-distance[-1]), after_1s_progress_m=remaining_progress(1.),
                  after_3s_progress_m=remaining_progress(3.), minimum_distance_m=float(distance.min()),
                  work_j=float(sum(works)), common_reward=float(sum(rewards)), ground_bodies=sorted(ground_bodies),
                  max_penetration_m=max(penetrations), max_joint_limit_error_rad=max(limits),
                  source_sha256={name:file_sha256(ROOT/name) for name in (
                      'app/evaluate_local_goal.py', 'app/local_goal.py', 'app/free_body.py', 'app/gpu_goal.py',
                      'app/goal_task.py', 'app/goal_checkpoint.py', 'app/mlp_policy.py')})
    write_json(output / 'summary.json', report)
    return report


def passes_gate(final: dict, initial: dict) -> bool:
    return bool(final['safe'] and final['net_progress_m'] >= .25
                and final['net_progress_m']-initial['net_progress_m'] >= .15
                and (final['reached'] or (final['after_1s_progress_m'] >= .20 and final['after_3s_progress_m'] >= .05)))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('native', 'cuda'), default='native')
    args = parser.parse_args()
    evaluate(args.checkpoint, args.state, args.output, args.backend)


if __name__ == '__main__':
    main()
