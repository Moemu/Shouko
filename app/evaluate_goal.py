"""Native MuJoCo evaluation of goal policies on a fixed, shared target deck."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .free_body import FreeBody, ROOT
from .goal_checkpoint import goal_contract, load_goal_policy
from .goal_task import GoalTask
from .mlp_policy import MLPPolicy


def file_sha256(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def fixed_deck() -> dict:
    episodes = []
    for index in range(16):
        angle = 2 * np.pi * index / 16
        direction = 1 if index % 2 == 0 else -1
        angles = angle + direction * 2 * np.pi / 3 * np.arange(3)
        targets = 2 * np.stack((np.cos(angles), np.sin(angles)), axis=1)
        episodes.append(dict(id=index, target_offsets_m=targets.tolist()))
    return dict(schema='yumi_goal_deck', version=1, episodes=episodes)


def read_deck(path: Path) -> dict:
    deck = json.loads(path.read_text(encoding='utf-8'))
    if deck.get('schema') != 'yumi_goal_deck' or deck.get('version') != 1 or not deck.get('episodes'):
        raise ValueError('Expected a nonempty version 1 goal deck')
    ids = set()
    for episode in deck['episodes']:
        offsets = np.asarray(episode['target_offsets_m'], dtype=float)
        if episode['id'] in ids:
            raise ValueError('Duplicate goal episode ID')
        ids.add(episode['id'])
        if offsets.shape != (3, 2) or not np.isfinite(offsets).all():
            raise ValueError('Expected three finite XY targets')
        if (not np.allclose(np.linalg.norm(offsets, axis=1), 2., atol=1e-8, rtol=0)
                or np.any(np.linalg.norm(np.diff(offsets, axis=0), axis=1) < 2 + GoalTask.radius)):
            raise ValueError('Goal deck violates target radius or switch distance')
    return deck


def assign_target(task: GoalTask, target: np.ndarray) -> None:
    task.target = target.copy()
    task.previous_distance = float(np.linalg.norm(target - task.body.center[:2]))
    task.dwell = task.target_elapsed = 0.


@torch.inference_mode()
def evaluate_episode(task: GoalTask, policy: MLPPolicy, episode: dict) -> tuple[dict, dict[str, np.ndarray]]:
    task.reset()
    body = task.body
    targets = task.origin + np.asarray(episode['target_offsets_m'])
    assign_target(task, targets[0])
    records = []
    trace = [np.r_[0., 0., body.center, body.velocity, task.target, 0., 0., body.data.qpos, body.action]]
    speed_sum = path_length = total_work = reward = 0.
    max_speed = max_penetration = max_limit_error = 0.
    max_contacts = max_constraints = steps = 0
    min_height = float(body.center[2])
    ground_bodies = set()
    target_index = 0

    def new_record() -> dict:
        distance = float(np.linalg.norm(task.target - body.center[:2]))
        return dict(index=target_index, target_m=task.target.tolist(), initial_distance_m=distance,
                    final_distance_m=distance, minimum_distance_m=distance, net_progress_m=0.,
                    duration_s=0., eligible_seconds=0., max_continuous_dwell_s=0., work_j=0., reached=False)

    record = new_record()
    while not task.done:
        observation = torch.from_numpy(task.observation()).unsqueeze(0)
        action = policy(observation)[0][0].numpy()
        old_center = body.center
        old_dwell = task.dwell
        active_target = task.target.copy()
        _, result = task.step(action)
        center, velocity = body.center, body.velocity
        if result.reason == 'nonfinite':
            raise FloatingPointError('Non-finite native evaluation transition')
        speed = float(np.linalg.norm(velocity[:2]))
        min_height = min(min_height, float(center[2]))
        distance = float(np.linalg.norm(active_target - center[:2]))
        eligible = distance <= task.radius and speed <= task.speed_limit
        dwell = old_dwell + body.dt if eligible else 0.
        record['duration_s'] += body.dt
        record['eligible_seconds'] += body.dt if eligible else 0.
        record['max_continuous_dwell_s'] = max(record['max_continuous_dwell_s'], dwell)
        record['minimum_distance_m'] = min(record['minimum_distance_m'], distance)
        record['final_distance_m'] = distance
        record['net_progress_m'] += result.progress
        record['work_j'] += result.work_j
        record['reached'] = result.reached
        speed_sum += speed
        max_speed = max(max_speed, speed)
        path_length += float(np.linalg.norm(center[:2] - old_center[:2]))
        total_work += result.work_j
        reward += result.reward
        contact = body.last_contacts
        max_penetration = max(max_penetration, contact.penetration)
        max_limit_error = max(max_limit_error, contact.limit_error)
        max_contacts = max(max_contacts, contact.count)
        max_constraints = max(max_constraints, contact.constraints)
        ground_bodies.update(contact.ground_bodies)
        steps += 1
        if steps % 5 == 0 or task.done or result.reached:
            trace.append(np.r_[steps * body.dt, target_index, center, velocity, active_target,
                               result.work_j, contact.penetration, body.data.qpos, body.action])
        if result.reached or task.done:
            records.append(record)
        if result.reached and not task.done:
            target_index += 1
            assign_target(task, targets[target_index])
            record = new_record()
    report = dict(id=episode['id'], arrivals=task.completed, complete=result.reason == 'complete',
                  reason=result.reason, steps=steps, duration_s=steps * body.dt, reward=reward,
                  net_progress_m=sum(row['net_progress_m'] for row in records),
                  path_length_m=path_length, work_j=total_work, mean_speed_m_s=speed_sum / steps,
                  max_speed_m_s=max_speed, max_penetration_m=max_penetration,
                  max_joint_limit_error_rad=max_limit_error, max_contacts=max_contacts,
                  max_constraints=max_constraints, ground_bodies=sorted(ground_bodies), targets=records)
    report.update(min_com_height_m=min_height, final_com_m=body.center.tolist())
    return report, {f"episode_{episode['id']}": np.asarray(trace)}


def evaluate(checkpoint: Path, deck_path: Path, output: Path) -> dict:
    deck = read_deck(deck_path)
    body = FreeBody()
    torch.set_num_threads(1)
    policy = load_goal_policy(checkpoint, body)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    episodes, traces = [], {}
    task = GoalTask(body)
    for episode in deck['episodes']:
        row, trace = evaluate_episode(task, policy, episode)
        episodes.append(row)
        traces.update(trace)
        print(json.dumps(row, allow_nan=False), flush=True)
    np.savez_compressed(output / 'trajectories.npz', **traces)
    report = dict(schema='yumi_goal_evaluation', version=1, backend='native_mujoco',
                  action_mode='deterministic_mean', checkpoint_sha256=file_sha256(checkpoint),
                  deck_sha256=file_sha256(deck_path), contract=asdict(goal_contract(body)),
                  episodes=episodes, episode_count=len(episodes), arrivals=sum(row['arrivals'] for row in episodes),
                  complete_episodes=sum(row['complete'] for row in episodes),
                  first_target_arrivals=sum(row['targets'][0]['reached'] for row in episodes),
                  attempted_targets=sum(len(row['targets']) for row in episodes),
                  scheduled_targets=3 * len(episodes),
                  termination_reasons=dict(Counter(row['reason'] for row in episodes)),
                  mean_net_progress_m=float(np.mean([row['net_progress_m'] for row in episodes])),
                  mean_first_target_min_distance_m=float(np.mean([row['targets'][0]['minimum_distance_m'] for row in episodes])),
                  mean_work_j=float(np.mean([row['work_j'] for row in episodes])),
                  mean_speed_m_s=float(np.mean([row['mean_speed_m_s'] for row in episodes])),
                  max_penetration_m=max(row['max_penetration_m'] for row in episodes),
                  elapsed_seconds=time.perf_counter() - started,
                  trace_columns=['time_s', 'target_index', 'com_xyz:3', 'velocity_xyz:3', 'target_xy:2',
                                 'control_step_work_j', 'control_step_peak_penetration_m',
                                 f'qpos:{body.model.nq}', 'action:18'],
                  trace_sampling='Every fifth control step (10 Hz), plus arrival and terminal steps')
    sources = ('app/evaluate_goal.py', 'app/free_body.py', 'app/goal_task.py', 'app/goal_checkpoint.py', 'app/mlp_policy.py')
    report['source_sha256'] = {name: file_sha256(ROOT / name) for name in sources}
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deck', type=Path, required=True)
    parser.add_argument('--create-deck', action='store_true')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.create_deck:
        args.deck.parent.mkdir(parents=True, exist_ok=True)
        with args.deck.open('x', encoding='utf-8') as handle:
            handle.write(json.dumps(fixed_deck(), indent=2) + '\n')
        print(file_sha256(args.deck))
    else:
        if args.checkpoint is None or args.output is None:
            parser.error('Evaluation requires --checkpoint and --output')
        evaluate(args.checkpoint, args.deck, args.output)


if __name__ == '__main__':
    main()
