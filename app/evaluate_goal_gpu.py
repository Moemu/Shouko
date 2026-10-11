"""Read-only CUDA backend check using the same deck and checkpoint as native evaluation."""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import time

import numpy as np
import torch

from .evaluate_goal import read_deck, file_sha256
from .free_body import ROOT
from .goal_checkpoint import goal_contract, load_goal_policy
from .gpu_goal import GPUGoalEnv


@torch.inference_mode()
def evaluate(checkpoint: Path, deck_path: Path, output: Path) -> dict:
    deck = read_deck(deck_path)
    torch.set_num_threads(1)
    env = GPUGoalEnv(len(deck['episodes']), 9001)
    policy = load_goal_policy(checkpoint, env.body, device='cuda')
    output.mkdir(parents=True, exist_ok=False)
    offsets = torch.tensor([e['target_offsets_m'] for e in deck['episodes']], device='cuda')
    targets = env.origin[:, None, :] + offsets
    index = torch.zeros(env.worlds, dtype=torch.long, device='cuda')
    active = torch.ones(env.worlds, dtype=torch.bool, device='cuda')
    env.target.copy_(targets[:, 0])
    env.previous_distance.copy_((env.target - env.center[:, :2]).norm(dim=1))
    progress = torch.zeros(env.worlds, device='cuda')
    work = torch.zeros_like(progress)
    speed_sum = torch.zeros_like(progress)
    min_distance = env.previous_distance.clone()
    steps = torch.zeros_like(index)
    arrivals = torch.zeros_like(index)
    max_penetration = torch.zeros_like(progress)
    max_dwell = torch.zeros_like(index)
    reasons = [''] * env.worlds
    traces = []
    started = time.perf_counter()
    while bool(active.any()):
        mean = policy(env.observation())[0]
        old_dwell = env.dwell_steps.clone()
        _, result = env.step(mean)
        if bool((active & result.numerical).any()):
            raise FloatingPointError('Numerical CUDA evaluation transition')
        distance = (targets[torch.arange(env.worlds, device='cuda'), index] - env.center[:, :2]).norm(dim=1)
        progress += torch.where(active, result.progress, 0.)
        work += torch.where(active, result.work_j, 0.)
        speed_sum += torch.where(active, env.velocity[:, :2].norm(dim=1), 0.)
        min_distance = torch.where(active & (index == 0), torch.minimum(min_distance, distance), min_distance)
        max_penetration = torch.where(active, torch.maximum(max_penetration, env.peak_penetration), max_penetration)
        measured_dwell = torch.where(result.reached, old_dwell + 1, env.dwell_steps)
        max_dwell = torch.where(active, torch.maximum(max_dwell, measured_dwell), max_dwell)
        steps += active.long()
        arrivals += (active & result.reached).long()
        ended = active & (result.terminated | result.truncated)
        for world in ended.nonzero().flatten().tolist():
            reasons[world] = ('complete' if result.complete[world] else 'timeout' if result.timeout[world]
                              else 'arena' if result.arena[world] else 'nonfinite')
        switched = active & result.reached & ~ended
        index += switched.long()
        if bool(switched.any()):
            env.target[switched] = targets[switched, index[switched]]
            env.previous_distance[switched] = (env.target[switched] - env.center[switched, :2]).norm(dim=1)
            env.dwell_steps[switched] = 0
            env.target_steps[switched] = 0
        if int(steps.max()) % 5 == 0 or bool(ended.any()):
            traces.append(torch.cat((env.center, active[:, None].float()), dim=1).cpu().numpy())
        active &= ~ended
        # Reset finished worlds through the same isolated reset path as training; ignore further metrics.
        done = result.terminated | result.truncated
        if bool(done.any()):
            env.reset(done)
    torch.cuda.synchronize()
    episodes = []
    for world, episode in enumerate(deck['episodes']):
        episodes.append(dict(id=episode['id'], arrivals=int(arrivals[world]), reason=reasons[world],
                             steps=int(steps[world]), net_progress_m=float(progress[world]),
                             first_target_min_distance_m=float(min_distance[world]), work_j=float(work[world]),
                             mean_speed_m_s=float(speed_sum[world] / steps[world]),
                             max_penetration_m=float(max_penetration[world]),
                             max_dwell_s=float(max_dwell[world]) * env.dt))
    np.savez_compressed(output / 'centers.npz', center_and_active=np.stack(traces))
    report = dict(schema='yumi_goal_backend_check', version=1, backend='mujoco_warp',
                  action_mode='deterministic_mean', checkpoint_sha256=file_sha256(checkpoint),
                  deck_sha256=file_sha256(deck_path), contract=asdict(goal_contract(env.body)),
                  episodes=episodes, arrivals=sum(row['arrivals'] for row in episodes),
                  complete_episodes=sum(row['reason'] == 'complete' for row in episodes),
                  termination_reasons=dict(Counter(reasons)), elapsed_seconds=time.perf_counter() - started,
                  mean_net_progress_m=float(progress.mean()), mean_work_j=float(work.mean()),
                  mean_first_target_min_distance_m=float(min_distance.mean()))
    sources = ('app/evaluate_goal_gpu.py', 'app/evaluate_goal.py', 'app/gpu_goal.py', 'app/goal_checkpoint.py')
    report['source_sha256'] = {name: file_sha256(ROOT / name) for name in sources}
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deck', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.deck, args.output)


if __name__ == '__main__':
    main()
