"""Held-out, teacher-free walking checks in native MuJoCo used by the live preview."""
import argparse
import hashlib
import json
import platform
import time

import numpy as np
import mujoco
import torch
import psutil

from .full_brain import ConnectomePolicy, ROOT
from .sim import Body


@torch.inference_mode()
def episode(model, body, seed, speed, seconds, lesion=False, push=False, heading_control=True):
    body.reset(seed)
    minimum_height, minimum_upright = 10.0, 1.0
    contacts = np.zeros(2, dtype=int)
    alternating = 0
    previous_side = None
    previous_contacts = [False, False]
    started = time.perf_counter()
    frames = []
    neural_ms = []
    for step in range(round(seconds / 0.02)):
        if push and step == 500:
            body.data.qvel[1] += 0.25
        _, yaw = body.observation()
        yaw_rate = float(np.clip(-yaw * 1.4, -0.2, 0.2)) if heading_control else 0.0
        observation = body.motor_observation([speed, 0, yaw_rate])
        neural_start = time.perf_counter()
        action, _ = model(torch.as_tensor(observation, device=model.weight.device)[None], lesion=lesion)
        joint_action = action[0].cpu().numpy()
        neural_ms.append((time.perf_counter() - neural_start) * 1000)
        state = body.step_joints(joint_action)
        minimum_height = min(minimum_height, state['height'])
        minimum_upright = min(minimum_upright, state['upright'])
        contacts += state['contacts']
        for side in range(2):
            if state['contacts'][side] and not previous_contacts[side]:
                alternating += int(previous_side is not None and previous_side != side)
                previous_side = side
        previous_contacts = state['contacts'].copy()
        if step % 5 == 0:
            frames.append(state)
        if state['fallen']:
            break
    duration = state['time']
    mean_speed = state['qpos'][0] / duration
    lateral_m = abs(state['qpos'][1])
    criteria = dict(survived=not state['fallen'] and duration >= seconds - 0.01,
                    speed=abs(mean_speed-speed) <= 0.15,
                    both_feet=min(state['foot_strikes']) >= int(seconds * 0.5),
                    alternation=alternating >= int(seconds),
                    upright=minimum_upright >= 0.7,
                    direction=lateral_m <= max(1, seconds * speed * 0.2))
    result = dict(seed=seed, target_speed=speed, requested_seconds=seconds, seconds=duration,
                  lesion='disconnected' if lesion else 'intact', push=push,
                  mean_speed=mean_speed, forward_m=state['qpos'][0], lateral_m=lateral_m,
                  fallen=state['fallen'], foot_strikes=state['foot_strikes'], alternations=alternating,
                  contact_fraction=(contacts / (step+1)).tolist(), minimum_height=minimum_height,
                  minimum_upright=minimum_upright, criteria=criteria, success=all(criteria.values()),
                  inference_ms_median=float(np.median(neural_ms)),
                  inference_ms_p95=float(np.percentile(neural_ms, 95)),
                  wall_seconds=time.perf_counter()-started)
    return result, frames


def main(args):
    torch.set_num_threads(4)
    checkpoint = ROOT / args.checkpoint
    with checkpoint.open('rb') as handle:
        checkpoint_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    model = ConnectomePolicy(device=args.device).eval()
    extra = model.load(checkpoint)
    body = Body(load_motor_policy=False, robot=args.robot)
    assert body.policy is None
    report = dict(kind='held_out_native_mujoco', robot=args.robot, checkpoint_sha256=checkpoint_hash,
                  graph_sha256=model.meta['graph_sha256'], checkpoint_updates=extra.get('updates'),
                  teacher_used=False, tests=[], controls=[], perturbations=[], long_walks=[],
                  runtime=dict(device=str(model.weight.device), torch=torch.__version__,
                               platform=platform.platform(), mujoco=mujoco.__version__,
                               gpu=torch.cuda.get_device_name() if args.device == 'cuda' else None),
                  heading_feedback='Clipped proportional target-yaw error supplied as a sensory command; no joint control outside the learned policy.',
                  successes=0, attempts=0, complete=False,
                  criterion='No fall; speed error <= 0.15 m/s; >= 0.5 strikes/s per foot; >= 1 alternating strike/s; upright >= 0.7; lateral deviation <= max(1m, 20% target forward distance).')
    target = ROOT / args.output
    target.parent.mkdir(parents=True, exist_ok=True)
    def save():
        temporary = target.with_suffix('.tmp')
        temporary.write_text(json.dumps(report, allow_nan=False, indent=2))
        temporary.replace(target)
    cases = [('tests', 8001+i, speed, args.seconds, False, False)
             for i, speed in enumerate([0.35, 0.5, 0.65] * 3)]
    if not args.quick:
        cases += [('long_walks', 8101, 0.5, 120, False, False)]
        cases += [('perturbations', 8201+i, 0.5, 30, False, True) for i in range(3)]
        cases += [('controls', 8001+i, speed, args.seconds, True, False)
                  for i, speed in enumerate([0.35, 0.5, 0.65])]
    for category, seed, speed, seconds, lesion, push in cases:
        result, frames = episode(model, body, seed, speed, seconds, lesion, push)
        report[category].append(result)
        report['successes'] = sum(row['success'] for row in report['tests'])
        report['attempts'] = len(report['tests'])
        save()
        print(json.dumps(dict(category=category, **result)), flush=True)
        if category == 'long_walks':
            target.with_name('heldout_replay.json').write_text(json.dumps(dict(meta=result, frames=frames)))
    report['complete'] = True
    report['runtime'].update(process_rss_mb=psutil.Process().memory_info().rss / 2**20,
                             peak_tensor_vram_mb=torch.cuda.max_memory_allocated() / 2**20 if args.device == 'cuda' else 0)
    save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='runs/cloud/best.pt')
    parser.add_argument('--output', default='runs/cloud/heldout.json')
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--robot', choices=['g1', 'yumi'], default='g1')
    main(parser.parse_args())
