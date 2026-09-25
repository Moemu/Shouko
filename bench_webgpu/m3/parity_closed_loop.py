"""M3 closed-loop parity reference: seed 4242, 500 control steps, CPU connectome.

Drives the same loop as app/studio_server.py (speed 0.5, target yaw 0, intact)
and records per-step joint actions and qpos for browser-side comparison.
"""
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from app.sim import Body
from app.full_brain import ConnectomePolicy, checkpoint_configuration

STEPS = 500
SPEED = 0.5
TARGET_YAW = 0.0


def main():
    checkpoint_path = ROOT / 'runs/yumi/best.pt'
    configuration = checkpoint_configuration(checkpoint_path)
    body = Body(load_motor_policy=False, robot='yumi',
                interface=configuration['physics_interface'],
                observation_size=configuration['observation_size'])
    brain = ConnectomePolicy(device='cpu', observation_size=configuration['observation_size'],
                             action_size=configuration['action_size'],
                             neural_steps=configuration['neural_steps']).eval()
    extra = brain.load(checkpoint_path, interface=body.interface)
    print('checkpoint extra:', json.dumps(extra, default=str)[:200], flush=True)

    body.reset(4242)
    initial_qvel = [float(v) for v in body.data.qvel[:2]]
    print(f'initial qvel[:2] = {initial_qvel}', flush=True)

    records = []
    start = time.perf_counter()
    for k in range(STEPS):
        gravity, yaw = body.observation()
        error = np.arctan2(np.sin(TARGET_YAW - yaw), np.cos(TARGET_YAW - yaw))
        obs = body.motor_observation([SPEED, 0, np.clip(error * 1.4, -0.2, 0.2)])
        with torch.inference_mode():
            action, _ = brain(torch.tensor(obs, device='cpu')[None], lesion=False)
        joint_action = action[0].numpy()
        state = body.step_joints(joint_action)
        records.append(dict(k=k, action=[float(v) for v in joint_action],
                            qpos=[float(v) for v in body.data.qpos]))
        if (k + 1) % 100 == 0:
            print(f'step {k + 1}/{STEPS} height={state["height"]:.4f} '
                  f'fallen={state["fallen"]} elapsed={time.perf_counter() - start:.1f}s', flush=True)

    final = records[-1]
    out = dict(
        steps=STEPS, seed=4242, speed=SPEED, target_yaw=TARGET_YAW,
        initial_qvel=initial_qvel,
        final_height=body.data.qpos[2], final_time=body.data.time,
        fallen=bool(body.data.qpos[2] < body.fall_height),
        distance=body.distance,
        records=records)
    target = Path(__file__).parent / 'parity_closed_loop.json'
    target.write_text(json.dumps(out))
    print(f'wrote {target} ({target.stat().st_size / 1e6:.1f} MB) in {time.perf_counter() - start:.1f}s', flush=True)
    print(f'final: height={out["final_height"]:.4f} fallen={out["fallen"]} distance={out["distance"]:.3f}', flush=True)


if __name__ == '__main__':
    main()
