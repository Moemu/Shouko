"""Sensitivity probe: does Python's own closed loop diverge past 1e-3 when the
action carries the same ~4e-5 noise measured between WebGPU and torch?

Runs the parity_closed_loop reference trajectory against a noise-perturbed
run (uniform +-5e-5 on every action element, i.i.d. per step) and reports
the qpos divergence curve. 200 steps cover the browser fall window.
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

STEPS = 200
SPEED = 0.5
TARGET_YAW = 0.0
NOISE = 5e-5


def rollout(brain, body, noise_rng=None):
    body.reset(4242)
    records = []
    for k in range(STEPS):
        gravity, yaw = body.observation()
        error = np.arctan2(np.sin(TARGET_YAW - yaw), np.cos(TARGET_YAW - yaw))
        obs = body.motor_observation([SPEED, 0, np.clip(error * 1.4, -0.2, 0.2)])
        with torch.inference_mode():
            action, _ = brain(torch.tensor(obs, device='cpu')[None], lesion=False)
        joint_action = action[0].numpy()
        if noise_rng is not None:
            joint_action = joint_action + noise_rng.uniform(-NOISE, NOISE, 12).astype(np.float32)
        body.step_joints(joint_action)
        records.append([float(v) for v in body.data.qpos])
    return records


def main():
    checkpoint_path = ROOT / 'runs/yumi/best.pt'
    configuration = checkpoint_configuration(checkpoint_path)
    body = Body(load_motor_policy=False, robot='yumi',
                interface=configuration['physics_interface'],
                observation_size=configuration['observation_size'])
    brain = ConnectomePolicy(device='cpu', observation_size=configuration['observation_size'],
                             action_size=configuration['action_size'],
                             neural_steps=configuration['neural_steps']).eval()
    brain.load(checkpoint_path, interface=body.interface)

    start = time.perf_counter()
    clean = rollout(brain, body)
    noisy = rollout(brain, body, noise_rng=np.random.default_rng(4242))
    diffs = [max(abs(a - b) for a, b in zip(qc, qn)) for qc, qn in zip(clean, noisy)]
    first = next((k for k, d in enumerate(diffs) if d > 1e-3), None)
    print(f'elapsed {time.perf_counter() - start:.0f}s')
    print(f'first qposDiff > 1e-3 at k={first}')
    print('diffs k=1..24: ' + ' '.join(f'{d:.2e}' for d in diffs[1:25]))
    print(f'final: clean_height={clean[-1][2]:.4f} noisy_height={noisy[-1][2]:.4f} '
          f'maxDiff={max(diffs):.3g}')


if __name__ == '__main__':
    main()
