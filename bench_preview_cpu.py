"""Scratch benchmark: CPU preview-loop timing, mirroring studio_server.load_checkpoint.

Run: .venv/Scripts/python.exe bench_preview_cpu.py [robot] [checkpoint]
ASCII-only output (GBK console).
"""
import sys
import time
from pathlib import Path

import numpy as np
import psutil
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from app.full_brain import ConnectomePolicy, checkpoint_configuration
from app.sim import Body

robot = sys.argv[1] if len(sys.argv) > 1 else 'yumi'
checkpoint = Path(sys.argv[2]) if len(sys.argv) > 2 else Path('runs/yumi/best.pt')

process = psutil.Process()


def rss_mb():
    return process.memory_info().rss / 2**20


print(f'robot={robot} checkpoint={checkpoint} rss_start={rss_mb():.0f}MB')

configuration = checkpoint_configuration(str(checkpoint))
body = Body(load_motor_policy=False, robot=robot,
            interface=configuration['physics_interface'],
            observation_size=configuration['observation_size'])
architecture = {key: configuration[key] for key in
                ('observation_size', 'action_size', 'neural_steps')}
brain = ConnectomePolicy(device='cpu', **architecture).eval()
extra = brain.load(str(checkpoint), interface=body.interface)
body.reset(4242)

control_dt = body.cfg['simulation_dt'] * body.cfg['control_decimation']
print(f'loaded rss={rss_mb():.0f}MB observation_size={architecture["observation_size"]} '
      f'action_size={architecture["action_size"]} neural_steps={architecture["neural_steps"]} '
      f'simulation_dt={body.cfg["simulation_dt"]} control_decimation={body.cfg["control_decimation"]} '
      f'control_dt={control_dt:.4f}s real_time_factor_1x_needs_loop_lt={control_dt*1000:.1f}ms')

speed, yaw = 0.5, 0.0
neural_ms = []
step_ms = []
loop_ms = []
N_WARMUP = 100
N_STEPS = 2000
running = True
obs = body.motor_observation([speed, 0, 0.0])
with torch.inference_mode():
    for i in range(N_WARMUP + N_STEPS):
        t0 = time.perf_counter()
        action, activity = brain(torch.tensor(obs, dtype=torch.float32)[None], lesion=False)
        joint_action = action[0].cpu().numpy()
        t1 = time.perf_counter()
        state = body.step_joints(joint_action)
        t2 = time.perf_counter()
        if i >= N_WARMUP:
            neural_ms.append((t1 - t0) * 1000)
            step_ms.append((t2 - t1) * 1000)
            loop_ms.append((t2 - t0) * 1000)
        gravity, cur_yaw = body.observation()
        error = np.arctan2(np.sin(yaw - cur_yaw), np.cos(yaw - cur_yaw))
        obs = body.motor_observation([speed, 0, np.clip(error * 1.4, -0.2, 0.2)])
        if state is not None and hasattr(state, 'get') is False and i % 500 == 0:
            pass

neural_ms = np.array(neural_ms)
step_ms = np.array(step_ms)
loop_ms = np.array(loop_ms)


def stats(arr):
    return (f'mean={np.mean(arr):.2f}ms p50={np.percentile(arr,50):.2f} '
            f'p95={np.percentile(arr,95):.2f} p99={np.percentile(arr,99):.2f} max={np.max(arr):.2f}')


print(f'neural: {stats(neural_ms)}')
print(f'sim_step: {stats(step_ms)}')
print(f'loop_total: {stats(loop_ms)}')
print(f'headroom_vs_1x_realtime: mean={control_dt*1000 - np.mean(loop_ms):.1f}ms '
      f'p99={control_dt*1000 - np.percentile(loop_ms,99):.1f}ms')
print(f'final_rss={rss_mb():.0f}MB')

# Approx concurrent sessions: how many loops fit in one control_dt on average.
ratio = control_dt * 1000 / np.mean(loop_ms)
print(f'avg_loops_fit_per_core_per_control_dt={ratio:.2f} (upper bound, no contention)')
