"""Test: does straightening the home posture mechanically lift the gait?

Mutates env.home (aliases the PD kernel buffer) plus initial_qpos, then runs
the standard screening evaluation. No retraining involved.
"""
import json
import sys

import torch

from app.full_brain import ConnectomePolicy
from app.gpu_body import GPUHumanoid
from app import train_full

CONFIGS = {
    'baseline': {},
    'knee-0.15': {3: 0.15, 9: 0.15},
    'straight': {0: -0.05, 3: 0.15, 4: -0.10, 6: -0.05, 9: 0.15, 10: -0.10},
}

which = sys.argv[1] if len(sys.argv) > 1 else 'baseline'
delta = CONFIGS[which]
env = GPUHumanoid(32, robot='yumi')
for idx, val in delta.items():
    env.home[idx] = val
env.initial_qpos[7:] = env.home
policy = ConnectomePolicy(neural_steps=4)
policy.load('runs/yumi/best.pt')
result = train_full.evaluate(policy, env, seconds=30)
print(json.dumps(dict(config=which, delta=delta, successes=result['successes'],
                      score=round(result['score'], 3),
                      mean_height=round(result.get('mean_height', 0), 3),
                      mean_duration=round(result['mean_duration'], 1))), flush=True)
