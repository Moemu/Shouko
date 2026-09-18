"""A/B whether yumi.yaml's 11:41 default_angles change is what broke the older checkpoints.

Non-destructive: overrides Body.home in memory, touches no config file.
`home` is the PD target baseline AND the observation reference (qpos[7:] - home),
so changing it shifts all 12 joint-position inputs the policy sees.
"""
import json
import sys

import numpy as np

from app.sim import Body
from app.full_brain import ConnectomePolicy
from app import evaluate_full

OLD_HOME = [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0]

checkpoint = sys.argv[1] if len(sys.argv) > 1 else 'runs/yumi/best.pt.bak'
use_old = (sys.argv[2] if len(sys.argv) > 2 else 'old') == 'old'

model = ConnectomePolicy(device='cuda').eval()
model.load(checkpoint)
body = Body(load_motor_policy=False, robot='yumi')
if use_old:
    body.home = np.array(OLD_HOME, dtype=np.float32)

print(json.dumps({'checkpoint': checkpoint, 'home': 'old(bak)' if use_old else 'new(current)',
                  'home_values': body.home.tolist()}), flush=True)
for seed, speed in [(8001, 0.35), (8002, 0.5), (8003, 0.65)]:
    row, _ = evaluate_full.episode(model, body, seed, speed, 10.0, False, False)
    print(json.dumps({k: row[k] for k in ['seed', 'target_speed', 'seconds', 'mean_speed',
                                          'forward_m', 'lateral_m', 'fallen', 'minimum_height', 'success']}), flush=True)
