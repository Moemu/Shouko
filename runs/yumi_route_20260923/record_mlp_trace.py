"""CPU replay for visual diagnostics; not a replacement for matched CUDA acceptance."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.full_brain import checkpoint_configuration
from app.mlp_policy import MLPPolicy
from app.sim import Body

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', required=True)
parser.add_argument('--output', required=True)
parser.add_argument('--seconds', type=float, default=30)
parser.add_argument('--seed', type=int, default=93002)
args = parser.parse_args()
output = Path(args.output)
if output.exists():
    raise FileExistsError(output)
config = checkpoint_configuration(args.checkpoint)
policy = MLPPolicy(device='cpu').eval()
policy.load(args.checkpoint)
body = Body(False, 'yumi', interface=config['physics_interface'], observation_size=50)
body.reset(args.seed)
ids = [np.flatnonzero(body.model.geom_bodyid == body.model.body(name).id)
       for name in ['left_ankle_roll_link', 'right_ankle_roll_link']]
rows = dict(time=[], qpos=[], clearance=[], contacts=[])
dt = body.cfg['simulation_dt']*body.cfg['control_decimation']
with torch.inference_mode():
    for _ in range(round(args.seconds/dt)):
        yaw = body.observation()[1]
        obs = body.motor_observation([.5, 0, np.clip(-yaw*1.4, -.2, .2)])
        action = policy(torch.from_numpy(obs)[None])[0].numpy()[0]
        state = body.step_joints(action)
        rows['time'].append(state['time'])
        rows['qpos'].append(state['qpos'])
        rows['contacts'].append(state['contacts'])
        rows['clearance'].append([min(body.data.geom_xpos[i, 2]-body.model.geom_size[i, 0]) for i in ids])
        if state['fallen']:
            break
np.savez_compressed(output, **{k: np.array(v) for k, v in rows.items()})
output.with_suffix('.json').write_text(json.dumps(dict(
    checkpoint=str(args.checkpoint), sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest(),
    device='cpu', seed=args.seed, command=.5, seconds=state['time'], fallen=state['fallen'],
    note='Supplemental visual replay; CUDA batch=9 native evaluation remains the acceptance path.'), indent=2))
print(json.dumps(dict(seconds=state['time'], fallen=state['fallen'], max_clearance=np.max(rows['clearance'], axis=0).tolist())))
