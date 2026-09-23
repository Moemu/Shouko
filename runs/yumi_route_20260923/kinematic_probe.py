"""Static clearance reachability; does not test dynamic single-support balance."""
import json
from pathlib import Path
import sys

import mujoco
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.full_brain import checkpoint_configuration
from app.sim import Body

configuration = checkpoint_configuration(ROOT/'runs/yumi_route_20260923/mlp_seed2026_best.pt')
body = Body(False, 'yumi', interface=configuration['physics_interface'], observation_size=50)
ids = [np.flatnonzero(body.model.geom_bodyid == body.model.body(n).id)
       for n in ['left_ankle_roll_link', 'right_ankle_roll_link']]
body.reset(0)
body.data.qpos[2] = .955


def solve(side, target):
    lo, hi = 0., 1.2
    for _ in range(40):
        knee = (lo+hi)/2
        body.data.qpos[np.array([7, 10, 11])+side*6] = [-knee/2, knee, -knee/2]
        mujoco.mj_forward(body.model, body.data)
        height = np.min(body.data.geom_xpos[ids[side], 2]-body.model.geom_size[ids[side], 0])
        if height < target:
            lo = knee
        else:
            hi = knee


solve(1, 0.)
rows = []
for target in [0., .025, .06]:
    solve(0, target)
    joints = body.data.qpos[7:]
    limits = body.model.jnt_range[1:]
    actions = (joints-body.home)/body.cfg['action_scale']
    rows.append(dict(target_clearance=target, joint_positions=joints.tolist(),
                     sole_min_z=[float(np.min(body.data.geom_xpos[i, 2]-body.model.geom_size[i, 0])) for i in ids],
                     within_joint_limits=bool(((joints>=limits[:, 0]) & (joints<=limits[:, 1])).all()),
                     max_abs_action=float(abs(actions).max()), within_action_limit=bool((abs(actions)<=8).all())))
output = ROOT/'runs/yumi_route_20260923/kinematic_clearance_feasible.json'
if output.exists():
    raise FileExistsError(output)
output.write_text(json.dumps(dict(pelvis_height=.955, rows=rows,
    limitation='Static forward kinematics, right sole at floor. No claim of dynamic balance, torque feasibility or learned control.'), indent=2))
print(json.dumps(rows))
