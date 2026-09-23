"""Supplemental trace uses the acceptance batch and ordered inference, with no teacher."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from app.evaluate_locomotion import load_policy
from app.ordered_inference import ordered_inference
from app.sim import Body

root=Path('runs/connectome_transfer_20260923')
output=root/'student_trace.npz'
if output.exists(): raise FileExistsError(output)
checkpoint=root/'dagger3/last.pt'
policy,cfg,kind=load_policy(checkpoint)
assert kind=='connectome'
bodies=[Body(False,'yumi',interface=cfg['physics_interface'],observation_size=50) for _ in range(9)]
for i,body in enumerate(bodies): body.reset(103001+i)
body=bodies[1]
ids=[np.flatnonzero(body.model.geom_bodyid==body.model.body(n).id) for n in ['left_ankle_roll_link','right_ankle_roll_link']]
rows=dict(time=[],qpos=[],contacts=[],clearance=[],actions=[],applied_torque=[])
speeds=[.35,.5,.65]*3
with ordered_inference():
    for step in range(1500):
        obs=np.stack([b.motor_observation([v,0,float(np.clip(-b.observation()[1]*1.4,-.2,.2))]) for b,v in zip(bodies,speeds)])
        actions=policy(torch.from_numpy(obs).cuda())[0].cpu().numpy()
        states=[b.step_joints(a) for b,a in zip(bodies,actions)]
        s=states[1]
        rows['time'].append(s['time']);rows['qpos'].append(s['qpos']);rows['contacts'].append(s['contacts'])
        rows['clearance'].append([float(np.min(body.data.geom_xpos[g,2]-body.model.geom_size[g,0])) for g in ids])
        rows['actions'].append(actions[1]);rows['applied_torque'].append(body.data.qfrc_actuator[6:].copy())
        if any(s['fallen'] for s in states): raise RuntimeError('Development replay unexpectedly fell')
np.savez_compressed(output,**{k:np.asarray(v) for k,v in rows.items()})
contact=np.asarray(rows['contacts'])[100:]
counts=contact.sum(1)
qpos=np.asarray(rows['qpos'])[100:]
report=dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),trace_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
    seed=103002,command=.5,seconds=rows['time'][-1],teacher_used=False,sparse_backend='ordered',batch=9,
    support_after_2s=dict(flight=float((counts==0).mean()),single=float((counts==1).mean()),double=float((counts==2).mean())),
    height_range=[float(qpos[:,2].min()),float(qpos[:,2].max())],
    limitation='Representative development trace, not a new independent acceptance cohort or naturalness validation.')
output.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
