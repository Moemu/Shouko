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

root=Path('runs/connectome_transfer_push_20260923');output=root/'candidate_trace.npz'
if output.exists():raise FileExistsError(output)
checkpoint=Path(json.loads((root/'execution.json').read_text())['candidate'])
model,cfg,kind=load_policy(checkpoint);assert kind=='connectome'
bodies=[Body(False,'yumi',interface=cfg['physics_interface'],observation_size=50) for _ in range(9)]
for i,b in enumerate(bodies):b.reset(103001+i)
body=bodies[1];ids=[np.flatnonzero(body.model.geom_bodyid==body.model.body(n).id) for n in ['left_ankle_roll_link','right_ankle_roll_link']]
rows=dict(time=[],qpos=[],contacts=[],clearance=[],actions=[]);alive=np.ones(9,dtype=bool)
with ordered_inference():
    for step in range(1500):
        obs=np.stack([b.motor_observation([v,0,float(np.clip(-b.observation()[1]*1.4,-.2,.2))]) for b,v in zip(bodies,[.35,.5,.65]*3)])
        actions=model(torch.from_numpy(obs).cuda())[0].cpu().numpy()
        for i,b in enumerate(bodies):
            if alive[i]:
                state=b.step_joints(actions[i]);alive[i]=not state['fallen']
                if i==1:
                    rows['time'].append(state['time']);rows['qpos'].append(state['qpos']);rows['contacts'].append(state['contacts'])
                    rows['clearance'].append([float(np.min(body.data.geom_xpos[g,2]-body.model.geom_size[g,0])) for g in ids]);rows['actions'].append(actions[1])
        if not alive[1]:break
np.savez_compressed(output,**{k:np.asarray(v) for k,v in rows.items()})
contact=np.asarray(rows['contacts'])[100:];n=contact.sum(1);qpos=np.asarray(rows['qpos'])[100:]
report=dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),trace_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
    seed=103002,command=.5,seconds=rows['time'][-1],fallen=not bool(alive[1]),teacher_used=False,batch=9,sparse_backend='ordered',
    support_after_2s=dict(flight=float((n==0).mean()),single=float((n==1).mean()),double=float((n==2).mean())),
    height_range=[float(qpos[:,2].min()),float(qpos[:,2].max())],limitation='Representative development trace; not independent acceptance or human naturalness validation.')
output.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report))
