"""Identify source-policy forward and turn responses without training or adapters."""
import json
from pathlib import Path
import time

import numpy as np
import torch

from app.full_brain import ConnectomePolicy, checkpoint_configuration
from app.sim import Body
from app.train_full import file_sha256
from ordered_csr import ordered_inference


torch.set_num_threads(8)
root=Path('runs/yumi_route_decision_20260922')
checkpoint=Path('runs/yumi_obs50/best.pt')
config=checkpoint_configuration(checkpoint)
policy=ConnectomePolicy(observation_size=50).eval()
policy.load(checkpoint)
bodies=[Body(False,'yumi',interface=config['physics_interface'],observation_size=50) for _ in range(9)]
seeds=list(range(94001,94010))
speeds=[0.35,0.5,0.65]*3
report=dict(kind='fixed_source_command_response',checkpoint_sha256=file_sha256(checkpoint),
            seeds=seeds,speeds=speeds,seconds=12,conditions=[],completed=False,
            limitation='Development system identification only; fixed-turn trials intentionally do not track world heading zero.')
with ordered_inference():
    for mode,turn in [('heading_feedback',None),('turn_negative',-0.2),('turn_zero',0.0),('turn_positive',0.2)]:
        for body,seed in zip(bodies,seeds):
            body.reset(seed)
        active=np.ones(9,dtype=bool)
        sums=np.zeros((9,6))
        counts=np.zeros(9,dtype=int)
        previous_yaw=np.zeros(9)
        unwrapped=np.zeros(9)
        last=[None]*9
        started=time.perf_counter()
        for step in range(600):
            observations=[]
            turn_commands=[]
            for body,speed in zip(bodies,speeds):
                _,yaw=body.observation()
                command=float(np.clip(-1.4*yaw,-0.2,0.2)) if turn is None else turn
                turn_commands.append(command)
                observations.append(body.motor_observation([speed,0,command]))
            actions=policy(torch.from_numpy(np.stack(observations)).cuda())[0].cpu().numpy()
            for i,body in enumerate(bodies):
                if not active[i]:
                    continue
                state=body.step_joints(actions[i])
                yaw=state['yaw']
                unwrapped[i]+=np.arctan2(np.sin(yaw-previous_yaw[i]),np.cos(yaw-previous_yaw[i]))
                previous_yaw[i]=yaw
                vx,vy=state['velocity'][:2]
                body_vx=vx*np.cos(yaw)+vy*np.sin(yaw)
                body_vy=-vx*np.sin(yaw)+vy*np.cos(yaw)
                sums[i]+=np.array([vx,body_vx,body_vy,np.hypot(vx,vy),abs(yaw),float(abs(turn_commands[i])>=0.199999)])
                counts[i]+=1
                active[i]=not state['fallen']
                last[i]=state
        tests=[]
        for i,state in enumerate(last):
            means=sums[i]/counts[i]
            tests.append(dict(seed=seeds[i],forward_command=speeds[i],turn_command=turn,seconds=state['time'],
                fallen=not bool(active[i]),forward_m=state['qpos'][0],lateral_m=state['qpos'][1],
                path_m=state['distance'],final_yaw=state['yaw'],unwrapped_yaw=float(unwrapped[i]),
                mean_yaw_rate=float(unwrapped[i]/state['time']),mean_world_vx=float(means[0]),
                mean_body_vx=float(means[1]),mean_body_vy=float(means[2]),mean_planar_speed=float(means[3]),
                mean_abs_yaw=float(means[4]),turn_saturation_fraction=float(means[5])))
        report['conditions'].append(dict(mode=mode,tests=tests,wall_seconds=time.perf_counter()-started))
        (root/'command_response.json').write_text(json.dumps(report,indent=2,allow_nan=False))
        print(json.dumps(dict(mode=mode,falls=sum(t['fallen'] for t in tests),
             mean_world_vx=float(np.mean([t['mean_world_vx'] for t in tests])),
             mean_body_vx=float(np.mean([t['mean_body_vx'] for t in tests])),
             mean_yaw_rate=float(np.mean([t['mean_yaw_rate'] for t in tests])))),flush=True)
report['completed']=True
assert file_sha256(checkpoint)==report['checkpoint_sha256']
(root/'command_response.json').write_text(json.dumps(report,indent=2,allow_nan=False))
