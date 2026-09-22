"""Native MuJoCo paired feedback checks with fixed-order neural inference."""
import hashlib
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from app.full_brain import ConnectomePolicy, checkpoint_configuration
from app.sim import Body
from app.train_full import file_sha256
from ordered_csr import ordered_inference


def evaluate_native(policy, bodies, seeds, speeds, zero_new, seconds=30, yaw_limit=0.2):
    for body, seed in zip(bodies,seeds):
        body.reset(seed)
    alive=np.ones(len(bodies),dtype=bool)
    min_upright=np.ones(len(bodies))
    min_height=np.full(len(bodies),10.0)
    alternations=np.zeros(len(bodies),dtype=int)
    previous_side=[None]*len(bodies)
    previous_contacts=[[False,False] for _ in bodies]
    last=[None]*len(bodies)
    digest=hashlib.sha256()
    start=time.perf_counter()
    for step in range(round(seconds/(bodies[0].model.opt.timestep*bodies[0].cfg['control_decimation']))):
        observations=[]
        for body,speed in zip(bodies,speeds):
            _,yaw=body.observation()
            observations.append(body.motor_observation([speed,0,float(np.clip(-yaw*1.4,-yaw_limit,yaw_limit))]))
        observations=np.stack(observations)
        if zero_new:
            observations[:,47:50]=0
        actions=policy(torch.from_numpy(observations).cuda())[0].cpu().numpy()
        for i,body in enumerate(bodies):
            if not alive[i]:
                continue
            state=body.step_joints(actions[i])
            min_upright[i]=min(min_upright[i],state['upright'])
            min_height[i]=min(min_height[i],state['height'])
            for side in range(2):
                if state['contacts'][side] and not previous_contacts[i][side]:
                    alternations[i]+=int(previous_side[i] is not None and previous_side[i]!=side)
                    previous_side[i]=side
            previous_contacts[i]=state['contacts'].copy()
            alive[i]=not state['fallen']
            last[i]=state
        digest.update(np.stack([body.data.qpos.copy() for body in bodies]).tobytes())
        digest.update(np.stack([body.data.qvel.copy() for body in bodies]).tobytes())
        if not alive.any():
            break
    tests=[]
    for i,(seed,speed,state) in enumerate(zip(seeds,speeds,last)):
        duration=state['time']
        velocity=state['qpos'][0]/duration
        lateral=abs(state['qpos'][1])
        criteria=dict(survived=bool(alive[i] and duration>=seconds-0.01),
                      speed=bool(abs(velocity-speed)<=0.15),
                      feet=bool(min(state['foot_strikes'])>=int(seconds*0.5)),
                      alternation=bool(alternations[i]>=int(seconds)),
                      upright=bool(min_upright[i]>=0.7), direction=bool(lateral<=max(1,seconds*speed*0.2)))
        tests.append(dict(seed=seed,target_speed=speed,seconds=duration,mean_speed=velocity,
                          lateral_m=lateral,fallen=not bool(alive[i]),foot_strikes=state['foot_strikes'],
                          alternations=int(alternations[i]),minimum_upright=float(min_upright[i]),
                          minimum_height=float(min_height[i]),criteria=criteria,success=all(criteria.values())))
    score=np.mean([min(t['seconds']/seconds,1)-abs(t['mean_speed']-t['target_speed'])-
                   t['lateral_m']/(seconds*t['target_speed']) for t in tests])
    return dict(tests=tests,score=float(score),successes=sum(t['success'] for t in tests),
                falls=sum(t['fallen'] for t in tests),lateral_m=float(np.mean([t['lateral_m'] for t in tests])),
                mean_speed=float(np.mean([t['mean_speed'] for t in tests])),trajectory_sha256=digest.hexdigest(),
                wall_seconds=time.perf_counter()-start)



def main():
    torch.set_num_threads(8)
    root=Path('runs/yumi_route_decision_20260922')
    output=root/'heading_limits.json'
    if output.exists():
        raise FileExistsError(output)
    checkpoint=Path('runs/yumi_obs50/best.pt')
    config=checkpoint_configuration(checkpoint)
    policy=ConnectomePolicy(observation_size=50).eval()
    policy.load(checkpoint)
    bodies=[Body(False,'yumi',interface=config['physics_interface'],observation_size=50) for _ in range(9)]
    prior=json.loads((root/'native_feedback27.json').read_text())
    report=dict(kind='source_heading_limit_development',checkpoint_sha256=file_sha256(checkpoint),
                limits=[0.2,0.4,0.8],heading_gain=1.4,conditions=[],completed=False,
                limitation='External heading command limit experiment, unchanged policy weights and speed objectives. Not new training or final acceptance.')
    with ordered_inference():
        for cohort in range(3):
            seeds=list(range(93001+cohort*10,93010+cohort*10))
            for limit in report['limits']:
                result=evaluate_native(policy,bodies,seeds,[0.35,0.5,0.65]*3,False,yaw_limit=limit)
                row=dict(cohort=cohort,yaw_limit=limit,seeds=seeds,**result)
                report['conditions'].append(row)
                if limit==0.2:
                    expected=next(r for r in prior['conditions'] if r['name']=='source' and r['cohort']==cohort)
                    assert row['trajectory_sha256']==expected['trajectory_sha256']
                output.write_text(json.dumps(report,indent=2,allow_nan=False))
                print(json.dumps({k:row[k] for k in ['cohort','yaw_limit','score','successes','falls','lateral_m']}),flush=True)
    assert file_sha256(checkpoint)==report['checkpoint_sha256']
    report['completed']=True
    report['baseline_trajectory_repeat_exact']=True
    output.write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
