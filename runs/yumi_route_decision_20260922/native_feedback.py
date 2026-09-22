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


def evaluate_native(policy, bodies, seeds, speeds, zero_new, seconds=30):
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
            observations.append(body.motor_observation([speed,0,float(np.clip(-yaw*1.4,-0.2,0.2))]))
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


def main(args):
    torch.set_num_threads(8)
    root=Path('runs/yumi_route_decision_20260922')
    output=root/args.output
    if output.exists():
        raise FileExistsError(output)
    source=Path('runs/yumi_obs50/best.pt')
    config=checkpoint_configuration(source)
    policy=ConnectomePolicy(observation_size=50).eval()
    bodies=[Body(False,'yumi',interface=config['physics_interface'],observation_size=50) for _ in range(9)]
    speeds=[0.35,0.5,0.65]*3
    report=dict(kind='native_development_feedback_ablation',cohorts=args.cohorts,speeds=speeds,
                neural_backend='fixed-order Warp CSR',physics='native MuJoCo',physics_interface=bodies[0].interface,
                conditions=[],completed=False,
                limitation='Development diagnostic using unchanged checkpoints and a different accumulation order; not final held-out acceptance.')
    conditions=[('source',source,False),('source_repeat',source,False),('source_zero',source,True)]
    for name in ['control','scaled']:
        path=Path('runs/yumi_actor_scale_ab_20260922')/name/'last.pt'
        conditions.extend([(name,path,False),(name+'_zero',path,True)])
    with ordered_inference():
        for cohort in range(args.cohorts):
            seeds=list(range(93001+cohort*10,93010+cohort*10))
            source_hash=None
            for name,path,zero in conditions:
                digest=file_sha256(path)
                policy.load(path,interface=bodies[0].interface)
                result=evaluate_native(policy,bodies,seeds,speeds,zero)
                row=dict(name=name,cohort=cohort,seeds=seeds,checkpoint=str(path),checkpoint_sha256=digest,zero_new_inputs=zero,**result)
                report['conditions'].append(row)
                output.write_text(json.dumps(report,indent=2,allow_nan=False))
                print(json.dumps({key:row[key] for key in ['name','cohort','score','successes','falls','lateral_m','trajectory_sha256','wall_seconds']}),flush=True)
                assert file_sha256(path)==digest
                if name=='source':
                    source_hash=result['trajectory_sha256']
                if name in ['source_repeat','source_zero']:
                    assert result['trajectory_sha256']==source_hash, 'Repeated or null-intervention trajectory differs'
    report['completed']=True
    report['source_repeat_and_null_intervention_exact']=True
    output.write_text(json.dumps(report,indent=2,allow_nan=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cohorts',type=int,choices=[1,3],default=1)
    parser.add_argument('--output',default='native_feedback.json')
    main(parser.parse_args())
