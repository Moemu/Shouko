"""Separate repeated neural inference from open-loop physics replay variation."""
import json
from pathlib import Path
import time

import torch

from app.full_brain import ConnectomePolicy, checkpoint_configuration
from app.gpu_body import GPUHumanoid
from app.train_full import file_sha256

torch.set_num_threads(8)
torch.manual_seed(20260922)
root=Path('runs/yumi_route_decision_20260922')
checkpoint=Path('runs/yumi_obs50/best.pt')
config=checkpoint_configuration(checkpoint)
policy=ConnectomePolicy(observation_size=50).eval()
policy.load(checkpoint)
batch=torch.load('rollout.pt',map_location='cpu',weights_only=True)
observations=batch['observations'].reshape(-1,50)[:32].cuda()
report=dict(kind='neural_and_physics_repeatability_probe',checkpoint_sha256=file_sha256(checkpoint),
            inference=[],physics=[],completed=False)
started=time.monotonic()


def save():
    report['elapsed_seconds']=time.monotonic()-started
    (root/'repeatability.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


with torch.inference_mode():
    for deterministic in [False,True]:
        torch.use_deterministic_algorithms(deterministic)
        try:
            initial=policy(observations)[0]
            torch.cuda.synchronize()
            start=time.perf_counter()
            deltas=[(policy(observations)[0]-initial).clone() for _ in range(16)]
            torch.cuda.synchronize()
            delta=torch.stack(deltas)
            report['inference'].append(dict(deterministic_algorithms=deterministic,
                rms=float(delta.double().square().mean().sqrt()),
                max_abs=float(delta.abs().max()),exact_repeats=int((delta==0).flatten(1).all(1).sum()),
                repeats=16,seconds=time.perf_counter()-start))
        except RuntimeError as error:
            report['inference'].append(dict(deterministic_algorithms=deterministic,error=str(error)))
        finally:
            torch.use_deterministic_algorithms(False)
        save()

    env=GPUHumanoid(32,robot='yumi',interface=config['physics_interface'],observation_size=50)
    mask=torch.ones(env.worlds,dtype=torch.bool,device='cuda')
    generator=torch.Generator(device='cuda').manual_seed(92001)
    initial_velocity=torch.randn(env.worlds,2,device='cuda',generator=generator)*0.015

    def reset():
        env.reset(mask)
        env.qvel[:,:2]=initial_velocity
        env.command.zero_()
        env.command[:,0]=torch.tensor([0.35,0.5,0.65],device='cuda').repeat(11)[:32]

    reset()
    tape=[]
    capture=[]
    for step in range(300):
        w,x,y,z=env.qpos[:,3:7].unbind(1)
        yaw=torch.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
        env.command[:,2]=(-yaw*1.4).clamp(-0.2,0.2)
        action=policy(env.observation())[0]
        tape.append(action.clone())
        env.step(action)
        capture.append(env.qpos.clone())
    capture=torch.stack(capture)
    references=None
    for repeat in range(3):
        reset()
        positions=[]
        for action in tape:
            env.step(action)
            positions.append(env.qpos.clone())
        positions=torch.stack(positions)
        delta=positions-capture
        repeat_delta=positions-references if references is not None else torch.zeros_like(positions)
        report['physics'].append(dict(repeat=repeat,steps=300,seconds=300*env.dt,
            capture_qpos_max_abs=float(delta.abs().max()),
            previous_replay_qpos_max_abs=float(repeat_delta.abs().max()) if references is not None else None,
            first_step_capture_max_abs=float(delta[0].abs().max()),
            final_capture_position_rms=float(delta[-1,:,:3].double().square().mean().sqrt()),
            sample_max_abs=[dict(step=i+1,delta=float(delta[i].abs().max())) for i in [0,49,149,299]]))
        references=positions
        env.check_physics()
        save()
report['completed']=True
report['limitation']='One 6-second fixed-action tape. Open-loop replay does not bound 30-second feedback trajectory variation.'
save()
print(json.dumps(report),flush=True)
