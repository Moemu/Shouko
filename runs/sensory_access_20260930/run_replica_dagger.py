"""At most two registered closed-loop rounds, without consuming replica holdouts."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import torch
from app.train_full import file_sha256

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--resume-evaluation',type=int,choices=[1,2]);args=parser.parse_args()
root=Path('runs/sensory_access_20260930');branch=root/'replica'
initial=json.loads((branch/'execution.json').read_text())
assert initial['completed'] and not initial['s1_development_passed']
normal=initial['results']['s1_normal'];assert normal['strict_passes']>=24 and normal['falls']<=1
out=branch/'dagger'
if args.resume_evaluation:
    assert out.is_dir()
    source=out/f'round{args.resume_evaluation}'/'last.pt'
else:
    out.mkdir(exist_ok=False)
    source=branch/'s1/last.pt'
prior=json.loads(source.with_name('fit.json').read_text())
assert prior['completed'] and file_sha256(source)==prior['checkpoint_sha256']
assert max(prior['final_input_response'].values())>0
training=prior['arguments']['train'].copy();validation=prior['arguments']['validation']
teacher=Path('runs/direction_control_20260929/teacher_control3029/last.pt')
report=dict(completed=False,jobs=[],results={},started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    limitation='Additional convergence rounds; not an equal-update replication of the main candidate.')
output=out/'execution.json'
if args.resume_evaluation:
    report=json.loads(output.read_text());assert not report['completed']
    report['resumed_evaluation_from_checkpoint_sha256']=file_sha256(source)
def persist():
    p=output.with_suffix('.tmp');p.write_text(json.dumps(report,indent=2));p.replace(output)
def run(name,args):
    command=[sys.executable,'-u',*args];row=dict(name=name,command=command);report['jobs'].append(row);persist();start=time.monotonic()
    with (out/(name+'.log')).open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=2400,
            env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8'))
    row.update(returncode=result.returncode,wall_seconds=time.monotonic()-start);persist()
    if result.returncode:raise RuntimeError(name)
used=set()
for p in training+validation:
    used.update(torch.load(p,map_location='cpu',weights_only=True,mmap=True)['episode_ids'].tolist())
persist()
for round_number in range(args.resume_evaluation or 1,3):
    seed_base=630001+(round_number-1)*4000
    collections=[] if args.resume_evaluation==round_number else [
        ('normal',['--seconds','12'],3),
        ('positive',['--seconds','30','--cohorts','1','--yaws','0','--push-velocity','.25'],1),
        ('negative',['--seconds','30','--cohorts','1','--yaws','0','--push-velocity','-.25'],1),
        ('highspeed',['--seconds','60','--cohorts','1','--yaws','0','--speeds','.65'],1)]
    for index,(label,extra,cohorts) in enumerate(collections):
        seed=seed_base+index*1000
        ids={seed+c*10+i for c in range(cohorts) for i in range(9)}
        assert not ids & used;used.update(ids)
        path=out/f'round{round_number}_{label}_data.pt'
        run(path.stem,['-m','app.imitate_yumi','collect','--teacher',str(teacher),'--student',str(source),
            '--beta','0','--seed-base',str(seed),'--output',str(path),*extra]);training.append(str(path))
    name=f'round{round_number}'
    if args.resume_evaluation!=round_number:
        run(name,['-m','app.train_sensory_readout','--source',str(source),'--train',*training,'--validation',*validation,
            '--output',str(out/name),'--sensory','--seed',str(4031+round_number)])
    source=out/name/'last.pt';fit=json.loads(source.with_name('fit.json').read_text());assert fit['completed']
    passed=True;normal_passed=False
    for label,extra,threshold in [('normal',['--cohorts','3'],24),('long',['--seconds','120','--trace'],9),
        ('yaw',['--yaws','-.2','.2'],16),('push_positive',['--push-velocity','.25'],8),('push_negative',['--push-velocity','-.25'],8)]:
        path=out/f'{name}_eval_{label}.json'
        run(path.stem,['-m','app.evaluate_direction','--checkpoints',str(source),'--output',str(path),'--seed-base','310201',*extra])
        data=json.loads(path.read_text());assert data['completed']
        s=data['conditions'][0]['summary'];report['results'][path.stem]=s;persist()
        print(json.dumps(dict(stage=path.stem,**s)),flush=True)
        accepted=s['strict_passes']>=threshold and s['falls']<=1 and (label!='long' or s['direction_passes']==9)
        if label=='normal':normal_passed=accepted
        if not accepted:passed=False;break
    report[f'{name}_development_passed']=passed;persist()
    if passed:
        report.update(candidate=str(source),candidate_sha256=file_sha256(source),rounds=round_number)
        break
    if not normal_passed:break
report.update(completed=True,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());persist()
