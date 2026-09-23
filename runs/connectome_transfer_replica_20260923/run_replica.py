import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
root=Path('runs/connectome_transfer_replica_20260923');root.mkdir(parents=True,exist_ok=True)
ridge=Path('runs/connectome_transfer_ridge_20260923');push=Path('runs/connectome_transfer_push_20260923');prior=Path('runs/prior');output=root/'execution.json'
if output.exists():raise FileExistsError(output)
report=dict(completed=False,jobs=[],development=[],started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    limitation='Extra on-policy rounds after fixed-schedule replication failed push. Not equal-budget replication.')
def persist():
    p=output.with_suffix('.tmp');p.write_text(json.dumps(report,indent=2));p.replace(output)
def run(name,args,timeout=1200):
    row=dict(name=name,command=[sys.executable,'-u',*args]);report['jobs'].append(row);persist();start=time.monotonic()
    with (root/(name+'.log')).open('w') as h:
        r=subprocess.run(row['command'],stdout=h,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8'),timeout=timeout)
    row.update(returncode=r.returncode,wall_seconds=time.monotonic()-start);persist()
    if r.returncode:raise RuntimeError(name)
def evaluate(label,checkpoint,seed,extra=()):
    path=root/(label+'.json');run(label,['-m','app.evaluate_locomotion','--checkpoints',str(checkpoint),'--seed-base',str(seed),*extra,'--output',str(path)])
    c=json.loads(path.read_text())['conditions'][0];ts=[t for g in c['results'] for t in g['tests']]
    return dict(strict=sum(t['success'] and all(f['qualifying_swings_per_second']>=1 for f in t['gait']['feet']) for t in ts),falls=c['summary']['falls'],summary=c['summary'])
persist();deadline=time.monotonic()+3600
while not (ridge/'candidate_trace.json').exists():
    if time.monotonic()>deadline:raise TimeoutError('Primary verification/trace not completed')
    time.sleep(10)
assert json.loads((ridge/'execution.json').read_text())['completed']
student=ridge/'replica/last.pt'
train=[str(prior/(n+'.pt')) for n in ['replica_train','replica1','replica2','replica3']]+[str(push/(n+'.pt')) for n in ['replica_push1','replica_push2']]
candidate=None
for iteration in range(1,3):
    label='extra'+str(iteration);data=root/(label+'.pt')
    run(label+'_collect',['-m','app.imitate_yumi','collect','--teacher','runs/teacher/last.pt','--student',str(student),
        '--beta','0','--push','--seconds','30','--seed-base',str(226001+100*iteration),'--output',str(data)],900)
    train.append(str(data))
    run(label+'_fit',['-m','app.imitate_yumi','fit','--source',str(student),'--train',*train,
        '--validation',str(prior/'replica_validation.pt'),'--mode','readout','--ridge','.0001','--output',str(root/label)],1800)
    student=root/label/'last.pt'
    normal=evaluate(label+'_normal',student,203001);pushed=evaluate(label+'_push',student,214001,['--push','--cohorts','1'])
    long=evaluate(label+'_long',student,215001,['--seconds','120','--cohorts','1'])
    passed=normal['strict']>=24 and pushed['strict']>=8 and long['strict']>=8 and max(normal['falls'],pushed['falls'],long['falls'])<=1
    report['development'].append(dict(round=iteration,normal=normal,push=pushed,long=long,passed=passed));persist()
    if passed:candidate=student;report['candidate']=str(candidate);persist();break
if candidate:
    for label,extra in [('holdout',['--seed-base','240001']),
        ('yaw_holdout',['--seed-base','241001','--cohorts','1','--yaws','-.2','.2']),
        ('long_holdout',['--seed-base','242001','--cohorts','1','--seconds','120']),
        ('push_holdout',['--seed-base','243001','--cohorts','1','--push']),
        ('native_csr',['--seed-base','203001','--sparse-backend','native'])]:
        run(label,['-m','app.evaluate_locomotion','--checkpoints',str(candidate),*extra,'--output',str(root/(label+'.json'))])
report.update(completed=True,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());persist()
