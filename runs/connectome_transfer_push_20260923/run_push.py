import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path('runs/connectome_transfer_push_20260923');root.mkdir(parents=True,exist_ok=True)
prior=Path('runs/prior');output=root/'execution.json'
if output.exists():raise FileExistsError(output)
report=dict(completed=False,jobs=[],primary=[],replication={},started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
def persist():
    tmp=output.with_suffix('.tmp');tmp.write_text(json.dumps(report,indent=2));tmp.replace(output)
def run(name,args,timeout=1800):
    row=dict(name=name,command=[sys.executable,'-u',*args]);report['jobs'].append(row);persist();start=time.monotonic()
    with (root/(name+'.log')).open('w') as handle:
        r=subprocess.run(row['command'],stdout=handle,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8'),timeout=timeout)
    row.update(returncode=r.returncode,wall_seconds=time.monotonic()-start);persist()
    if r.returncode:raise RuntimeError(name)
def evaluate(label,checkpoint,seed,push=False):
    path=root/(label+'.json')
    run(label,['-m','app.evaluate_locomotion','--checkpoints',str(checkpoint),'--seed-base',str(seed),
        *(['--push','--cohorts','1'] if push else []),'--output',str(path)],900)
    c=json.loads(path.read_text())['conditions'][0];ts=[t for g in c['results'] for t in g['tests']]
    return dict(strict=sum(t['success'] and all(f['qualifying_swings_per_second']>=1 for f in t['gait']['feet']) for t in ts),
                falls=c['summary']['falls'],summary=c['summary'])
def fit_round(label,student,train,validation,seed):
    data=root/(label+'.pt')
    run(label+'_collect',['-m','app.imitate_yumi','collect','--teacher','runs/teacher/last.pt',
        '--student',str(student),'--beta','0','--push','--seconds','30','--seed-base',str(seed),'--output',str(data)],900)
    train.append(str(data))
    run(label+'_fit',['-m','app.imitate_yumi','fit','--source',str(student),'--train',*train,
        '--validation',str(validation),'--mode','readout','--output',str(root/label)],1800)
    return root/label/'last.pt'
persist();deadline=time.monotonic()+2400
while not (prior/'student_trace.json').exists():
    if time.monotonic()>deadline:raise TimeoutError('Prior verification/trace not finished')
    time.sleep(10)
student=prior/'dagger3/last.pt'
train=[str(prior/(n+'.pt')) for n in ['train','dagger1','dagger2','dagger3']]
candidate=None
for iteration in range(1,4):
    label='push'+str(iteration)
    student=fit_round(label,student,train,prior/'validation.pt',214001+100*iteration)
    normal=evaluate(label+'_native',student,103001)
    pushed=evaluate(label+'_push',student,114001,True)
    report['primary'].append(dict(round=iteration,normal=normal,push=pushed));persist()
    if normal['strict']>=24 and normal['falls']<=1 and pushed['strict']>=8 and pushed['falls']<=1:
        candidate=student;report['candidate']=str(candidate);report['candidate_round']=iteration;persist();break
if candidate:
    student=prior/'replica3/last.pt'
    train=[str(prior/(n+'.pt')) for n in ['replica_train','replica1','replica2','replica3']]
    for iteration in range(1,report['candidate_round']+1):
        student=fit_round('replica_push'+str(iteration),student,train,prior/'replica_validation.pt',224001+100*iteration)
    report['replication']=dict(normal=evaluate('replica_native',student,203001),push=evaluate('replica_push',student,214001,True));persist()
    for label,extra in [
        ('holdout',['--seed-base','120001']),
        ('yaw_holdout',['--seed-base','121001','--cohorts','1','--yaws','-.2','.2']),
        ('long_holdout',['--seed-base','122001','--cohorts','1','--seconds','120']),
        ('push_holdout',['--seed-base','123001','--cohorts','1','--push']),
        ('lesion',['--seed-base','103001','--cohorts','1','--lesion','--repeat']),
        ('native_csr',['--seed-base','103001','--sparse-backend','native'])]:
        run(label,['-m','app.evaluate_locomotion','--checkpoints',str(candidate),*extra,'--output',str(root/(label+'.json'))],1200)
report.update(completed=True,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());persist()
