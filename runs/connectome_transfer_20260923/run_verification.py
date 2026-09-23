import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path('runs/connectome_transfer_20260923')
output=root/'verification_execution.json'
if output.exists():
    raise FileExistsError(output)
report=dict(completed=False,jobs=[],replication=[],started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
            holdout_candidate=str(root/'dagger3/last.pt'))
def persist():
    tmp=output.with_suffix('.tmp'); tmp.write_text(json.dumps(report,indent=2)); tmp.replace(output)
def run(name,args,timeout=1800):
    row=dict(name=name,command=[sys.executable,'-u',*args]); report['jobs'].append(row); persist()
    start=time.monotonic()
    with (root/(name+'.log')).open('w') as handle:
        r=subprocess.run(row['command'],stdout=handle,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8'),timeout=timeout)
    row.update(returncode=r.returncode,wall_seconds=time.monotonic()-start); persist()
    if r.returncode: raise RuntimeError(name)
def gate(path):
    c=json.loads(path.read_text())['conditions'][0]
    t=[t for g in c['results'] for t in g['tests']]
    return dict(strict=sum(t['success'] and all(f['qualifying_swings_per_second']>=1 for f in t['gait']['feet']) for t in t),
                falls=c['summary']['falls'],summary=c['summary'])
for name,seed in [('replica_train',201001),('replica_validation',202001)]:
    run(name,['-m','app.imitate_yumi','collect','--teacher','runs/teacher/last.pt','--seed-base',str(seed),
        '--output',str(root/(name+'.pt'))],300)
train=[str(root/'replica_train.pt')]
student=Path('runs/source/best.pt')
for iteration in range(4):
    label='replica'+str(iteration)
    if iteration:
        data=root/(label+'.pt')
        run(label+'_collect',['-m','app.imitate_yumi','collect','--teacher','runs/teacher/last.pt',
            '--student',str(student),'--beta','0','--seed-base',str(204001+100*iteration),'--output',str(data)],600)
        train.append(str(data))
    run(label+'_fit',['-m','app.imitate_yumi','fit','--source',str(student),'--train',*train,
        '--validation',str(root/'replica_validation.pt'),'--mode','readout','--output',str(root/label)])
    student=root/label/'last.pt'
run('replica3_native',['-m','app.evaluate_locomotion','--checkpoints',str(student),'--seed-base','203001',
    '--output',str(root/'replica3_native.json')],900)
report['replication']=gate(root/'replica3_native.json'); persist()
candidate=report['holdout_candidate']
for label,extra in [
    ('student_holdout',['--seed-base','110001']),
    ('student_yaw_holdout',['--seed-base','111001','--cohorts','1','--yaws','-.2','.2']),
    ('student_long_holdout',['--seed-base','112001','--cohorts','1','--seconds','120']),
    ('student_push_holdout',['--seed-base','113001','--cohorts','1','--push']),
    ('student_lesion',['--seed-base','103001','--cohorts','1','--lesion','--repeat']),
    ('student_native_csr',['--seed-base','103001','--sparse-backend','native'])]:
    run(label,['-m','app.evaluate_locomotion','--checkpoints',candidate,*extra,'--output',str(root/(label+'.json'))],1200)
train=[str(root/(name+'.pt')) for name in ['train','dagger1','dagger2','dagger3']]
run('clipped_readout_fit',['-m','app.imitate_yumi','fit','--source',candidate,'--train',*train,
    '--validation',str(root/'validation.pt'),'--mode','readout','--clip-targets','--output',str(root/'clipped_readout')])
run('clipped_readout_native',['-m','app.evaluate_locomotion','--checkpoints',str(root/'clipped_readout/last.pt'),
    '--seed-base','103001','--output',str(root/'clipped_readout_native.json')],900)
report.update(completed=True,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()); persist()
