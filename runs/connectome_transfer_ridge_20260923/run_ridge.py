import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path('runs/connectome_transfer_ridge_20260923');root.mkdir(parents=True,exist_ok=True)
push=Path('runs/connectome_transfer_push_20260923');balance=Path('runs/connectome_transfer_balance_20260923');output=root/'execution.json'
if output.exists():raise FileExistsError(output)
report=dict(completed=False,jobs=[],development=[],started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
def persist():
    p=output.with_suffix('.tmp');p.write_text(json.dumps(report,indent=2));p.replace(output)
def run(name,args,timeout=1200):
    row=dict(name=name,command=[sys.executable,'-u',*args]);report['jobs'].append(row);persist();start=time.monotonic()
    with (root/(name+'.log')).open('w') as h:
        r=subprocess.run(row['command'],stdout=h,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8'),timeout=timeout)
    row.update(returncode=r.returncode,wall_seconds=time.monotonic()-start);persist()
    if r.returncode:raise RuntimeError(name)
def evaluate(label,checkpoint,seed,extra=()):
    path=root/(label+'.json');run(label,['-m','app.evaluate_locomotion','--checkpoints',str(checkpoint),
        '--seed-base',str(seed),*extra,'--output',str(path)])
    c=json.loads(path.read_text())['conditions'][0];ts=[t for g in c['results'] for t in g['tests']]
    return dict(strict=sum(t['success'] and all(f['qualifying_swings_per_second']>=1 for f in t['gait']['feet']) for t in ts),falls=c['summary']['falls'],summary=c['summary'])
def development(label,checkpoint,seeds):
    normal=evaluate(label+'_normal',checkpoint,seeds[0]);pushed=evaluate(label+'_push',checkpoint,seeds[1],['--push','--cohorts','1'])
    long=evaluate(label+'_long',checkpoint,seeds[2],['--seconds','120','--cohorts','1'])
    passed=normal['strict']>=24 and pushed['strict']>=8 and long['strict']>=8 and max(normal['falls'],pushed['falls'],long['falls'])<=1
    return dict(normal=normal,push=pushed,long=long,passed=passed)
persist();deadline=time.monotonic()+3600
while not json.loads((balance/'execution.json').read_text())['completed']:
    if time.monotonic()>deadline:raise TimeoutError('Balance stage not finished')
    time.sleep(10)
if json.loads((balance/'execution.json').read_text()).get('candidate'):
    report.update(completed=True,skipped='Interpolation candidate exists; reassess its validation first');persist();sys.exit(0)
teacher=evaluate('teacher_matched_long','runs/teacher/last.pt',122001,['--seconds','120','--cohorts','1'])
report['teacher_matched_long']=teacher;persist()
if teacher['strict']<8 or teacher['falls']>1:
    report.update(completed=True,stopped='Teacher itself fails the matched long-walk contract');persist();sys.exit(0)
candidate=None
for ridge,label in [(.0001,'ridge1e4'),(.00001,'ridge1e5'),(.01,'ridge1e2')]:
    run(label+'_fit',[str(root/'refit_cached.py'),'--reference',str(push/'push2'),'--ridge',str(ridge),'--output',str(root/label)])
    path=root/label/'last.pt';gate=development(label,path,[103001,114001,115001]);report['development'].append(dict(ridge=ridge,**gate));persist()
    if gate['passed']:
        candidate=path;report.update(candidate=str(path),ridge=ridge);persist();break
if candidate:
    run('replica_fit',[str(root/'refit_cached.py'),'--reference',str(push/'replica_push2'),'--ridge',str(report['ridge']),'--output',str(root/'replica')])
    report['replication']=development('replica',root/'replica/last.pt',[203001,214001,215001]);persist()
    for label,extra in [('holdout',['--seed-base','140001']),
        ('yaw_holdout',['--seed-base','141001','--cohorts','1','--yaws','-.2','.2']),
        ('long_holdout',['--seed-base','142001','--cohorts','1','--seconds','120']),
        ('push_holdout',['--seed-base','143001','--cohorts','1','--push']),
        ('lesion',['--seed-base','103001','--cohorts','1','--lesion','--repeat']),
        ('native_csr',['--seed-base','103001','--sparse-backend','native'])]:
        run(label,['-m','app.evaluate_locomotion','--checkpoints',str(candidate),*extra,'--output',str(root/(label+'.json'))])
report.update(completed=True,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());persist()
