"""Reproduce the restored source, collect independent validation, then test one weighted fit."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path('runs/sensory_access_20260930');root.mkdir(exist_ok=True)
source=Path('runs/direction_control_20260929/transfer_main/round4/last.pt')
teacher=Path('runs/direction_control_20260929/teacher_control3029/last.pt')
output=root/'initial_execution.json'
if output.exists():raise FileExistsError(output)
report=dict(completed=False,jobs=[],results={},started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
def persist():
    p=output.with_suffix('.tmp');p.write_text(json.dumps(report,indent=2));p.replace(output)
def run(name,arguments,timeout=1500):
    command=[sys.executable,'-u',*arguments];row=dict(name=name,command=command);report['jobs'].append(row);persist()
    start=time.monotonic()
    with (root/(name+'.log')).open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=timeout,
            env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8'))
    row.update(returncode=result.returncode,wall_seconds=time.monotonic()-start);persist()
    if result.returncode:raise RuntimeError(name)
def evaluate(name,path,extra):
    target=root/(name+'.json')
    run(name,['-m','app.evaluate_direction','--checkpoints',str(path),'--output',str(target),'--seed-base','310201',*extra])
    data=json.loads(target.read_text());assert data['completed']
    summary=data['conditions'][0]['summary'];report['results'][name]=summary;persist()
    print(json.dumps(dict(stage=name,**summary)),flush=True);return summary
persist()
for label,extra in [('normal',['--cohorts','3']),('long',['--seconds','120','--trace']),
    ('push_positive',['--push-velocity','.25']),('push_negative',['--push-velocity','-.25'])]:
    s=evaluate('source_'+label,source,extra)
    assert s['falls']==0 and s['strict_passes']==s['episodes'],s
    if label=='long':assert s['direction_passes']==6 and abs(s['lateral_m']-1.122782339)<.01,s
for label,seed,extra in [('normal',510001,['--seconds','12']),
    ('positive',511001,['--seconds','30','--cohorts','1','--yaws','0','--push-velocity','.25']),
    ('negative',512001,['--seconds','30','--cohorts','1','--yaws','0','--push-velocity','-.25'])]:
    run('validation_'+label,['-m','app.imitate_yumi','collect','--teacher',str(teacher),'--student',str(source),
        '--beta','0','--seed-base',str(seed),'--output',str(root/('validation_'+label+'.pt')),*extra])
run('weighted_fit',['-m','runs.sensory_access_20260930.weighted_refit'])
candidate=root/'weighted/last.pt';passed=True
for label,extra,threshold in [('normal',['--cohorts','3'],24),('long',['--seconds','120','--trace'],9),
    ('yaw',['--yaws','-.2','.2'],16),('push_positive',['--push-velocity','.25'],8),('push_negative',['--push-velocity','-.25'],8)]:
    s=evaluate('weighted_'+label,candidate,extra)
    if s['strict_passes']<threshold or s['falls']>1 or (label=='long' and s['direction_passes']!=9):
        passed=False;break
report['weighted_development_passed']=passed
if passed:
    report.update(development_candidate=str(candidate),candidate_sha256=hashlib.sha256(candidate.read_bytes()).hexdigest())
report.update(completed=True,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());persist()
