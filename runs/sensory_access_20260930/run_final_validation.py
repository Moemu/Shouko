"""Freeze a development-selected checkpoint before consuming reserved holdouts."""
import argparse
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from app.train_full import file_sha256

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--replica',action='store_true')
parser.add_argument('--replica-dagger',action='store_true');args=parser.parse_args()
if args.replica_dagger and not args.replica:parser.error('--replica-dagger requires --replica')
root=Path('runs/sensory_access_20260930')
branch=root/'replica' if args.replica else root
if args.replica_dagger:branch=branch/'dagger'
execution=json.loads((branch/('execution.json' if args.replica else 'pair_execution.json')).read_text())
assert execution['completed']
if args.replica_dagger:
    source=Path(execution['candidate']);selected=source.parent.name
    assert file_sha256(source)==execution['candidate_sha256']
else:
    selected='s1' if execution['s1_development_passed'] else ('c0' if execution['c0_development_passed'] else None)
    assert selected is not None
    source=branch/selected/'last.pt'
fit=json.loads(source.with_name('fit.json').read_text())
assert fit['completed'] and file_sha256(source)==fit['checkpoint_sha256']
out=branch/'final_validation';out.mkdir(exist_ok=False)
report=dict(completed=False,selected=selected,checkpoint=str(source),checkpoint_sha256=file_sha256(source),
    frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source_sha256=file_sha256(__file__),
    selection_rule=('First complete development pass within two additional S1 DAgger rounds; no final holdout tuning.' if args.replica_dagger else
                    'Prefer S1 if all development gates pass; otherwise C0 if all pass. No final holdout tuning.'),jobs=[],results={})
def persist():
    p=out/'execution.tmp';p.write_text(json.dumps(report,indent=2));p.replace(out/'execution.json')
persist();passed=True;base=450001 if args.replica else 350001
for name,seed,extra,threshold in [('normal',base,['--cohorts','3'],24),('yaw',base+1000,['--yaws','-.2','.2'],16),
    ('long',base+2000,['--seconds','120','--trace'],9),('push_positive',base+3000,['--push-velocity','.25'],8),
    ('push_negative',base+4000,['--push-velocity','-.25'],8)]:
    assert file_sha256(source)==report['checkpoint_sha256']
    command=[sys.executable,'-u','-m','app.evaluate_direction','--checkpoints',str(source),
        '--output',str(out/(name+'.json')),'--seed-base',str(seed),*extra]
    row=dict(name=name,command=command);report['jobs'].append(row);persist();start=time.monotonic()
    with (out/(name+'.log')).open('w') as log:
        result=subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=2400,
            env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8'))
    row.update(returncode=result.returncode,wall_seconds=time.monotonic()-start);persist()
    if result.returncode:raise RuntimeError(name)
    data=json.loads((out/(name+'.json')).read_text());assert data['completed']
    assert data['conditions'][0]['checkpoint_sha256']==report['checkpoint_sha256']
    s=data['conditions'][0]['summary'];report['results'][name]=s
    passed &= s['strict_passes']>=threshold and s['falls']<=1 and (name!='long' or s['direction_passes']==9)
    persist();print(json.dumps(dict(stage=name,**s)),flush=True)
report.update(completed=True,passed=passed,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());persist()
