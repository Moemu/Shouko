"""Verify fixed teacher continuation across seeds and new evaluation starts."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('runs/direction_control_20260929')
output = root/'teacher_repro_execution.json'
if output.exists():
    raise FileExistsError(output)
report = dict(completed=False, jobs=[], results={}, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
def persist():
    temp=output.with_suffix('.tmp'); temp.write_text(json.dumps(report, indent=2)); temp.replace(output)
def run(name, arguments, timeout=1800):
    command=[sys.executable, '-u', *arguments]
    row=dict(name=name, command=command); report['jobs'].append(row); persist()
    started=time.monotonic()
    with (root/(name+'.log')).open('w') as log:
        result=subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
            env=dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8'), timeout=timeout)
    row.update(returncode=result.returncode, wall_seconds=time.monotonic()-started); persist()
    if result.returncode:
        raise RuntimeError(name)
persist()
deadline=time.monotonic()+1200
while not json.loads((root/'yaw_gain28_execution.json').read_text())['completed']:
    if time.monotonic()>deadline:
        raise TimeoutError('Feedback diagnostic still running')
    time.sleep(10)
run('teacher_control_replica_driver', [str(root/'run_teacher_control_replica.py')], 2400)
replica=json.loads((root/'teacher_control_replica_execution.json').read_text())
report['replica_development_passed']=replica['control_replica_passed']; persist()
if replica['control_replica_passed']:
    checkpoints=[str(root/f'teacher_control{seed}/last.pt') for seed in [3029,3030]]
    report['fixed_candidates']={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in checkpoints}; persist()
    passed=True
    for label, delta, extra, required in [('normal',0,['--cohorts','3'],24), ('yaw',1000,['--yaws','-.2','.2'],16),
        ('long',2000,['--seconds','120','--trace'],8), ('push_positive',3000,['--push-velocity','.25'],8),
        ('push_negative',4000,['--push-velocity','-.25'],8)]:
        path=root/('teachers_holdout_'+label+'.json')
        run('teachers_holdout_'+label, ['-m','app.evaluate_direction','--checkpoints',*checkpoints,
            '--seed-base',str(550001+delta),'--output',str(path),*extra])
        data=json.loads(path.read_text()); assert data['completed']
        summaries=[c['summary'] for c in data['conditions']]
        report['results'][label]=summaries
        passed=passed and all(s['strict_passes']>=required and s['falls']<=1 and
            (label!='long' or s['direction_passes']==9) for s in summaries)
        persist(); print(json.dumps(dict(stage=label, summaries=summaries)), flush=True)
    report['both_teachers_heldout_passed']=passed; persist()
    run('teacher_source_matched_long',['-m','app.evaluate_direction','--checkpoints','runs/source_teacher/last.pt',
        '--seed-base','552001','--seconds','120','--trace','--output',str(root/'teacher_source_matched_long.json')])
report.update(completed=True, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()); persist()
