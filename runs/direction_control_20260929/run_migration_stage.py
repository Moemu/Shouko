"""Serialize migration after the preregistered teacher comparison releases the GPU."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('runs/direction_control_20260929')
output = root/'migration_execution.json'
if output.exists():
    raise FileExistsError(output)
report = dict(completed=False, phase='waiting_for_teacher', jobs=[], started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())


def persist():
    temp = output.with_suffix('.tmp'); temp.write_text(json.dumps(report, indent=2)); temp.replace(output)


def run(name, script, branch):
    command = [sys.executable, '-u', str(root/script), '--branch', branch]
    report['phase'] = name
    row = dict(name=name, command=command); report['jobs'].append(row); persist()
    started = time.monotonic()
    with (root/(name+'.log')).open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8'), timeout=5400)
    row.update(returncode=result.returncode, wall_seconds=time.monotonic()-started); persist()
    if result.returncode:
        raise RuntimeError(name)


persist()
deadline = time.monotonic()+2400
while not json.loads((root/'teacher_execution.json').read_text())['completed']:
    if time.monotonic() > deadline:
        raise TimeoutError('Teacher comparison has not completed; no migration started')
    time.sleep(10)
for branch in ['main', 'replica']:
    run('transfer_'+branch, 'run_transfer.py', branch)
    result = json.loads((root/('transfer_'+branch)/'execution.json').read_text())
    assert result['completed']
    if not result.get('development_candidate'):
        report['stopped_at_development'] = branch; persist(); break
    run('holdout_'+branch, 'run_transfer_holdout.py', branch)
report.update(completed=True, phase='complete', finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()); persist()
