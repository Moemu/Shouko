import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('runs/connectome_transfer_20260923')
root.mkdir(parents=True, exist_ok=True)
output = root/'initial_execution.json'
if output.exists():
    raise FileExistsError(output)
report = dict(completed=False, jobs=[], started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())

def persist():
    tmp = output.with_suffix('.tmp')
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(output)

def run(name, args, timeout=1800):
    row = dict(name=name, command=[sys.executable, '-u', *args])
    report['jobs'].append(row)
    persist()
    start = time.monotonic()
    with (root/(name+'.log')).open('w') as handle:
        result = subprocess.run(row['command'], stdout=handle, stderr=subprocess.STDOUT,
            env=dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8'), timeout=timeout)
    row.update(returncode=result.returncode, wall_seconds=time.monotonic()-start)
    persist()
    if result.returncode:
        raise RuntimeError(name)

teacher = 'runs/teacher/last.pt'
for label, seed in [('train', 101001), ('validation', 102001)]:
    run('collect_'+label, ['-m', 'app.imitate_yumi', 'collect', '--teacher', teacher,
        '--seed-base', str(seed), '--output', str(root/(label+'.pt'))], 300)
run('readout', ['-m', 'app.imitate_yumi', 'fit', '--source', 'runs/source/best.pt',
    '--train', str(root/'train.pt'), '--validation', str(root/'validation.pt'),
    '--mode', 'readout', '--output', str(root/'readout')], 1800)
run('readout_native', ['-m', 'app.evaluate_locomotion', '--checkpoints', str(root/'readout/last.pt'),
    '--seed-base', '103001', '--output', str(root/'readout_native.json')], 900)
report.update(completed=True, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
persist()
