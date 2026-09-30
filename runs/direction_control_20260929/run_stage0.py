"""Bounded paired fixed-policy direction diagnostics; no training or weight writes."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('runs/direction_control_20260929')
output = root/'stage0_execution.json'
if output.exists():
    raise FileExistsError(output)
models = ['runs/source_teacher/last.pt', 'runs/source_main/last.pt', 'runs/source_replica/last.pt']
expected = ['9c5ca9339b26f5af2e8acf94e7edd3b9371ef5efb41d647bcbd7904243ac50e1',
            'f1a200718ebfec845cbf9c02ba743e937ea94c747fba53e7b12dfd12f8d3e1f9',
            '8841868cf1cbbbd01be19585bc18e751c11901b3e806556705d124bb207616cf']
for model, digest in zip(models, expected):
    assert hashlib.sha256(Path(model).read_bytes()).hexdigest() == digest
report = dict(completed=False, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), jobs=[],
              source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('app').glob('*.py')})


def persist():
    temp = output.with_suffix('.tmp')
    temp.write_text(json.dumps(report, indent=2)); temp.replace(output)


def run(name, checkpoints, extra):
    command = [sys.executable, '-u', '-m', 'app.evaluate_direction', '--checkpoints', *checkpoints,
               '--output', str(root/(name+'.json')), '--seed-base', '310001', *extra]
    row = dict(name=name, command=command, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    report['jobs'].append(row); persist()
    start = time.monotonic()
    with (root/(name+'.log')).open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8'), timeout=1200)
    row.update(returncode=result.returncode, wall_seconds=time.monotonic()-start); persist()
    if result.returncode:
        raise RuntimeError(name)
    result = json.loads((root/(name+'.json')).read_text())
    assert result['completed']
    print(json.dumps(dict(name=name, wall_seconds=row['wall_seconds'], summaries=[c['summary'] for c in result['conditions']])), flush=True)


persist()
run('baseline_long', models, ['--seconds', '120', '--trace'])
for sign, value in [('positive', '.25'), ('negative', '-.25')]:
    run('push_'+sign, models, ['--push-velocity', value, '--trace'])
for option in ['vy-command', 'hip-roll-offset', 'ankle-roll-offset']:
    for sign, value in [('positive', '.05'), ('negative', '-.05')]:
        run(option.replace('-', '_')+'_'+sign, models[:2], ['--'+option, value])
report.update(completed=True, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()); persist()
