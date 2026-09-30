"""Validate every preregistered probe passing the paired short-screen gate."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('runs/direction_control_20260929')
output = root/'probe_validation_execution.json'
if output.exists():
    raise FileExistsError(output)
analysis = json.loads((root/'stage0_analysis.json').read_text())
assert analysis['completed']
report = dict(completed=False, jobs=[], candidates=[], started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())


def persist():
    temp = output.with_suffix('.tmp'); temp.write_text(json.dumps(report, indent=2)); temp.replace(output)


def evaluate(probe, label, extra):
    option, sign = probe['name'].rsplit('_', 1)
    name = f'validate_{probe["name"]}_m{probe["model"]}_{label}'
    path = root/(name+'.json')
    command = [sys.executable, '-u', '-m', 'app.evaluate_direction', '--checkpoints', probe['checkpoint'],
               '--output', str(path), '--seed-base', '310001', '--'+option.replace('_', '-'), '.05' if sign == 'positive' else '-.05', *extra]
    row = dict(name=name, command=command); report['jobs'].append(row); persist()
    started = time.monotonic()
    with (root/(name+'.log')).open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8'), timeout=1200)
    row.update(returncode=result.returncode, wall_seconds=time.monotonic()-started); persist()
    if result.returncode:
        raise RuntimeError(name)
    result = json.loads(path.read_text()); assert result['completed']
    summary = result['conditions'][0]['summary']
    print(json.dumps(dict(name=name, **summary)), flush=True)
    return summary


persist()
for probe in analysis['probes']:
    if not probe['qualifies_for_long']:
        continue
    candidate = dict(name=probe['name'], model=probe['model'], checkpoint=probe['checkpoint'], passed=False)
    report['candidates'].append(candidate); persist()
    candidate['long'] = evaluate(probe, 'long', ['--seconds', '120', '--trace']); persist()
    if candidate['long']['direction_passes'] != 9:
        continue
    passed = True
    for label, extra, minimum in [('normal', ['--cohorts', '3'], 24), ('yaw', ['--yaws', '-.2', '.2'], 16),
                                 ('push_positive', ['--push-velocity', '.25', '--trace'], 8),
                                 ('push_negative', ['--push-velocity', '-.25', '--trace'], 8)]:
        candidate[label] = evaluate(probe, label, extra); persist()
        passed = passed and candidate[label]['strict_passes'] >= minimum and candidate[label]['falls'] <= 1
    candidate['passed'] = passed; persist()
report.update(completed=True, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()); persist()
