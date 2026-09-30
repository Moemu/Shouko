"""One bounded readout-only PPO feasibility experiment after fixed-data refit failed."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

root = Path('runs/direction_control_20260929')
source = root/'readout_ppo_warm3029/last.pt'
assert not json.loads((root/'transfer_main/refit_execution.json').read_text())['passed']
output = root/'yaw_gain28_execution.json'
if output.exists():
    raise FileExistsError(output)
report = dict(completed=False, jobs=[], results={}, source_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('app').glob('*.py')})


def persist():
    temp = output.with_suffix('.tmp'); temp.write_text(json.dumps(report, indent=2)); temp.replace(output)


def run(name, arguments, timeout=1200):
    row = dict(name=name, command=[sys.executable, '-u', *arguments]); report['jobs'].append(row); persist()
    start = time.monotonic()
    with (root/(name+'.log')).open('w') as log:
        result = subprocess.run(row['command'], stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8'), timeout=timeout)
    row.update(returncode=result.returncode, wall_seconds=time.monotonic()-start); persist()
    if result.returncode:
        raise RuntimeError(name)


def evaluate(label, checkpoint, extra):
    path = root/(label+'.json')
    run(label, ['-m', 'app.evaluate_direction', '--checkpoints', str(checkpoint), '--seed-base', '310201',
               '--output', str(path), *extra])
    data = json.loads(path.read_text()); assert data['completed']
    summary = data['conditions'][0]['summary']; report['results'][label] = summary; persist()
    print(json.dumps(dict(stage=label, **summary)), flush=True)
    return summary


persist()
checkpoint = root/'readout_ppo_warm3029/last.pt'
passed = True
for label, extra, required in [('long', ['--seconds', '120', '--trace'], 8),
    ('normal', ['--cohorts', '3'], 24), ('yaw', ['--yaws', '-.2', '.2'], 16),
    ('push_positive', ['--push-velocity', '.25'], 8), ('push_negative', ['--push-velocity', '-.25'], 8)]:
    summary = evaluate('yaw_gain28_'+label, checkpoint, ['--yaw-feedback-gain', '2.8', *extra])
    if summary['strict_passes'] < required or summary['falls'] > 1 or (label == 'long' and summary['direction_passes'] != 9):
        passed = False; break
report.update(development_passed=passed, checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), yaw_feedback_gain=2.8)
report.update(completed=True, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()); persist()
