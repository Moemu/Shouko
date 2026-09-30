"""Evaluate a previously fixed readout candidate on untouched direction holdouts."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--branch', choices=['main', 'replica'], required=True)
parser.add_argument('--selection-file', default=None)
args = parser.parse_args()
root = Path('runs/direction_control_20260929')
folder = root/('transfer_'+args.branch)
prior = json.loads(Path(args.selection_file or folder/'execution.json').read_text())
assert prior['completed'] and prior.get('development_candidate')
checkpoint = prior['development_candidate']
assert hashlib.sha256(Path(checkpoint).read_bytes()).hexdigest() == prior['candidate_sha256']
output = folder/'holdout_execution.json'
if output.exists():
    raise FileExistsError(output)
base = 350001 if args.branch == 'main' else 450001
report = dict(completed=False, jobs=[], results={}, checkpoint=checkpoint, checkpoint_sha256=prior['candidate_sha256'],
              started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())


def persist():
    temporary = output.with_suffix('.tmp'); temporary.write_text(json.dumps(report, indent=2)); temporary.replace(output)


def evaluate(label, checkpoints, seed, extra):
    path = folder/(label+'.json')
    command = [sys.executable, '-u', '-m', 'app.evaluate_direction', '--checkpoints', *checkpoints,
               '--seed-base', str(seed), '--output', str(path), *extra]
    row = dict(name=label, command=command); report['jobs'].append(row); persist()
    started = time.monotonic()
    with (folder/(label+'.log')).open('w') as log:
        result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8'), timeout=1200)
    row.update(returncode=result.returncode, wall_seconds=time.monotonic()-started); persist()
    if result.returncode:
        raise RuntimeError(label)
    data = json.loads(path.read_text()); assert data['completed']
    summary = data['conditions'][0]['summary']; report['results'][label] = summary; persist()
    print(json.dumps(dict(branch=args.branch, stage=label, **summary)), flush=True)
    return data


persist()
passed = True
for label, delta, extra, required in [('holdout_normal', 0, ['--cohorts', '3'], 24),
    ('holdout_yaw', 1000, ['--yaws', '-.2', '.2'], 16),
    ('holdout_long', 2000, ['--seconds', '120', '--trace'], 8),
    ('holdout_push_positive', 3000, ['--push-velocity', '.25', '--trace'], 8),
    ('holdout_push_negative', 4000, ['--push-velocity', '-.25', '--trace'], 8)]:
    data = evaluate(label, [checkpoint], base+delta, extra)
    summary = data['conditions'][0]['summary']
    passed = passed and summary['strict_passes'] >= required and summary['falls'] <= 1
    if label == 'holdout_long':
        passed = passed and summary['direction_passes'] == 9
report['heldout_passed'] = passed; persist()
evaluate('source_matched_long', [f'runs/source_{args.branch}/last.pt'], base+2000, ['--seconds', '120', '--trace'])
native = evaluate('native_normal', [checkpoint], base, ['--cohorts', '3', '--sparse-backend', 'native'])
report['native_passed'] = native['conditions'][0]['summary']['strict_passes'] >= 24 and native['conditions'][0]['summary']['falls'] <= 1
if args.branch == 'main':
    first = evaluate('repeat_first', [checkpoint], base, [])
    second = evaluate('repeat_second', [checkpoint], base, [])
    report['repeat_identical'] = first['conditions'][0]['results'][0]['trajectory_sha256'] == second['conditions'][0]['results'][0]['trajectory_sha256']
report.update(completed=True, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()); persist()
