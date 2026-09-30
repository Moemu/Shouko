"""One predeclared ridge counterfactual on round-four data, with no new samples."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--branch', choices=['main', 'replica'], required=True)
args = parser.parse_args()
root = Path('runs/direction_control_20260929'); folder = root/('transfer_'+args.branch)
prior = json.loads((folder/'execution.json').read_text())
assert prior['completed'] and not prior.get('development_candidate') and len(prior['rounds']) == 4
output = folder/'refit_execution.json'
if output.exists():
    raise FileExistsError(output)
report = dict(completed=False, jobs=[], results={}, source_execution_sha256=hashlib.sha256((folder/'execution.json').read_bytes()).hexdigest())


def persist():
    temp = output.with_suffix('.tmp'); temp.write_text(json.dumps(report, indent=2)); temp.replace(output)


def run(name, arguments):
    row = dict(name=name, command=[sys.executable, '-u', *arguments]); report['jobs'].append(row); persist()
    start = time.monotonic()
    with (folder/(name+'.log')).open('w') as log:
        result = subprocess.run(row['command'], stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8'), timeout=1200)
    row.update(returncode=result.returncode, wall_seconds=time.monotonic()-start); persist()
    if result.returncode:
        raise RuntimeError(name)


persist()
refit = folder/'ridge1e5'
run('refit_ridge1e5', [str(root/'refit_cached.py'), '--reference', str(folder/'round4'), '--ridge', '.00001', '--output', str(refit)])
checkpoint = refit/'last.pt'
old = torch.load(folder/'round4/last.pt', map_location='cpu', weights_only=True, mmap=True)
new = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
changed = [key for key, value in old['state_dict'].items() if not torch.equal(value, new['state_dict'][key])]
assert set(changed).issubset({'readout.weight', 'readout.bias'}), changed
del old, new
report.update(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(), changed_tensors=changed); persist()
passed = True
for label, extra, required in [('normal', ['--cohorts', '3'], 24), ('long', ['--seconds', '120', '--trace'], 8),
    ('yaw', ['--yaws', '-.2', '.2'], 16), ('push_positive', ['--push-velocity', '.25'], 8), ('push_negative', ['--push-velocity', '-.25'], 8)]:
    path = folder/('ridge1e5_'+label+'.json')
    run('ridge1e5_'+label, ['-m', 'app.evaluate_direction', '--checkpoints', str(checkpoint),
        '--seed-base', '310201' if args.branch == 'main' else '410201', '--output', str(path), *extra])
    data = json.loads(path.read_text()); assert data['completed']
    summary = data['conditions'][0]['summary']; report['results'][label] = summary; persist()
    print(json.dumps(dict(branch=args.branch, stage=label, **summary)), flush=True)
    if summary['strict_passes'] < required or summary['falls'] > 1 or (label == 'long' and summary['direction_passes'] != 9):
        passed = False
        break
if passed:
    report.update(development_candidate=str(checkpoint), candidate_sha256=report['checkpoint_sha256'])
report.update(completed=True, passed=passed); persist()
