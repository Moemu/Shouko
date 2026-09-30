"""Bounded readout-only migration with signed disturbance DAgger and cached features."""
import argparse
import datetime
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
root = Path('runs/direction_control_20260929')
folder = root/('transfer_'+args.branch)
folder.mkdir(exist_ok=False)
selection = json.loads((root/'teacher_selection.json').read_text())
teacher = selection['checkpoint']
assert hashlib.sha256(Path(teacher).read_bytes()).hexdigest() == selection['checkpoint_sha256']
source = f'runs/source_{args.branch}/last.pt'
seed = 320001 if args.branch == 'main' else 420001
development_seed = 310201 if args.branch == 'main' else 410201
output = folder/'execution.json'
report = dict(completed=False, jobs=[], rounds=[], teacher=selection, source=source,
              source_checkpoint_sha256=hashlib.sha256(Path(source).read_bytes()).hexdigest(),
              started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('app').glob('*.py')})


def persist():
    temporary = output.with_suffix('.tmp'); temporary.write_text(json.dumps(report, indent=2)); temporary.replace(output)


def run(name, arguments, timeout=1500):
    row = dict(name=name, command=[sys.executable, '-u', *arguments]); report['jobs'].append(row); persist()
    started = time.monotonic()
    with (folder/(name+'.log')).open('w') as log:
        result = subprocess.run(row['command'], stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8'), timeout=timeout)
    row.update(returncode=result.returncode, wall_seconds=time.monotonic()-started); persist()
    if result.returncode:
        raise RuntimeError(name)


def collect(name, collection_seed, extra):
    path = folder/(name+'.pt')
    run(name, ['-m', 'app.imitate_yumi', 'collect', '--teacher', teacher, '--seed-base', str(collection_seed),
               '--output', str(path), *extra])
    return str(path)


def evaluate(checkpoint, iteration, label, extra, required):
    name = f'round{iteration}_{label}'; path = folder/(name+'.json')
    run(name, ['-m', 'app.evaluate_direction', '--checkpoints', str(checkpoint), '--seed-base', str(development_seed),
               '--output', str(path), *extra])
    data = json.loads(path.read_text()); assert data['completed']
    summary = data['conditions'][0]['summary']
    report['rounds'][-1][label] = summary; persist()
    print(json.dumps(dict(branch=args.branch, round=iteration, stage=label, **summary)), flush=True)
    return summary['strict_passes'] >= required and summary['falls'] <= 1 and (label != 'long' or summary['direction_passes'] == 9)


persist()
training = [collect('teacher_train', seed, ['--seconds', '12'])]
validation = collect('teacher_validation', seed+1000, ['--seconds', '12'])
student = source
for iteration in range(1, 5):
    report['rounds'].append(dict(round=iteration)); persist()
    base = seed+2000+iteration*100
    training.append(collect(f'round{iteration}_normal_data', base,
        ['--student', str(student), '--beta', '0', '--seconds', '12']))
    for label, delta, value in [('positive', 30, '.25'), ('negative', 40, '-.25')]:
        training.append(collect(f'round{iteration}_{label}_data', base+delta,
            ['--student', str(student), '--beta', '0', '--seconds', '30', '--cohorts', '1', '--yaws', '0', '--push-velocity', value]))
    fit = folder/f'round{iteration}'
    run(f'round{iteration}_fit', ['-m', 'app.imitate_yumi', 'fit', '--source', str(student), '--train', *training,
        '--validation', validation, '--mode', 'readout', '--ridge', '.0001', '--feature-cache', str(folder/'feature_cache'),
        '--output', str(fit)])
    student = fit/'last.pt'
    old = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
    new = torch.load(student, map_location='cpu', weights_only=True, mmap=True)
    changed = [name for name, value in old['state_dict'].items() if not torch.equal(value, new['state_dict'][name])]
    assert set(changed).issubset({'readout.weight', 'readout.bias'}), changed
    del old, new
    report['rounds'][-1].update(checkpoint_sha256=hashlib.sha256(student.read_bytes()).hexdigest(), changed_tensors=changed); persist()
    passed = True
    for label, extra, required in [('normal', ['--cohorts', '3'], 24), ('long', ['--seconds', '120'], 8),
        ('yaw', ['--yaws', '-.2', '.2'], 16), ('push_positive', ['--push-velocity', '.25'], 8),
        ('push_negative', ['--push-velocity', '-.25'], 8)]:
        if not evaluate(student, iteration, label, extra, required):
            passed = False
            break
    report['rounds'][-1]['passed'] = passed; persist()
    if passed:
        report.update(development_candidate=str(student), candidate_sha256=hashlib.sha256(student.read_bytes()).hexdigest())
        break
report.update(completed=True, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()); persist()
