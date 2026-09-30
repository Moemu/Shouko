"""Bake the first passing bounded trim into readout bias, then evaluate unseen seeds."""
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
output = root/'calibrated_execution.json'
if output.exists():
    raise FileExistsError(output)
prior = json.loads((root/'probe_validation_execution.json').read_text())
assert prior['completed']
passed = [c for c in prior['candidates'] if c['model'] == 1 and c['passed']]
assert passed, 'No main-model probe passed the full development gate'
chosen = passed[0]['name']
assert chosen == 'hip_roll_offset_negative', 'This preregistration covers the first hip correction only'
report = dict(completed=False, jobs=[], branches={}, chosen=chosen,
              started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('app').glob('*.py')})


def persist():
    temp = output.with_suffix('.tmp'); temp.write_text(json.dumps(report, indent=2)); temp.replace(output)


def bake(label, source):
    target = root/label/'last.pt'; target.parent.mkdir(exist_ok=False)
    source_hash = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    checkpoint = torch.load(source, weights_only=True, map_location='cpu', mmap=True)
    original_bias = checkpoint['state_dict']['readout.bias'].clone()
    checkpoint['state_dict']['readout.bias'] = original_bias.clone()
    checkpoint['state_dict']['readout.bias'][[1, 7]] -= .05
    checkpoint['extra'] = dict(method='bounded readout-bias calibration', source_sha256=source_hash,
                              changed_indices=[1, 7], action_offset=-.05, development_probe=chosen)
    torch.save(checkpoint, target)
    reloaded = torch.load(target, weights_only=True, map_location='cpu', mmap=True)
    original = torch.load(source, weights_only=True, map_location='cpu', mmap=True)
    changed = [key for key, value in original['state_dict'].items() if not torch.equal(value, reloaded['state_dict'][key])]
    assert changed == ['readout.bias'], changed
    changed_indices = torch.nonzero(original_bias != reloaded['state_dict']['readout.bias']).flatten().tolist()
    assert changed_indices == [1, 7]
    assert hashlib.sha256(Path(source).read_bytes()).hexdigest() == source_hash
    report['branches'][label] = dict(source=source, source_sha256=source_hash,
        checkpoint=str(target), checkpoint_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),
        changed_tensors=changed, changed_indices=changed_indices, all_other_tensors_identical=True, results={})
    persist()
    return target


def evaluate(label, checkpoint, suffix, extra):
    name = label+'_'+suffix; path = root/(name+'.json')
    command = [sys.executable, '-u', '-m', 'app.evaluate_direction', '--checkpoints', str(checkpoint), '--output', str(path), *extra]
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
    report['branches'][label]['results'][suffix] = summary; persist()
    print(json.dumps(dict(name=name, **summary)), flush=True)
    return summary


persist()
for label, source, base in [('calibrated_main', 'runs/source_main/last.pt', 350001),
                             ('calibrated_replica', 'runs/source_replica/last.pt', 450001)]:
    checkpoint = bake(label, source)
    if label == 'calibrated_main':
        dev = evaluate(label, checkpoint, 'baked_development_long', ['--seed-base', '310001', '--seconds', '120'])
        if dev['direction_passes'] != 9:
            report['stopped_reason'] = 'Baked candidate failed the fixed development gate'; persist(); break
    passed = True
    for suffix, delta, extra, minimum in [('normal', 0, ['--cohorts', '3'], 24),
        ('yaw', 1000, ['--yaws', '-.2', '.2'], 16), ('long', 2000, ['--seconds', '120', '--trace'], 8),
        ('push_positive', 3000, ['--push-velocity', '.25', '--trace'], 8),
        ('push_negative', 4000, ['--push-velocity', '-.25', '--trace'], 8)]:
        summary = evaluate(label, checkpoint, suffix, ['--seed-base', str(base+delta), *extra])
        passed = passed and summary['strict_passes'] >= minimum and summary['falls'] <= 1
        if suffix == 'long':
            passed = passed and summary['direction_passes'] == 9
    report['branches'][label]['heldout_passed'] = passed; persist()
report.update(completed=True, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()); persist()
