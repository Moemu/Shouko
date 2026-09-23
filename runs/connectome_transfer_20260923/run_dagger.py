import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('runs/connectome_transfer_20260923')
output = root/'dagger_execution.json'
if output.exists():
    raise FileExistsError(output)
report = dict(completed=False, jobs=[], rounds=[], started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())

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

student = root/'readout/last.pt'
train = [str(root/'train.pt')]
for iteration in range(1, 7):
    label = f'dagger{iteration}'
    data = root/(label+'.pt')
    run(label+'_collect', ['-m', 'app.imitate_yumi', 'collect', '--teacher', 'runs/teacher/last.pt',
        '--student', str(student), '--beta', '0', '--seed-base', str(104001+100*iteration),
        '--output', str(data)], 600)
    train.append(str(data))
    run(label+'_fit', ['-m', 'app.imitate_yumi', 'fit', '--source', str(student), '--train', *train,
        '--validation', str(root/'validation.pt'), '--mode', 'readout', '--output', str(root/label)], 1800)
    student = root/label/'last.pt'
    native = root/(label+'_native.json')
    run(label+'_native', ['-m', 'app.evaluate_locomotion', '--checkpoints', str(student),
        '--seed-base', '103001', '--output', str(native)], 900)
    condition = json.loads(native.read_text())['conditions'][0]
    tests = [t for g in condition['results'] for t in g['tests']]
    strict = sum(t['success'] and all(f['qualifying_swings_per_second'] >= 1 for f in t['gait']['feet']) for t in tests)
    row = dict(round=iteration, strict=strict, falls=condition['summary']['falls'],
               mean_seconds=sum(t['seconds'] for t in tests)/len(tests), summary=condition['summary'])
    report['rounds'].append(row)
    persist()
    if strict >= 24 and row['falls'] <= 1:
        report['development_candidate'] = str(student)
        break
report.update(completed=True, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
persist()
