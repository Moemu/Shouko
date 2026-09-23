"""Replicate the complete two-stage schedule only after its first strict development pass."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('runs/yumi_route_20260923')
output = root/'support_replication_execution.json'
if output.exists():
    raise FileExistsError(output)
report = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), completed=False,
              phase='waiting_for_extension', jobs=[],
              source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('app').glob('*.py')},
              initialization_sha256=hashlib.sha256(Path('runs/yumi_obs50/best.pt').read_bytes()).hexdigest())


def persist():
    temp = output.with_suffix('.tmp')
    temp.write_text(json.dumps(report, indent=2))
    temp.replace(output)


def strict_gate(path):
    c = json.loads(path.read_text())['conditions'][0]
    tests = [t for g in c['results'] for t in g['tests']]
    strict = sum(t['success'] and all(f['qualifying_swings_per_second'] >= 1 for f in t['gait']['feet']) for t in tests)
    return dict(strict_passes=strict, episodes=len(tests), falls=c['summary']['falls'],
                passed=strict >= 24 and c['summary']['falls'] <= 1)


def run(name, args, timeout):
    row = dict(name=name, command=[sys.executable, '-u', *args], timeout_seconds=timeout,
               started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    report['jobs'].append(row)
    persist()
    start = time.monotonic()
    with (root/(name+'.log')).open('w') as handle:
        try:
            result = subprocess.run(row['command'], stdout=handle, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8'), timeout=timeout)
            row['returncode'] = result.returncode
        except subprocess.TimeoutExpired:
            row['returncode'] = 'timeout'
    row['wall_seconds'] = time.monotonic()-start
    persist()
    if row['returncode'] != 0:
        raise RuntimeError(name)


persist()
deadline = time.monotonic()+1500
while True:
    prior = json.loads((root/'support_extension_execution.json').read_text())
    if prior['completed']:
        break
    if time.monotonic() > deadline:
        raise TimeoutError('Extension did not finish')
    time.sleep(10)
gate = strict_gate(root/'support_extend1024_native.json')
report['first_seed_gate'] = gate
persist()
if not gate['passed']:
    report.update(completed=True, phase='first_seed_gate_failed')
    persist()
    sys.exit(0)
report['phase'] = 'replicating_seed2027'
persist()
common = ['-m', 'app.ppo_yumi', '--policy', 'mlp', '--gait-reward', 'phase_support',
          '--steps', '128', '--epochs', '4', '--lr', '.0003', '--lr-final-frac', '1',
          '--knee-gate', 'stance', '--target-kl', '.02', '--std0', '.4', '--eval-seconds', '30',
          '--episode-seconds', '6', '--max-iterations', '400', '--max-seconds', '1200', '--seed', '2027']
first = 'support_replicate_seed2027'
second = 'support_replicate2027_extend1024'
run(first, [*common, '--resume', 'runs/yumi_obs50/best.pt', '--worlds', '256', '--minibatch', '1024',
            '--value-warmup', '8', '--eval-every', '100', '--runs-dir', str(root/first)], 1350)
assert json.loads((root/first/'training.json').read_text())['iteration'] == 400
run(second, [*common, '--resume', str(root/first/'last.pt'), '--resume-state', '--worlds', '1024',
             '--minibatch', '4096', '--value-warmup', '0', '--eval-every', '200', '--runs-dir', str(root/second)], 1350)
assert json.loads((root/second/'training.json').read_text())['iteration'] == 400
checkpoint = str(root/second/'last.pt')
native = root/'support_replicate2027_native.json'
run('support_replicate2027_native', ['-m', 'app.evaluate_locomotion', '--checkpoints', checkpoint,
                                    '--output', str(native)], 360)
report['second_seed_gate'] = strict_gate(native)
persist()
if report['second_seed_gate']['passed']:
    # Candidate is fixed to the second seed before any holdout results are observed.
    report.update(phase='holdout', holdout_candidate=checkpoint)
    persist()
    for name, extra in [
        ('support_extended_holdout', ['--seed-base', '97001']),
        ('support_extended_yaw_holdout', ['--seed-base', '97101', '--cohorts', '1', '--yaws', '-.2', '.2']),
        ('support_extended_long_holdout', ['--seed-base', '97201', '--cohorts', '1', '--seconds', '120']),
        ('support_extended_push_holdout', ['--seed-base', '97301', '--cohorts', '1', '--push']),
    ]:
        run(name, ['-m', 'app.evaluate_locomotion', '--checkpoints', checkpoint, *extra,
                   '--output', str(root/(name+'.json'))], 480)
report.update(completed=True, phase='complete', finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
persist()
