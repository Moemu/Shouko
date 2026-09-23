"""One fixed convergence extension after the exact seed-2027 schedule failed direction."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('runs/yumi_route_20260923')
output = root/'support_refinement_execution.json'
if output.exists():
    raise FileExistsError(output)
checkpoint = root/'support_replicate2027_extend1024/last.pt'
prior = json.loads((root/'support_replication_execution.json').read_text())
assert prior['completed'] and not prior['second_seed_gate']['passed']
report = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), completed=False,
              phase='waiting_for_control', jobs=[], initialization_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              prior_transitions=65536000, added_transitions=200*1024*128,
              source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('app').glob('*.py')},
              limitation='Preserves the earlier failed 18/27 result. This is extra convergence work, not an equal-duration seed replication.')


def persist():
    temp = output.with_suffix('.tmp')
    temp.write_text(json.dumps(report, indent=2))
    temp.replace(output)


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
deadline = time.monotonic()+1800
while True:
    prior = json.loads((root/'equal_budget_execution.json').read_text())
    if prior['completed']:
        break
    if time.monotonic() > deadline:
        raise TimeoutError('Control did not finish')
    time.sleep(10)
report['phase'] = 'refining_seed2027'
persist()
name = 'support_refine2027'
run(name, ['-m', 'app.ppo_yumi', '--resume', str(checkpoint), '--resume-state', '--policy', 'mlp',
    '--gait-reward', 'phase_support', '--worlds', '1024', '--steps', '128', '--epochs', '4',
    '--minibatch', '4096', '--lr', '.0003', '--lr-final-frac', '1', '--knee-gate', 'stance',
    '--target-kl', '.02', '--std0', '.4', '--value-warmup', '0', '--eval-seconds', '30',
    '--episode-seconds', '6', '--eval-every', '200', '--max-iterations', '200', '--max-seconds', '900',
    '--seed', '2027', '--runs-dir', str(root/name)], 1050)
assert json.loads((root/name/'training.json').read_text())['iteration'] == 200
checkpoint = str(root/name/'last.pt')
native = root/'support_refine2027_native.json'
run(name+'_native', ['-m', 'app.evaluate_locomotion', '--checkpoints', checkpoint, '--output', str(native)], 360)
condition = json.loads(native.read_text())['conditions'][0]
tests = [t for g in condition['results'] for t in g['tests']]
strict = sum(t['success'] and all(f['qualifying_swings_per_second'] >= 1 for f in t['gait']['feet']) for t in tests)
report['development_gate'] = dict(strict_passes=strict, episodes=len(tests), falls=condition['summary']['falls'])
persist()
if strict >= 24 and condition['summary']['falls'] <= 1:
    report.update(phase='holdout', holdout_candidate=checkpoint)
    persist()
    for label, extra in [
        ('support_refined_holdout', ['--seed-base', '97001']),
        ('support_refined_yaw_holdout', ['--seed-base', '97101', '--cohorts', '1', '--yaws', '-.2', '.2']),
        ('support_refined_long_holdout', ['--seed-base', '97201', '--cohorts', '1', '--seconds', '120']),
        ('support_refined_push_holdout', ['--seed-base', '97301', '--cohorts', '1', '--push']),
    ]:
        run(label, ['-m', 'app.evaluate_locomotion', '--checkpoints', checkpoint, *extra,
                    '--output', str(root/(label+'.json'))], 480)
else:
    report['stopped_at_development_gate'] = True
report.update(completed=True, phase='complete', finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
persist()
