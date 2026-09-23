"""Fresh-actor support-timing probe; only successful development policies reach holdout."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('runs/yumi_route_20260923')
execution = ROOT/'support_execution.json'
if execution.exists():
    raise FileExistsError(execution)
previous = json.loads((Path('../route-gait-20260923')/ROOT/'gait_execution.json').read_text())
assert previous['completed']
assert all(j['returncode'] == 0 for j in previous['jobs'])
report = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), jobs=[], completed=False,
              source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('app').glob('*.py')},
              initialization_sha256=hashlib.sha256(Path('runs/yumi_obs50/best.pt').read_bytes()).hexdigest())


def persist():
    tmp = execution.with_suffix('.tmp')
    tmp.write_text(json.dumps(report, indent=2))
    tmp.replace(execution)


def run(name, args, timeout):
    entry = dict(name=name, command=[sys.executable, '-u', *args], timeout_seconds=timeout,
                 started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    report['jobs'].append(entry)
    persist()
    started = time.monotonic()
    with (ROOT/(name+'.log')).open('w') as handle:
        try:
            result = subprocess.run(entry['command'], stdout=handle, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8'), timeout=timeout)
            entry['returncode'] = result.returncode
        except subprocess.TimeoutExpired:
            entry['returncode'] = 'timeout'
    entry['wall_seconds'] = time.monotonic()-started
    persist()
    if entry['returncode'] != 0:
        raise RuntimeError(name)


persist()
run('support_regression', ['-m', 'app.test_gait_reward'], 60)
common = ['-m', 'app.ppo_yumi', '--resume', 'runs/yumi_obs50/best.pt', '--policy', 'mlp',
          '--gait-reward', 'phase_support', '--worlds', '256', '--steps', '128', '--epochs', '4',
          '--minibatch', '1024', '--lr', '.0003', '--lr-final-frac', '1', '--knee-gate', 'stance',
          '--target-kl', '.02', '--std0', '.4', '--value-warmup', '8', '--eval-seconds', '30',
          '--episode-seconds', '6', '--eval-every', '100', '--max-iterations', '400', '--max-seconds', '1200']
passed = []
for seed in [2026, 2027]:
    name = f'support_seed{seed}'
    run(name, [*common, '--seed', str(seed), '--runs-dir', str(ROOT/name)], 1350)
    status = json.loads((ROOT/name/'training.json').read_text())
    if status['iteration'] != 400:
        raise RuntimeError('Incomplete development budget')
    output = ROOT/(name+'_native.json')
    run(name+'_native', ['-m', 'app.evaluate_locomotion', '--checkpoints', str(ROOT/name/'last.pt'),
                         '--output', str(output)], 360)
    condition = json.loads(output.read_text())['conditions'][0]
    tests = [t for g in condition['results'] for t in g['tests']]
    strict = sum(t['success'] and all(f['qualifying_swings_per_second'] >= 1 for f in t['gait']['feet']) for t in tests)
    report[name+'_strict_passes'] = strict
    persist()
    if strict < 24 or condition['summary']['falls'] > 1:
        report['stopped_at_development_gate'] = name
        break
    passed.append(name)
if len(passed) == 2:
    # Select the second independent training run in advance, without ranking holdout scores.
    checkpoint = str(ROOT/passed[-1]/'last.pt')
    report['holdout_candidate'] = checkpoint
    persist()
    for name, extra in [
        ('support_holdout', ['--seed-base', '97001']),
        ('support_yaw_holdout', ['--seed-base', '97101', '--cohorts', '1', '--yaws', '-.2', '.2']),
        ('support_long_holdout', ['--seed-base', '97201', '--cohorts', '1', '--seconds', '120']),
        ('support_push_holdout', ['--seed-base', '97301', '--cohorts', '1', '--push']),
    ]:
        run(name, ['-m', 'app.evaluate_locomotion', '--checkpoints', checkpoint, *extra,
                   '--output', str(ROOT/(name+'.json'))], 480)
report['completed'] = True
report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
persist()
