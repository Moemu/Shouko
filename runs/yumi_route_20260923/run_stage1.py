"""Bounded single-GPU baseline and paired recovery jobs; never changes instance power."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('runs/yumi_route_20260923')
execution = ROOT/'stage1_execution.json'
if execution.exists():
    raise FileExistsError(execution)
report = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), jobs=[], completed=False)


def run(name, args, timeout):
    entry = dict(name=name, command=[sys.executable, '-u', *args], timeout_seconds=timeout,
                 started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    report['jobs'].append(entry)
    execution.write_text(json.dumps(report, indent=2))
    started = time.monotonic()
    with (ROOT/(name+'.log')).open('w') as handle:
        try:
            result = subprocess.run(entry['command'], stdout=handle, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8'), timeout=timeout)
            entry['returncode'] = result.returncode
        except subprocess.TimeoutExpired:
            entry['returncode'] = 'timeout'
    entry['wall_seconds'] = time.monotonic()-started
    execution.write_text(json.dumps(report, indent=2))
    print(json.dumps(entry), flush=True)
    if entry['returncode'] != 0:
        raise RuntimeError(name)


deadline = time.monotonic()+600
while True:
    baseline_path = ROOT/'baseline_native.json'
    baseline = json.loads(baseline_path.read_text()) if baseline_path.exists() else {}
    if baseline.get('completed'):
        break
    if time.monotonic() > deadline:
        raise TimeoutError('Baseline calibration did not complete')
    time.sleep(10)
assert baseline['completed']
conditions = baseline['conditions']
assert all(c['summary']['successes'] == 2 for c in conditions)
for cohort in range(3):
    assert len({c['results'][cohort]['trajectory_sha256'] for c in conditions}) == 1

run('cpu_regression', ['-m', 'app.test_recovery_curriculum'], 60)
common = ['-m', 'app.ppo_yumi', '--resume', 'runs/yumi_obs50/best.pt', '--steps', '128',
          '--epochs', '4', '--lr-final-frac', '1', '--knee-gate', 'stance', '--target-kl', '0.02',
          '--eval-seconds', '30', '--episode-seconds', '6']
run('mlp_seed2026', [*common, '--policy', 'mlp', '--worlds', '256', '--minibatch', '1024',
                    '--lr', '0.0003', '--value-warmup', '8', '--eval-every', '100',
                    '--max-iterations', '400', '--max-seconds', '1200', '--seed', '2026',
                    '--runs-dir', str(ROOT/'mlp_seed2026')], 1350)
run('mlp_seed2026_native', ['-m', 'app.evaluate_locomotion', '--checkpoints',
                          str(ROOT/'mlp_seed2026/best.pt'), str(ROOT/'mlp_seed2026/last.pt'),
                          '--output', str(ROOT/'mlp_seed2026_native.json')], 360)
for seed in [2026, 2027]:
    for condition, yaws in [('control', ['0']), ('recovery', ['0', '-0.1', '0.1', '-0.2', '0.2'])]:
        name = f'{condition}_seed{seed}'
        run(name, [*common, '--resume-state', '--worlds', '128', '--minibatch', '512',
                   '--freeze-brain', '--lr', '0.000005', '--value-warmup', '1', '--eval-every', '21',
                   '--max-iterations', '21', '--max-seconds', '1200', '--seed', str(seed),
                   '--initial-yaws', *yaws, '--runs-dir', str(ROOT/name)], 1350)
        status = json.loads((ROOT/name/'training.json').read_text())
        if status['iteration'] != 21:
            raise RuntimeError('Incomplete paired training budget: '+name)
    checkpoints = [str(ROOT/f'{c}_seed{seed}/last.pt') for c in ['control', 'recovery']]
    run(f'paired_seed{seed}', ['-m', 'app.evaluate_locomotion', '--checkpoints', *checkpoints,
                             '--output', str(ROOT/f'paired_seed{seed}.json')], 480)
    run(f'recovery_seed{seed}_native', ['-m', 'app.evaluate_locomotion', '--checkpoints',
                               'runs/yumi_obs50/best.pt', *checkpoints, '--seed-base', '95001',
                               '--cohorts', '1', '--seconds', '12', '--yaws', '-0.2', '0.2',
                               '--output', str(ROOT/f'recovery_seed{seed}_native.json')], 480)
report['completed'] = True
report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
execution.write_text(json.dumps(report, indent=2))
