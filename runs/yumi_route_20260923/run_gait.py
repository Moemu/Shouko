"""Single-GPU paired reward probe, after the original recovery queue completes."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path('runs/yumi_route_20260923')
stage1 = Path('../route-20260923')/ROOT/'stage1_execution.json'
execution = ROOT/'gait_execution.json'
if execution.exists():
    raise FileExistsError(execution)
report = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), jobs=[], completed=False)


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
    print(json.dumps(entry), flush=True)
    if entry['returncode'] != 0:
        raise RuntimeError(name)


persist()
deadline = time.monotonic()+7200
while True:
    try:
        first = json.loads(stage1.read_text())
    except json.JSONDecodeError:
        time.sleep(2)
        continue
    if first.get('completed'):
        break
    if time.monotonic() > deadline:
        raise TimeoutError('Recovery queue did not finish; GPU not taken over')
    if any(j.get('returncode', 0) != 0 for j in first['jobs']):
        raise RuntimeError('Recovery queue failed; inspect before proceeding')
    time.sleep(15)

source = Path('../route-20260923')/ROOT/'mlp_seed2026/best.pt'
report['initialization_sha256'] = hashlib.sha256(source.read_bytes()).hexdigest()
report['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('app').glob('*.py')}
persist()
run('gait_regression', ['-m', 'app.test_gait_reward'], 60)
common = ['-m', 'app.ppo_yumi', '--resume', str(source), '--policy', 'mlp', '--worlds', '256',
          '--steps', '128', '--epochs', '4', '--minibatch', '1024', '--lr', '.0001', '--lr-final-frac', '1',
          '--knee-gate', 'stance', '--target-kl', '.02', '--std0', '.25', '--value-warmup', '8',
          '--eval-seconds', '30', '--episode-seconds', '6', '--eval-every', '100',
          '--max-iterations', '400', '--max-seconds', '1200']
for seed in [3026, 3027]:
    for condition, reward in [('control', 'none'), ('phase', 'phase_clearance')]:
        name = f'gait_{condition}_seed{seed}'
        run(name, [*common, '--seed', str(seed), '--gait-reward', reward, '--runs-dir', str(ROOT/name)], 1350)
        status = json.loads((ROOT/name/'training.json').read_text())
        if status['iteration'] != 400:
            raise RuntimeError('Incomplete paired reward training: '+name)
    output = ROOT/f'gait_paired_seed{seed}.json'
    run(f'gait_paired_seed{seed}', ['-m', 'app.evaluate_locomotion', '--checkpoints',
                                  *[str(ROOT/f'gait_{c}_seed{seed}/last.pt') for c in ['control', 'phase']],
                                  '--output', str(output)], 360)
    results = json.loads(output.read_text())['conditions']
    control, phase = [c['summary']['qualifying_swings_per_second'] for c in results]
    if phase <= max(control, .1):
        report['second_seed_skipped'] = 'No meaningful swing signal in pilot; inspect shaping before more budget'
        break
report['completed'] = True
report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
persist()
