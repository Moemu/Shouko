"""Bounded sequential A/B and paired evaluation. Never changes instance power."""
import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('runs/yumi_actor_scale_ab_20260922')
report = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              instance_policy='keep_running', jobs=[])
environment = dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8')


def run(name, arguments, timeout):
    command = [sys.executable, '-u', *arguments]
    entry = dict(name=name, command=command, timeout_seconds=timeout)
    report['jobs'].append(entry)
    started = time.monotonic()
    with (root / (name + '.log')).open('w') as output:
        try:
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT,
                                    env=environment, timeout=timeout)
            entry['returncode'] = result.returncode
        except subprocess.TimeoutExpired:
            entry['returncode'] = 'timeout'
    entry['wall_seconds'] = time.monotonic() - started
    (root / 'execution.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(entry), flush=True)
    if entry['returncode'] != 0:
        raise RuntimeError(f'{name} failed; dependent jobs skipped')


run('precheck', ['precheck.py'], 90)
common = ['-m', 'app.ppo_yumi', '--resume', 'runs/yumi_obs50/best.pt', '--resume-state',
          '--worlds', '128', '--steps', '128', '--minibatch', '512', '--epochs', '4',
          '--freeze-brain', '--lr', '0.000005', '--lr-final-frac', '1',
          '--value-warmup', '1', '--knee-gate', 'stance', '--target-kl', '0.02',
          '--eval-seconds', '30', '--eval-every', '7', '--max-iterations', '7',
          '--max-seconds', '480', '--seed', '2026']
for name, extra in [('control', []), ('scaled', ['--actor-new-input-std',
                      '0.08117300271987915', '0.06434640288352966', '0.05'])]:
    directory = root / name
    if directory.exists():
        raise FileExistsError(f'Fresh run directory required: {directory}')
    run(name, [*common, '--runs-dir', str(directory), *extra], 600)
    status = json.loads((directory / 'training.json').read_text())
    if status['iteration'] != 7 or sum(not row['value_warmup'] for row in status['history']) != 6:
        raise RuntimeError(f'{name} did not complete matched actor rounds')
run('paired', ['-m', 'app.compare_policies', '--checkpoints', 'runs/yumi_obs50/best.pt',
               str(root / 'control/last.pt'), str(root / 'scaled/last.pt'),
               '--seeds', '92001', '92002', '92003', '--worlds', '32', '--seconds', '30',
               '--output', str(root / 'paired.json')], 360)
report['completed_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
(root / 'execution.json').write_text(json.dumps(report, indent=2))
