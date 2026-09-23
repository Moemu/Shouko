"""Complete the old-reward counterfactual at the same two-stage sample budget."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('runs/yumi_route_20260923')
output = root/'equal_budget_execution.json'
if output.exists():
    raise FileExistsError(output)
checkpoint = Path('../route-20260923')/root/'mlp_seed2026/last.pt'
report = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), completed=False,
              phase='waiting_for_replication', jobs=[],
              source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('app').glob('*.py')},
              initialization_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              first_stage_transitions=400*256*128, second_stage_transitions=400*1024*128,
              limitation='Matched two-stage sample/world/minibatch schedule and seed; ordinary CUDA training is not promised bitwise deterministic.')


def persist():
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(output)


persist()
deadline = time.monotonic()+7200
while True:
    prior = json.loads((root/'support_replication_execution.json').read_text())
    if prior['completed']:
        break
    if time.monotonic() > deadline:
        raise TimeoutError('Replication did not finish')
    time.sleep(10)
report['phase'] = 'equal_budget_control'
persist()
commands = [
    ('old_reward_extend1024', ['-m', 'app.ppo_yumi', '--resume', str(checkpoint), '--resume-state',
      '--policy', 'mlp', '--gait-reward', 'none', '--worlds', '1024', '--steps', '128', '--epochs', '4',
      '--minibatch', '4096', '--lr', '.0003', '--lr-final-frac', '1', '--knee-gate', 'stance',
      '--target-kl', '.02', '--std0', '.4', '--value-warmup', '0', '--eval-seconds', '30',
      '--episode-seconds', '6', '--eval-every', '200', '--max-iterations', '400', '--max-seconds', '1200',
      '--seed', '2026', '--runs-dir', str(root/'old_reward_extend1024')], 1350),
    ('old_reward_extend1024_native', ['-m', 'app.evaluate_locomotion', '--checkpoints',
      str(root/'old_reward_extend1024/last.pt'), '--output', str(root/'old_reward_extend1024_native.json')], 360),
]
for name, args, timeout in commands:
    row = dict(name=name, command=[sys.executable, '-u', *args], timeout_seconds=timeout,
               started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    report['jobs'].append(row)
    persist()
    started = time.monotonic()
    with (root/(name+'.log')).open('w') as handle:
        try:
            result = subprocess.run(row['command'], stdout=handle, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8'), timeout=timeout)
            row['returncode'] = result.returncode
        except subprocess.TimeoutExpired:
            row['returncode'] = 'timeout'
    row['wall_seconds'] = time.monotonic()-started
    persist()
    if row['returncode'] != 0:
        raise RuntimeError(name)
    if name == 'old_reward_extend1024':
        assert json.loads((root/name/'training.json').read_text())['iteration'] == 400
report.update(completed=True, phase='complete', finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
persist()
