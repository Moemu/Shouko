"""Fixed 52.4M-transition feasibility extension after the registered 400-round gate."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('runs/yumi_route_20260923')
output = root/'support_extension_execution.json'
if output.exists():
    raise FileExistsError(output)
previous = json.loads((root/'support_execution.json').read_text())
assert previous['completed'] and previous.get('stopped_at_development_gate') == 'support_seed2026'
checkpoint = root/'support_seed2026/last.pt'
report = dict(started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), completed=False, jobs=[],
              source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('app').glob('*.py')},
              initialization_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
              prior_transitions=400*256*128, added_transition_budget=400*1024*128,
              limitation='Full optimizer warm-start; new environment/RNG launch. World/minibatch sizes changed for throughput; not a one-factor control.')


def persist():
    temporary = output.with_suffix('.tmp')
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(output)


def run(name, args, timeout):
    row = dict(name=name, command=[sys.executable, '-u', *args], timeout_seconds=timeout)
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


persist()
run('support_extend1024', ['-m', 'app.ppo_yumi', '--resume', str(checkpoint), '--resume-state',
    '--policy', 'mlp', '--gait-reward', 'phase_support', '--worlds', '1024', '--steps', '128',
    '--epochs', '4', '--minibatch', '4096', '--lr', '.0003', '--lr-final-frac', '1',
    '--knee-gate', 'stance', '--target-kl', '.02', '--std0', '.4', '--value-warmup', '0',
    '--eval-seconds', '30', '--episode-seconds', '6', '--eval-every', '200',
    '--max-iterations', '400', '--max-seconds', '1200', '--seed', '2026',
    '--runs-dir', str(root/'support_extend1024')], 1350)
status = json.loads((root/'support_extend1024/training.json').read_text())
assert status['iteration'] == 400
run('support_extend1024_native', ['-m', 'app.evaluate_locomotion', '--checkpoints',
    str(root/'support_extend1024/last.pt'), '--output', str(root/'support_extend1024_native.json')], 360)
report['completed'] = True
report['finished_utc'] = datetime.datetime.now(datetime.timezone.utc).isoformat()
persist()
