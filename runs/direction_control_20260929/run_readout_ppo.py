"""One bounded readout-only PPO feasibility experiment after fixed-data refit failed."""
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
source = root/'transfer_main/round4/last.pt'
assert not json.loads((root/'transfer_main/refit_execution.json').read_text())['passed']
output = root/'readout_ppo_execution.json'
if output.exists():
    raise FileExistsError(output)
report = dict(completed=False, jobs=[], results={}, source_checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
              started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('app').glob('*.py')})


def persist():
    temp = output.with_suffix('.tmp'); temp.write_text(json.dumps(report, indent=2)); temp.replace(output)


def run(name, arguments, timeout=1200):
    row = dict(name=name, command=[sys.executable, '-u', *arguments]); report['jobs'].append(row); persist()
    start = time.monotonic()
    with (root/(name+'.log')).open('w') as log:
        result = subprocess.run(row['command'], stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8'), timeout=timeout)
    row.update(returncode=result.returncode, wall_seconds=time.monotonic()-start); persist()
    if result.returncode:
        raise RuntimeError(name)


def evaluate(label, checkpoint, extra):
    path = root/(label+'.json')
    run(label, ['-m', 'app.evaluate_direction', '--checkpoints', str(checkpoint), '--seed-base', '310201',
               '--output', str(path), *extra])
    data = json.loads(path.read_text()); assert data['completed']
    summary = data['conditions'][0]['summary']; report['results'][label] = summary; persist()
    print(json.dumps(dict(stage=label, **summary)), flush=True)
    return summary


persist()
baseline = evaluate('readout_source_native', source, ['--cohorts', '3', '--sparse-backend', 'native'])
assert baseline['strict_passes'] >= 24 and baseline['falls'] <= 1
for label, value in [('positive', '.25'), ('negative', '-.25')]:
    evaluate('readout_source_push_'+label, source, ['--push-velocity', value])
folder = root/'readout_ppo3029'
run('readout_ppo3029', ['-m', 'app.ppo_yumi', '--resume', str(source), '--policy', 'connectome', '--readout-only',
    '--gait-reward', 'phase_support', '--lateral-precision', '6', '--worlds', '128', '--steps', '128', '--epochs', '4',
    '--minibatch', '512', '--lr', '.00001', '--lr-final-frac', '1', '--target-kl', '.01', '--std0', '.05', '--entropy', '0',
    '--value-warmup', '16', '--value-abort-vloss', '10', '--episode-seconds', '6', '--knee-gate', 'stance',
    '--eval-seconds', '30', '--eval-every', '160', '--max-iterations', '160', '--max-seconds', '1500',
    '--seed', '3029', '--runs-dir', str(folder)], 1650)
state = json.loads((folder/'training.json').read_text())
assert state['trainable_actor_parameters'] == 9792
checkpoint = folder/'last.pt'
old = torch.load(source, map_location='cpu', weights_only=True, mmap=True)
new = torch.load(checkpoint, map_location='cpu', weights_only=True, mmap=True)
changed = [key for key, value in old['state_dict'].items() if not torch.equal(value, new['state_dict'][key])]
assert set(changed).issubset({'readout.weight', 'readout.bias'}), changed
report.update(training_phase=state['phase'], iterations=state['iteration'], changed_tensors=changed,
              checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest()); persist()
if state['phase'] != 'value_not_converged':
    passed = True
    for label, extra, required in [('normal', ['--cohorts', '3'], 24), ('long', ['--seconds', '120', '--trace'], 8),
        ('yaw', ['--yaws', '-.2', '.2'], 16), ('push_positive', ['--push-velocity', '.25'], 8), ('push_negative', ['--push-velocity', '-.25'], 8)]:
        summary = evaluate('readout_ppo_'+label, checkpoint, extra)
        if summary['strict_passes'] < required or summary['falls'] > 1 or (label == 'long' and summary['direction_passes'] != 9):
            passed = False; break
    report['development_passed'] = passed
else:
    report['development_passed'] = False
report.update(completed=True, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()); persist()
