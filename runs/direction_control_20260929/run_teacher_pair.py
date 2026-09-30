"""Fixed-budget lateral-reward comparison, with conditional independent training."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root = Path('runs/direction_control_20260929')
output = root/'teacher_execution.json'
if output.exists():
    raise FileExistsError(output)
assert json.loads((root/'stage0_execution.json').read_text())['completed']
assert (root/'teacher_preregistration.json').exists()
report = dict(completed=False, jobs=[], branches={}, started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in Path('app').glob('*.py')},
              preregistration_sha256=hashlib.sha256((root/'teacher_preregistration.json').read_bytes()).hexdigest())


def persist():
    temp = output.with_suffix('.tmp'); temp.write_text(json.dumps(report, indent=2)); temp.replace(output)


def run(name, args, timeout=1500):
    row = dict(name=name, command=[sys.executable, '-u', *args], started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat())
    report['jobs'].append(row); persist()
    started = time.monotonic()
    with (root/(name+'.log')).open('w') as log:
        result = subprocess.run(row['command'], stdout=log, stderr=subprocess.STDOUT,
                                env=dict(os.environ, OMP_NUM_THREADS='8', MKL_NUM_THREADS='8'), timeout=timeout)
    row.update(returncode=result.returncode, wall_seconds=time.monotonic()-started); persist()
    if result.returncode:
        raise RuntimeError(name)


def evaluate(label, checkpoint, suffix, extra):
    result_path = root/(label+'_'+suffix+'.json')
    run(label+'_'+suffix, ['-m', 'app.evaluate_direction', '--checkpoints', str(checkpoint),
                          '--seed-base', '310101', '--output', str(result_path), *extra])
    data = json.loads(result_path.read_text()); assert data['completed']
    summary = data['conditions'][0]['summary']
    report['branches'][label][suffix] = summary; persist()
    print(json.dumps(dict(branch=label, stage=suffix, **summary)), flush=True)
    return summary


def branch(label, precision, seed):
    folder = root/label
    run(label, ['-m', 'app.ppo_yumi', '--resume', 'runs/source_teacher/last.pt', '--resume-state', '--policy', 'mlp',
        '--gait-reward', 'phase_support', '--lateral-precision', str(precision), '--worlds', '1024', '--steps', '128',
        '--epochs', '4', '--minibatch', '4096', '--lr', '.0003', '--lr-final-frac', '1', '--knee-gate', 'stance',
        '--target-kl', '.02', '--std0', '.4', '--value-warmup', '8', '--eval-seconds', '30', '--episode-seconds', '6',
        '--eval-every', '400', '--max-iterations', '400', '--max-seconds', '1200', '--seed', str(seed), '--runs-dir', str(folder)])
    state = json.loads((folder/'training.json').read_text())
    assert state['iteration'] == 400
    checkpoint = folder/'last.pt'
    report['branches'][label] = dict(checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                                    configuration=state['configuration']); persist()
    normal = evaluate(label, checkpoint, 'normal', ['--cohorts', '3'])
    if normal['strict_passes'] < 24 or normal['falls'] > 1:
        return False
    long = evaluate(label, checkpoint, 'long', ['--seconds', '120', '--trace'])
    if long['direction_passes'] != 9 or long['falls']:
        return False
    for suffix, extra, threshold in [('yaw', ['--yaws', '-.2', '.2'], 16),
                                      ('push_positive', ['--push-velocity', '.25'], 8),
                                      ('push_negative', ['--push-velocity', '-.25'], 8)]:
        summary = evaluate(label, checkpoint, suffix, extra)
        if summary['strict_passes'] < threshold or summary['falls'] > 1:
            return False
    return True


persist()
report['control_passed'] = branch('teacher_control3029', 6, 3029); persist()
report['refined_passed'] = branch('teacher_lateral3029', 60, 3029); persist()
if report['refined_passed']:
    report['replica_passed'] = branch('teacher_lateral3030', 60, 3030); persist()
report.update(completed=True, finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()); persist()
