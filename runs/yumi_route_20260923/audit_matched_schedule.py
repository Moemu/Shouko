"""Record configuration and normalization comparability of the completed reward control."""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parent
output = root/'matched_schedule_audit.json'
if output.exists():
    raise FileExistsError(output)
report = dict(stages=[], limitations=['Different reward produces different sampled trajectories and Adam state.',
                                      'Ordinary CUDA training is not promised bitwise deterministic.'])
for old_name, new_name in [('mlp_seed2026', 'support_seed2026'),
                          ('old_reward_extend1024', 'support_extend1024')]:
    paths = [root/name/'training.json' for name in [old_name, new_name]]
    old, new = [json.loads(p.read_text()) for p in paths]
    assert old['iteration'] == new['iteration'] == 400
    differences = {k: [old['configuration'].get(k), new['configuration'].get(k)]
                   for k in set(old['configuration']) | set(new['configuration'])
                   if old['configuration'].get(k) != new['configuration'].get(k)}
    assert set(differences) <= {'gait_reward', 'runs_dir', 'resume'}
    assert old['observation_normalization'] == new['observation_normalization']
    report['stages'].append(dict(old=old_name, phase_support=new_name, configuration_differences=differences,
                                 normalization_equal=True, iterations=400,
                                 training_transitions=old['env_steps'],
                                 training_json_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths]))
with tarfile.open(root/'stage1_final.tgz') as old, tarfile.open(root/'support_initial.tgz') as new:
    old_paths = {n.split('/app/', 1)[1]:n for n in old.getnames() if '/app/' in n}
    new_paths = {n.split('/app/', 1)[1]:n for n in new.getnames() if '/app/' in n}
    names = ['mlp_policy.py', 'recovery_curriculum.py', 'gpu_body.py']
    report['unchanged_implementations'] = {
        n: old.extractfile(old_paths[n]).read() == new.extractfile(new_paths[n]).read() for n in names}
    assert all(report['unchanged_implementations'].values())
report['total_transitions_each'] = sum(s['training_transitions'] for s in report['stages'])
output.write_text(json.dumps(report, indent=2))
print(json.dumps(report))
