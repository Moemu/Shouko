"""Recalculate the fixed final acceptance gates from preserved episode-level files."""
import hashlib
import json
from pathlib import Path

root = Path('runs/connectome_transfer_replica_20260923')
def sha(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()

result = dict(candidates=[], limitation='Independent data with one extra round, not equal-budget replication.')
for folder, checkpoint, expected in [
    (Path('runs/connectome_transfer_ridge_20260923'), 'ridge1e4/last.pt', 'f1a200718ebfec845cbf9c02ba743e937ea94c747fba53e7b12dfd12f8d3e1f9'),
    (root, 'extra1/last.pt', '8841868cf1cbbbd01be19585bc18e751c11901b3e806556705d124bb207616cf'),
]:
    assert sha(folder/checkpoint) == expected
    groups = []
    used_ids = set()
    for name, count, required in [('holdout',27,24), ('yaw_holdout',18,16), ('long_holdout',9,8), ('push_holdout',9,8)]:
        path = folder/(name+'.json')
        data = json.loads(path.read_text())
        assert data['completed']
        condition = data['conditions'][0]
        assert condition['checkpoint_sha256'] == expected
        tests = [test for cohort in condition['results'] for test in cohort['tests']]
        seeds = {test['seed'] for test in tests}
        assert not seeds & used_ids
        used_ids |= seeds
        assert len(tests) == count
        passes = sum(t['success'] and all(f['qualifying_swings_per_second']>=1 for f in t['gait']['feet']) for t in tests)
        falls = sum(t['fallen'] for t in tests)
        assert passes >= required and falls <= 1
        groups.append(dict(file=str(path), sha256=sha(path), strict=passes, count=count, falls=falls,
                           precise_yaw_recovered=sum(t['gait']['recovery_seconds'] is not None for t in tests),
                           **{k:condition['summary'][k] for k in ['speed_mae','lateral_m','qualifying_swings_per_second']}))
    result['candidates'].append(dict(checkpoint=str(folder/checkpoint), checkpoint_sha256=expected, groups=groups))
result['all_final_gates_passed'] = True
(root/'final_summary.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
