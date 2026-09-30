"""Summarize paired diagnostic episodes without fitting or selecting holdout weights."""
import json
from pathlib import Path
import numpy as np

root = Path('runs/direction_control_20260929')
baseline = json.loads((root/'baseline_long.json').read_text())
assert baseline['completed']
report = dict(baselines=[], probes=[], completed=False)
baseline_y = {}
for model, condition in enumerate(baseline['conditions']):
    rows = condition['results'][0]['tests']
    with np.load(root/condition['results'][0]['trace']) as trace:
        thirty = [float(trace[f'episode{i}_qpos'][np.argmin(abs(trace[f'episode{i}_time']-30)), 1]) for i in range(9)]
    baseline_y[model] = np.abs(thirty)
    report['baselines'].append(dict(checkpoint=condition['checkpoint'], summary=condition['summary'],
        signed_y_30s=thirty, per_speed=[dict(speed=rows[i]['target_speed'],
            mean_end_y_m=float(np.mean([rows[j]['direction']['signed_end_y_m'] for j in range(i, 9, 3)])),
            mean_steady_body_lateral_mps=float(np.mean([rows[j]['direction']['steady_body_lateral_mps'] for j in range(i, 9, 3)])),
            mean_body_lateral_rms_mps=float(np.mean([rows[j]['direction']['steady_body_lateral_rms_mps'] for j in range(i, 9, 3)])),
            mean_heading_y_m=float(np.mean([rows[j]['direction']['heading_y_m'] for j in range(i, 9, 3)]))) for i in range(3)]))
names = [f'{option}_{sign}' for option in ['vy_command', 'hip_roll_offset', 'ankle_roll_offset'] for sign in ['positive', 'negative']]
for name in names:
    path = root/(name+'.json')
    if not path.exists():
        continue
    data = json.loads(path.read_text())
    if not data['completed']:
        continue
    for model, condition in enumerate(data['conditions']):
        rows = condition['results'][0]['tests']; values = np.array([r['lateral_m'] for r in rows])
        reduced = [bool(values[i::3].mean() < baseline_y[model][i::3].mean()) for i in range(3)]
        passed = condition['summary']['strict_passes'] == 9 and condition['summary']['falls'] == 0 and all(reduced)
        report['probes'].append(dict(name=name, model=model, checkpoint=condition['checkpoint'],
            summary=condition['summary'], per_speed_lateral_m=[float(values[i::3].mean()) for i in range(3)],
            improved_all_speeds=all(reduced), qualifies_for_long=passed))
report['completed'] = len(report['probes']) == 12
(root/'stage0_analysis.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
