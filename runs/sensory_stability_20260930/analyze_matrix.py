"""Summarize every fixed cell without selecting a winning seed."""
import json
from pathlib import Path
from statistics import mean

root = Path('runs/sensory_stability_20260930')
matrix = json.loads((root/'matrix.json').read_text())
assert matrix['completed']
report = dict(stages={}, changes={}, execution_counts={})
for stage, cells in matrix['stages'].items():
    rows = {}
    for arm in ['c0', 's1']:
        selected = [cells[f'{data}_{seed}_{arm}'] for data in ['A', 'B'] for seed in [3031, 4031]]
        rows[arm] = dict(passed_cells=sum(cell['development_passed'] for cell in selected),
            long_mean_range_m=[min(cell['results']['long']['lateral_m'] for cell in selected),
                               max(cell['results']['long']['lateral_m'] for cell in selected)],
            long_mean_across_cells_m=mean(cell['results']['long']['lateral_m'] for cell in selected))
    rows['c0_minus_s1_long_mean_m'] = {
        f'{data}_{seed}':cells[f'{data}_{seed}_c0']['results']['long']['lateral_m']-
                         cells[f'{data}_{seed}_s1']['results']['long']['lateral_m']
        for data in ['A', 'B'] for seed in [3031, 4031]}
    report['stages'][stage] = rows
for key, cell in matrix['stages']['2'].items():
    first = matrix['stages']['1'][key]['results']['long']
    second = cell['results']['long']
    report['changes'][key] = dict(long_mean_change_m=second['lateral_m']-first['lateral_m'],
        signed_endpoint_change_by_speed_m={speed:second['by_speed'][speed]['signed_endpoint_mean_m']-
            first['by_speed'][speed]['signed_endpoint_mean_m'] for speed in ['0.35', '0.5', '0.65']})
new_cells = [cell for cells in matrix['stages'].values() for cell in cells.values() if not cell['reused']]
report['execution_counts'] = dict(new_training_calls=len(new_cells),
    new_development_groups=sum(len(cell['results']) for cell in new_cells),
    new_development_episodes=sum(result['episodes'] for cell in new_cells for result in cell['results'].values()))
if (root/'final.json').exists():
    final = json.loads((root/'final.json').read_text())
    assert final['completed']
    report['execution_counts'].update(new_final_groups=sum(len(branch['results']) for branch in final['branches'].values()),
        new_final_episodes=sum(result['episodes'] for branch in final['branches'].values() for result in branch['results'].values()))
(root/'analysis.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report['stages'], indent=2))
