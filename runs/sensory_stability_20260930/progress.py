"""Print compact, read-only progress from atomic experiment reports."""
import json
from pathlib import Path

root = Path('runs/sensory_stability_20260930')
for path in sorted(root.glob('stage*/*/execution.json')):
    item = json.loads(path.read_text())
    results = item['results']
    long = results.get('long', {})
    print(json.dumps(dict(cell=str(path.parent.relative_to(root)), completed=item['completed'],
        passed=item.get('development_passed'), error=item.get('error'),
        groups=list(results), active_job=(item['jobs'][-1]['name'] if item['jobs'] else 'reused'),
        long_corridor=long.get('direction_passes'), long_mean=long.get('lateral_m'),
        long_speed=long.get('by_speed'))))
matrix = json.loads((root/'matrix.json').read_text())
print(json.dumps({key:matrix[key] for key in ['completed', 'errors', 'repeat_parameter_max_difference',
    'stability_screen_passed', 'c0_passed_cells'] if key in matrix}))
