"""Infer actual push exposure from completed collection episode durations."""
import hashlib
import json
from pathlib import Path

root=Path('runs/direction_control_20260929')
results=[]
for path in sorted((root/'transfer_main').glob('round*_data.json')):
    data=json.loads(path.read_text())
    velocity=data.get('effective_push_velocity',0)
    if not velocity:
        continue
    episodes=data['episodes']
    applied=[e for e in episodes if e['seconds']>10+1e-6]
    results.append(dict(source=str(path), source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        configured_push_velocity=velocity, episodes=len(episodes), inferred_exposed_episodes=len(applied),
        by_speed={str(speed):dict(episodes=sum(e['command']==speed for e in episodes),
            inferred_exposed=sum(e['command']==speed for e in applied)) for speed in [.35,.5,.65]}))
report=dict(method='A surviving episode must complete a control step after the ten-second impulse.',
    limitation='Inferred from durations and collection source, not a separately recorded event flag.', results=results)
(root/'push_exposure_audit.json').write_text(json.dumps(report, indent=2))
for row in results:
    print(Path(row['source']).name, row['inferred_exposed_episodes'], '/', row['episodes'])
