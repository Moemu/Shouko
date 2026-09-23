"""Run one native-CSR, batch-one episode from the extracted model package."""
import hashlib
import json
from pathlib import Path
import torch
from app.evaluate_locomotion import load_policy, evaluate_batch
from app.sim import Body

root=Path(__file__).resolve().parent
manifest=json.loads((root/'manifest.json').read_text())
for row in manifest['files']:
    path=root/row['path']
    with path.open('rb') as handle:
        assert hashlib.file_digest(handle,'sha256').hexdigest()==row['sha256'],row['path']
torch.set_num_threads(8)
checkpoint=root/manifest['models']['primary']['path']
policy,cfg,kind=load_policy(str(checkpoint))
body=Body(False,'yumi',interface=cfg['physics_interface'],observation_size=cfg['observation_size'])
result=evaluate_batch(policy,[body],[4242],[.5],[0.],30.)
test=result['tests'][0]
strict=test['success'] and all(f['qualifying_swings_per_second']>=1 for f in test['gait']['feet'])
report=dict(checkpoint_sha256=manifest['models']['primary']['sha256'],policy_kind=kind,
    backend='native CSR',batch=1,strict=bool(strict),result=result,
    limitation='Package and single-world execution smoke; not new independent acceptance or UI verification.')
(root/'package_smoke.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
assert strict,'Package smoke failed'
