"""Audit the original cached training residuals and actual push coverage."""
import json
from pathlib import Path
import torch
from app.train_sensory_readout import grouped_metrics
from app.train_full import file_sha256

torch.set_num_threads(8)
reference=Path('runs/direction_control_20260929/transfer_main/round4')
fit=json.loads((reference/'fit.json').read_text())
cache=torch.load(reference/'features.pt',map_location='cpu',weights_only=True,mmap=True)
state=torch.load(reference/'last.pt',map_location='cpu',weights_only=True,mmap=True)['state_dict']
offset=0;rows=[]
for path in fit['arguments']['train']:
    assert file_sha256(path)==fit['data_sha256'][path]
    data=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    size=len(data['observations'])
    features=cache['train'][offset:offset+size].double();offset+=size
    predicted=features@state['readout.weight'].double().T+state['readout.bias'].double()
    velocity=data.get('effective_push_velocity',0)
    episodes=data['episodes']
    rows.append(dict(path=path,sha256=file_sha256(path),samples=size,episodes=len(episodes),
        push_velocity=velocity, inferred_push_exposure=sum(e['seconds']>10+1e-6 for e in episodes) if velocity else 0,
        vy_world_quantiles=torch.quantile(data['observations'][:,48]/data['physics_interface']['linear_velocity_scale'],
            torch.tensor([0,.01,.25,.5,.75,.99,1])).tolist(), **grouped_metrics(predicted,data)))
assert offset==len(cache['train'])
report=dict(source_checkpoint_sha256=file_sha256(reference/'last.pt'),cache_sha256=file_sha256(reference/'features.pt'),
    method='Float64 affine readout of original cached frozen features; push exposure inferred from duration > 10 s + 1e-6.',
    source_sha256=file_sha256(__file__),samples=offset,datasets=rows)
Path('runs/sensory_access_20260930/training_coverage.json').write_text(json.dumps(report,indent=2))
print(json.dumps(dict(samples=offset,files=len(rows),push_exposure=[(Path(r['path']).name,r['inferred_push_exposure']) for r in rows if r['push_velocity']])))
