"""One preregistered sample-weighted ridge solve on the exact round-four cache."""
import hashlib
import json
from pathlib import Path
import time
import torch

torch.set_num_threads(8)
root=Path('runs/sensory_access_20260930'); reference=Path('runs/direction_control_20260929/transfer_main/round4')
out=root/'weighted'; out.mkdir(exist_ok=False)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
fit=json.loads((reference/'fit.json').read_text()); started=time.monotonic()
for p,h in fit['data_sha256'].items():assert sha(p)==h
cache=torch.load(reference/'features.pt',map_location='cpu',weights_only=True,mmap=True)
training=[torch.load(p,map_location='cpu',weights_only=True,mmap=True) for p in fit['arguments']['train']]
targets=torch.cat([d['targets'] for d in training]).double()
obs=torch.cat([d['observations'] for d in training])
mean,scale=cache['mean'],cache['scale']
x=(cache['train'].double()-mean)/scale
x=torch.cat([x,torch.ones(len(x),1,dtype=x.dtype)],1)
reg=torch.eye(x.shape[1],dtype=x.dtype)*1e-4;reg[-1,-1]=0
baseline=torch.linalg.solve(x.T@x/len(x)+reg,x.T@targets/len(x))
checkpoint=torch.load(reference/'last.pt',map_location='cpu',weights_only=True,mmap=True)
state=checkpoint['state_dict']; original={k:v.clone() for k,v in state.items() if k.startswith('readout.')}
def unpack(coef):
    weight=(coef[:-1]/scale[:,None]).T.float()
    bias=(coef[-1]-mean@(coef[:-1]/scale[:,None])).float()
    return weight,bias
bw,bb=unpack(baseline)
error=max(float((bw-original['readout.weight']).abs().max()),float((bb-original['readout.bias']).abs().max()))
assert error<1e-6,error
velocity=obs[:,48].double()/training[0]['physics_interface']['linear_velocity_scale']
weights=1+4*(velocity.abs()/.02).clamp(max=1)
coef=torch.linalg.solve(x.T@(x*weights[:,None])/weights.sum()+reg,x.T@(targets*weights[:,None])/weights.sum())
state['readout.weight'],state['readout.bias']=unpack(coef)
checkpoint['extra']=dict(method='sample-weighted ridge, fixed round4 features',source_sha256=sha(reference/'last.pt'),
    ridge=1e-4,weight_definition='1 + 4 * min(abs(world_vy_mps) / 0.02, 1)',normalization='sum(weights)')
torch.save(checkpoint,out/'last.pt')
report=dict(completed=True,checkpoint_sha256=sha(out/'last.pt'),reference_checkpoint_sha256=sha(reference/'last.pt'),
    cache_sha256=sha(reference/'features.pt'),data_sha256=fit['data_sha256'],baseline_reconstruction_max_error=error,
    weight_min=float(weights.min()),weight_max=float(weights.max()),weight_mean=float(weights.mean()),
    observations=len(x),training_scale_vy=float(obs[:,48].std().clamp_min(1e-4)),
    training_mse=float((x@coef-targets).square().mean()),wall_seconds=time.monotonic()-started,
    source_sha256=sha(__file__),changed_tensors=['readout.weight','readout.bias'])
(out/'fit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report),flush=True)
