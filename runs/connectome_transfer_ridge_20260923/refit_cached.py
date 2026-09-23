"""A single-variable ridge comparison on identical frozen-connectome features."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import torch

torch.set_num_threads(8)
p=argparse.ArgumentParser();p.add_argument('--reference',required=True);p.add_argument('--ridge',type=float,required=True);p.add_argument('--output',required=True);a=p.parse_args()
reference=Path(a.reference);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);started=time.monotonic()
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
fit=json.loads((reference/'fit.json').read_text());assert fit['completed'] and not fit['arguments']['clip_targets']
for filename,digest in fit['data_sha256'].items():assert sha(filename)==digest,filename
features=torch.load(reference/'features.pt',weights_only=True,map_location='cpu')
train=[torch.load(path,weights_only=True,map_location='cpu') for path in fit['arguments']['train']]
validation=[torch.load(path,weights_only=True,map_location='cpu') for path in fit['arguments']['validation']]
y=torch.cat([d['targets'] for d in train]).double();vy=torch.cat([d['targets'] for d in validation]).double()
assert not set(torch.cat([d['episode_ids'] for d in train]).tolist())&set(torch.cat([d['episode_ids'] for d in validation]).tolist())
mean,scale=features['mean'],features['scale'];x=(features['train'].double()-mean)/scale
assert len(x)==len(y)
x=torch.cat([x,torch.ones(len(x),1,dtype=torch.float64)],1);gram=x.T@x/len(x)
eigenvalues=torch.linalg.eigvalsh(gram)
reg=torch.eye(x.shape[1],dtype=x.dtype)*a.ridge;reg[-1,-1]=0
rhs=x.T@y/len(x)
coef=torch.linalg.solve(gram+reg,rhs)
checkpoint=torch.load(reference/'last.pt',weights_only=True,map_location='cpu')
original_reg=torch.eye(x.shape[1],dtype=x.dtype)*fit['arguments']['ridge'];original_reg[-1,-1]=0
original_coef=torch.linalg.solve(gram+original_reg,rhs)
recreated_weight=(original_coef[:-1]/scale[:,None]).T.float()
recreated_bias=(original_coef[-1]-mean@(original_coef[:-1]/scale[:,None])).float()
baseline_error=max(float((recreated_weight-checkpoint['state_dict']['readout.weight']).abs().max()),
                   float((recreated_bias-checkpoint['state_dict']['readout.bias']).abs().max()))
torch.testing.assert_close(recreated_weight,checkpoint['state_dict']['readout.weight'],rtol=1e-5,atol=1e-6)
torch.testing.assert_close(recreated_bias,checkpoint['state_dict']['readout.bias'],rtol=1e-5,atol=1e-6)
checkpoint['state_dict']['readout.weight']=(coef[:-1]/scale[:,None]).T.float()
checkpoint['state_dict']['readout.bias']=(coef[-1]-mean@(coef[:-1]/scale[:,None])).float()
checkpoint['extra']=dict(method='cached full-graph ridge refit',ridge=a.ridge,reference_checkpoint_sha256=sha(reference/'last.pt'))
torch.save(checkpoint,out/'last.pt')
# Command masks use the recorded command scale, not an assumed observation coordinate scale.
obs=torch.cat([d['observations'] for d in train]);vobs=torch.cat([d['observations'] for d in validation])
command_scale=train[0]['physics_interface']['cmd_scale'][0]
def by_command(raw,target,observations):
    prediction=raw.double()@checkpoint['state_dict']['readout.weight'].double().T+checkpoint['state_dict']['readout.bias'].double()
    error=prediction-target
    return dict(mse=float(error.square().mean()),executed_mse=float((prediction.clamp(-8,8)-target.clamp(-8,8)).square().mean()),
        per_command={str(speed):float(error[(observations[:,6]-speed*command_scale).abs()<1e-4].square().mean()) for speed in [.35,.5,.65]})
report=dict(completed=True,ridge=a.ridge,reference=str(reference),reference_fit_sha256=sha(reference/'fit.json'),
    refit_source_sha256=sha(__file__),baseline_cache_reproduction_max_error=baseline_error,
    feature_cache_sha256=sha(reference/'features.pt'),data_sha256=fit['data_sha256'],checkpoint_sha256=sha(out/'last.pt'),
    eigenvalue_quantiles=torch.quantile(eigenvalues,torch.tensor([0,.01,.1,.5,.9,.99,1.],dtype=torch.float64)).tolist(),
    eigenvalues_above_ridge=int((eigenvalues>a.ridge).sum()),readout_weight_norm=float(checkpoint['state_dict']['readout.weight'].norm()),
    training=by_command(features['train'],y,obs),validation=by_command(features['validation'],vy,vobs),
    wall_seconds=time.monotonic()-started,method='No new observations, labels, graph features, normalization or physics.')
(out/'fit.json').write_text(json.dumps(report,indent=2,allow_nan=False));print(json.dumps(report),flush=True)
