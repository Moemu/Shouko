import hashlib
import json
from pathlib import Path
import torch
torch.set_num_threads(8)
root=Path('runs/connectome_transfer_ridge_20260923');output=root/'cache_and_freeze_audit.json'
if output.exists():raise FileExistsError(output)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
source=torch.load('runs/source/best.pt',weights_only=True,map_location='cpu',mmap=True)['state_dict'];rows=[]
for p in sorted(root.glob('*/fit.json')):
    fit=json.loads(p.read_text());ref=Path(fit['reference']);original=json.loads((ref/'fit.json').read_text())
    f=torch.load(ref/'features.pt',weights_only=True,map_location='cpu')
    assert sha(ref/'features.pt')==fit['feature_cache_sha256']
    y=torch.cat([torch.load(name,weights_only=True,map_location='cpu')['targets'] for name in original['arguments']['train']]).double()
    for name,digest in fit['data_sha256'].items():assert sha(name)==digest
    mean,scale=f['mean'],f['scale'];x=(f['train'].double()-mean)/scale
    x=torch.cat([x,torch.ones(len(x),1,dtype=torch.float64)],1);gram=x.T@x/len(x);rhs=x.T@y/len(x)
    errors={}
    for label,lam,checkpoint in [('baseline',original['arguments']['ridge'],ref/'last.pt'),('candidate',fit['ridge'],p.parent/'last.pt')]:
        reg=torch.eye(x.shape[1],dtype=x.dtype)*lam;reg[-1,-1]=0;c=torch.linalg.solve(gram+reg,rhs)
        w=(c[:-1]/scale[:,None]).T.float();b=(c[-1]-mean@(c[:-1]/scale[:,None])).float()
        s=torch.load(checkpoint,weights_only=True,map_location='cpu',mmap=True)['state_dict']
        torch.testing.assert_close(s['readout.weight'],w,rtol=1e-5,atol=1e-6);torch.testing.assert_close(s['readout.bias'],b,rtol=1e-5,atol=1e-6)
        errors[label]=max(float((s['readout.weight']-w).abs().max()),float((s['readout.bias']-b).abs().max()))
        assert all(torch.equal(s[n],source[n]) for n in source if not n.startswith('readout.'))
    assert sha(p.parent/'last.pt')==fit['checkpoint_sha256']
    rows.append(dict(fit=str(p),checkpoint_sha256=fit['checkpoint_sha256'],reproduction_max_errors=errors,
        all_non_readout_tensors_bitwise_equal=True,data_and_feature_hashes_verified=True))
output.write_text(json.dumps(dict(verified=rows),indent=2));print(json.dumps(dict(verified_fits=len(rows),maximum_error=max(max(r['reproduction_max_errors'].values()) for r in rows))))
