import hashlib
import json
from pathlib import Path
import torch

torch.set_num_threads(4)
root=Path('runs/connectome_transfer_push_20260923');prior=Path('runs/prior')
output=root/'chain_audit.json'
if output.exists():raise FileExistsError(output)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
expected_teacher='9c5ca9339b26f5af2e8acf94e7edd3b9371ef5efb41d647bcbd7904243ac50e1'
assert sha('runs/teacher/last.pt')==expected_teacher
assert sha(prior/'dagger3/last.pt')=='d69567c552f0e74ae8c46dedfb897ca7e7fd4fc40c35f6b90f1b723b5095b916'
assert json.loads((prior/'verification_execution.json').read_text())['completed']
source=torch.load('runs/source/best.pt',map_location='cpu',weights_only=True,mmap=True)['state_dict']
rows=[]
for path in sorted(root.glob('*/fit.json')):
    d=json.loads(path.read_text());args=d['arguments'];checkpoint=path.parent/'last.pt'
    assert d['completed'] and d['checkpoint_sha256']==sha(checkpoint)
    assert d['source_checkpoint_sha256']==sha(args['source'])
    for filename,digest in d['data_sha256'].items():assert sha(filename)==digest,filename
    data_path=root/(path.parent.name+'.pt')
    data=torch.load(data_path,map_location='cpu',weights_only=True)
    assert data['student_sha256']==d['source_checkpoint_sha256']
    assert data['teacher_sha256']==expected_teacher and data['arguments']['beta']==0 and data['arguments']['push']
    train_ids=set()
    for file in args['train']:
        part=torch.load(file,map_location='cpu',weights_only=True);ids=set(part['episode_ids'].tolist())
        assert not train_ids&ids,'Repeated training episode identifiers'
        train_ids|=ids
    validation=torch.load(args['validation'][0],map_location='cpu',weights_only=True)
    assert not train_ids & set(validation['episode_ids'].tolist())
    model=torch.load(checkpoint,map_location='cpu',weights_only=True,mmap=True)['state_dict']
    changed=[n for n in source if not torch.equal(source[n],model[n])]
    assert all(n.startswith('readout.') for n in changed)
    y=data['targets'];assert torch.isfinite(y).all() and torch.isfinite(data['observations']).all()
    rows.append(dict(fit=str(path),checkpoint_sha256=sha(checkpoint),source_checkpoint_sha256=d['source_checkpoint_sha256'],
        samples=len(y),student_sampling_chain_verified=True,data_hashes_verified=True,train_validation_disjoint=True,
        changed_keys=changed,teacher_clip_fraction=float((y.abs()>8).float().mean()),
        raw_executed_target_mse=float((y-y.clamp(-8,8)).square().mean())))
output.write_text(json.dumps(dict(verified=rows,source_frozen_except_readout=True,teacher_sha256=expected_teacher),indent=2))
print(json.dumps(dict(verified_fits=len(rows),all_chains_valid=True)))
