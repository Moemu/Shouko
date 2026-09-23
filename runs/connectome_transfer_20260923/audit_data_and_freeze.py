import hashlib
import json
from pathlib import Path
import torch
import argparse

torch.set_num_threads(4)
root=Path('runs/connectome_transfer_20260923')
p=argparse.ArgumentParser();p.add_argument('--output',default=str(root/'data_freeze_audit.json'));args=p.parse_args()
output=Path(args.output)
if output.exists():
    raise FileExistsError(output)
rows=[]
for path in sorted(root.glob('*.pt')):
    d=torch.load(path,map_location='cpu',weights_only=True)
    if 'targets' not in d:
        continue
    x,y=d['observations'],d['targets']
    assert x.ndim==2 and x.shape[1]==50 and y.shape==(len(x),12)
    assert torch.isfinite(x).all() and torch.isfinite(y).all()
    rows.append(dict(file=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(), samples=len(x),
        episodes=len(torch.unique(d['episode_ids'])), teacher_sha256=d['teacher_sha256'],
        student_sha256=d['student_sha256'], beta=d['arguments']['beta'],
        teacher_raw_clip_fraction=float((y.abs()>8).float().mean()),
        teacher_per_joint_clip_fraction=(y.abs()>8).float().mean(0).tolist(),
        teacher_max_absolute=float(y.abs().max()), teacher_raw_executed_mse=float((y-y.clamp(-8,8)).square().mean())))
source=torch.load('runs/source/best.pt',map_location='cpu',weights_only=True,mmap=True)['state_dict']
frozen=[]
for p in sorted(root.glob('*/last.pt')):
    model=torch.load(p,map_location='cpu',weights_only=True,mmap=True)['state_dict']
    changed=[name for name in source if not torch.equal(source[name],model[name])]
    only_readout=all(name.startswith('readout.') for name in changed)
    assert only_readout,(str(p),changed)
    frozen.append(dict(file=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest(),
                       changed_keys=changed,all_other_tensors_bitwise_equal=True))
output.write_text(json.dumps(dict(data=rows,weights=frozen,device='cpu',
    note='Labels are raw teacher actions. Zero clip fraction establishes equality to executed targets on these recorded states.'),indent=2))
print(json.dumps(dict(datasets=len(rows),max_teacher_clip_fraction=max(r['teacher_raw_clip_fraction'] for r in rows),
                     readout_only_checkpoints=len(frozen))))
