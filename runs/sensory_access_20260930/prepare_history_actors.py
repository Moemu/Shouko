"""Transfer only differing readouts after checking every frozen tensor locally."""
import json
from pathlib import Path
import torch
from app.train_full import file_sha256

torch.set_num_threads(8)
root=Path('runs/sensory_access_20260930')
base=Path('runs/direction_control_20260929/transfer_main/round4/last.pt')
reference=torch.load(base,map_location='cpu',weights_only=True,mmap=True)
paths=[Path('runs/connectome_transfer_ridge_20260923/ridge1e4/last.pt')]+[
    Path(f'runs/direction_control_20260929/transfer_main/round{i}/last.pt') for i in [1,2,3]]
rows=[];patches=[]
for i,path in enumerate(paths):
    saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    changed=[k for k,v in saved['state_dict'].items() if not torch.equal(v,reference['state_dict'][k])]
    assert set(changed)=={'readout.weight','readout.bias'},changed
    expected=json.loads(Path(f'runs/direction_control_20260929/transfer_main/round{i+1}_normal_data.json').read_text())['student_sha256']
    assert file_sha256(path)==expected
    patches.append(dict(metadata={k:v for k,v in saved.items() if k!='state_dict'},
        readout={k:saved['state_dict'][k].clone() for k in changed},original_checkpoint_sha256=expected))
    rows.append(dict(label=f'history{i}',original_checkpoint_sha256=expected,changed_tensors=changed))
torch.save(dict(base_sha256=file_sha256(base),actors=patches),root/'history_readouts.pt')
(root/'history_readouts.json').write_text(json.dumps(dict(base_sha256=file_sha256(base),actors=rows,
    payload_sha256=file_sha256(root/'history_readouts.pt'),source_sha256=file_sha256(__file__),
    method='Every tensor outside readout verified bitwise equal to base. Reconstructed serialization hash may differ.'),indent=2))
print(json.dumps(rows))
