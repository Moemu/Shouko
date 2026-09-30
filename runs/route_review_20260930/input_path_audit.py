"""Read-only checkpoint and counterfactual observation-path audit."""
from contextlib import nullcontext
import hashlib
import json
from pathlib import Path

import torch

from app.evaluate_locomotion import load_policy
from app.ordered_inference import ordered_inference

torch.set_num_threads(4)
root=Path('runs/direction_control_20260929')
output=Path('runs/route_review_20260930/input_path_audit.json')
def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()
paths=[root/'transfer_main/round4/last.pt', root/'transfer_main/ridge1e5/last.pt',
       root/'readout_ppo_warm3029/last.pt', root/'teacher_control3029/last.pt',
       root/'teacher_control3030/last.pt']
data_path=root/'transfer_main/round4_normal_data.pt'
data=torch.load(data_path,map_location='cpu',weights_only=True,mmap=True)
rows=torch.linspace(0,len(data['observations'])-1,9).long()
base=data['observations'][rows].clone()
report=dict(kind='read_only_input_path_audit',data_sha256=sha(data_path),rows=rows.tolist(),
    coordinate_system='MuJoCo free-joint translational qvel[:2], world frame; observation indices 47 and 48. Height offset at 49.',
    limitation='Input interventions hold other observed fields fixed; they are not physically rolled-out states or causal locomotion tests.',
    source_sha256={str(p):sha(p) for p in [Path(__file__),Path('app/full_brain.py'),Path('app/sim.py')]},checkpoints=[])
for path in paths:
    digest=sha(path)
    saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    state=saved['state_dict']; weight=state['encoder.weight']
    item=dict(checkpoint=str(path),checkpoint_sha256=digest,
        encoder_shape=list(weight.shape),new_input_column_norms=weight[:,47:50].norm(dim=0).tolist(),
        new_input_nonzero=[int(torch.count_nonzero(weight[:,i])) for i in [47,48,49]],
        new_input_means=state['obs_mean'][47:50].tolist(),new_input_scales=state['obs_std'][47:50].tolist())
    if path.name=='last.pt' and ('round4' in path.parts or 'teacher_control3029' in path.parts):
        model,cfg,kind=load_policy(path)
        batches=[base]
        channels=[('world_vx',47,.05*cfg['physics_interface']['linear_velocity_scale']),
                  ('world_vy',48,.05*cfg['physics_interface']['linear_velocity_scale']),('height',49,.01)]
        for name,column,delta in channels:
            for sign in [-1,1]:
                altered=base.clone(); altered[:,column]+=sign*delta; batches.append(altered)
        with torch.inference_mode(), ordered_inference() if kind=='connectome' else nullcontext():
            actions,_=model(torch.cat(batches).cuda())
        pieces=actions.cpu().split(len(base)); baseline=pieces[0]
        item['counterfactual_action_change']={}
        for i,(name,column,delta) in enumerate(channels):
            errors=torch.cat([pieces[1+2*i]-baseline,pieces[2+2*i]-baseline])
            item['counterfactual_action_change'][name]=dict(observation_delta_abs=delta,
                max_abs=float(errors.abs().max()),rms=float(errors.square().mean().sqrt()))
        del model, actions
        torch.cuda.empty_cache()
    assert sha(path)==digest
    report['checkpoints'].append(item)
    print(json.dumps(item),flush=True)
output.write_text(json.dumps(report,indent=2),encoding='utf-8')
