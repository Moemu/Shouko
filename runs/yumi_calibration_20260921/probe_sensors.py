import hashlib,json
from pathlib import Path
import torch
from app.full_brain import ConnectomePolicy
root=Path('runs/yumi_calibration_20260921')
batch=torch.load(root/'rollout.pt',map_location='cpu',weights_only=True)
obs=batch['observations'].reshape(-1,50)
indices=torch.linspace(0,len(obs)-1,64).long()
probe=obs[indices].cuda()
std=obs.std(0)
policy=ConnectomePolicy(observation_size=50).eval()
rows=[]
with torch.inference_mode():
    for name in ['runs/yumi_obs50/best.pt',str(root/'short128_normfix/last.pt'),str(root/'short128_low_lr/last.pt')]:
        policy.load(name)
        baseline,_=policy(probe)
        ablated=probe.clone();ablated[:,47:]=0
        action,_=policy(ablated)
        shifts={}
        for column,label in [(47,'vx'),(48,'vy'),(49,'height')]:
            changed=probe.clone();changed[:,column]+=std[column].cuda()
            prediction,_=policy(changed)
            shifts[label]=dict(raw_observation_std=float(std[column]),normalized_std=float(std[column]/policy.obs_std[column].cpu()),action_rms_change=float((prediction-baseline).square().mean().sqrt()))
        with open(name,'rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
        rows.append(dict(checkpoint=name,checkpoint_sha256=digest,new_encoder_rms=float(policy.encoder.weight[:,47:].square().mean().sqrt()),zero_new_inputs_action_rms=float((action-baseline).square().mean().sqrt()),one_std_input_shift=shifts))
report=dict(kind='fixed_observation_sensitivity',source_rollout_checkpoint=batch['checkpoint_sha256'],observations=64,limitation='Input interventions only, not a locomotion experiment or a proof of causality.',results=rows)
(root/'sensor_probe.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report))
