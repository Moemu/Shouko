"""Read small encoder tensors and saved Adam moments; no full graph or CUDA."""
import json
from pathlib import Path

import torch

from app.train_full import file_sha256, validated_checkpoint_state


torch.set_num_threads(2)
root = Path('runs/yumi_calibration_20260921')
source_path = root / 'migrated/runs/yumi_obs50/best.pt'
source = torch.load(source_path, map_location='cpu', mmap=True, weights_only=True)['state_dict']
batch = torch.load(root / 'rollout.pt', map_location='cpu', weights_only=True)
assert batch['checkpoint_sha256'] == file_sha256(source_path)
observations = batch['observations'].reshape(-1, 50)
rows = []
for name in ['short128_normfix', 'short128_low_lr']:
    checkpoint = root / name / 'last.pt'
    weights = torch.load(checkpoint, map_location='cpu', mmap=True, weights_only=True)['state_dict']
    saved = validated_checkpoint_state(checkpoint, root / name / 'ppo_state.pt', 'cpu')
    rms_sums = torch.zeros(2, dtype=torch.float64)
    count = 0
    for part in observations.split(128):
        x = ((part - weights['obs_mean']) / weights['obs_std']).clamp(-10, 10)
        old = x[:, :47] @ weights['encoder.weight'][:, :47].T
        new = x[:, 47:] @ weights['encoder.weight'][:, 47:].T
        rms_sums += torch.tensor([old.double().square().sum(), new.double().square().sum()])
        count += old.numel()
    rates = saved['optimizer']['param_groups']
    encoder_group_index = saved['optimizer_names'].index(['encoder.weight'])
    group = rates[encoder_group_index]
    moment = saved['optimizer']['state'][group['params'][0]]
    step = int(moment['step'])
    beta1, beta2 = group['betas']
    direction = ((moment['exp_avg'] / (1 - beta1**step)) /
                 ((moment['exp_avg_sq'] / (1 - beta2**step)).sqrt() + group['eps']))
    delta = weights['encoder.weight'] - source['encoder.weight']
    row = dict(checkpoint=checkpoint.as_posix(), checkpoint_sha256=file_sha256(checkpoint),
               encoder_adam_step=step, encoder_lr=group['lr'],
               sensory_drive_rms_old_new=(rms_sums / count).sqrt().tolist(),
               encoder_delta_rms_old=float(delta[:, :47].square().mean().sqrt()),
               encoder_delta_rms_new=float(delta[:, 47:].square().mean().sqrt()),
               adam_direction_rms_old=float(direction[:, :47].square().mean().sqrt()),
               adam_direction_rms_new=float(direction[:, 47:].square().mean().sqrt()))
    rows.append(row)
parameter = torch.nn.Parameter(torch.zeros(2))
optimizer = torch.optim.Adam([parameter], lr=2.5e-6)
parameter.grad = torch.tensor([0.1, 2.0])
optimizer.step()
report = dict(kind='saved_encoder_and_adam_audit', source_sha256=file_sha256(source_path),
              results=rows, synthetic_adam_example=dict(gradients=[0.1, 2.0], updates=parameter.detach().tolist()),
              limitation='Encoder drive and saved Adam state only; no complete-policy gradients or simulated walking.')
output = Path('runs/yumi_input_audit_20260921/encoder_updates.json')
output.write_text(json.dumps(report, indent=2), encoding='utf-8')
print(json.dumps(report, indent=2))
