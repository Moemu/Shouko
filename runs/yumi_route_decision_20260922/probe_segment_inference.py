"""Measure repeatability of ordered CSR row sums without changing checkpoint weights."""
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch

from app.full_brain import ConnectomePolicy, SparseMessage
from app.train_full import file_sha256


def segment_forward(ctx, values, activity, ptr, pre):
    return torch.segment_reduce(activity[pre] * values[:, None], 'sum', offsets=ptr, axis=0)


torch.set_num_threads(8)
torch.manual_seed(20260922)
root=Path('runs/yumi_route_decision_20260922')
checkpoint=Path('runs/yumi_obs50/best.pt')
policy=ConnectomePolicy(observation_size=50).eval()
policy.load(checkpoint)
batch=torch.load('rollout.pt',map_location='cpu',weights_only=True)
observations=batch['observations'].reshape(-1,50)[:32].cuda()
report=dict(kind='ordered_csr_inference_probe',checkpoint_sha256=file_sha256(checkpoint),trials=[],completed=False)
with torch.inference_mode():
    for size in [4,32]:
        obs=observations[:size]
        reference=policy(obs)[0]
        torch.cuda.reset_peak_memory_stats()
        with patch.object(SparseMessage,'forward',staticmethod(segment_forward)):
            initial=policy(obs)[0]
            torch.cuda.synchronize()
            started=time.perf_counter()
            delta=torch.stack([policy(obs)[0]-initial for _ in range(5)])
            torch.cuda.synchronize()
            report['trials'].append(dict(batch=size,repeat_max_abs=float(delta.abs().max()),
                repeat_rms=float(delta.double().square().mean().sqrt()),
                csr_reference_max_abs=float((initial-reference).abs().max()),
                csr_reference_rms=float((initial-reference).double().square().mean().sqrt()),
                seconds_per_forward=(time.perf_counter()-started)/5,
                peak_tensor_gib=torch.cuda.max_memory_allocated()/2**30))
report['completed']=True
report['limitation']='Inference-only ordered reduction; arithmetic order differs from native CSR. Not a gait test or a training change.'
(root/'segment_inference.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report),flush=True)
