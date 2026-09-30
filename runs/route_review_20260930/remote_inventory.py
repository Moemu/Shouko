"""Read-only inventory; omit machine identifiers and connection details."""
import hashlib
import importlib.metadata
import json
from pathlib import Path
import shutil
import subprocess

import torch

root=Path('/root/autodl-tmp/neuromechfly')
def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()
weights=[]
for path in sorted((root/'runs').rglob('*.pt')):
    saved=torch.load(path,map_location='cpu',weights_only=True,mmap=True)
    state=saved.get('state_dict',{})
    if 'encoder.weight' not in state:
        continue
    weights.append(dict(path=str(path.relative_to(root)),sha256=sha(path),bytes=path.stat().st_size,
        observation_size=saved.get('observation_size'),policy_kind=saved.get('policy_kind','connectome'),
        encoder_shape=list(state['encoder.weight'].shape) if 'encoder.weight' in state else None))
files=['app/ppo_yumi.py','app/imitate_yumi.py','app/evaluate_direction.py','app/yumi_description/yumi.xml']
report=dict(review_date='2026-09-30',weights=weights,required_files={p:(root/p).exists() for p in files},
    graph_sha256=sha(root/'data/full_graph.npz'),disk_free_bytes=shutil.disk_usage(root).free,
    runtime={p:importlib.metadata.version(p) for p in ['torch','mujoco','mujoco-warp','warp-lang']},
    gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,memory.used','--format=csv,noheader'],text=True).strip(),
    gpu_processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True).strip())
print(json.dumps(report,indent=2))
