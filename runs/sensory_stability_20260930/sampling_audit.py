"""Describe the fixed datasets and exact minibatch streams without changing training."""
import hashlib
import json
from pathlib import Path
import torch

torch.set_num_threads(8)
root = Path('runs/sensory_stability_20260930')
old = Path('runs/sensory_access_20260930')
recipes = {name:json.loads(path.read_text()) for name, path in
           [('A', old/'s1/fit.json'), ('B', old/'replica/s1/fit.json')]}
extra = json.loads((old/'replica/dagger/round1/fit.json').read_text())['arguments']['train'][-4:]
report = {}

def describe(x):
    speed = x[:, 6]/2
    vy = x[:, 48]/.25
    return dict(samples=len(x), speed_counts={str(v):int(((speed-v).abs()<1e-5).sum()) for v in [.35, .5, .65]},
                vy_mean_mps=float(vy.mean()), vy_std_mps=float(vy.std()),
                vy_tail_counts=dict(negative=int((vy<-.02).sum()), central=int((vy.abs()<=.02).sum()),
                                    positive=int((vy>.02).sum())))

for stage in [1, 2]:
    for name, recipe in recipes.items():
        files = recipe['arguments']['train']+(extra if stage == 2 else [])
        observations = torch.cat([torch.load(p, map_location='cpu', weights_only=True, mmap=True)['observations'] for p in files])
        row = dict(dataset=describe(observations), streams={})
        for seed in [3031, 4031]:
            generator = torch.Generator().manual_seed(seed+(stage == 2))
            indices = torch.stack([torch.randint(len(observations), (128,), generator=generator) for _ in range(500)])
            digest = hashlib.sha256(indices.numpy().tobytes()).hexdigest()
            if stage == 1 and (name, seed) in [('A', 3031), ('B', 4031)]:
                assert digest == recipe['sample_indices_sha256']
            row['streams'][str(seed)] = dict(sample_indices_sha256=digest,
                all_updates=describe(observations[indices.flatten()]),
                final_50_updates=describe(observations[indices[-50:].flatten()]))
        report[f'stage{stage}_{name}'] = row
(root/'sampling_audit.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report, indent=2))
