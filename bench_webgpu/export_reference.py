"""Scratch: export WebGPU benchmark fixtures from runs/yumi/best.pt.

Outputs to bench_webgpu/fixtures/: raw binary tensors + reference outputs.
Run: .venv-gpu/Scripts/python.exe bench_webgpu/export_reference.py
"""
import json
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from app.full_brain import ConnectomePolicy

OUT = Path(__file__).resolve().parent / 'fixtures'
OUT.mkdir(exist_ok=True)

brain = ConnectomePolicy(device='cpu', observation_size=50).eval()
brain.load(str(ROOT / 'runs/yumi/best.pt'))

with torch.no_grad():
    values = (brain.weight * (1 + 0.9 * torch.tanh(brain.edge_delta))).to(torch.float32)
    obs_mean, obs_std = brain.obs_mean, brain.obs_std
    signal = torch.zeros(brain.n, 1)
    activity = torch.zeros_like(signal)

    fixtures = {}
    rng = torch.Generator().manual_seed(4242)
    observations = [torch.zeros(1, brain.encoder.in_features),
                    (torch.randn(1, brain.encoder.in_features, generator=rng) * 0.5)]
    for tag, obs in [('zero', observations[0]), ('rand', observations[1])]:
        x = ((obs - obs_mean) / obs_std).clamp(-10, 10)
        signal = torch.zeros(brain.n, 1)
        signal = signal.index_copy(0, brain.inputs, brain.encoder(x).T)
        activity = torch.zeros_like(signal)
        values_ = values
        for _ in range(brain.neural_steps):
            activity = 0.2 * activity + 0.8 * torch.tanh(
                3 * torch.sparse.mm(torch.sparse_csr_tensor(brain.ptr, brain.pre, values_,
                                           size=(brain.n, brain.n), check_invariants=False),
                                    activity) + signal + brain.neuron_bias[:, None])
        out_act = activity[brain.outputs, 0]
        motor = brain.normalizer(out_act[None])
        action = brain.readout(motor)
        fixtures[tag] = dict(
            obs=obs[0].tolist(),
            action=action[0].tolist(),
            output_activity=out_act.tolist(),
            signal_nonzero=signal[:, 0].tolist(),
        )
    # step-by-step full references for the 'rand' fixture (debug isolation)
    x = ((observations[1] - obs_mean) / obs_std).clamp(-10, 10)
    signal = torch.zeros(brain.n, 1).index_copy(0, brain.inputs, brain.encoder(x).T)
    activity = torch.zeros_like(signal)
    (signal + brain.neuron_bias[:, None]).squeeze(1).to(torch.float32).numpy().tofile(OUT / 'sbias.f32.bin')
    csr = torch.sparse_csr_tensor(brain.ptr, brain.pre, values,
                                  size=(brain.n, brain.n), check_invariants=False)
    for step in range(brain.neural_steps):
        activity = 0.2 * activity + 0.8 * torch.tanh(
            3 * torch.sparse.mm(csr, activity) + signal + brain.neuron_bias[:, None])
        activity.squeeze(1).to(torch.float32).numpy().tofile(OUT / f'act_step{step + 1}.f32.bin')

values.numpy().tofile(OUT / 'values.f32.bin')
brain.pre.numpy().tofile(OUT / 'pre.i32.bin')
brain.ptr.numpy().tofile(OUT / 'ptr.i32.bin')
brain.inputs.to(torch.int32).numpy().tofile(OUT / 'inputs.i32.bin')
brain.neuron_bias.detach().numpy().tofile(OUT / 'bias.f32.bin')
brain.encoder.weight.detach().to(torch.float32).numpy().tofile(OUT / 'encoder.f32.bin')
brain.readout.weight.detach().to(torch.float32).numpy().tofile(OUT / 'readout.f32.bin')
brain.readout.bias.detach().to(torch.float32).numpy().tofile(OUT / 'readout_bias.f32.bin')
brain.normalizer.weight.detach().numpy().tofile(OUT / 'norm_w.f32.bin')
brain.normalizer.bias.detach().numpy().tofile(OUT / 'norm_b.f32.bin')
obs_mean.numpy().tofile(OUT / 'obs_mean.f32.bin')
obs_std.numpy().tofile(OUT / 'obs_std.f32.bin')
brain.outputs.to(torch.int32).numpy().tofile(OUT / 'outputs.i32.bin')

meta = dict(
    n=int(brain.n), nnz=int(values.numel()),
    observation_size=int(brain.encoder.in_features),
    inputs_count=int(len(brain.inputs)), outputs_count=int(len(brain.outputs)),
    neural_steps=int(brain.neural_steps), action_size=int(brain.readout.out_features),
    fixtures=fixtures,
)
(OUT / 'reference.json').write_text(json.dumps(meta))
sizes = {p.name: f'{p.stat().st_size/2**20:.1f}MB' for p in OUT.glob('*.bin')}
print('exported:', json.dumps(sizes, indent=0).replace('\n', ' '))
print(f"n={meta['n']} nnz={meta['nnz']} obs={meta['observation_size']} "
      f"inputs={meta['inputs_count']} outputs={meta['outputs_count']}")
