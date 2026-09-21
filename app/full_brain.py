"""Trainable sensory-to-joint policy whose recurrent edges are the full connectome."""
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]


def checkpoint_configuration(path):
    """Read small metadata using memory mapping, without loading the graph or CUDA."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True, mmap=True)
    size = int(checkpoint.get('observation_size') or 47)
    if size not in (47, 50):
        raise ValueError(f'Unsupported checkpoint observation size: {size}')
    if checkpoint['state_dict']['encoder.weight'].shape[1] != size:
        raise ValueError('Checkpoint encoder and observation metadata disagree')
    return dict(observation_size=size, action_size=int(checkpoint.get('action_size', 12)),
                neural_steps=int(checkpoint.get('neural_steps', 4)),
                physics_interface=checkpoint.get('physics_interface'))


class SparseMessage(torch.autograd.Function):
    """Compute gradients only at measured edges; native CSR backward densifies N by N."""
    @staticmethod
    def forward(ctx, values, activity, ptr, pre):
        n = activity.shape[0]
        matrix = torch.sparse_csr_tensor(ptr, pre, values, size=(n, n), check_invariants=False)
        ctx.save_for_backward(values, activity, ptr, pre)
        return torch.sparse.mm(matrix, activity)

    @staticmethod
    def backward(ctx, gradient):
        values, activity, ptr, pre = ctx.saved_tensors
        n = activity.shape[0]
        matrix = torch.sparse_csr_tensor(ptr, pre, values, size=(n, n), check_invariants=False)
        gradient = gradient.contiguous()
        edge_gradient = torch.sparse.sampled_addmm(matrix, gradient, activity.T, beta=0).values()
        activity_gradient = torch.sparse.mm(matrix.transpose(0, 1), gradient)
        return edge_gradient, activity_gradient, None, None


class ConnectomePolicy(nn.Module):
    def __init__(self, observation_size=47, action_size=12, neural_steps=4, device='cuda'):
        super().__init__()
        self.meta = json.loads((ROOT / 'data/full_graph.json').read_text())
        with (ROOT / 'data/full_graph.npz').open('rb') as handle:
            if hashlib.file_digest(handle, 'sha256').hexdigest() != self.meta['graph_sha256']:
                raise ValueError('Full connectome graph hash mismatch')
        graph = np.load(ROOT / 'data/full_graph.npz', allow_pickle=False)
        self.n = len(graph['ids'])
        self.neural_steps = neural_steps
        for name in ['ptr', 'pre', 'weight', 'inputs', 'outputs']:
            self.register_buffer(name, torch.as_tensor(graph[name], device=device))
        self.edge_delta = nn.Parameter(torch.zeros_like(self.weight))
        self.neuron_bias = nn.Parameter(torch.zeros(self.n, device=device))
        self.encoder = nn.Linear(observation_size, len(self.inputs), bias=False, device=device)
        self.normalizer = nn.LayerNorm(len(self.outputs), device=device)
        self.readout = nn.Linear(len(self.outputs), action_size, device=device)
        nn.init.normal_(self.encoder.weight, std=0.12)
        nn.init.normal_(self.readout.weight, std=0.005)
        nn.init.zeros_(self.readout.bias)
        self.register_buffer('obs_mean', torch.zeros(observation_size, device=device))
        self.register_buffer('obs_std', torch.ones(observation_size, device=device))
        self.physics_interface = None

    def forward(self, observation, state=None, lesion=False):
        batch = observation.shape[0]
        x = ((observation - self.obs_mean) / self.obs_std).clamp(-10, 10)
        signal = torch.zeros((self.n, batch), device=x.device, dtype=x.dtype)
        signal = signal.index_copy(0, self.inputs, self.encoder(x).T)
        activity = torch.zeros_like(signal) if state is None else state
        values = self.weight * (1 + 0.9 * torch.tanh(self.edge_delta))
        if lesion:
            values = values * 0
        for _ in range(self.neural_steps):
            activity = 0.2 * activity + 0.8 * torch.tanh(
                3 * SparseMessage.apply(values, activity, self.ptr, self.pre) + signal + self.neuron_bias[:, None])
        motor = self.normalizer(activity[self.outputs].T)
        return self.readout(motor), activity

    @torch.no_grad()
    def set_observation_stats(self, observations):
        self.obs_mean.copy_(observations.mean(0))
        self.obs_std.copy_(observations.std(0).clamp_min(0.05))

    def save(self, path, extra=None, physics=None):
        target = Path(path)
        temporary = target.with_suffix('.tmp')
        torch.save(dict(state_dict=self.state_dict(), graph_sha256=self.meta['graph_sha256'],
                        observation_size=self.encoder.in_features, action_size=self.readout.out_features,
                        neural_steps=self.neural_steps, physics_interface=physics, extra=extra or {}),
                   temporary)
        temporary.replace(target)

    def load(self, path, interface=None):
        checkpoint = torch.load(path, map_location=self.weight.device, weights_only=True)
        if checkpoint['graph_sha256'] != self.meta['graph_sha256']:
            raise ValueError('Checkpoint graph mismatch')
        if checkpoint['neural_steps'] != self.neural_steps:
            raise ValueError('Checkpoint neural update count mismatch')
        recorded = checkpoint.get('physics_interface')
        if recorded is not None and interface is not None and any(
                interface.get(key) != value for key, value in recorded.items()):
            raise ValueError('Checkpoint physics interface mismatch: home/scales differ from the live body config')
        self.physics_interface = recorded
        self.load_state_dict(checkpoint['state_dict'])
        return checkpoint['extra']
