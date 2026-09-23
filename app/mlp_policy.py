"""Small policy baseline for matched locomotion experiments, never a preview default."""
from pathlib import Path

import torch
from torch import nn


class MLPPolicy(nn.Module):
    def __init__(self, observation_size=50, action_size=12, device='cuda'):
        super().__init__()
        self.encoder = nn.Linear(observation_size, 256, device=device)
        self.hidden = nn.Sequential(nn.ELU(), nn.Linear(256, 256, device=device), nn.ELU(),
                                    nn.Linear(256, 128, device=device), nn.ELU())
        self.readout = nn.Linear(128, action_size, device=device)
        nn.init.normal_(self.readout.weight, std=0.005)
        nn.init.zeros_(self.readout.bias)
        self.register_buffer('obs_mean', torch.zeros(observation_size, device=device))
        self.register_buffer('obs_std', torch.ones(observation_size, device=device))
        self.physics_interface = None

    def forward(self, observation):
        x = ((observation-self.obs_mean)/self.obs_std).clamp(-10, 10)
        features = self.hidden(self.encoder(x))
        return self.readout(features), features

    def load(self, path, interface=None):
        data = torch.load(path, map_location=self.obs_mean.device, weights_only=True)
        recorded = data.get('physics_interface')
        if recorded is not None and interface is not None and any(
                interface.get(key) != value for key, value in recorded.items()):
            raise ValueError('Baseline checkpoint interface mismatch')
        if data.get('policy_kind') == 'mlp':
            self.load_state_dict(data['state_dict'])
            extra = data['extra']
        else:
            # Same observation coordinates and body, with an independently initialized actor.
            self.obs_mean.copy_(data['state_dict']['obs_mean'])
            self.obs_std.copy_(data['state_dict']['obs_std'])
            extra = {}
        self.physics_interface = recorded
        return extra

    def save(self, path, extra=None, physics=None):
        target = Path(path)
        temporary = target.with_suffix('.tmp')
        torch.save(dict(policy_kind='mlp', state_dict=self.state_dict(),
                        observation_size=self.encoder.in_features,
                        action_size=self.readout.out_features, neural_steps=0,
                        physics_interface=physics, extra=extra or {}), temporary)
        temporary.replace(target)
