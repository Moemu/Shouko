"""PPO critic input coordinates, saved independently of the actor checkpoint."""
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class ValueObservationStats:
    mean: torch.Tensor
    std: torch.Tensor

    @classmethod
    def restore(cls, state: dict | None, actor_mean: torch.Tensor, actor_std: torch.Tensor):
        if state is None or 'value_observation_stats' not in state:
            mean, std = actor_mean, actor_std
        else:
            saved = state['value_observation_stats']
            if not isinstance(saved, dict) or set(saved) != {'mean', 'std'}:
                raise ValueError('Invalid critic observation statistics')
            mean, std = saved['mean'], saved['std']
        if not isinstance(mean, torch.Tensor) or not isinstance(std, torch.Tensor):
            raise ValueError('Critic observation statistics must be tensors')
        if mean.shape != actor_mean.shape or std.shape != actor_std.shape:
            raise ValueError('Critic observation statistics do not match observation size')
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (std > 0).all():
            raise ValueError('Critic observation statistics must be finite with positive scales')
        return cls(mean.detach().to(actor_mean).clone(), std.detach().to(actor_std).clone())

    def normalize(self, observations: torch.Tensor) -> torch.Tensor:
        return (observations - self.mean) / self.std

    def state_dict(self) -> dict[str, torch.Tensor]:
        return dict(mean=self.mean.detach().cpu().clone(), std=self.std.detach().cpu().clone())


@torch.no_grad()
def set_new_actor_scales(policy, scales: list[float]):
    """Change only zero-initialized columns, preserving the actor's initial map."""
    if policy.encoder.in_features != 50:
        raise ValueError('New actor input scales require 50 observations')
    candidate = torch.as_tensor(scales, device=policy.obs_std.device, dtype=policy.obs_std.dtype)
    if candidate.shape != (3,) or not torch.isfinite(candidate).all() or not (candidate > 0).all():
        raise ValueError('Provide three finite positive actor input scales')
    if torch.count_nonzero(policy.encoder.weight[:, 47:]):
        raise ValueError('Cannot rescale learned actor columns without changing the policy')
    policy.obs_std[47:].copy_(candidate)
