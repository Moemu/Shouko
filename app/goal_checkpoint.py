"""Versioned, atomic goal-PPO checkpoint; deliberately separate from preview weights."""
from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path

import torch
from torch import nn

from .free_body import FreeBody, JOINT_NAMES
from .goal_task import GoalTask, OBSERVATION_FIELDS, OBSERVATION_SIZE
from .mlp_policy import MLPPolicy
from .train_full import restore_optimizer


@dataclass(frozen=True)
class GoalContract:
    body_sha256: str
    graph_sha256: str | None
    joint_names: tuple[str, ...]
    observation_fields: tuple[tuple[str, int], ...]
    observation_size: int
    action_size: int
    simulation_dt: float
    control_decimation: int
    action_scale: float
    action_clip: float
    home: tuple[float, ...]
    kp: tuple[float, ...]
    kd: tuple[float, ...]
    joint_ranges: tuple[tuple[float, float], ...]
    torque_limits: tuple[float, ...]
    task: dict[str, float | int]


def goal_contract(body: FreeBody, graph_sha256: str | None = None) -> GoalContract:
    return GoalContract(hashlib.sha256(body.xml.encode()).hexdigest(), graph_sha256,
                        JOINT_NAMES, OBSERVATION_FIELDS, OBSERVATION_SIZE, 18,
                        float(body.model.opt.timestep), body.decimation, body.action_scale, 8.,
                        tuple(body.home.tolist()), tuple(body.kp.tolist()), tuple(body.kd.tolist()),
                        tuple(tuple(row) for row in body.ranges.tolist()), tuple(body.limits.tolist()), GoalTask.parameters())


def optimizer_names(policy: nn.Module, value: nn.Module, log_std: nn.Parameter,
                    optimizer: torch.optim.Optimizer) -> list[list[str]]:
    names = {id(parameter): 'policy.' + name for name, parameter in policy.named_parameters()}
    names.update({id(parameter): 'value.' + name for name, parameter in value.named_parameters()})
    names[id(log_std)] = 'log_std'
    return [[names[id(parameter)] for parameter in group['params']] for group in optimizer.param_groups]


def save_goal_checkpoint(path: Path, *, contract: GoalContract, policy: nn.Module, value: nn.Module,
                         log_std: nn.Parameter, optimizer: torch.optim.Optimizer,
                         value_optimizer: torch.optim.Optimizer, iteration: int, env_steps: int,
                         settings: dict[str, float | int]) -> str:
    if not isinstance(policy, MLPPolicy) or contract.graph_sha256 is not None:
        raise ValueError('Version 1 goal-PPO checkpoints support MLP with no connectome graph')
    payload = dict(schema='yumi_goal_ppo', version=1, policy_kind='mlp', contract=asdict(contract),
                   policy=policy.state_dict(), value=value.state_dict(), log_std=log_std.detach(),
                   optimizer=optimizer.state_dict(), value_optimizer=value_optimizer.state_dict(),
                   optimizer_names=optimizer_names(policy, value, log_std, optimizer),
                   value_optimizer_names=optimizer_names(policy, value, log_std, value_optimizer),
                   iteration=iteration, env_steps=env_steps, settings=settings,
                   torch_rng=torch.get_rng_state(),
                   cuda_rng=torch.cuda.get_rng_state_all() if log_std.is_cuda else [])
    temporary = path.with_name(path.name + '.pending')
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def read_goal_payload(path: Path, contract: GoalContract, device: str | torch.device = 'cpu') -> dict:
    payload = torch.load(path, map_location=device, weights_only=True)
    if payload.get('schema') != 'yumi_goal_ppo' or payload.get('version') != 1 or payload.get('policy_kind') != 'mlp':
        raise ValueError('Expected a version 1 MLP goal-PPO checkpoint')
    if payload.get('contract') != asdict(contract) or contract.graph_sha256 is not None:
        raise ValueError('Goal checkpoint body/graph/control/observation/task contract mismatch')
    if payload['iteration'] < 0 or payload['env_steps'] < 0:
        raise ValueError('Invalid goal checkpoint counters')
    for tensor in payload['policy'].values():
        if not torch.isfinite(tensor).all():
            raise ValueError('Non-finite goal checkpoint policy')
    if not (payload['policy']['obs_std'] > 0).all():
        raise ValueError('Goal observation scales must be positive')
    return payload


def load_goal_policy(path: Path, body: FreeBody, device: str = 'cpu') -> MLPPolicy:
    payload = read_goal_payload(path, goal_contract(body), device)
    policy = MLPPolicy(OBSERVATION_SIZE, 18, device=device)
    policy.load_state_dict(payload['policy'], strict=True)
    return policy.eval()


def load_goal_checkpoint(path: Path, *, contract: GoalContract, policy: nn.Module, value: nn.Module,
                         log_std: nn.Parameter, optimizer: torch.optim.Optimizer,
                         value_optimizer: torch.optim.Optimizer,
                         settings: dict[str, float | int]) -> tuple[int, int]:
    payload = read_goal_payload(path, contract, log_std.device)
    if not isinstance(policy, MLPPolicy) or contract.graph_sha256 is not None:
        raise ValueError('Version 1 goal-PPO checkpoints support only MLP')
    if payload.get('settings') != settings:
        raise ValueError('Goal checkpoint PPO settings mismatch')
    if (payload.get('optimizer_names') != optimizer_names(policy, value, log_std, optimizer)
            or payload.get('value_optimizer_names') != optimizer_names(policy, value, log_std, value_optimizer)):
        raise ValueError('Goal checkpoint optimizer parameter names/order mismatch')
    if payload['iteration'] < 0 or payload['env_steps'] < 0:
        raise ValueError('Invalid goal checkpoint counters')
    for name in ('policy', 'value'):
        for tensor in payload[name].values():
            if not torch.isfinite(tensor).all():
                raise ValueError(f'Non-finite goal checkpoint {name}')
    if not (payload['policy']['obs_std'] > 0).all():
        raise ValueError('Goal observation scales must be positive')
    for name in ('optimizer', 'value_optimizer'):
        for slot in payload[name]['state'].values():
            if any(torch.is_tensor(tensor) and not torch.isfinite(tensor).all() for tensor in slot.values()):
                raise ValueError(f'Non-finite goal checkpoint {name}')
    stored_std = payload['log_std']
    if stored_std.shape != log_std.shape or not torch.isfinite(stored_std).all():
        raise ValueError('Invalid goal exploration state')
    policy.load_state_dict(payload['policy'], strict=True)
    value.load_state_dict(payload['value'], strict=True)
    if (not restore_optimizer(optimizer, payload['optimizer'], payload.get('optimizer_names'),
                              optimizer_names(policy, value, log_std, optimizer))
            or not restore_optimizer(value_optimizer, payload['value_optimizer'], payload.get('value_optimizer_names'),
                                     optimizer_names(policy, value, log_std, value_optimizer))):
        raise ValueError('Goal checkpoint optimizer slots do not match')
    with torch.no_grad():
        log_std.copy_(stored_std)
    torch.set_rng_state(payload['torch_rng'].cpu())
    if log_std.is_cuda:
        torch.cuda.set_rng_state_all([state.cpu() for state in payload['cuda_rng']])
    return int(payload['iteration']), int(payload['env_steps'])
