"""CPU diagnostics for a real PPO rollout, without resampling actions or physics."""
import argparse
import json
from pathlib import Path

import torch


def gae_targets(rewards, values, dones, gamma, lam):
    """Terminal transitions do not bootstrap through the reset state."""
    advantages = torch.zeros_like(rewards)
    gae = torch.zeros_like(rewards[0])
    for t in reversed(range(len(rewards))):
        alive = 1 - dones[t]
        delta = rewards[t] + gamma * values[t + 1] * alive - values[t]
        gae = delta + gamma * lam * alive * gae
        advantages[t] = gae
    return advantages, advantages + values[:-1]


def regression_metrics(predictions, targets):
    predictions, targets = predictions.double().reshape(-1), targets.double().reshape(-1)
    error = predictions - targets
    variance = float(targets.var(unbiased=False))
    mse = float(error.square().mean())
    return dict(samples=len(targets), target_mean=float(targets.mean()),
                target_variance=variance, prediction_mean=float(predictions.mean()),
                bias=float(error.mean()), mse=mse, rmse=mse ** 0.5,
                r2=1 - mse / variance if variance > 0 else None,
                explained_variance=1 - float(error.var(unbiased=False)) / variance if variance > 0 else None)


def save_rollout(path, *, observations, actions, rewards, dones, values, std,
                 advantages, returns, gamma, lam, interface, iteration, seed, checkpoint_sha256, reward_config):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tensors = dict(observations=observations, actions=actions, rewards=rewards,
                   dones=dones, values=values, std=std, advantages=advantages, returns=returns)
    batch = dict(schema=1, stage='before_ppo_update', gamma=gamma, lam=lam,
                 physics_interface=interface, iteration=iteration, seed=seed,
                 checkpoint_sha256=checkpoint_sha256, reward_config=reward_config,
                 **{name: value.detach().cpu() for name, value in tensors.items()})
    temporary = target.with_suffix('.tmp')
    torch.save(batch, temporary)
    temporary.replace(target)


def diagnose(batch):
    if batch.get('schema') != 1 or batch.get('stage') != 'before_ppo_update':
        raise ValueError('Expected a version 1 training rollout before the PPO update')
    rewards, values, dones = (batch[key] for key in ('rewards', 'values', 'dones'))
    observations, actions, std = (batch[key] for key in ('observations', 'actions', 'std'))
    if rewards.ndim != 2 or min(rewards.shape) < 1:
        raise ValueError('Expected nonempty time x world rewards')
    steps, worlds = rewards.shape
    if (values.shape != (steps + 1, worlds) or dones.shape != rewards.shape or
            observations.ndim != 3 or observations.shape[:2] != rewards.shape or
            observations.shape[2] not in (47, 50) or actions.shape != (steps, worlds, 12) or
            std.shape != (12,)):
        raise ValueError('Rollout tensor shapes do not align')
    for key in ('advantages', 'returns'):
        if batch[key].shape != rewards.shape:
            raise ValueError(f'Unaligned rollout tensor: {key}')
    for key in ('observations', 'actions', 'rewards', 'dones', 'values', 'std', 'advantages', 'returns'):
        if not bool(torch.isfinite(batch[key]).all()):
            raise ValueError(f'Non-finite rollout tensor: {key}')
    if not bool(((dones == 0) | (dones == 1)).all()) or not bool((std > 0).all()):
        raise ValueError('Invalid terminal mask or exploration standard deviation')
    gamma, lam = batch['gamma'], batch['lam']
    if not 0 <= gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError('Gamma and lambda must be in [0, 1]')
    expected_advantages, expected_targets = gae_targets(rewards, values, dones, gamma, lam)
    advantages, targets = batch['advantages'], batch['returns']
    if (not torch.allclose(advantages, expected_advantages, rtol=1e-4, atol=1e-5) or
            not torch.allclose(targets, expected_targets, rtol=1e-4, atol=1e-5)):
        raise ValueError('Saved targets disagree with rollout rewards, values or terminal masks')
    terminal = dones.bool()
    command_scale = batch['physics_interface']['cmd_scale'][0]
    if not command_scale > 0:
        raise ValueError('Forward command scale must be positive')
    commands = observations[..., 6] / command_scale
    per_command = []
    for command in torch.unique(commands):
        selected = commands == command
        per_command.append(dict(target_speed=float(command),
                                **regression_metrics(values[:-1][selected], targets[selected])))
    return dict(kind='training_rollout_diagnostic', iteration=batch['iteration'], seed=batch['seed'],
                checkpoint_sha256=batch['checkpoint_sha256'], reward_config=batch['reward_config'],
                stage=batch['stage'], gamma=gamma, lam=lam,
                physics_interface=batch['physics_interface'],
                observation_size=observations.shape[2], worlds=worlds, steps=steps,
                exploration_std=std.tolist(), reward_mean=float(rewards.mean()),
                advantage_mean=float(advantages.mean()),
                all=regression_metrics(values[:-1], targets),
                per_command=per_command,
                terminal=regression_metrics(values[:-1][terminal], targets[terminal]) if terminal.any() else None,
                nonterminal=regression_metrics(values[:-1][~terminal], targets[~terminal]) if (~terminal).any() else None,
                limitation='In-sample bootstrapped targets; these metrics do not prove value accuracy, irreducible noise, or walking success.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = diagnose(torch.load(args.rollout, map_location='cpu', weights_only=True))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(report, allow_nan=False))


if __name__ == '__main__':
    main()
