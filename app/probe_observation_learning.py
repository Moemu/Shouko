"""Compare one offline PPO update with and without scaling the three new inputs.

Uses a recorded rollout, never steps physics, and never saves policy weights.
The zero-column baseline is required so both conditions start with the same actor.
"""
import argparse
import copy
import json
import time
from pathlib import Path

import psutil
import torch

from .full_brain import ConnectomePolicy, checkpoint_configuration
from .ppo_yumi import Value, freeze_policy_core
from .train_full import file_sha256, restore_optimizer, validated_checkpoint_state
from .value_diagnostics import diagnose, regression_metrics
from .value_observation import ValueObservationStats


def rms(tensor):
    return float(tensor.detach().double().square().mean().sqrt())


def critic_probe(state, observation, targets, mean, old_scale, new_scale):
    value = Value(50)
    value.load_state_dict(state['value'])
    with torch.no_grad():
        baseline = value((observation - mean) / old_scale)
        naive = value((observation - mean) / new_scale)
        # Scale-only coordinate change; the mean stays fixed.
        value.net[0].weight.mul_(new_scale / old_scale)
        preserved = value((observation - mean) / new_scale)
    torch.testing.assert_close(preserved, baseline, rtol=1e-5, atol=1e-4)
    return dict(baseline=regression_metrics(baseline, targets),
                naive_shared_scale=regression_metrics(naive, targets),
                reparameterized=regression_metrics(preserved, targets),
                naive_output_rms_change=rms(naive - baseline),
                preserved_max_abs_error=float((preserved - baseline).abs().max()),
                note='Analytic CPU check only. No critic parameters or optimizer state are saved.')


def optimizer_for(policy, log_std, saved, learning_rate):
    names = [['neuron_bias'], ['readout.weight', 'readout.bias'],
             ['encoder.weight'], ['log_std']]
    parameters = dict(policy.named_parameters(), log_std=log_std)
    rates = [learning_rate * 0.1, learning_rate, learning_rate * 0.5, learning_rate * 0.1]
    optimizer = torch.optim.Adam([
        dict(params=[parameters[name] for name in group], lr=rate)
        for group, rate in zip(names, rates)], foreach=False)
    if not restore_optimizer(optimizer, copy.deepcopy(saved['optimizer']),
                             saved.get('optimizer_names'), names):
        raise ValueError('Cannot reproduce the source optimizer groups')
    # The cloud probe applies these rates after its value-only warmup.
    for group, rate in zip(optimizer.param_groups, rates):
        group['lr'] = rate
    return optimizer


def main(args):
    if args.samples < 2 or args.microbatch < 1 or args.max_seconds <= 0:
        raise ValueError('Invalid sample, microbatch or time limit')
    started = time.perf_counter()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    available = psutil.virtual_memory().available
    # Checkpoint tensors stay memory-mapped; full-graph work uses small GPU chunks.
    minimum_ram = (256 if args.statistics_only else 768) * 1024**2
    if available < minimum_ram:
        raise RuntimeError(f'Need at least {minimum_ram / 2**30:g} GiB available RAM for this probe')
    if args.device == 'cuda' and not args.statistics_only:
        battery = psutil.sensors_battery()
        if battery is not None and not battery.power_plugged:
            raise RuntimeError('Connect laptop power before the CUDA probe')
        if torch.cuda.mem_get_info()[0] < 2 * 1024**3:
            raise RuntimeError('Need at least 2 GiB free device memory')
        torch.cuda.set_per_process_memory_fraction(0.3)

    batch = torch.load(args.rollout, map_location='cpu', weights_only=True)
    diagnose(batch)
    digest = file_sha256(args.checkpoint)
    if batch['checkpoint_sha256'] != digest:
        raise ValueError('Rollout and checkpoint hashes differ')
    source = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    weights = source['state_dict']
    if weights['encoder.weight'].shape[1] != 50 or torch.count_nonzero(weights['encoder.weight'][:, 47:]):
        raise ValueError('Probe requires the unchanged 50-input zero-column baseline')
    state = validated_checkpoint_state(args.checkpoint, args.state, 'cpu')
    value_stats = ValueObservationStats.restore(state, weights['obs_mean'], weights['obs_std'])
    observation = batch['observations'].reshape(-1, 50)
    old_scale = weights['obs_std']
    new_scale = old_scale.clone()
    new_scale[47:] = observation[:, 47:].std(0).clamp_min(0.05)
    x = (observation - weights['obs_mean']) / old_scale
    report = dict(kind='offline_one_update_input_scale_probe', completed=False,
                  phase='statistics_complete', checkpoint_sha256=digest,
                  rollout_sha256=file_sha256(args.rollout), state_sha256=file_sha256(args.state),
                  device='cpu' if args.statistics_only else args.device,
                  available_ram_gib=available / 2**30,
                  source_optimizer_slots=len(state['optimizer']['state']),
                  seed=args.seed, learning_rate=args.lr, entropy_coefficient=0.001,
                  original_scales=old_scale.tolist(),
                  candidate_scales=new_scale.tolist(), conditions=[],
                  observation_statistics=[dict(column=i, mean=float(observation[:, i].mean()),
                                               raw_std=float(observation[:, i].std()),
                                               normalized_std=float(x[:, i].std()),
                                               clamp_fraction=float((x[:, i].abs() > 10).float().mean()))
                                          for i in range(50)],
                  critic=critic_probe(state, observation, batch['returns'].reshape(-1),
                                      value_stats.mean, value_stats.std, new_scale),
                  limitation='One stored rollout and one offline update; no on-policy continuation, walking or causal success claim. Scaling uses this diagnostic batch, not held-out data.')

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report['elapsed_seconds'] = time.perf_counter() - started
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

    def check_budget():
        if time.perf_counter() - started >= args.max_seconds:
            report['phase'] = 'time_limit'
            save()
            raise TimeoutError('Offline probe time budget reached')
        if psutil.virtual_memory().available < 256 * 1024**2:
            report['phase'] = 'low_memory'
            save()
            raise RuntimeError('Stopped probe: available system RAM fell below 256 MiB')

    save()
    if args.statistics_only:
        report['statistics_only'] = True
        report['completed'] = True
        save()
        return
    report['phase'] = 'actor_probe'
    config = checkpoint_configuration(args.checkpoint)
    policy = ConnectomePolicy(observation_size=50, action_size=config['action_size'],
                               neural_steps=config['neural_steps'], device=args.device)
    if source['graph_sha256'] != policy.meta['graph_sha256']:
        raise ValueError('Source checkpoint belongs to a different graph')
    freeze_policy_core(policy)
    selected = torch.randperm(len(observation))[:min(args.samples, len(observation))]
    obs = observation[selected].to(args.device)
    actions = batch['actions'].reshape(-1, 12)[selected].to(args.device)
    all_advantages = batch['advantages'].reshape(-1)
    advantages = ((all_advantages - all_advantages.mean()) /
                  (all_advantages.std() + 1e-8))[selected].to(args.device)
    report['selected_flat_indices'] = selected.tolist()
    report['microbatch'] = args.microbatch

    @torch.no_grad()
    def predict():
        predictions = []
        for part in obs.split(args.microbatch):
            check_budget()
            predictions.append(policy(part)[0])
        return torch.cat(predictions)

    policy.load_state_dict(weights)
    reference = predict()
    reference_std = state['log_std'].to(args.device).exp().clamp(0.05, 1.5)
    old_logp = torch.distributions.Normal(reference, reference_std).log_prob(actions).sum(-1)
    for label, scale in [('original', old_scale), ('new_inputs_scaled', new_scale)]:
        check_budget()
        policy.load_state_dict(weights)
        policy.zero_grad(set_to_none=True)
        with torch.no_grad():
            policy.obs_std.copy_(scale)
        initial = predict()
        torch.testing.assert_close(initial, reference, rtol=1e-5, atol=1e-6)
        log_std = torch.nn.Parameter(state['log_std'].to(args.device).clone())
        optimizer = optimizer_for(policy, log_std, state, args.lr)
        optimizer.zero_grad(set_to_none=True)
        for first in range(0, len(obs), args.microbatch):
            check_budget()
            sl = slice(first, min(first + args.microbatch, len(obs)))
            mean, _ = policy(obs[sl])
            distribution = torch.distributions.Normal(mean, log_std.exp().clamp(0.05, 1.5))
            ratio = (distribution.log_prob(actions[sl]).sum(-1) - old_logp[sl]).exp()
            objective = torch.minimum(ratio * advantages[sl], ratio.clamp(0.8, 1.2) * advantages[sl])
            entropy = distribution.entropy().sum(-1)
            (-(objective + 0.001 * entropy).sum() / len(obs)).backward()
        encoder_gradient = policy.encoder.weight.grad.detach().clone()
        gradient_norms = {
            name: float(torch.stack([p.grad.detach().square().sum() for p in group['params']]).sum().sqrt())
            for name, group in zip(('neuron_bias', 'readout', 'encoder', 'log_std'), optimizer.param_groups)
        }
        parameters = [p for group in optimizer.param_groups for p in group['params']]
        total_norm = torch.nn.utils.clip_grad_norm_(parameters, 5.0, foreach=False)
        if not torch.isfinite(total_norm):
            report['phase'] = 'nonfinite_gradient'
            save()
            raise FloatingPointError('Offline policy gradient is non-finite')
        optimizer.step()
        updated = predict()
        delta = policy.encoder.weight.detach().cpu() - weights['encoder.weight']
        row = dict(condition=label, initial_action_max_abs_error=float((initial-reference).abs().max()),
                   initial_action_rms_error=rms(initial-reference),
                   unclipped_gradient_norm=float(total_norm), clip_multiplier=min(1., 5./(float(total_norm)+1e-6)),
                   gradient_norm_by_group=gradient_norms,
                   encoder_gradient_rms_by_column=encoder_gradient.double().square().mean(0).sqrt().cpu().tolist(),
                   encoder_nonzero_gradient_fraction_by_column=(encoder_gradient != 0).float().mean(0).cpu().tolist(),
                   encoder_update_rms_by_column=delta.double().square().mean(0).sqrt().tolist(),
                   action_rms_change=rms(updated-reference),
                   mean_kl_fixed_std=float(((updated-reference).square()/(2*reference_std.square())).sum(-1).mean()))
        new_columns = policy.encoder.weight[:, 47:].detach().clone()
        with torch.no_grad():
            policy.encoder.weight[:, 47:] = 0
        without_new = predict()
        row['new_channel_action_rms'] = rms(updated-without_new)
        row['old_path_action_rms_change'] = rms(without_new-reference)
        with torch.no_grad():
            policy.encoder.weight[:, 47:] = new_columns
        updated_parameters = {name: p.detach().clone() for name, p in policy.named_parameters()
                              if p.requires_grad}
        route_effects = {}
        for route in ('neuron_bias', 'readout', 'encoder_old', 'encoder_new'):
            with torch.no_grad():
                for name, parameter in policy.named_parameters():
                    if name in updated_parameters:
                        parameter.copy_(weights[name])
                        if name == route or name.startswith(route + '.'):
                            parameter.copy_(updated_parameters[name])
                if route.startswith('encoder_'):
                    columns = slice(0, 47) if route == 'encoder_old' else slice(47, 50)
                    policy.encoder.weight[:, columns] = updated_parameters['encoder.weight'][:, columns]
            route_effects[route] = rms(predict() - reference)
        with torch.no_grad():
            for name, parameter in policy.named_parameters():
                if name in updated_parameters:
                    parameter.copy_(updated_parameters[name])
        row['isolated_step_action_rms'] = route_effects
        row['route_effect_note'] = 'Apply one parameter group delta at a time to the initial actor. Nonlinear effects need not add up.'
        report['conditions'].append(row)
        save()
        print(json.dumps({key: value for key, value in row.items() if not isinstance(value, list)}), flush=True)
    if args.device == 'cuda':
        report['peak_tensor_gib'] = torch.cuda.max_memory_allocated() / 2**30
    report['completed'] = True
    report['phase'] = 'completed'
    save()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--state', required=True, type=Path)
    parser.add_argument('--rollout', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--statistics-only', action='store_true',
                        help='read observations and the small critic on CPU without loading the graph')
    parser.add_argument('--samples', type=int, default=64)
    parser.add_argument('--microbatch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=5e-6)
    parser.add_argument('--seed', type=int, default=20260921)
    parser.add_argument('--max-seconds', type=float, default=180)
    main(parser.parse_args())
