"""Bounded MLP PPO runner for the independent 18-joint goal task, without a teacher."""
import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import time
import math

import torch
import warp as wp

from .free_body import ROOT
from .goal_checkpoint import goal_contract, load_goal_checkpoint, save_goal_checkpoint
from .goal_task import OBSERVATION_SIZE, GoalTask
from .gpu_goal import GPUGoalEnv
from .ppo_yumi import Value
from .mlp_policy import MLPPolicy
from .recovery_curriculum import bootstrap_timeouts
from .training_control import TrainingControl
from .train_full import atomic_json
from .value_diagnostics import gae_targets


@dataclass(frozen=True)
class PPOSettings:
    seed: int = 11
    worlds: int = 16
    steps: int = 64
    epochs: int = 2
    minibatch: int = 256
    lr: float = 3e-4
    value_lr: float = 1e-3
    gamma: float = .99
    lam: float = .95
    clip: float = .2
    entropy: float = .001
    target_kl: float = .02
    initial_std: float = .35


def train(output: Path, settings: PPOSettings, iterations: int, max_seconds: float,
          resume: Path | None = None, *, work_cost: float = GoalTask.work_cost) -> dict:
    if min(settings.worlds, settings.steps, settings.epochs, settings.minibatch, iterations) < 1:
        raise ValueError('PPO dimensions and iteration budget must be positive')
    if settings.worlds * settings.steps < 2 or not math.isfinite(max_seconds) or max_seconds <= 0:
        raise ValueError('Use at least two rollout samples and a finite positive time budget')
    if not math.isfinite(work_cost) or work_cost < 0 or not math.isfinite(settings.initial_std) or not .05 <= settings.initial_std <= 1.5:
        raise ValueError('Use finite nonnegative work cost and initial standard deviation in [0.05, 1.5]')
    checkpoint_settings = asdict(settings)
    if work_cost != GoalTask.work_cost:
        checkpoint_settings['training_work_cost'] = work_cost
    output.mkdir(parents=True, exist_ok=False)
    torch.manual_seed(settings.seed)
    torch.set_num_threads(1)
    started = time.perf_counter()
    status = dict(phase='initializing', method='mlp_goal_ppo', history=[], settings=checkpoint_settings,
                  training_reward=dict(progress=1., arrival=GoalTask.arrival_reward, work_cost=work_cost),
                  resume_kind='optimizer_warm_start_new_environment' if resume else 'random_initialization',
                  max_seconds=max_seconds, max_launch_iterations=iterations)
    atomic_json(output / 'training.json', status)
    env = GPUGoalEnv(settings.worlds, settings.seed)
    contract = goal_contract(env.body)
    policy = MLPPolicy(OBSERVATION_SIZE, 18, device='cuda')
    value = Value(OBSERVATION_SIZE).cuda()
    log_std = torch.nn.Parameter(torch.full((18,), settings.initial_std, device='cuda').log())
    parameters = [*policy.parameters(), log_std]
    optimizer = torch.optim.Adam(parameters, lr=settings.lr, foreach=False)
    value_optimizer = torch.optim.Adam(value.parameters(), lr=settings.value_lr, foreach=False)
    iteration = env_steps = 0
    if resume is not None:
        iteration, env_steps = load_goal_checkpoint(resume, contract=contract, policy=policy, value=value,
                                                   log_std=log_std, optimizer=optimizer, value_optimizer=value_optimizer,
                                                   settings=checkpoint_settings)
        with resume.open('rb') as handle:
            status['source_checkpoint_sha256'] = hashlib.file_digest(handle, 'sha256').hexdigest()
    initial_checkpoint_sha256 = save_goal_checkpoint(output / 'initial.pt', contract=contract, policy=policy, value=value,
                                                    log_std=log_std, optimizer=optimizer, value_optimizer=value_optimizer,
                                                    iteration=iteration, env_steps=env_steps, settings=checkpoint_settings)
    initial_weights = {name: tensor.detach().clone() for name, tensor in policy.named_parameters()}
    torch.cuda.synchronize()
    status['initialization_seconds'] = time.perf_counter() - started
    status['initial_iteration'], status['initial_env_steps'] = iteration, env_steps
    control = TrainingControl(max_seconds)
    torch.cuda.reset_peak_memory_stats()
    observation = env.observation()
    launch_iterations = 0
    launch_steps = 0

    def write_status(phase: str | None = None) -> None:
        status.update(phase=phase or status['phase'], iteration=iteration, env_steps=env_steps, **control.fields())
        atomic_json(output / 'training.json', status)

    try:
        while launch_iterations < iterations and control.safe_point(write_status) and not control.budget_exceeded():
            write_status('training')
            rollout_started = time.perf_counter()
            observations, actions, logps, rewards, values, dones = [], [], [], [], [], []
            arrivals = completed_episodes = timeout_episodes = arena_episodes = 0
            raw_reward = task_reward_sum = progress_sum = work_sum = 0.
            max_contacts = max_constraints = 0
            max_penetration = 0.
            with torch.no_grad():
                for _ in range(settings.steps):
                    if control.budget_exceeded():
                        break
                    mean, _ = policy(observation)
                    distribution = torch.distributions.Normal(mean, log_std.exp().clamp(.05, 1.5))
                    action = distribution.sample()
                    predicted = value(observation)
                    next_observation, result = env.step(action)
                    if bool(result.numerical.any()) or not bool(torch.isfinite(next_observation).all()):
                        raise FloatingPointError('Numerical goal transition; PPO update aborted')
                    training_reward = result.reward + (GoalTask.work_cost - work_cost) * result.work_j
                    reward = training_reward
                    if bool(result.truncated.any()):
                        reward = bootstrap_timeouts(reward, value(next_observation), result.truncated, settings.gamma)
                    done = result.terminated | result.truncated
                    observations.append(observation)
                    actions.append(action)
                    logps.append(distribution.log_prob(action).sum(-1))
                    rewards.append(reward)
                    values.append(predicted)
                    dones.append(done.float())
                    arrivals += int(result.reached.sum())
                    completed_episodes += int(result.complete.sum())
                    timeout_episodes += int(result.timeout.sum())
                    arena_episodes += int(result.arena.sum())
                    raw_reward += float(training_reward.sum())
                    task_reward_sum += float(result.reward.sum())
                    progress_sum += float(result.progress.sum())
                    work_sum += float(result.work_j.sum())
                    max_contacts = max(max_contacts, int(env.peak_contacts_total.max()))
                    max_constraints = max(max_constraints, int(env.peak_constraints.max()))
                    max_penetration = max(max_penetration, float(env.peak_penetration.max()))
                    observation = env.reset(done) if bool(done.any()) else next_observation
                if not observations:
                    break
                next_value = value(observation)
            advantage, returns = gae_targets(torch.stack(rewards), torch.stack(values + [next_value]),
                                             torch.stack(dones), settings.gamma, settings.lam)
            if not bool(torch.isfinite(advantage).all() and torch.isfinite(returns).all()):
                raise FloatingPointError('Non-finite goal PPO targets')
            obs_batch, act_batch, old_logp = torch.cat(observations), torch.cat(actions), torch.cat(logps)
            advantage, returns = advantage.flatten(), returns.flatten()
            advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
            torch.cuda.synchronize()
            rollout_seconds = time.perf_counter() - rollout_started
            update_started = time.perf_counter()
            update_count = 0
            pi_loss_sum = value_loss_sum = kl_sum = 0.
            stop_early = False
            for _ in range(settings.epochs):
                permutation = torch.randperm(len(obs_batch), device='cuda')
                for start in range(0, len(obs_batch), settings.minibatch):
                    mb = permutation[start:start + settings.minibatch]
                    mean, _ = policy(obs_batch[mb])
                    distribution = torch.distributions.Normal(mean, log_std.exp().clamp(.05, 1.5))
                    logratio = distribution.log_prob(act_batch[mb]).sum(-1) - old_logp[mb]
                    ratio = logratio.exp()
                    kl = ((ratio - 1) - logratio).mean()
                    pi_loss = -torch.minimum(ratio * advantage[mb], ratio.clamp(1-settings.clip, 1+settings.clip) * advantage[mb]).mean()
                    pi_loss -= settings.entropy * distribution.entropy().sum(-1).mean()
                    optimizer.zero_grad(set_to_none=True)
                    pi_loss.backward()
                    gradient = torch.nn.utils.clip_grad_norm_(parameters, 5., foreach=False)
                    if not bool(torch.isfinite(gradient)):
                        raise FloatingPointError('Non-finite goal actor gradient')
                    optimizer.step()
                    value_loss = (value(obs_batch[mb]) - returns[mb]).square().mean()
                    value_optimizer.zero_grad(set_to_none=True)
                    value_loss.backward()
                    gradient = torch.nn.utils.clip_grad_norm_(value.parameters(), 5., foreach=False)
                    if not bool(torch.isfinite(gradient)):
                        raise FloatingPointError('Non-finite goal value gradient')
                    value_optimizer.step()
                    update_count += 1
                    pi_loss_sum += float(pi_loss.detach())
                    value_loss_sum += float(value_loss.detach())
                    kl_sum += float(kl.detach())
                    if float(kl.detach()) > 1.5 * settings.target_kl:
                        stop_early = True
                        break
                if stop_early:
                    break
            torch.cuda.synchronize()
            update_seconds = time.perf_counter() - update_started
            iteration += 1
            launch_iterations += 1
            transitions = len(obs_batch)
            env_steps += transitions
            launch_steps += transitions
            row = dict(iteration=iteration, transitions=transitions, actor_optimizer_steps=update_count,
                       value_optimizer_steps=update_count, reward_mean=raw_reward / transitions,
                       task_reward_mean=task_reward_sum / transitions, progress_sum_m=progress_sum,
                       work_penalty=work_cost * work_sum,
                       absolute_work_j=work_sum, arrivals=arrivals, completed_episodes=completed_episodes,
                       timeout_episodes=timeout_episodes, arena_episodes=arena_episodes,
                       pi_loss=pi_loss_sum / update_count, value_loss=value_loss_sum / update_count,
                       approximate_kl=kl_sum / update_count, kl_stop=stop_early,
                       rollout_seconds=rollout_seconds, update_seconds=update_seconds,
                       rollout_samples_per_second=transitions / rollout_seconds,
                       training_samples_per_second=transitions / (rollout_seconds + update_seconds),
                       max_contacts_total=max_contacts, max_constraints_per_world=max_constraints,
                       max_penetration_m=max_penetration)
            status['history'].append(row)
            write_status('training')
            print(json.dumps(row, allow_nan=False), flush=True)
    except Exception as error:
        status['error'] = f'{type(error).__name__}: {error}'
        write_status('failed')
        raise

    save_started = time.perf_counter()
    checkpoint_sha256 = save_goal_checkpoint(output / 'last.pt', contract=contract, policy=policy, value=value,
                                            log_std=log_std, optimizer=optimizer, value_optimizer=value_optimizer,
                                            iteration=iteration, env_steps=env_steps, settings=checkpoint_settings)
    actor_change = max(float((tensor.detach() - initial_weights[name]).abs().max()) for name, tensor in policy.named_parameters())
    if launch_steps and actor_change == 0:
        raise AssertionError('PPO sampled transitions without changing the actor')
    history = status['history']
    duration = sum(row['rollout_seconds'] + row['update_seconds'] for row in history)
    report = dict(kind='goal_ppo_run', device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                  contract=asdict(contract), settings=checkpoint_settings, checkpoint_sha256=checkpoint_sha256,
                  training_reward=status['training_reward'],
                  initial_checkpoint_sha256=initial_checkpoint_sha256,
                  resume_kind=status['resume_kind'], source_checkpoint_sha256=status.get('source_checkpoint_sha256'),
                  initial_iteration=status['initial_iteration'], final_iteration=iteration,
                  initial_env_steps=status['initial_env_steps'], final_env_steps=env_steps,
                  launch_iterations=launch_iterations, launch_transitions=launch_steps,
                  actor_max_change=actor_change, actor_optimizer_steps=sum(row['actor_optimizer_steps'] for row in history),
                  arrivals=sum(row['arrivals'] for row in history),
                  completed_episodes=sum(row['completed_episodes'] for row in history),
                  timeout_episodes=sum(row['timeout_episodes'] for row in history),
                  arena_episodes=sum(row['arena_episodes'] for row in history),
                  training_samples_per_second=launch_steps / duration if duration else 0.,
                  peak_torch_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                  peak_torch_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                  warp_pool_current_mib=wp.get_mempool_used_mem_current('cuda:0') / 2**20,
                  warp_pool_high_mib=wp.get_mempool_used_mem_high('cuda:0') / 2**20,
                  cuda_free_mib=torch.cuda.mem_get_info()[0] / 2**20,
                  initialization_seconds=status['initialization_seconds'],
                  save_seconds=time.perf_counter() - save_started, elapsed_seconds=time.perf_counter() - started,
                  history=history, limitation='Training metrics only; capability requires separate held-out evaluation.')
    sources = ('app/free_body.py', 'app/goal_task.py', 'app/gpu_goal.py', 'app/goal_checkpoint.py', 'app/ppo_goal.py',
               'app/mlp_policy.py', 'app/ppo_yumi.py', 'app/recovery_curriculum.py', 'app/value_diagnostics.py')
    report['source_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    atomic_json(output / 'summary.json', report)
    write_status('complete')
    print(json.dumps({key: val for key, val in report.items() if key not in ('history', 'contract', 'source_sha256')}, allow_nan=False), flush=True)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True, help='New output directory; existing directories are rejected')
    parser.add_argument('--worlds', type=int, default=16)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--iterations', type=int, default=32, help='Maximum additional iterations in this launch')
    parser.add_argument('--max-seconds', type=float, default=120., help='Training budget, excluding initialization and saving')
    parser.add_argument('--seed', type=int, default=11)
    parser.add_argument('--initial-std', type=float, default=.35)
    parser.add_argument('--work-cost', type=float, default=GoalTask.work_cost, help='Training objective only; task and common evaluation stay unchanged')
    parser.add_argument('--resume', type=Path, help='Restore matching MLP/critic/Adam, then reset the environment')
    args = parser.parse_args()
    train(args.output, PPOSettings(seed=args.seed, worlds=args.worlds, steps=args.steps, initial_std=args.initial_std),
          args.iterations, args.max_seconds, args.resume, work_cost=args.work_cost)


if __name__ == '__main__':
    main()
