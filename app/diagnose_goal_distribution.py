"""Read-only paired mean/sampled rollouts for existing goal checkpoints."""
from pathlib import Path
import time

import numpy as np
import torch
import warp as wp

from .diagnose_goal_d0 import write_json
from .evaluate_goal import file_sha256
from .free_body import ROOT
from .goal_checkpoint import goal_contract, read_goal_payload
from .goal_task import OBSERVATION_FIELDS, OBSERVATION_SIZE
from .gpu_goal import GPUGoalEnv
from .mlp_policy import MLPPolicy
from .ppo_yumi import Value
from .recovery_curriculum import bootstrap_timeouts
from .value_diagnostics import gae_targets, regression_metrics


def observation_statistics(observations: np.ndarray, mean: np.ndarray, std: np.ndarray) -> list[dict]:
    values = observations.reshape(-1, OBSERVATION_SIZE)
    normalized = (values - mean) / std
    rows = []
    offset = 0
    for name, size in OBSERVATION_FIELDS:
        for coordinate in range(size):
            i = offset + coordinate
            x = values[:, i]
            rows.append(dict(field=name, coordinate=coordinate, mean=float(x.mean()), std=float(x.std()),
                             quantiles=np.quantile(x, [0., .01, .5, .99, 1.]).tolist(),
                             constant=bool(np.ptp(x) < 1e-8), actor_clipped_fraction=float(np.mean(np.abs(normalized[:, i]) > 10.))))
        offset += size
    return rows


def contact_state(env: GPUGoalEnv) -> np.ndarray:
    counts = int(wp.to_torch(env.data.nacon)[0])
    worlds = wp.to_torch(env.data.contact.worldid)[:counts].cpu().numpy()
    geometries = wp.to_torch(env.data.contact.geom)[:counts].cpu().numpy()
    result = np.zeros((env.worlds, env.body.model.nbody), dtype=bool)
    for world, pair in zip(worlds, geometries):
        if 0 <= world < env.worlds and env.body.floor in pair:
            other = int(pair[1] if pair[0] == env.body.floor else pair[0])
            if other >= 0:
                result[world, env.body.model.geom_bodyid[other]] = True
    return result


@torch.inference_mode()
def paired_rollout(checkpoint: Path, seed: int, mode: str, output: Path, deadline: float) -> dict:
    env = GPUGoalEnv(worlds=8, seed=180000 + seed)
    payload = read_goal_payload(checkpoint, goal_contract(env.body), 'cuda')
    policy = MLPPolicy(OBSERVATION_SIZE, 18, device='cuda').eval().requires_grad_(False)
    policy.load_state_dict(payload['policy'])
    value = Value(OBSERVATION_SIZE).cuda().eval().requires_grad_(False)
    value.load_state_dict(payload['value'])
    std = payload['log_std'].exp().clamp(.05, 1.5)
    before_policy = {k: x.clone() for k, x in policy.state_dict().items()}
    before_value = {k: x.clone() for k, x in value.state_dict().items()}
    before_std = payload['log_std'].clone()
    torch.manual_seed(190000 + seed)
    initial_qpos = env.qpos.cpu().numpy().copy()
    initial_target = env.target.cpu().numpy().copy()
    observation = env.observation()
    records: dict[str, list[np.ndarray]] = {k: [] for k in (
        'observation', 'mean', 'action', 'log_prob', 'value', 'next_value', 'reward', 'progress', 'work_j',
        'done', 'truncated', 'bootstrap_reward', 'center', 'next_center', 'qpos', 'qvel',
        'ground_contact', 'penetration', 'joint_limit_error')}
    reason = ''
    for _ in range(256):
        if time.time() >= deadline:
            reason = 'budget'
            break
        mu, _ = policy(observation)
        distribution = torch.distributions.Normal(mu, std)
        action = distribution.sample() if mode == 'sampled' else mu
        predicted = value(observation)
        center, qpos, qvel = env.center.clone(), env.qpos.clone(), env.qvel.clone()
        next_obs, result = env.step(action)
        if bool(result.numerical.any()):
            reason = 'numerical'
            break
        next_value = value(next_obs)
        reward = bootstrap_timeouts(result.reward, next_value, result.truncated, .99)
        limits = torch.as_tensor(env.body.ranges, dtype=env.qpos.dtype, device='cuda')
        q = env.qpos[:, env.qadr]
        limit_error = torch.maximum(limits[:, 0] - q, q - limits[:, 1]).clamp_min(0).amax(1)
        tensors = dict(observation=observation, mean=mu, action=action, log_prob=distribution.log_prob(action).sum(-1),
                       value=predicted, next_value=next_value, reward=result.reward, progress=result.progress,
                       work_j=result.work_j, done=result.terminated | result.truncated, truncated=result.truncated,
                       bootstrap_reward=reward, center=center, next_center=env.center.clone(), qpos=qpos, qvel=qvel,
                       penetration=env.peak_penetration, joint_limit_error=limit_error)
        for key, tensor in tensors.items():
            records[key].append(tensor.cpu().numpy().copy())
        records['ground_contact'].append(contact_state(env))
        if float(env.peak_penetration.max()) > .030 or float(limit_error.max()) > .15:
            reason = 'physical_guard'
            break
        observation = env.reset(result.terminated | result.truncated) if bool((result.terminated | result.truncated).any()) else next_obs
    arrays = {k: np.asarray(v) for k, v in records.items()}
    count = len(arrays['reward'])
    advantages, returns, td = [], [], []
    for start in range(0, count, 64):
        stop = min(start + 64, count)
        reward = torch.from_numpy(arrays['bootstrap_reward'][start:stop])
        values = torch.from_numpy(np.concatenate((arrays['value'][start:stop], arrays['next_value'][stop-1:stop])))
        dones = torch.from_numpy(arrays['done'][start:stop].astype(np.float32))
        adv, ret = gae_targets(reward, values, dones, .99, .95)
        advantages.append(adv.numpy()); returns.append(ret.numpy())
        td.append((reward + .99 * values[1:] * (1 - dones) - values[:-1]).numpy())
    if count:
        arrays.update(advantage=np.concatenate(advantages), returns=np.concatenate(returns), td_residual=np.concatenate(td))
    arrays.update(initial_qpos=initial_qpos, initial_target=initial_target,
                  obs_mean=policy.obs_mean.cpu().numpy(), obs_std=policy.obs_std.cpu().numpy(), std=std.cpu().numpy())
    np.savez_compressed(output / f'{seed}_{mode}.npz', **arrays)
    assert all(torch.equal(x, before_policy[k]) for k, x in policy.state_dict().items())
    assert all(torch.equal(x, before_value[k]) for k, x in value.state_dict().items())
    assert torch.equal(payload['log_std'], before_std)
    report = dict(seed=seed, mode=mode, complete=count == 256 and not reason, reason=reason,
                  steps=count, transitions=count * env.worlds, checkpoint_sha256=file_sha256(checkpoint),
                  trace_sha256=file_sha256(output / f'{seed}_{mode}.npz'), weights_unchanged=True,
                  gamma=.99, lam=.95, gae_chunk_steps=64, dt=env.dt,
                  observation_fields=OBSERVATION_FIELDS, body_names=[env.body.model.body(i).name for i in range(env.body.model.nbody)])
    if count:
        report.update(observations=observation_statistics(arrays['observation'], arrays['obs_mean'], arrays['obs_std']),
                      value_metrics=regression_metrics(torch.from_numpy(arrays['value']), torch.from_numpy(arrays['returns'])),
                      mean_work_per_transition_j=float(arrays['work_j'].mean()),
                      mean_reward=float(arrays['reward'].mean()), mean_progress_m=float(arrays['progress'].mean()),
                      max_penetration_m=float(arrays['penetration'].max()), max_limit_error_rad=float(arrays['joint_limit_error'].max()))
        phases = []
        seconds = np.arange(count) * env.dt
        for name, mask in [('first', seconds < 1), ('middle', (seconds >= 1) & (seconds < 3)), ('late', seconds >= 3)]:
            if mask.any():
                phases.append(dict(phase=name, transitions=int(mask.sum() * env.worlds),
                                   mean_work_j=float(arrays['work_j'][mask].mean()),
                                   mean_progress_m=float(arrays['progress'][mask].mean()),
                                   mean_advantage=float(arrays['advantage'][mask].mean()),
                                   mean_absolute_td=float(np.abs(arrays['td_residual'][mask]).mean()),
                                   value_metrics=regression_metrics(torch.from_numpy(arrays['value'][mask]), torch.from_numpy(arrays['returns'][mask]))))
        report['phases'] = phases
    write_json(output / f'{seed}_{mode}.json', report)
    print(f"distribution {seed} {mode}: {count * env.worlds} transitions, {reason or 'complete'}", flush=True)
    return report


def run_distribution(output: Path, deadline: float, checkpoint_root: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(1)
    rows = []
    pairs = []
    for seed in (404, 505, 606):
        checkpoint = checkpoint_root / f'baseline_{seed}/last.pt'
        before = file_sha256(checkpoint)
        modes = []
        for mode in ('sampled', 'mean'):
            if time.time() >= deadline:
                break
            row = paired_rollout(checkpoint, seed, mode, output, deadline)
            rows.append(row); modes.append(row)
        assert file_sha256(checkpoint) == before
        if len(modes) == 2:
            with np.load(output / f'{seed}_sampled.npz') as a, np.load(output / f'{seed}_mean.npz') as b:
                assert np.array_equal(a['initial_qpos'], b['initial_qpos'])
                assert np.array_equal(a['initial_target'], b['initial_target'])
                pairs.append(dict(seed=seed, initial_conditions_equal=True,
                                  sampled_work_j=float(a['work_j'].sum()), mean_work_j=float(b['work_j'].sum()),
                                  ratio=float(a['work_j'].sum() / max(1e-12, b['work_j'].sum()))))
        write_json(output / 'progress.json', dict(runs=rows, pairs=pairs))
    report = dict(complete=len(rows) == 6 and all(row['complete'] for row in rows),
                  transitions=sum(row['transitions'] for row in rows), runs=rows, pairs=pairs,
                  source_sha256={name: file_sha256(ROOT / name) for name in (
                      'app/diagnose_goal_distribution.py', 'app/mlp_policy.py', 'app/ppo_yumi.py',
                      'app/gpu_goal.py', 'app/recovery_curriculum.py', 'app/value_diagnostics.py')})
    write_json(output / 'summary.json', report)
    return report
