"""PPO fine-tune of the full-connectome policy on the Yumi body.

The existing G1 teacher cannot stabilize the Yumi body (falls in ~1.7 s).
Here we fine-tune the G1-trained checkpoint
with PPO on a locomotion reward: velocity tracking, upright, heading, alive
bonus, small action penalty, terminal fall penalty.

The policy is memoryless per control step (observation -> 12 joint actions),
so PPO treats each decision step independently; gradients flow through the
measured connectome edges exactly as in DAgger training.
"""
import argparse
import json
import os
import time

import numpy as np
import torch
from torch import nn

from .full_brain import ConnectomePolicy, ROOT
from .gpu_body import GPUHumanoid
from . import train_full

RUNS = ROOT / 'runs/yumi'
train_full.RUNS = RUNS
evaluate, atomic_json = train_full.evaluate, train_full.atomic_json


class Value(nn.Module):
    def __init__(self, obs_size):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(obs_size, 256), nn.ELU(),
                                 nn.Linear(256, 256), nn.ELU(), nn.Linear(256, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


def quad_yaw(qpos):
    w, x, y, z = qpos[:, 3:7].unbind(1)
    return torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def train(args):
    RUNS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    history = []
    status = dict(phase='initializing', robot='yumi',
                  method='full-connectome PPO fine-tune on the Yumi body (no teacher)',
                  elapsed=0, history=history, seed=args.seed, pid=os.getpid())

    def write_status(phase, **extra):
        status.update(phase=phase, elapsed=time.perf_counter()-started, **extra)
        atomic_json(RUNS/'training.json', status)

    write_status('initializing')
    device = 'cuda'
    env = GPUHumanoid(args.worlds, robot='yumi')
    policy = ConnectomePolicy(neural_steps=args.neural_steps)
    resumed = policy.load(args.resume)
    value = Value(policy.encoder.in_features).to(device)
    log_std = nn.Parameter(torch.full((12,), np.log(args.std0), device=device))

    optimizer = torch.optim.Adam([
        dict(params=[policy.edge_delta], lr=args.lr*2),
        dict(params=[p for n, p in policy.named_parameters() if n != 'edge_delta'], lr=args.lr),
        dict(params=[log_std], lr=args.lr),
    ], foreach=False)
    value_opt = torch.optim.Adam(value.parameters(), lr=1e-3)
    base_lrs = [g['lr'] for g in optimizer.param_groups]
    ppo_state_path = RUNS/'ppo_state.pt'
    if args.resume_state:
        # A best checkpoint pairs with the PPO state saved when it was selected.
        state_path = ppo_state_path
        if os.path.basename(args.resume).startswith('best') and (RUNS/'ppo_state_best.pt').exists():
            state_path = RUNS/'ppo_state_best.pt'
        if state_path.exists():
            state = torch.load(state_path, weights_only=True, map_location=device)
            value.load_state_dict(state['value'])
            log_std.data.copy_(state['log_std'])
            print(json.dumps(dict(resumed_ppo_state=str(state_path))), flush=True)

    base = evaluate(policy, env, seconds=args.eval_seconds)
    print(json.dumps(dict(iteration=-1, successes=base['successes'], score=round(base['score'], 3))), flush=True)
    best_score = base['score']
    best_path = RUNS/'best.pt'
    old_score = float('-inf')
    if best_path.exists():
        try:
            old_score = torch.load(best_path, map_location='cpu', weights_only=True).get('extra', {}).get('evaluation', {}).get('score', float('-inf'))
        except Exception as e:
            print(json.dumps(dict(best_load_warning=str(e))), flush=True)
    if old_score > best_score:
        best_score = old_score
        print(json.dumps(dict(kept_historical_best=round(old_score, 3), base_score=round(base['score'], 3))), flush=True)
    else:
        policy.save(best_path, dict(evaluation=base, updates=resumed.get('updates', 0), ppo=True))

    mask = torch.ones(env.worlds, dtype=torch.bool, device=device)
    env.reset(mask, randomize=True)
    env.command.zero_()
    env.command[:, 0] = torch.linspace(0.15, 0.75, env.worlds, device=device)
    # Turn command follows the same yaw feedback used in evaluation/acceptance.
    # Gait shaping: track which ankle is lowest; reward alternation.
    prev_low = torch.zeros(env.worlds, dtype=torch.long, device=device)

    def reset_extras(done_mask):
        nonlocal prev_low
        prev_low = torch.where(done_mask, torch.zeros_like(prev_low), prev_low)

    iteration = 0
    while time.perf_counter()-started < args.max_seconds:
        # ---- collect ----
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        policy.eval()
        with torch.no_grad():
            for t in range(args.steps):
                env.command[:, 2] = (-quad_yaw(env.qpos) * 1.4).clamp(-0.2, 0.2)
                obs = env.observation()
                mean, _ = policy(obs)
                std = log_std.exp().clamp(0.05, 1.5)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                val = value((obs - policy.obs_mean) / policy.obs_std)
                prev_action = env.actions.clone()
                fallen = env.step(action)
                vx, vy = env.qvel[:, 0], env.qvel[:, 1]
                yaw = quad_yaw(env.qpos)
                upright = -env.gravity()[:, 2]
                vel_err = (vx - env.command[:, 0]).square() + vy.square()
                feet_z = env.xpos[:, env.feet, 2]
                low = feet_z.argmin(dim=1)
                switched = (low != prev_low) & (feet_z.max(dim=1).values - feet_z.min(dim=1).values > 0.03)
                prev_low = low
                reward = (1.5 * torch.exp(-4 * vel_err) + 0.5 * upright.clamp(0, 1)
                          + 0.3 * torch.exp(-3 * yaw.square()) + 0.2
                          + 0.4 * switched.float()
                          - 0.002 * action.square().mean(dim=1)
                          - 0.01 * (action - prev_action).square().mean(dim=1))
                reward = torch.where(fallen, reward - 1.0, reward)
                obs_buf.append(obs); act_buf.append(action); logp_buf.append(dist.log_prob(action).sum(-1))
                rew_buf.append(reward); val_buf.append(val); done_buf.append(fallen.float())
                if bool(fallen.any()):
                    env.reset(fallen, randomize=True)
                    reset_extras(fallen)
        with torch.no_grad():
            next_val = value((env.observation() - policy.obs_mean) / policy.obs_std)
        rewards = torch.stack(rew_buf)          # T x W
        values = torch.stack(val_buf + [next_val])
        dones = torch.stack(done_buf)
        advantages = torch.zeros_like(rewards)
        gae = torch.zeros(env.worlds, device=device)
        for t in reversed(range(args.steps)):
            delta = rewards[t] + args.gamma * values[t+1] * (1-dones[t]) - values[t]
            gae = delta + args.gamma * args.lam * (1-dones[t]) * gae
            advantages[t] = gae
        returns = advantages + values[:-1]
        obs_b = torch.cat(obs_buf); act_b = torch.cat(act_buf); logp_b = torch.cat(logp_buf)
        adv_b = advantages.reshape(-1); ret_b = returns.reshape(-1)
        adv_b = (adv_b - adv_b.mean()) / (adv_b.std() + 1e-8)

        # ---- update ----
        policy.train()
        pi_loss_total = v_loss_total = kl_total = clipfrac_total = 0.0
        count = 0
        stop_early = False
        for epoch in range(args.epochs):
            if stop_early:
                break
            perm = torch.randperm(len(obs_b), device=device)
            for start in range(0, len(obs_b), args.minibatch):
                mb = perm[start:start+args.minibatch]
                mean, _ = policy(obs_b[mb])
                std = log_std.exp().clamp(0.05, 1.5)
                dist = torch.distributions.Normal(mean, std)
                logp = dist.log_prob(act_b[mb]).sum(-1)
                logratio = logp - logp_b[mb]
                ratio = logratio.exp()
                approx_kl = ((ratio - 1) - logratio).mean()
                clipfrac = ((ratio - 1).abs() > args.clip).float().mean()
                s1 = ratio * adv_b[mb]
                s2 = ratio.clamp(1-args.clip, 1+args.clip) * adv_b[mb]
                pi_loss = -torch.min(s1, s2).mean() - args.entropy * dist.entropy().sum(-1).mean()
                optimizer.zero_grad(set_to_none=True)
                pi_loss.backward()
                torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + [log_std], 5.0, foreach=False)
                optimizer.step()
                v = value((obs_b[mb] - policy.obs_mean) / policy.obs_std)
                v_loss = (v - ret_b[mb]).square().mean()
                value_opt.zero_grad(set_to_none=True)
                v_loss.backward()
                value_opt.step()
                pi_loss_total += float(pi_loss); v_loss_total += float(v_loss)
                kl_total += float(approx_kl); clipfrac_total += float(clipfrac); count += 1
                if float(approx_kl) > 1.5 * args.target_kl:
                    stop_early = True
                    break
        iteration += 1
        frac = min((time.perf_counter()-started) / args.max_seconds, 1.0)
        for group, base in zip(optimizer.param_groups, base_lrs):
            group['lr'] = base * (1 - (1 - args.lr_final_frac) * frac)
        row = dict(iteration=iteration, reward=float(rewards.mean()), pi_loss=pi_loss_total/count,
                   v_loss=v_loss_total/count, std=float(log_std.exp().mean()),
                   approx_kl=kl_total/count, clipfrac=clipfrac_total/count,
                   lr=optimizer.param_groups[0]['lr'], kl_stop=stop_early,
                   elapsed=time.perf_counter()-started)
        history.append(row)
        print(json.dumps(row), flush=True)
        write_status('training', iteration=iteration, peak_vram_gib=torch.cuda.max_memory_allocated()/2**30)

        if iteration % args.eval_every == 0:
            result = evaluate(policy, env, seconds=args.eval_seconds, record=True)
            result['iteration'] = iteration
            atomic_json(RUNS/'evaluation.json', result)
            print(json.dumps(dict(iteration=iteration, successes=result['successes'],
                                  score=round(result['score'], 3),
                                  mean_duration=round(result['mean_duration'], 1))), flush=True)
            policy.save(RUNS/'last.pt', dict(evaluation=result, updates=iteration, ppo=True))
            torch.save(dict(value=value.state_dict(), log_std=log_std.detach().cpu()), ppo_state_path)
            if result['score'] > best_score:
                best_score = result['score']
                policy.save(RUNS/'best.pt', dict(evaluation=result, updates=iteration, ppo=True))
                torch.save(dict(value=value.state_dict(), log_std=log_std.detach().cpu()),
                           RUNS/'ppo_state_best.pt')
            status['evaluation'] = result
            # Return to collection distribution after evaluation resets commands.
            env.reset(torch.ones(env.worlds, dtype=torch.bool, device=device), randomize=True)
            env.command.zero_()
            env.command[:, 0] = torch.linspace(0.15, 0.75, env.worlds, device=device)
            reset_extras(torch.ones(env.worlds, dtype=torch.bool, device=device))
    policy.save(RUNS/'last.pt', dict(evaluation=status.get('evaluation'), updates=iteration, ppo=True, final=True))
    torch.save(dict(value=value.state_dict(), log_std=log_std.detach().cpu()), ppo_state_path)
    write_status('budget_finished', iteration=iteration,
                 note='PPO fine-tune ended. Walking acceptance requires the held-out evaluation.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worlds', type=int, default=32)
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--minibatch', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--neural-steps', type=int, default=4)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--lam', type=float, default=0.95)
    parser.add_argument('--clip', type=float, default=0.2)
    parser.add_argument('--entropy', type=float, default=0.001)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--lr-final-frac', type=float, default=0.1)
    parser.add_argument('--target-kl', type=float, default=0.03)
    parser.add_argument('--std0', type=float, default=0.4)
    parser.add_argument('--eval-seconds', type=float, default=30)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--max-seconds', type=float, default=270)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--resume', default=str(ROOT/'runs/cloud/best.pt'))
    parser.add_argument('--resume-state', action='store_true')
    train(parser.parse_args())
