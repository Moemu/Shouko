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
from pathlib import Path

import numpy as np
import torch
import yaml
from torch import nn

from .full_brain import ConnectomePolicy, ROOT
from .gpu_body import GPUHumanoid
from .sim import observation_interface
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


def checkpoint_observation_size(path):
    """Peek the observation size recorded in a checkpoint (47 before 2026-09-20)."""
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    return int(checkpoint.get('observation_size') or 47)


def same_weights(stored, live):
    """Exact state-dict comparison; True when an evaluation is of unchanged weights."""
    if not stored:
        return False
    live_cpu = {key: value.detach().cpu() for key, value in live.items()}
    if set(stored) != set(live_cpu):
        return False
    return all(torch.equal(stored[key], live_cpu[key]) for key in stored)


def transfer_slope(result):
    """Per-command mean speeds and the cmd -> vx slope fitted across tiers.

    Score on this lineage has a +-0.09 evaluation spread; the tier means and
    their slope are the primary monitoring quantities for the 50-dim
    experiment (observation closure should first move the dead 0.35 tier).
    """
    per_cmd = {}
    for row in result.get('tests', []):
        per_cmd.setdefault(round(row['target_speed'], 2), []).append(row['mean_speed'])
    tiers = {cmd: round(sum(v) / len(v), 3) for cmd, v in sorted(per_cmd.items())}
    if len(tiers) < 2:
        return None, tiers
    xs = list(tiers)
    ys = [tiers[x] for x in xs]
    mx, my = sum(xs) / len(xs), sum(ys) / len(ys)
    slope = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sum((x - mx) ** 2 for x in xs)
    return round(slope, 3), tiers


def train(args):
    global RUNS
    if args.runs_dir:
        # Experiment isolation: a 50-dim checkpoint must never land on
        # runs/yumi/best.pt, where the 47-dim native acceptance path would break.
        RUNS = ROOT / args.runs_dir
        train_full.RUNS = RUNS
    RUNS.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    if args.resume is None:
        args.resume = str(RUNS / 'best.pt')
    control = train_full.TrainingControl(args.max_seconds)
    history = []
    status = dict(phase='initializing', robot='yumi',
                  method='full-connectome PPO fine-tune on the Yumi body (no teacher)',
                  elapsed=0, history=history, seed=args.seed, pid=os.getpid())

    active_phase = 'initializing'
    def write_status(phase=None, **extra):
        nonlocal active_phase
        if phase is not None and phase not in ('paused', 'stopping'):
            active_phase = phase
        fields = control.fields()
        status.update(phase=phase or active_phase, elapsed=fields['wall_elapsed'], **fields, **extra)
        atomic_json(RUNS/'training.json', status)

    write_status('initializing')
    device = 'cuda'
    interface = None
    if args.interface_yaml:
        # Pair the body config with the checkpoint explicitly (lesson of
        # HOME_REFERENCE_MISMATCH_20260916): the upright lineage trains against
        # runs/local/yumi_tall.yaml, not whatever yumi.yaml currently holds.
        interface = observation_interface(
            yaml.safe_load((ROOT / args.interface_yaml).read_text(encoding='utf-8')))
    observation_size = checkpoint_observation_size(args.resume)
    env = GPUHumanoid(args.worlds, robot='yumi', interface=interface,
                      observation_size=observation_size)
    policy = ConnectomePolicy(neural_steps=args.neural_steps, observation_size=observation_size)
    resumed = policy.load(args.resume, interface=env.interface)
    value = Value(policy.encoder.in_features).to(device)
    log_std = nn.Parameter(torch.full((12,), np.log(args.std0), device=device))

    if args.freeze_brain:
        # Freeze ONLY the 25.6M synaptic edges (the topology stabilizer).
        # Train neuron_bias (166k operating-point thresholds) + readout +
        # encoder (~210k params total). Readout-only (40k) was under-capacity
        # for the new straight-leg gait (ppo12 plateaued at score 0.2-0.3);
        # full-network updates destroy the recurrent attractor within 10-15
        # iterations (ppo9/ppo10/ppo13). This is the stable middle ground.
        policy.edge_delta.requires_grad = False
        optimizer = torch.optim.Adam([
            # neuron_bias sits inside a 4-step recurrent tanh; large updates
            # saturate the activations and shatter the limit cycle (~iter 30
            # in ppo14). Keep its lr an order of magnitude below readout.
            dict(params=[policy.neuron_bias], lr=args.lr*0.1),
            dict(params=[p for n, p in policy.named_parameters() if n.startswith('readout')], lr=args.lr),
            dict(params=[p for n, p in policy.named_parameters() if n.startswith('encoder')], lr=args.lr*0.5),
            dict(params=[log_std], lr=args.lr*0.1),
        ], foreach=False)
    else:
        optimizer = torch.optim.Adam([
            dict(params=[policy.edge_delta], lr=args.lr*2),
            dict(params=[p for n, p in policy.named_parameters() if n != 'edge_delta'], lr=args.lr),
            dict(params=[log_std], lr=args.lr),
        ], foreach=False)
    value_opt = torch.optim.Adam(value.parameters(), lr=1e-3)
    base_lrs = [g['lr'] for g in optimizer.param_groups]
    parameter_names = {id(p): n for n, p in policy.named_parameters()}
    parameter_names[id(log_std)] = 'log_std'
    optimizer_names = [[parameter_names[id(p)] for p in group['params']] for group in optimizer.param_groups]
    if args.resume_state:
        # Cross-launch optimizer warm start, not live environment/RNG continuation.
        checkpoint = Path(args.resume)
        state_path = checkpoint.with_name('ppo_state_best.pt' if checkpoint.stem == 'best' else
                                          'ppo_state.pt' if checkpoint.stem == 'last' else
                                          checkpoint.stem + '_state.pt')
        state = train_full.validated_checkpoint_state(checkpoint, state_path, device)
        value.load_state_dict(state['value'])
        log_std.data.copy_(state['log_std'])
        opt_loaded = train_full.restore_optimizer(optimizer, state.get('optimizer'),
                                                 state.get('optimizer_names'), optimizer_names)
        if not train_full.restore_optimizer(value_opt, state.get('value_optimizer')):
            raise ValueError('Incompatible PPO value optimizer state')
        print(json.dumps(dict(resumed_ppo_state=str(state_path), policy_optimizer=opt_loaded,
                              optimizer_state_skipped=None if opt_loaded else 'incompatible parameter groups')), flush=True)
    if args.std_override > 0:
        # Fine-tuning a converged gait: the saved exploration noise (std 0.4)
        # is far too large — every iteration churns the policy (clipfrac ~0.3)
        # and the gait collapses within ~15 iterations (ppo9 peaked at iter 10
        # and never recovered; ppo10 runs 1-3 reproduced the collapse).
        log_std.data.fill_(np.log(args.std_override))
        print(json.dumps(dict(std_override=args.std_override)), flush=True)

    def combined(ev):
        # Selection metric: tracking score plus a posture bonus, so short-burst
        # restarts ratchet toward taller walking, not just equal tracking.
        # Plain score-only selection drifts: PPO fine-tuning from the optimum
        # degrades monotonically after ~10 iterations (ppo9 peak at iter 10;
        # ppo10 control with zero posture reward reproduced the collapse).
        if not ev:
            return float('-inf')
        h = ev.get('mean_height', 0.826)
        return ev['score'] + args.posture_select_w * min(max((h - 0.80) / 0.13, 0), 1)

    def save_checkpoint(name, evaluation=None, updates=0, final=False, combined_bar=None):
        state_path = RUNS/('ppo_state_best.pt' if name == 'best' else 'ppo_state.pt')
        extra = dict(evaluation=evaluation, updates=updates, ppo=True, final=final)
        if combined_bar is not None:
            # Honest selection bar: the mean the candidate was accepted under,
            # so a later launch recomputes the same threshold instead of a
            # lucky draw's combined value.
            extra['combined_bar'] = combined_bar
        train_full.atomic_model_save(policy, RUNS/f'{name}.pt', extra,
                                    dict(value=value.state_dict(), log_std=log_std.detach().cpu(),
                                         optimizer=optimizer.state_dict(), optimizer_names=optimizer_names,
                                         value_optimizer=value_opt.state_dict()), state_path,
                                    physics=env.interface)

    if not control.safe_point(write_status):
        save_checkpoint('last', updates=0, final=True)
        write_status('stopped', iteration=0)
        return
    write_status('evaluating', iteration=0)
    base = evaluate(policy, env, seconds=args.eval_seconds)
    base_slope, base_tiers = transfer_slope(base)
    base['transfer_slope'], base['per_cmd_mean_speed'] = base_slope, base_tiers
    print(json.dumps(dict(iteration=-1, successes=base['successes'], score=round(base['score'], 3),
                          mean_height=round(base.get('mean_height', 0), 3),
                          transfer_slope=base_slope, per_cmd=base_tiers)), flush=True)
    best_path = RUNS/'best.pt'
    stored_bar = None
    same_as_best = False
    if best_path.exists():
        try:
            stored = torch.load(best_path, map_location='cpu', weights_only=True)
            stored_extra = stored.get('extra') or {}
            stored_bar = stored_extra.get('combined_bar')
            if stored_bar is None:
                stored_bar = combined(stored_extra.get('evaluation'))
            same_as_best = same_weights(stored.get('state_dict'), policy.state_dict())
        except Exception as e:
            print(json.dumps(dict(best_load_warning=str(e))), flush=True)
    if same_as_best:
        # Re-evaluating unchanged weights must never raise the selection bar
        # (winner's curse, PPO_OBS50_20260920: 0.844 -> 0.869 on identical
        # tensors). Any stored number for these weights is just one draw;
        # re-anchor to the fresh one and persist it as the explicit bar.
        best_combined = combined(base)
        save_checkpoint('best', base, updates=resumed.get('updates', 0), combined_bar=best_combined)
        print(json.dumps(dict(reevaluation_reanchor=round(best_combined, 3),
                              stale_stored_bar=None if stored_bar is None else round(stored_bar, 3))), flush=True)
    elif stored_bar is not None and stored_bar > combined(base):
        best_combined = stored_bar
        print(json.dumps(dict(kept_historical_best=round(stored_bar, 3), base_score=round(base['score'], 3))), flush=True)
    else:
        # Different weights taking over at startup face the same confirmation
        # rule as mid-run candidates.
        confirm = evaluate(policy, env, seconds=args.eval_seconds)
        pair = (combined(base) + combined(confirm)) / 2
        print(json.dumps(dict(ratchet_candidate=round(combined(base), 3),
                              ratchet_confirm=round(combined(confirm), 3),
                              ratchet_mean=round(pair, 3))), flush=True)
        threshold = (stored_bar if stored_bar is not None else float('-inf')) + args.ratchet_margin
        if pair > threshold:
            best_combined = pair
            save_checkpoint('best', base, updates=resumed.get('updates', 0), combined_bar=pair)
        else:
            best_combined = stored_bar if stored_bar is not None else combined(base)
            print(json.dumps(dict(kept_historical_best=round(best_combined, 3))), flush=True)
    status['evaluation'] = base

    mask = torch.ones(env.worlds, dtype=torch.bool, device=device)
    env.reset(mask, randomize=True)
    env.command.zero_()
    env.command[:, 0] = torch.linspace(0.15, 0.75, env.worlds, device=device)
    # Turn command follows the same yaw feedback used in evaluation/acceptance.
    # Gait shaping: track which ankle is lowest; reward alternation.
    prev_low = torch.zeros(env.worlds, dtype=torch.long, device=device)
    knee_idx = torch.tensor([10, 16], device=device)  # qpos[7+3], qpos[7+9]

    def reset_extras(done_mask):
        nonlocal prev_low
        prev_low = torch.where(done_mask, torch.zeros_like(prev_low), prev_low)

    iteration = 0
    env_steps = 0  # This launch's training rollout transitions; excludes evaluations.
    while control.safe_point(write_status) and not control.budget_exceeded():
        write_status('training', iteration=iteration, env_steps=env_steps)
        # ---- collect ----
        obs_buf, act_buf, logp_buf, rew_buf, val_buf, done_buf = [], [], [], [], [], []
        knee_sum = height_sum = 0.0; pose_count = 0
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
                feet_z = env.xpos[:, env.feet, 2]
                low = feet_z.argmin(dim=1)
                switched = (low != prev_low) & (feet_z.max(dim=1).values - feet_z.min(dim=1).values > 0.03)
                prev_low = low
                # Independent terms so drifting sideways or off-heading cannot be
                # subsidized by the packed velocity/survival terms (ppo5/ppo6 drift).
                reward = (1.5 * torch.exp(-4 * (vx - env.command[:, 0]).square())
                          + 0.6 * torch.exp(-6 * vy.square())
                          + 0.8 * torch.exp(-3 * yaw.square())
                          + 0.3 * upright.clamp(0, 1) + 0.1
                          + 0.4 * switched.float()
                          - 0.002 * action.square().mean(dim=1)
                          - 0.01 * (action - prev_action).square().mean(dim=1))
                # Posture shaping (ppo10): the ppo9 gait walks in a deep squat
                # (pelvis ~0.83 m vs 0.96 straight, knees 0.7-1.3 rad) because
                # nothing in the reward preferred extended legs. Add a pelvis
                # height Gaussian and a soft knee-flexion penalty. Both are
                # gated on actually walking at the commanded speed: ungated,
                # PPO collapsed to standing still and farming the posture
                # bonus (observed: score 0.87 -> 0.33 with zero falls).
                knee = env.qpos[:, knee_idx]
                height = env.qpos[:, 2]
                walk_gate = (vx / env.command[:, 0].clamp_min(0.05)).clamp(0, 1)
                # Linear height ramp, not a Gaussian: at the current squat
                # (0.84 m) a sigma=0.05 Gaussian around 0.93 is ~exp(-3.2) with
                # ~zero gradient, so it silently taxed walking without pulling
                # the posture up (ppo10 run 4). The ramp gives constant pull.
                height_ramp = ((height - 0.80) / (args.height_target - 0.80)).clamp(0, 1)
                if args.knee_gate == 'stance':
                    # GAIT_MEASUREMENT_20260918: swing-leg knee flexion is the
                    # only foot-clearance mechanism this body has (per-bout lift
                    # 1-2 cm); taxing it subsidizes the shuffle. Tax the loaded
                    # (lowest) leg's knee only.
                    knee_tax = torch.relu(knee.gather(1, low[:, None]).squeeze(1) - 0.5)
                else:
                    knee_tax = torch.relu(knee - 0.5).mean(dim=1)
                reward = (reward
                          + walk_gate * (args.posture_w * height_ramp
                                         - args.knee_w * knee_tax))
                reward = torch.where(fallen, reward - 1.0, reward)
                knee_sum += float(knee.mean()); height_sum += float(height.mean()); pose_count += 1
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
        # Value warmup: with a fresh value network the first advantages are
        # noise; updating the policy on them destroys a good baseline within a
        # few iterations (observed in ppo7: 12/32 -> 0/32 in 5 iterations).
        policy_frozen = iteration < args.value_warmup
        pi_loss_total = v_loss_total = kl_total = clipfrac_total = 0.0
        count = 0
        stop_early = False
        for epoch in range(args.epochs * (2 if policy_frozen else 1)):
            if stop_early:
                break
            perm = torch.randperm(len(obs_b), device=device)
            for start in range(0, len(obs_b), args.minibatch):
                mb = perm[start:start+args.minibatch]
                if not policy_frozen:
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
                    torch.nn.utils.clip_grad_norm_([p for p in policy.parameters() if p.requires_grad] + [log_std], 5.0, foreach=False)
                    optimizer.step()
                    pi_loss_total += float(pi_loss)
                    kl_total += float(approx_kl); clipfrac_total += float(clipfrac)
                v = value((obs_b[mb] - policy.obs_mean) / policy.obs_std)
                v_loss = (v - ret_b[mb]).square().mean()
                value_opt.zero_grad(set_to_none=True)
                v_loss.backward()
                value_opt.step()
                v_loss_total += float(v_loss); count += 1
                if not policy_frozen and float(approx_kl) > 1.5 * args.target_kl:
                    stop_early = True
                    break
        iteration += 1
        env_steps += args.steps * env.worlds
        frac = min(control.active_elapsed() / args.max_seconds, 1.0)
        for group, base in zip(optimizer.param_groups, base_lrs):
            group['lr'] = base * (1 - (1 - args.lr_final_frac) * frac)
        row = dict(iteration=iteration, reward=float(rewards.mean()), pi_loss=pi_loss_total/count,
                   v_loss=v_loss_total/count, std=float(log_std.exp().mean()),
                   approx_kl=kl_total/max(count, 1), clipfrac=clipfrac_total/max(count, 1),
                   lr=optimizer.param_groups[0]['lr'], kl_stop=stop_early, value_warmup=policy_frozen,
                   mean_knee=round(knee_sum / max(pose_count, 1), 3),
                   mean_height=round(height_sum / max(pose_count, 1), 3),
                   elapsed=control.wall_elapsed(), active_elapsed=control.active_elapsed(),
                   env_steps=env_steps)
        history.append(row)
        print(json.dumps(row), flush=True)
        write_status('training', iteration=iteration, peak_vram_gib=torch.cuda.max_memory_allocated()/2**30)

        if (args.value_abort_vloss > 0 and iteration == args.value_warmup
                and len(history) >= 2):
            # Pre-registered falsification point: if the value net cannot get
            # its loss under the threshold even with exclusive training, the
            # advantage signal stays noise and policy updates are skipped.
            recent = [abs(h['v_loss']) for h in history[-min(5, len(history)):]]
            v_ma = sum(recent) / len(recent)
            print(json.dumps(dict(value_warmup_end=True, v_loss_ma=round(v_ma, 2),
                                  abort_threshold=args.value_abort_vloss)), flush=True)
            if v_ma > args.value_abort_vloss:
                save_checkpoint('last', status.get('evaluation'), updates=iteration, final=True)
                write_status('value_not_converged', iteration=iteration, v_loss_ma=round(v_ma, 2),
                             note='Pre-registered abort: value loss above threshold after warmup; policy updates skipped.')
                return

        if iteration % args.eval_every == 0:
            write_status('evaluating', iteration=iteration, env_steps=env_steps)
            result = evaluate(policy, env, seconds=args.eval_seconds, record=True)
            result['iteration'] = iteration
            slope, tiers = transfer_slope(result)
            result['transfer_slope'], result['per_cmd_mean_speed'] = slope, tiers
            atomic_json(RUNS/'evaluation.json', result)
            print(json.dumps(dict(iteration=iteration, successes=result['successes'],
                                  score=round(result['score'], 3),
                                  mean_height=round(result.get('mean_height', 0), 3),
                                  mean_duration=round(result['mean_duration'], 1),
                                  transfer_slope=slope, per_cmd=tiers)), flush=True)
            save_checkpoint('last', result, updates=iteration)
            if combined(result) > best_combined:
                # Confirmation draw: accept on the mean only, and store the mean
                # as the bar (single-eval max selection was the winner's curse).
                confirm = evaluate(policy, env, seconds=args.eval_seconds)
                pair = (combined(result) + combined(confirm)) / 2
                print(json.dumps(dict(ratchet_candidate=round(combined(result), 3),
                                      ratchet_confirm=round(combined(confirm), 3),
                                      ratchet_mean=round(pair, 3), bar=round(best_combined, 3))), flush=True)
                if pair > best_combined + args.ratchet_margin:
                    best_combined = pair
                    result['combined_bar'] = pair
                    save_checkpoint('best', result, updates=iteration, combined_bar=pair)
            status['evaluation'] = result
            # Return to collection distribution after evaluation resets commands.
            env.reset(torch.ones(env.worlds, dtype=torch.bool, device=device), randomize=True)
            env.command.zero_()
            env.command[:, 0] = torch.linspace(0.15, 0.75, env.worlds, device=device)
            reset_extras(torch.ones(env.worlds, dtype=torch.bool, device=device))
    save_checkpoint('last', status.get('evaluation'), updates=iteration, final=True)
    write_status('stopped' if control.finish_requested else 'budget_finished', iteration=iteration, env_steps=env_steps,
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
    parser.add_argument('--value-warmup', type=int, default=8,
                        help='iterations that train only the value network before touching the policy')
    parser.add_argument('--std0', type=float, default=0.4)
    parser.add_argument('--std-override', type=float, default=-1,
                        help='if > 0, reset exploration noise to this std after resuming')
    parser.add_argument('--freeze-brain', action='store_true',
                        help='freeze connectome edges/biases; fine-tune readout+encoder only')
    parser.add_argument('--posture-w', type=float, default=0.4,
                        help='weight of the pelvis-height Gaussian posture term')
    parser.add_argument('--knee-w', type=float, default=0.2,
                        help='weight of the soft knee-flexion penalty above 0.5 rad')
    parser.add_argument('--knee-gate', choices=['stance', 'both'], default='both',
                        help="tax only the stance (lowest) leg's knee ('both' = legacy two-leg penalty)")
    parser.add_argument('--height-target', type=float, default=0.93,
                        help='target pelvis height (m); straight-leg is ~0.96')
    parser.add_argument('--posture-select-w', type=float, default=0.5,
                        help='weight of mean pelvis height in the best-checkpoint selection metric')
    parser.add_argument('--ratchet-margin', type=float, default=0.03,
                        help='mean of two evaluations must beat the bar by this much to take over best')
    parser.add_argument('--value-abort-vloss', type=float, default=-1,
                        help='abort before policy updates if mean v_loss over the last warmup iterations exceeds this')
    parser.add_argument('--eval-seconds', type=float, default=30)
    parser.add_argument('--eval-every', type=int, default=5)
    parser.add_argument('--max-seconds', type=float, default=270)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--resume', default=None,
                        help='checkpoint to warm start from (default <runs-dir>/best.pt)')
    parser.add_argument('--runs-dir', default=None,
                        help='experiment directory for checkpoints/status (default runs/yumi)')
    parser.add_argument('--interface-yaml', default=None,
                        help='body yaml to pair with the checkpoint (e.g. runs/local/yumi_tall.yaml); '
                             'default is the robot spec yaml')
    parser.add_argument('--resume-state', action='store_true',
                        help='Restore verified PPO state beside --resume; not a live-process resume')
    train(parser.parse_args())
