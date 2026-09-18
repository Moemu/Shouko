"""End-to-end connectome imitation and DAgger; the teacher is used only for training."""
import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .full_brain import ConnectomePolicy, ROOT
from .gpu_body import GPUHumanoid, UNITREE
from .teacher import load_teacher, reset_teacher

RUNS = ROOT / 'runs/cloud'


def atomic_json(path, value):
    target = Path(path)
    temporary = target.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, allow_nan=False))
    temporary.replace(target)


@torch.no_grad()
def evaluate(model, env, seconds=30, record=False):
    mask = torch.ones(env.worlds, device='cuda', dtype=torch.bool)
    env.reset(mask)
    generator = torch.Generator(device='cuda').manual_seed(1001)
    env.qvel[:, :2] = torch.randn(env.worlds, 2, device='cuda', generator=generator) * 0.015
    env.command.zero_()
    env.command[:, 0] = torch.tensor([0.35, 0.5, 0.65], device='cuda').repeat((env.worlds+2)//3)[:env.worlds]
    alive = mask.clone()
    duration = torch.zeros(env.worlds, device='cuda')
    forward = torch.zeros_like(duration)
    lateral = torch.zeros_like(duration)
    min_height = torch.ones_like(duration) * 10
    frames = []
    started = time.perf_counter()
    model.eval()
    for step in range(round(seconds/env.dt)):
        record_alive = bool(alive[0])
        w, x, y, z = env.qpos[:, 3:7].unbind(1)
        yaw = torch.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))
        env.command[:, 2] = (-yaw * 1.4).clamp(-0.2, 0.2)
        observation = env.observation()
        action, activity = model(observation)
        fallen = env.step(action)
        if step % 50 == 0:
            env.check_physics()
        duration[alive] += env.dt
        forward[alive] = env.qpos[alive, 0]
        lateral[alive] = env.qpos[alive, 1].abs()
        min_height[alive] = torch.minimum(min_height[alive], env.qpos[alive, 2])
        alive &= ~fallen
        if record and record_alive and step % 5 == 0:
            frame = env.snapshot()
            frame['command'] = action[0].cpu().tolist()
            frames.append(frame)
        if not bool(alive.any()):
            break
        env.reset(~alive)
    speed = forward / duration.clamp_min(env.dt)
    target = env.command[:, 0]
    results = [dict(seed=1001, environment=i, target_speed=float(target[i]), seconds=float(duration[i]),
                    mean_speed=float(speed[i]), forward_m=float(forward[i]),
                    minimum_height=float(min_height[i]), fallen=not bool(alive[i]),
                    lateral_m=float(lateral[i]),
                    success=bool(alive[i] and abs(speed[i]-target[i]) < 0.15 and lateral[i] < seconds*target[i]*0.2)) for i in range(env.worlds)]
    summary = dict(tests=results, successes=sum(row['success'] for row in results), attempts=env.worlds,
                   mean_duration=float(duration.mean()), mean_speed=float(speed.mean()),
                   score=float((duration/seconds).mean() - (speed-target).abs().mean() - (lateral/(seconds*target)).mean()),
                   wall_seconds=time.perf_counter()-started,
                   teacher_used=False,
                   criterion='Duration and forward-speed screening; final foot-contact and held-out tests remain required.')
    if record:
        atomic_json(RUNS/'replay.json', dict(meta=summary, frames=frames))
    model.train()
    return summary


@torch.no_grad()
def collect(env, teacher, model, steps, beta):
    observations, targets = [], []
    env.reset(torch.ones(env.worlds, device='cuda', dtype=torch.bool), randomize=True)
    reset_teacher(teacher, torch.ones(env.worlds, device='cuda', dtype=torch.bool))
    env.command.zero_()
    env.command[:, 0] = torch.linspace(0.15, 0.75, env.worlds, device='cuda')
    env.command[:, 2] = torch.linspace(-0.2, 0.2, env.worlds, device='cuda')
    for step in range(steps):
        observation = env.observation()
        target = teacher(observation)
        observations.append(observation.cpu())
        targets.append(target.cpu())
        if beta == 1:
            action = target
        else:
            prediction, _ = model(observation)
            action = beta * target + (1-beta) * prediction
        fallen = env.step(action)
        if step % 50 == 0:
            env.check_physics()
        if step % 80 == 79:
            env.qvel[:, :2] += torch.randn_like(env.qvel[:, :2]) * 0.07
        if bool(fallen.any()):
            env.reset(fallen, randomize=True)
            reset_teacher(teacher, fallen)
    return torch.cat(observations), torch.cat(targets)


def train(args):
    RUNS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    history = []
    status = dict(phase='initializing', method='full-connectome end-to-end imitation and DAgger',
                  teacher_role='Training labels only. Independent evaluation and deployment use direct brain actions.',
                  elapsed=0, history=history, seed=args.seed, hourly_price_yuan=args.hourly_price, pid=os.getpid())
    def write_status(phase, **extra):
        elapsed = time.perf_counter()-started
        status.update(phase=phase, elapsed=elapsed, estimated_compute_yuan=elapsed/3600*args.hourly_price, **extra)
        atomic_json(RUNS/'training.json', status)
    write_status('initializing')
    env = GPUHumanoid(args.worlds)
    model = ConnectomePolicy(neural_steps=args.neural_steps)
    teacher = load_teacher(args.worlds)
    write_status('collecting', round=0)
    dataset_path = RUNS/'demonstrations.pt'
    if dataset_path.exists():
        dataset = torch.load(dataset_path, weights_only=True, map_location='cpu')
        observations, targets = dataset['observations'], dataset['targets']
    else:
        observations, targets = collect(env, teacher, model, args.collect_steps, beta=1)
        torch.save(dict(observations=observations, targets=targets,
                        teacher='Unitree G1 pretrained policy, training data only'), dataset_path)
    model.set_observation_stats(observations.cuda())
    if args.resume:
        resumed = model.load(args.resume)
    else:
        resumed = {}
    optimizer = torch.optim.Adam([
        dict(params=[model.edge_delta], lr=args.lr*2),
        dict(params=[parameter for name, parameter in model.named_parameters() if name != 'edge_delta'], lr=args.lr),
    ], foreach=False)
    best_score = evaluate(model, env, seconds=args.eval_seconds)['score'] if args.resume else -float('inf')
    training_steps = resumed.get('updates', 0)
    initial_steps = training_steps
    for round_index in range(args.rounds):
        if round_index:
            write_status('collecting', round=round_index)
            # Gradually expose the student to the physical consequences of its own errors.
            beta = max(0.0, 1-round_index/4)
            new_observations, new_targets = collect(env, teacher, model, args.collect_steps//2, beta)
            observations = torch.cat([observations, new_observations])[-200000:]
            targets = torch.cat([targets, new_targets])[-200000:]
        train_x, train_y = observations.cuda(), targets.cuda()
        write_status('training', round=round_index, examples=len(observations))
        total_loss = 0
        for update in range(args.updates):
            if time.perf_counter()-started >= args.max_seconds:
                break
            indices = torch.randint(len(train_x), (args.batch,), device='cuda')
            optimizer.zero_grad(set_to_none=True)
            predicted, activity = model(train_x[indices])
            loss = (predicted-train_y[indices]).square().mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Non-finite full-connectome loss')
            loss.backward()
            if training_steps == initial_steps:
                status['edge_gradient_max'] = float(model.edge_delta.grad.abs().max())
                assert status['edge_gradient_max'] > 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, foreach=False)
            optimizer.step()
            total_loss += float(loss)
            training_steps += 1
            if update % 20 == 0:
                row = dict(samples=training_steps*args.batch, action_mse=float(loss),
                           loss_kind='training joint-action MSE', round=round_index,
                           elapsed=time.perf_counter()-started)
                history.append(row)
                write_status('training', update=training_steps, edge_delta_rms=float(model.edge_delta.square().mean().sqrt()),
                             peak_vram_gib=torch.cuda.max_memory_allocated()/2**30)
                print(json.dumps(row), flush=True)
        write_status('evaluating', round=round_index)
        result = evaluate(model, env, seconds=args.eval_seconds, record=True)
        result['round'] = round_index
        result['updates'] = training_steps
        atomic_json(RUNS/'evaluation.json', result)
        print(json.dumps(result), flush=True)
        model.save(RUNS/'last.pt', dict(evaluation=result, updates=training_steps))
        if result['score'] > best_score:
            best_score = result['score']
            model.save(RUNS/'best.pt', dict(evaluation=result, updates=training_steps))
        status['evaluation'] = result
        if time.perf_counter()-started >= args.max_seconds:
            break
    temporary = dataset_path.with_suffix('.tmp')
    torch.save(dict(observations=observations, targets=targets,
                    teacher='Unitree G1 pretrained policy, training data only'), temporary)
    temporary.replace(dataset_path)
    write_status('budget_finished', update=training_steps,
                 note='This run ended. Passing final walking acceptance has not been inferred from training loss.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worlds', type=int, default=16)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--neural-steps', type=int, default=4)
    parser.add_argument('--collect-steps', type=int, default=600)
    parser.add_argument('--updates', type=int, default=300)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--eval-seconds', type=float, default=15)
    parser.add_argument('--max-seconds', type=float, default=1800)
    parser.add_argument('--hourly-price', type=float, default=1.88)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--resume')
    train(parser.parse_args())
