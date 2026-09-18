"""Fine-tune the full-connectome policy on the Yumi body via DAgger.

Reuses train_full's collect/evaluate with a Yumi GPUHumanoid. The G1 teacher
still supplies action labels; the student starts from the G1-trained checkpoint
(which walks ~1.6 s on Yumi before falling), so collection begins with a
teacher/student mix (beta=0.5) instead of pure teacher rollout.
"""
import argparse
import json
import os
import time

import torch

from .full_brain import ConnectomePolicy, ROOT
from .gpu_body import GPUHumanoid
from .teacher import load_teacher
from . import train_full

RUNS = ROOT / 'runs/yumi'
train_full.RUNS = RUNS  # evaluate() writes replay/status under this directory
collect, evaluate, atomic_json = train_full.collect, train_full.evaluate, train_full.atomic_json


def train(args):
    RUNS.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(8)
    torch.manual_seed(args.seed)
    started = time.perf_counter()
    history = []
    status = dict(phase='initializing', robot='yumi',
                  method='full-connectome DAgger fine-tune on the Yumi body',
                  teacher_role='G1 teacher supplies action labels only; evaluation is teacher-free.',
                  elapsed=0, history=history, seed=args.seed, pid=os.getpid())

    def write_status(phase, **extra):
        status.update(phase=phase, elapsed=time.perf_counter()-started, **extra)
        atomic_json(RUNS/'training.json', status)

    write_status('initializing')
    env = GPUHumanoid(args.worlds, robot='yumi')
    model = ConnectomePolicy(neural_steps=args.neural_steps)
    teacher = load_teacher(args.worlds)
    resumed = model.load(args.resume)
    write_status('collecting', round=0)
    dataset_path = RUNS/'demonstrations.pt'
    if dataset_path.exists():
        dataset = torch.load(dataset_path, weights_only=True, map_location='cpu')
        observations, targets = dataset['observations'], dataset['targets']
    else:
        observations, targets = collect(env, teacher, model, args.collect_steps, beta=args.beta0)
        torch.save(dict(observations=observations, targets=targets,
                        teacher='Unitree G1 pretrained policy, training data only'), dataset_path)
    model.set_observation_stats(observations.cuda())
    # Restore the checkpoint's observation normalizer: re-fitting it to the
    # initial mixed-rollout distribution would shift every input the brain sees.
    if args.resume:
        model.load(args.resume)
    optimizer = torch.optim.Adam([
        dict(params=[model.edge_delta], lr=args.lr*2),
        dict(params=[p for n, p in model.named_parameters() if n != 'edge_delta'], lr=args.lr),
    ], foreach=False)
    best_score = evaluate(model, env, seconds=args.eval_seconds)['score']
    training_steps = resumed.get('updates', 0)
    initial_steps = training_steps
    for round_index in range(args.rounds):
        write_status('collecting', round=round_index)
        beta = max(0.0, args.beta0 - round_index/4)
        new_observations, new_targets = collect(env, teacher, model, args.collect_steps//2, beta)
        observations = torch.cat([observations, new_observations])[-200000:]
        targets = torch.cat([targets, new_targets])[-200000:]
        train_x, train_y = observations.cuda(), targets.cuda()
        write_status('training', round=round_index, examples=len(observations))
        for update in range(args.updates):
            if time.perf_counter()-started >= args.max_seconds:
                break
            indices = torch.randint(len(train_x), (args.batch,), device='cuda')
            optimizer.zero_grad(set_to_none=True)
            predicted, _ = model(train_x[indices])
            loss = (predicted-train_y[indices]).square().mean()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError('Non-finite full-connectome loss')
            loss.backward()
            if training_steps == initial_steps:
                status['edge_gradient_max'] = float(model.edge_delta.grad.abs().max())
                assert status['edge_gradient_max'] > 0
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0, foreach=False)
            optimizer.step()
            training_steps += 1
            if update % 20 == 0:
                row = dict(samples=training_steps*args.batch, action_mse=float(loss),
                           loss_kind='training joint-action MSE', round=round_index,
                           elapsed=time.perf_counter()-started)
                history.append(row)
                write_status('training', update=training_steps,
                             edge_delta_rms=float(model.edge_delta.square().mean().sqrt()),
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
    write_status('budget_finished', update=training_steps,
                 note='Fine-tune ended. Passing final walking acceptance has not been inferred from training loss.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--worlds', type=int, default=16)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--neural-steps', type=int, default=4)
    parser.add_argument('--collect-steps', type=int, default=600)
    parser.add_argument('--updates', type=int, default=300)
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--eval-seconds', type=float, default=15)
    parser.add_argument('--max-seconds', type=float, default=1500)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--beta0', type=float, default=0.5)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--resume', default=str(ROOT/'runs/cloud/best.pt'))
    train(parser.parse_args())
