"""Measure collection and frozen-edge backward costs without saving model updates."""
import argparse
import gc
import json
import time
from pathlib import Path
from unittest.mock import patch

import torch

from .full_brain import ConnectomePolicy, SparseMessage, checkpoint_configuration
from .gpu_body import GPUHumanoid
from .gpu_benchmark import sparse_check


def legacy_backward(ctx, gradient):
    """Reference implementation from before frozen-edge gradient skipping."""
    values, activity, ptr, pre = ctx.saved_tensors
    matrix = torch.sparse_csr_tensor(ptr, pre, values, size=(activity.shape[0], activity.shape[0]),
                                     check_invariants=False)
    gradient = gradient.contiguous()
    edge = torch.sparse.sampled_addmm(matrix, gradient, activity.T, beta=0).values()
    inputs = torch.sparse.mm(matrix.transpose(0, 1), gradient)
    return edge, inputs, None, None


def backward_trial(policy, observation, repeats, backward):
    elapsed = []
    torch.cuda.reset_peak_memory_stats()
    with patch.object(SparseMessage, 'backward', staticmethod(backward)):
        for index in range(repeats + 1):
            policy.zero_grad(set_to_none=True)
            torch.cuda.synchronize()
            started = time.perf_counter()
            action, _ = policy(observation)
            loss = action.square().mean()
            loss.backward()
            torch.cuda.synchronize()
            if index:
                elapsed.append(time.perf_counter() - started)
    gradients = {name: p.grad.detach().cpu().clone() for name, p in policy.named_parameters()
                 if p.requires_grad and p.grad is not None}
    return dict(seconds=sum(elapsed) / len(elapsed),
                samples_per_second=len(observation) / (sum(elapsed) / len(elapsed)),
                peak_tensor_gib=torch.cuda.max_memory_allocated() / 2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30), gradients


def main(args):
    torch.set_num_threads(8)
    torch.manual_seed(2026)
    torch.cuda.set_per_process_memory_fraction(0.8)
    configuration = checkpoint_configuration(args.checkpoint)
    policy = ConnectomePolicy(**{k: configuration[k] for k in
                               ('observation_size', 'action_size', 'neural_steps')}).eval()
    policy.load(args.checkpoint)
    policy.edge_delta.requires_grad_(False)
    report = dict(kind='collection_and_backward_microbenchmark', gpu=torch.cuda.get_device_name(),
                  torch=str(torch.__version__), checks=sparse_check('cuda'),
                  scope='Full graph plus physics collection; isolated frozen-edge backward. No optimizer steps, reward or value fitting.',
                  checkpoint=str(args.checkpoint), collection=[], backward=[])

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

    sample = None
    for worlds in args.worlds:
        env = GPUHumanoid(worlds, robot='yumi', interface=configuration['physics_interface'],
                          observation_size=configuration['observation_size'])
        env.command[:, 0] = torch.linspace(0.15, 0.75, worlds, device='cuda')
        fall_count = 0
        with torch.no_grad():
            for _ in range(10):
                action, _ = policy(env.observation())
                env.step(action)
            env.reset(torch.ones(worlds, dtype=torch.bool, device='cuda'), randomize=True)
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            started = time.perf_counter()
            for _ in range(args.steps):
                observation = env.observation()
                action, _ = policy(observation)
                fallen = env.step(action)
                if bool(fallen.any()):
                    fall_count += int(fallen.sum())
                    env.reset(fallen, randomize=True)
            torch.cuda.synchronize()
            seconds = time.perf_counter() - started
            env.check_physics()
            sample = observation[:1].detach().clone()
        row = dict(worlds=worlds, steps=args.steps, seconds=seconds,
                   samples_per_second=worlds * args.steps / seconds, reset_count=fall_count,
                   peak_tensor_gib=torch.cuda.max_memory_allocated() / 2**30,
                   device_used_gib=(torch.cuda.mem_get_info()[1] - torch.cuda.mem_get_info()[0]) / 2**30)
        report['collection'].append(row)
        print(json.dumps(dict(collection=row)), flush=True)
        save()
        del env
        gc.collect()
        torch.cuda.empty_cache()

    optimized_backward = SparseMessage.backward
    for batch in args.minibatches:
        observation = sample.expand(batch, -1).contiguous()
        reference, old_gradients = backward_trial(policy, observation, args.repeats, legacy_backward)
        optimized, gradients = backward_trial(policy, observation, args.repeats, optimized_backward)
        errors = {name: float((gradients[name] - value).abs().max())
                  for name, value in old_gradients.items()}
        for name, value in old_gradients.items():
            torch.testing.assert_close(gradients[name], value, rtol=1e-5, atol=1e-6)
        row = dict(minibatch=batch, legacy=reference, optimized=optimized,
                   speedup=reference['seconds'] / optimized['seconds'], gradient_max_abs_error=errors)
        report['backward'].append(row)
        print(json.dumps(dict(backward=row)), flush=True)
        save()
        policy.zero_grad(set_to_none=True)
        gc.collect()
        torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--worlds', type=int, nargs='+', default=[32, 64, 128])
    parser.add_argument('--minibatches', type=int, nargs='+', default=[128, 256, 512])
    parser.add_argument('--steps', type=int, default=128)
    parser.add_argument('--repeats', type=int, default=3)
    main(parser.parse_args())
