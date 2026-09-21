"""Fit a fixed real rollout on CPU, holding out complete simulation worlds."""
import argparse
import json
import time
from pathlib import Path

import psutil
import torch

from .ppo_yumi import Value
from .train_full import file_sha256, validated_checkpoint_state
from .value_diagnostics import diagnose, regression_metrics


def main(args):
    available = psutil.virtual_memory().available
    if available < 1024**3:
        raise RuntimeError('CPU probe needs at least 1 GiB available system memory')
    torch.set_num_threads(2)
    torch.manual_seed(20260921)
    batch = torch.load(args.rollout, map_location='cpu', weights_only=True)
    diagnose(batch)
    digest = file_sha256(args.checkpoint)
    if batch['checkpoint_sha256'] != digest:
        raise ValueError('Rollout and checkpoint hashes differ')
    state = validated_checkpoint_state(args.checkpoint, args.state, 'cpu')
    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=True, mmap=True)
    mean, scale = (checkpoint['state_dict'][key] for key in ('obs_mean', 'obs_std'))
    observation = (batch['observations'] - mean) / scale
    steps, worlds, size = observation.shape
    if worlds < 2:
        raise ValueError('At least two worlds are needed for a held-out group')
    held_out = torch.arange(worlds) % 5 == 0
    validation = held_out.expand(steps, -1).reshape(-1)
    x = observation.reshape(-1, size)
    y = batch['returns'].reshape(-1)
    indices = torch.where(~validation)[0]
    value = Value(size)
    value.load_state_dict(state['value'])
    optimizer = torch.optim.Adam(value.parameters(), lr=1e-3)
    optimizer.load_state_dict(state['value_optimizer'])
    started = time.perf_counter()
    report = dict(kind='fixed_rollout_cpu_value_fit', checkpoint_sha256=digest,
                  split='world index modulo 5 == 0 held out, all time steps and resets stay in their world',
                  train_worlds=int((~held_out).sum()), held_out_worlds=int(held_out.sum()),
                  threads=2, available_ram_gib=available/2**30, history=[],
                  limitation='One bootstrapped rollout, not independent return accuracy or walking evidence; no model weights saved.')

    def measure(epoch):
        with torch.no_grad():
            prediction = value(x)
        row = dict(epoch=epoch, elapsed=time.perf_counter()-started,
                   train=regression_metrics(prediction[~validation], y[~validation]),
                   held_out=regression_metrics(prediction[validation], y[validation]))
        report['history'].append(row)
        print(json.dumps(row), flush=True)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')

    measure(0)
    for epoch in range(1, args.epochs+1):
        if time.perf_counter()-started >= args.max_seconds:
            break
        for selected in indices[torch.randperm(len(indices))].split(256):
            optimizer.zero_grad(set_to_none=True)
            loss = (value(x[selected]) - y[selected]).square().mean()
            loss.backward()
            optimizer.step()
        measure(epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rollout', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--max-seconds', type=float, default=60)
    main(parser.parse_args())
