"""Measure full-connectome forward/backward/Adam, never infer speed from GPU specs."""
import argparse
import json
import time

import torch

from .full_brain import ConnectomePolicy, SparseMessage, ROOT


def sparse_check(device):
    ptr = torch.tensor([0, 1, 3], device=device, dtype=torch.int32)
    cols = torch.tensor([1, 0, 1], device=device, dtype=torch.int32)
    values = torch.tensor([0.4, -0.2, 0.3], device=device, requires_grad=True)
    x = torch.tensor([[0.2, -0.3], [0.7, 0.4]], device=device, requires_grad=True)
    dense_values = values.detach().clone().requires_grad_()
    dense_x = x.detach().clone().requires_grad_()
    result = SparseMessage.apply(values, x, ptr, cols)
    reference = torch.stack([torch.stack([dense_values[0] * 0, dense_values[0]]), dense_values[1:]]) @ dense_x
    assert torch.allclose(result, reference, atol=1e-6)
    result.square().sum().backward()
    reference.square().sum().backward()
    assert torch.allclose(values.grad, dense_values.grad, atol=1e-6)
    assert torch.allclose(x.grad, dense_x.grad, atol=1e-6)
    return 'sparse direction, forward and parameter/input gradients match dense reference'


def benchmark(batches, neural_steps, repeats):
    torch.manual_seed(2026)
    torch.set_num_threads(8)
    check = sparse_check('cuda')
    model = ConnectomePolicy(neural_steps=neural_steps)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, foreach=False)
    results = []
    for batch in batches:
        observation = torch.randn(batch, 47, device='cuda')
        target = torch.randn(batch, 12, device='cuda')
        torch.cuda.reset_peak_memory_stats()
        times = []
        try:
            for i in range(repeats + 1):
                torch.cuda.synchronize()
                start = time.perf_counter()
                optimizer.zero_grad(set_to_none=True)
                action, activity = model(observation)
                loss = (action - target).square().mean()
                loss.backward()
                if i == 0:
                    edge_grad = float(model.edge_delta.grad.abs().max())
                    encoder_grad = float(model.encoder.weight.grad.abs().max())
                    changed_fraction = float((model.edge_delta.grad != 0).float().mean())
                    assert edge_grad > 0 and encoder_grad > 0
                optimizer.step()
                torch.cuda.synchronize()
                if i:
                    times.append(time.perf_counter() - start)
            item = dict(batch=batch, neural_steps=neural_steps, seconds=sum(times)/len(times),
                        observations_per_second=batch/(sum(times)/len(times)),
                        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,
                        loss=float(loss), edge_gradient_max=edge_grad, encoder_gradient_max=encoder_grad,
                        edge_gradient_nonzero_fraction=changed_fraction,
                        neuron_activity_nonzero_fraction=float((activity.abs() > 1e-8).float().mean()))
            del action, activity, loss
        except torch.cuda.OutOfMemoryError as exc:
            item = dict(batch=batch, error='CUDA OOM', detail=str(exc))
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        results.append(item)
        print(json.dumps(item), flush=True)
    report = dict(gpu=torch.cuda.get_device_name(), torch=torch.__version__, cuda=torch.version.cuda,
                  graph=model.meta, parameters=sum(p.numel() for p in model.parameters()), checks=check,
                  measurement='Full sensory encoder, complete sparse graph, joint readout, backward and Adam; excludes physics.',
                  results=results)
    path = ROOT / 'runs/cloud/brain_benchmark.json'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--batches', type=int, nargs='+', default=[8, 16, 32, 64])
    parser.add_argument('--neural-steps', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    benchmark(args.batches, args.neural_steps, args.repeats)
