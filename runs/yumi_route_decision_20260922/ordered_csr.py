"""Inference-only, fixed-order row sums for controlled evaluation experiments."""
from contextlib import contextmanager
import json
from pathlib import Path
import time
from unittest.mock import patch

import torch
import warp as wp

from app.full_brain import ConnectomePolicy, SparseMessage
from app.train_full import file_sha256


@wp.kernel
def ordered_rows(ptr: wp.array(dtype=wp.int32), pre: wp.array(dtype=wp.int32),
                 values: wp.array(dtype=wp.float32), activity: wp.array2d(dtype=wp.float32),
                 result: wp.array2d(dtype=wp.float32)):
    row, batch = wp.tid()
    total = float(0.0)
    for edge in range(ptr[row], ptr[row + 1]):
        total = total + values[edge] * activity[pre[edge], batch]
    result[row, batch] = total


def ordered_forward(ctx, values, activity, ptr, pre):
    if torch.is_grad_enabled():
        raise RuntimeError('Ordered CSR probe is inference-only')
    if values.dtype != torch.float32 or activity.dtype != torch.float32:
        raise TypeError('Ordered CSR probe requires float32')
    if ptr.dtype != torch.int32 or pre.dtype != torch.int32:
        raise TypeError('Ordered CSR probe requires int32 graph indices')
    result = torch.empty_like(activity)
    wp.launch(ordered_rows, dim=activity.shape,
              inputs=[wp.from_torch(ptr), wp.from_torch(pre), wp.from_torch(values), wp.from_torch(activity)],
              outputs=[wp.from_torch(result)], stream=wp.stream_from_torch(torch.cuda.current_stream()))
    return result


@contextmanager
def ordered_inference():
    wp.init()
    with torch.inference_mode(), patch.object(SparseMessage, 'forward', staticmethod(ordered_forward)):
        yield


def main():
    torch.set_num_threads(8)
    torch.manual_seed(20260922)
    wp.init()
    root = Path('runs/yumi_route_decision_20260922')
    checkpoint = Path('runs/yumi_obs50/best.pt')
    policy = ConnectomePolicy(observation_size=50).eval()
    policy.load(checkpoint)
    batch = torch.load('rollout.pt', map_location='cpu', weights_only=True)
    observations = batch['observations'].reshape(-1, 50)[:32].cuda()
    report = dict(kind='fixed_order_warp_csr_probe', checkpoint_sha256=file_sha256(checkpoint), trials=[])
    # Direction, empty rows, signed values, and accumulation are checked against a dense matrix.
    ptr = torch.tensor([0, 2, 2, 5], dtype=torch.int32, device='cuda')
    pre = torch.tensor([0, 2, 0, 1, 2], dtype=torch.int32, device='cuda')
    values = torch.tensor([0.4, -0.2, 0.8, -0.5, 0.3], device='cuda')
    activity = torch.randn(3, 9, device='cuda')
    dense = torch.tensor([[0.4, 0, -0.2], [0, 0, 0], [0.8, -0.5, 0.3]], device='cuda')
    with torch.inference_mode():
        result = ordered_forward(None, values, activity, ptr, pre)
        torch.testing.assert_close(result, dense @ activity, atol=1e-6, rtol=1e-6)
        for size in [1, 9, 32]:
            obs = observations[:size]
            reference = policy(obs)[0]
            torch.cuda.reset_peak_memory_stats()
            with ordered_inference():
                initial = policy(obs)[0]
                torch.cuda.synchronize()
                started = time.perf_counter()
                delta = torch.stack([policy(obs)[0]-initial for _ in range(10)])
                torch.cuda.synchronize()
                report['trials'].append(dict(batch=size, repeat_max_abs=float(delta.abs().max()),
                    csr_reference_max_abs=float((initial-reference).abs().max()),
                    seconds_per_forward=(time.perf_counter()-started)/10,
                    peak_tensor_gib=torch.cuda.max_memory_allocated()/2**30))
                torch.testing.assert_close(initial, reference, atol=1e-5, rtol=1e-5)
                assert not torch.count_nonzero(delta), 'Ordered inference still varies on the sampled batch'
    report['small_dense_check'] = 'passed'
    report['completed'] = True
    report['limitation'] = 'Test-only inference backend. Float accumulation order differs; no weights or training changes.'
    (root/'ordered_csr.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
