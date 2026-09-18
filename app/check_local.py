"""Compare the trained full graph on CPU and the local GPU, then measure inference."""
import hashlib
import json
import platform
import time

import numpy as np
import psutil
import torch

from .full_brain import ConnectomePolicy, ROOT
from .sim import Body


@torch.inference_mode()
def main():
    torch.set_num_threads(4)
    body = Body(load_motor_policy=False)
    assert body.policy is None
    model = ConnectomePolicy(device='cpu').eval()
    checkpoint = ROOT/'runs/cloud/best.pt'
    model.load(checkpoint)
    assert model.n == 166700 and model.weight.numel() == 25582938
    observations = torch.from_numpy(np.stack([body.motor_observation([speed, 0, 0]) for speed in [0.35, 0.5, 0.65]]))
    reference, _ = model(observations)
    cpu_ms = []
    for _ in range(8):
        start = time.perf_counter()
        model(observations[:1])
        cpu_ms.append((time.perf_counter()-start)*1000)
    model.to('cuda')
    actual, _ = model(observations.to('cuda'))
    torch.testing.assert_close(actual.cpu(), reference, rtol=1e-4, atol=1e-5)
    latency = []
    for i in range(110):
        start = time.perf_counter()
        action, activity = model(observations[:1].to('cuda'))
        action.cpu().numpy()
        activity[model.outputs, 0].cpu().numpy()
        if i >= 10:
            latency.append((time.perf_counter()-start)*1000)
    with checkpoint.open('rb') as handle:
        checkpoint_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
    result = dict(passed=True, checkpoint_sha256=checkpoint_hash, graph_sha256=model.meta['graph_sha256'],
                  neurons=model.n, edges=model.weight.numel(), teacher_used=False,
                  device=torch.cuda.get_device_name(), platform=platform.platform(), torch=torch.__version__,
                  cpu_median_ms=float(np.median(cpu_ms)), gpu_median_ms=float(np.median(latency)),
                  gpu_p95_ms=float(np.percentile(latency, 95)), gpu_samples=len(latency),
                  cpu_gpu_max_action_error=float((actual.cpu()-reference).abs().max()),
                  peak_tensor_vram_mb=torch.cuda.max_memory_allocated()/2**20,
                  process_rss_mb=psutil.Process().memory_info().rss/2**20,
                  ram_available_mb=psutil.virtual_memory().available/2**20)
    target = ROOT/'runs/local/inference.json'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()
