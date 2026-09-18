"""Check teacher reference walking and measure GPU physics independently of the brain."""
import json
import time

import torch
import warp as wp

from .gpu_body import GPUHumanoid, UNITREE, ROOT
from .teacher import load_teacher


def check(worlds=16, steps=1500):
    torch.set_num_threads(8)
    start = time.perf_counter()
    env = GPUHumanoid(worlds)
    teacher = load_teacher(worlds)
    torch.cuda.synchronize()
    compile_seconds = time.perf_counter()-start
    fallen = torch.zeros(worlds, device='cuda', dtype=torch.bool)
    start = time.perf_counter()
    with torch.no_grad():
        for i in range(steps):
            action = teacher(env.observation())
            fallen |= env.step(action)
            if bool(fallen.any()):
                break
    torch.cuda.synchronize()
    elapsed = time.perf_counter()-start
    overflow = wp.to_torch(env.data.overflow)
    result = dict(worlds=worlds, control_steps=i+1, simulated_seconds=(i+1)*env.dt,
                  compile_seconds=compile_seconds, wall_seconds=elapsed,
                  control_steps_per_second=worlds*(i+1)/elapsed,
                  fallen=int(fallen.sum()), overflow=overflow.cpu().tolist(),
                  state=env.snapshot(), method='Frozen teacher reference solely to validate physics; no fly training.')
    path = ROOT/'runs/cloud/physics_benchmark.json'
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    assert not bool(overflow.any()), 'Physics constraint/contact capacity overflow'
    assert not bool(fallen.any()), 'Teacher failed: correct the GPU physics before training'
    assert float(env.qpos[:, 0].mean()) > 0.25 * ((i+1)*env.dt)


if __name__ == '__main__':
    check()
