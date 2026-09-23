"""Read-only MLP collection benchmark, run only while no training job uses the GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.full_brain import checkpoint_configuration
from app.gpu_body import GPUHumanoid
from app.mlp_policy import MLPPolicy
from app.ppo_yumi import quad_yaw

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--worlds', type=int, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
torch.set_num_threads(8)
torch.manual_seed(3917)
cfg = checkpoint_configuration(args.checkpoint)
env = GPUHumanoid(args.worlds, 'yumi', cfg['physics_interface'], cfg['observation_size'])
policy = MLPPolicy(cfg['observation_size']).eval()
policy.load(args.checkpoint)
env.command[:, 0] = torch.linspace(.15, .75, args.worlds, device='cuda')
rows = []
with torch.no_grad():
    for trial in range(4):
        env.reset(torch.ones(args.worlds, dtype=torch.bool, device='cuda'), randomize=True)
        torch.cuda.synchronize()
        start = time.perf_counter()
        for step in range(128):
            env.command[:, 2] = (-quad_yaw(env.qpos)*1.4).clamp(-.2, .2)
            mean, _ = policy(env.observation())
            fallen = env.step(mean+torch.randn_like(mean)*.38)
            if bool(fallen.any()):
                env.reset(fallen, randomize=True)
        torch.cuda.synchronize()
        elapsed = time.perf_counter()-start
        env.check_physics()
        if trial:
            rows.append(dict(seconds=elapsed, samples_per_second=128*args.worlds/elapsed))
report = dict(worlds=args.worlds, collection=rows,
              mean_samples_per_second=sum(r['samples_per_second'] for r in rows)/len(rows),
              checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
              limitation='Collection only; excludes reward/critic/optimizer. No weights are updated.',
              memory_used_gib=(torch.cuda.mem_get_info()[1]-torch.cuda.mem_get_info()[0])/2**30)
args.output.write_text(json.dumps(report, indent=2))
print(json.dumps(report))
