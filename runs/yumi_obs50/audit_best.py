"""Audit: is best.pt tensor-identical to the expanded source, and did any
trained state actually improve per-command velocity tracking?

- state-dict comparison (instant, decisive for the '+0.09 improvement' claim)
- paired 30 s evaluations of source / best / last in one process, with the
  cmd->vx slope fitted per model (the experiment's real success metric)
"""
import json

import torch
import yaml

from app.full_brain import ConnectomePolicy
from app.gpu_body import GPUHumanoid
from app.sim import observation_interface
from app import train_full

out = {}

src = torch.load('runs/yumi_obs50/best_tall_obs50.pt', map_location='cpu', weights_only=True)
best = torch.load('runs/yumi_obs50/best.pt', map_location='cpu', weights_only=True)
last = torch.load('runs/yumi_obs50/last.pt', map_location='cpu', weights_only=True)
diffs = {k: float((src['state_dict'][k] - best['state_dict'][k]).abs().max())
         for k in src['state_dict'] if src['state_dict'][k].shape == best['state_dict'][k].shape}
shape_mismatch = [k for k in src['state_dict'] if src['state_dict'][k].shape != best['state_dict'][k].shape]
out['best_vs_source'] = dict(max_tensor_abs_diff=max(diffs.values()), shape_mismatch=shape_mismatch,
                             best_updates=best.get('extra', {}).get('updates'))
out['last_vs_source_encoder_max'] = float(
    (src['state_dict']['encoder.weight'] - last['state_dict']['encoder.weight']).abs().max())
print(json.dumps(out), flush=True)

interface = observation_interface(yaml.safe_load(open('runs/local/yumi_tall.yaml', encoding='utf-8')))
env = GPUHumanoid(32, robot='yumi', interface=interface, observation_size=50)
rows = {}
for label, ckpt in [('source', 'runs/yumi_obs50/best_tall_obs50.pt'),
                    ('best', 'runs/yumi_obs50/best.pt'),
                    ('last', 'runs/yumi_obs50/last.pt')]:
    policy = ConnectomePolicy(observation_size=50).eval()
    policy.load(ckpt, interface=env.interface)
    s = train_full.evaluate(policy, env, seconds=30)
    per_cmd = {}
    for r in s['tests']:
        per_cmd.setdefault(round(r['target_speed'], 2), []).append(r['mean_speed'])
    means = {c: round(sum(v) / len(v), 4) for c, v in sorted(per_cmd.items())}
    xs = sorted(means)
    n = len(xs)
    mx, my = sum(xs) / n, sum(means[x] for x in xs) / n
    slope = sum((x - mx) * (means[x] - my) for x in xs) / sum((x - mx) ** 2 for x in xs)
    rows[label] = dict(score=round(s['score'], 3), successes=s['successes'],
                       mean_height=round(s['mean_height'], 3),
                       per_cmd_mean_speed=means, transfer_slope=round(slope, 3))
    print(json.dumps({label: rows[label]}), flush=True)
    del policy
out['paired_evals'] = rows
json.dump(out, open('runs/yumi_obs50/audit_best.json', 'w'), indent=2)
print('AUDIT_DONE', flush=True)
