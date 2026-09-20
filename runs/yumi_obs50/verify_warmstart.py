"""Warm-start equivalence check, one config per process (GPU-footprint friendly).

Run twice in separate processes (separate CUDA/MuJoCo contexts, never stacked):

  python runs/yumi_obs50/verify_warmstart.py --config source_47
  python runs/yumi_obs50/verify_warmstart.py --config expanded_50

Each run evaluates its checkpoint through train_full.evaluate on the tall
interface and appends to warmstart_check.json. Zero-initialized new encoder
columns must reproduce the source policy's behaviour bit-for-bit up to GPU
nondeterminism.
"""
import argparse
import json

import torch
import yaml

ROOT = 'D:/Project/Neuromechfly'

CONFIGS = dict(
    source_47=dict(observation_size=47, checkpoint=ROOT + '/runs/yumi/best_tall.pt'),
    expanded_50=dict(observation_size=50, checkpoint=ROOT + '/runs/yumi_obs50/best_tall_obs50.pt'),
)


def main(args):
    from app.full_brain import ConnectomePolicy
    from app.gpu_body import GPUHumanoid
    from app.sim import observation_interface
    from app import train_full

    spec = CONFIGS[args.config]
    interface = observation_interface(
        yaml.safe_load(open(ROOT + '/runs/local/yumi_tall.yaml', encoding='utf-8').read()))
    torch.manual_seed(2026)
    env = GPUHumanoid(args.worlds, robot='yumi', interface=interface,
                      observation_size=spec['observation_size'])
    policy = ConnectomePolicy(observation_size=spec['observation_size']).eval()
    policy.load(spec['checkpoint'], interface=env.interface)
    summary = train_full.evaluate(policy, env, seconds=args.seconds)
    result = dict(score=round(summary['score'], 6), successes=summary['successes'],
                  mean_speed=round(summary['mean_speed'], 6),
                  mean_height=round(summary['mean_height'], 6),
                  per_world_mean_speed=[round(row['mean_speed'], 6) for row in summary['tests']],
                  per_world_fallen=[row['fallen'] for row in summary['tests']])
    print(json.dumps({args.config: result}), flush=True)

    path = ROOT + '/runs/yumi_obs50/warmstart_check.json'
    try:
        store = json.load(open(path))
    except Exception:
        store = {}
    store[args.config] = result
    store.setdefault('meta', dict(worlds=args.worlds, seconds=args.seconds))
    if 'source_47' in store and 'expanded_50' in store and store.get('meta', {}).get('worlds') == args.worlds:
        a, b = store['source_47'], store['expanded_50']
        max_delta = max(abs(x - y) for x, y in zip(a['per_world_mean_speed'], b['per_world_mean_speed']))
        store['verdict'] = dict(max_mean_speed_delta=round(max_delta, 6),
                                fallen_mismatch=sum(x != y for x, y in zip(a['per_world_fallen'], b['per_world_fallen'])),
                                score_delta=round(abs(a['score'] - b['score']), 6),
                                bit_equivalent=bool(max_delta < 1e-3 and
                                                    sum(x != y for x, y in zip(a['per_world_fallen'], b['per_world_fallen'])) == 0))
    with open(path, 'w') as handle:
        json.dump(store, handle, indent=2)
    print(json.dumps(store.get('verdict', 'waiting_for_other_config')), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', choices=sorted(CONFIGS), required=True)
    parser.add_argument('--worlds', type=int, default=16)
    parser.add_argument('--seconds', type=float, default=10)
    main(parser.parse_args())
