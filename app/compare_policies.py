"""Paired training-screen comparisons; these are not held-out walking acceptance."""
import argparse
import hashlib
import json
from pathlib import Path

import torch

from .full_brain import ConnectomePolicy, checkpoint_configuration
from .gpu_body import GPUHumanoid
from .train_full import evaluate


def main(args):
    torch.set_num_threads(8)
    config = checkpoint_configuration(args.checkpoints[0])
    env = GPUHumanoid(args.worlds, robot='yumi', interface=config['physics_interface'],
                      observation_size=config['observation_size'])
    policy = ConnectomePolicy(**{key: config[key] for key in
                               ('observation_size', 'action_size', 'neural_steps')}).eval()
    report = dict(kind='paired_training_screen', seeds=args.seeds, worlds=args.worlds,
                  requested_seconds=args.seconds, physics_interface=env.interface, results=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for seed in args.seeds:
        for checkpoint in args.checkpoints:
            policy.load(checkpoint, interface=env.interface)
            with checkpoint.open('rb') as f:
                digest = hashlib.file_digest(f, 'sha256').hexdigest()
            result = evaluate(policy, env, seconds=args.seconds, seed=seed)
            row = dict(checkpoint=str(checkpoint), checkpoint_sha256=digest, seed=seed, evaluation=result)
            report['results'].append(row)
            args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
            print(json.dumps(dict(checkpoint=str(checkpoint), seed=seed,
                                  score=result['score'], successes=result['successes'],
                                  mean_duration=result['mean_duration'], mean_speed=result['mean_speed'],
                                  mean_height=result['mean_height'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[91001, 91002, 91003])
    parser.add_argument('--worlds', type=int, default=32)
    parser.add_argument('--seconds', type=float, default=30)
    main(parser.parse_args())
