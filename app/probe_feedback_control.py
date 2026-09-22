"""Fixed-checkpoint feedback ablation with repeated intact development evaluations."""
import argparse
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
import yaml

from .full_brain import ConnectomePolicy, checkpoint_configuration
from .gpu_body import GPUHumanoid
from .sim import ROBOTS, apply_interface, configure_observation, observation_interface
from .train_full import evaluate, file_sha256


class FeedbackIntervention(nn.Module):
    def __init__(self, policy: ConnectomePolicy):
        super().__init__()
        self.policy = policy
        self.zero_new_inputs = False

    def forward(self, observation):
        if self.zero_new_inputs:
            observation = observation.clone()
            observation[:, 47:50] = 0
        return self.policy(observation)


def main(args):
    torch.set_num_threads(8)
    torch.manual_seed(20260922)
    if args.worlds < 1 or args.seconds <= 0 or args.max_seconds <= 0:
        raise ValueError('Worlds and time budgets must be positive')
    if args.output.exists():
        raise FileExistsError('Use a fresh output path to preserve prior evidence')
    config = checkpoint_configuration(args.checkpoints[0])
    if config['observation_size'] != 50:
        raise ValueError('Feedback ablation requires the 50-input interface')
    spec = ROBOTS['yumi']
    interfaces = []
    for path in args.checkpoints:
        other = checkpoint_configuration(path)
        if any(other[key] != config[key] for key in ['observation_size', 'action_size', 'neural_steps']):
            raise ValueError('Compared checkpoints must share policy dimensions and neural steps')
        # Legacy checkpoints omit fields whose values are supplied by the body defaults.
        resolved = configure_observation(apply_interface(
            yaml.safe_load(spec['cfg'].read_text(encoding='utf-8')), other['physics_interface']),
            other['observation_size'], 0.8, spec['initial_height'])
        interfaces.append(observation_interface(resolved))
    if any(interface != interfaces[0] for interface in interfaces[1:]):
        raise ValueError('Compared checkpoints must resolve to the same complete physics interface')
    env = GPUHumanoid(args.worlds, robot='yumi', interface=config['physics_interface'], observation_size=50)
    policy = ConnectomePolicy(observation_size=50, action_size=config['action_size'],
                               neural_steps=config['neural_steps'])
    model = FeedbackIntervention(policy)
    hashes = {str(path): file_sha256(path) for path in args.checkpoints}
    report = dict(kind='fixed_weight_feedback_ablation_development_screen', checkpoints=hashes,
                  worlds=args.worlds, seconds=args.seconds, seeds=args.seeds,
                  physics_interface=env.interface, results=[], completed=False,
                  intervention='Set raw observations 47:50 to zero at every decision; never update weights.',
                  repeat_control='Two intact runs per checkpoint/seed estimate same-seed trajectory variation.',
                  limitation='Development diagnostic. No training repetitions, promotion or held-out walking acceptance.')
    schedule = []
    rng = random.Random(20260922)
    for seed in args.seeds:
        conditions = [(path, mode) for path in args.checkpoints for mode in ['intact_1', 'intact_2', 'zero_new']]
        rng.shuffle(conditions)
        schedule.extend((seed, path, mode) for path, mode in conditions)
    report['schedule'] = [dict(seed=seed, checkpoint=str(path), mode=mode) for seed, path, mode in schedule]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    for seed, checkpoint, mode in schedule:
        if time.monotonic()-started >= args.max_seconds:
            raise TimeoutError('Feedback probe budget reached; partial evidence retained')
        policy.load(checkpoint, interface=env.interface)
        model.zero_new_inputs = mode == 'zero_new'
        result = evaluate(model, env, seconds=args.seconds, seed=seed)
        row = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashes[str(checkpoint)],
                   seed=seed, mode=mode, evaluation=result)
        report['results'].append(row)
        report['elapsed_seconds'] = time.monotonic()-started
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps(dict(checkpoint=str(checkpoint), seed=seed, mode=mode,
                              score=result['score'], successes=result['successes'],
                              falls=sum(t['fallen'] for t in result['tests']),
                              lateral_m=sum(t['lateral_m'] for t in result['tests'])/env.worlds)), flush=True)
    assert all(file_sha256(Path(path)) == digest for path, digest in hashes.items())
    report['completed'] = True
    report['checkpoint_files_unchanged'] = True
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoints', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seeds', type=int, nargs='+', default=[92001, 92002, 92003])
    parser.add_argument('--worlds', type=int, default=32)
    parser.add_argument('--seconds', type=float, default=30)
    parser.add_argument('--max-seconds', type=float, default=900)
    main(parser.parse_args())
