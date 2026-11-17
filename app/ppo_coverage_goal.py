"""Bounded actor-initialized PPO for single-state versus fixed mixed starts."""
import argparse
from pathlib import Path

from .coverage_goal import CoverageGoalSetup
from .ppo_goal import PPOSettings, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--source-checkpoint', type=Path, required=True)
    parser.add_argument('--source-state', type=Path, required=True)
    parser.add_argument('--arm', choices=('near', 'mixed'), required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=420.)
    args = parser.parse_args()
    train(args.output, PPOSettings(seed=args.seed, initial_std=.15), 256, args.max_seconds, work_cost=0.,
          local=CoverageGoalSetup(args.pool, args.source_checkpoint, args.source_state, args.arm))


if __name__ == '__main__':
    main()
