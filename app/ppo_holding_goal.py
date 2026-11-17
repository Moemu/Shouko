"""Random-initialized PPO for a versioned, finite near-field holding reset."""
import argparse
from pathlib import Path

from .braking_task import HoldingGoalSetup
from .ppo_goal import PPOSettings, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--potential-scale', type=float, choices=(0., .5), required=True)
    parser.add_argument('--max-seconds', type=float, default=420.)
    args = parser.parse_args()
    train(args.output, PPOSettings(seed=args.seed, initial_std=.15), 256, args.max_seconds,
          local=HoldingGoalSetup(args.state, args.potential_scale))


if __name__ == '__main__':
    main()
