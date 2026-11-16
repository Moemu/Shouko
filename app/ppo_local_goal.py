"""PPO entry for the separately identified fallen-state local diagnostic task."""
import argparse
from pathlib import Path

from .local_goal import LocalGoalSetup
from .ppo_goal import PPOSettings, train


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--state', type=Path, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--rho', type=float, choices=(0., .9), required=True)
    parser.add_argument('--work-cost', type=float, choices=(0., .001), required=True)
    parser.add_argument('--iterations', type=int, default=256)
    parser.add_argument('--worlds', type=int, default=16)
    parser.add_argument('--steps', type=int, default=64)
    parser.add_argument('--max-seconds', type=float, default=420.)
    args = parser.parse_args()
    train(args.output, PPOSettings(seed=args.seed, worlds=args.worlds, steps=args.steps, initial_std=.7),
          args.iterations, args.max_seconds, work_cost=args.work_cost, local=LocalGoalSetup(args.state, args.rho))


if __name__ == '__main__':
    main()
