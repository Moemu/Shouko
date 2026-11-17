"""Evaluate one coverage checkpoint at the original or near-field frozen start."""
import argparse
from pathlib import Path

from .coverage_goal import load_pool
from .evaluate_holding_goal import evaluate as evaluate_finite


def evaluate(checkpoint: Path, pool: Path, output: Path, backend: str, start: str) -> dict:
    states = load_pool(pool)
    if start not in ('original', 'near'):
        raise ValueError('Expected original or near start')
    state = states[0 if start == 'original' else 7]
    return evaluate_finite(checkpoint, pool/'manifest.json', output, backend, transfer_state=state.path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--pool', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--backend', choices=('native', 'cuda'), default='native')
    parser.add_argument('--start', choices=('original', 'near'), required=True)
    args = parser.parse_args()
    evaluate(args.checkpoint, args.pool, args.output, args.backend, args.start)


if __name__ == '__main__':
    main()
