"""Real short PPO checks for reward variants and legacy default warm starts."""
import argparse
import json
from pathlib import Path

import torch

from .ppo_goal import train, PPOSettings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--legacy', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    settings = PPOSettings(seed=741, worlds=4, steps=8)
    default = train(args.output / 'default', settings, 2, 30.)
    zero = train(args.output / 'zero', settings, 2, 30., work_cost=0.)
    a = torch.load(args.output / 'default/initial.pt', map_location='cpu', weights_only=True)
    b = torch.load(args.output / 'zero/initial.pt', map_location='cpu', weights_only=True)
    for key in a['policy']:
        torch.testing.assert_close(a['policy'][key], b['policy'][key], rtol=0, atol=0)
    for report, cost in ((default, .001), (zero, 0.)):
        assert report['training_reward']['work_cost'] == cost
        for row in report['history']:
            total = row['reward_mean'] * row['transitions']
            expected = row['progress_sum_m'] + row['arrivals'] - row['work_penalty']
            assert abs(total - expected) < 1e-5, (total, expected)
            assert abs(row['work_penalty'] - cost * row['absolute_work_j']) < 1e-9
    try:
        train(args.output / 'wrong_cost', settings, 1, 30., resume=args.output / 'zero/last.pt')
    except ValueError as error:
        assert 'settings mismatch' in str(error)
    else:
        raise AssertionError('Wrong training reward accepted on restore')
    resumed = train(args.output / 'zero_resumed', settings, 1, 30., resume=args.output / 'zero/last.pt', work_cost=0.)
    assert resumed['initial_env_steps'] == 64 and resumed['final_env_steps'] == 96
    legacy_checked = False
    if args.legacy:
        legacy = train(args.output / 'legacy_resumed', PPOSettings(), 1, 30., resume=args.legacy)
        assert legacy['initial_env_steps'] == 32768 and legacy['final_env_steps'] == 33792
        legacy_checked = True
    result = dict(initial_policy_equal=True, reward_decomposition=True, wrong_cost_rejected=True,
                  zero_cost_restored=True, legacy_default_restored=legacy_checked)
    (args.output / 'checks.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print('PASS: goal training objective variants')


if __name__ == '__main__':
    main()
