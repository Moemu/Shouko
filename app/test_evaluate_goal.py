"""Fixed-deck validation and native evaluation metric consistency."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch

from .evaluate_goal import evaluate_episode, fixed_deck, read_deck
from .free_body import FreeBody
from .goal_task import GoalTask, OBSERVATION_SIZE
from .mlp_policy import MLPPolicy


def main() -> None:
    torch.set_num_threads(1)
    deck = fixed_deck()
    with TemporaryDirectory() as directory:
        path = Path(directory) / 'deck.json'
        path.write_text(json.dumps(deck), encoding='utf-8')
        assert read_deck(path) == deck
        damaged = fixed_deck()
        damaged['episodes'][0]['target_offsets_m'][1] = [2., 0.]
        path.write_text(json.dumps(damaged), encoding='utf-8')
        try:
            read_deck(path)
        except ValueError:
            pass
        else:
            raise AssertionError('Insufficient switch distance accepted')
    policy = MLPPolicy(OBSERVATION_SIZE, 18, device='cpu')
    with torch.no_grad():
        for parameter in policy.parameters():
            parameter.zero_()
    task = GoalTask(FreeBody())
    first, trace1 = evaluate_episode(task, policy, deck['episodes'][0])
    second, trace2 = evaluate_episode(task, policy, deck['episodes'][0])
    assert first == second
    np.testing.assert_array_equal(trace1['episode_0'], trace2['episode_0'])
    assert first['reason'] == 'timeout' and first['steps'] == 1500 and first['arrivals'] == 0
    target = first['targets'][0]
    assert abs(target['net_progress_m'] - (target['initial_distance_m'] - target['final_distance_m'])) < 1e-10
    assert abs(target['duration_s'] - 30.) < 1e-9
    assert first['work_j'] > 0 and first['max_penetration_m'] > 0
    print(json.dumps(dict(deterministic_repeat=True, progress_telescopes=True,
                          invalid_switch_rejected=True, zero_action_episode=first), indent=2))
    print('PASS: native goal evaluation')


if __name__ == '__main__':
    main()
