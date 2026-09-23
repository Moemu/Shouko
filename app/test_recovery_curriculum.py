"""Reset coverage, time-limit bootstrap, and baseline persistence regression checks."""
from pathlib import Path
import json
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch

from .evaluate_locomotion import gait_summary
from .mlp_policy import MLPPolicy
from .recovery_curriculum import RecoveryCurriculum, bootstrap_timeouts
from .value_diagnostics import gae_targets


def main():
    env = SimpleNamespace(qpos=torch.zeros(6, 19), worlds=6, dt=0.02, episode_steps=torch.zeros(6))
    curriculum = RecoveryCurriculum(env, [0, -0.2, 0.2], 4)
    mask = torch.ones(6, dtype=torch.bool)
    for _ in range(3):
        curriculum.apply(mask)
    assert torch.equal(curriculum.starts, torch.ones(6, 3, dtype=torch.long))
    torch.testing.assert_close(env.qpos[:, 3:7].square().sum(1), torch.ones(6))
    env.episode_steps[:] = 200
    fallen = torch.tensor([True, False, False, False, False, False])
    assert not curriculum.truncated(fallen)[0] and curriculum.truncated(fallen)[1:].all()
    corrected = bootstrap_timeouts(torch.tensor([1.]), torch.tensor([5.]), torch.tensor([True]), .9)
    rewards = torch.stack([corrected, torch.tensor([2.])])
    advantage, _ = gae_targets(rewards, torch.tensor([[3.], [9.], [4.]]), torch.tensor([[1.], [0.]]), .9, .95)
    torch.testing.assert_close(advantage, torch.tensor([[2.5], [-3.4]]))
    model = MLPPolicy(device='cpu')
    obs = torch.randn(4, 50)
    with TemporaryDirectory() as directory:
        path = Path(directory)/'baseline.pt'
        model.save(path, physics={'gait_period_s': .8})
        restored = MLPPolicy(device='cpu')
        restored.load(path, interface={'gait_period_s': .8})
        torch.testing.assert_close(model(obs)[0], restored(obs)[0], rtol=0, atol=0)
        try:
            restored.load(path, interface={'gait_period_s': 1.})
        except ValueError:
            pass
        else:
            raise AssertionError('Mismatched baseline body accepted')
    contacts = np.ones((50, 2), dtype=bool)
    contacts[10:20, 0] = False
    clearance = np.zeros((50, 2)); clearance[10:20, 0] = .03
    gait = gait_summary(dict(contacts=contacts, clearance=clearance, feet_xy=np.zeros((50, 2, 2)),
                             yaw=np.zeros(50), y=np.zeros(50), height=np.ones(50)*.9,
                             knees=np.zeros((50, 2))), .02)
    assert gait['feet'][0]['qualifying_swings'] == 1 and gait['feet'][1]['qualifying_swings'] == 0
    json.dumps(gait, allow_nan=False)
    print('PASS: balanced recovery starts, fall/timeout distinction, GAE reset boundary, MLP reload, gait events')


if __name__ == '__main__':
    main()
