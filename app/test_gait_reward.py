"""Check that the added reward separates lifting from shuffling and hopping."""
import torch
import numpy as np
import warp as wp
from types import SimpleNamespace

from .gait_reward import SoleClearanceReward, phase_clearance_reward, phase_support_components
from .sim import Body


def main():
    phase = torch.linspace(0, 2*torch.pi, 400)[:-1]
    sine = torch.sin(phase)
    alternating = torch.stack((sine.clamp_min(0), (-sine).clamp_min(0)), dim=1)*.06
    shuffle = torch.full_like(alternating, .001)
    hopping = sine.clamp_min(0)[:, None].expand(-1, 2)*.06
    reverse = alternating.flip(1)
    target = phase_clearance_reward(alternating, phase)
    assert torch.allclose(target, torch.ones_like(target))
    for impostor in [shuffle, hopping, reverse]:
        assert phase_clearance_reward(impostor, phase).mean() < .85
    speed = torch.zeros_like(alternating)
    target_components = phase_support_components(alternating, phase, speed)
    for impostor in [shuffle, hopping, reverse]:
        assert phase_support_components(impostor, phase, speed).sum(1).mean() < target_components.sum(1).mean()
    moving = phase_support_components(alternating, phase, torch.ones_like(speed))
    assert (moving[:, 2] <= 0).all() and moving.sum(1).mean() < target_components.sum(1).mean()
    body = Body(False, 'yumi')
    body.reset(93001)
    ids = [np.flatnonzero(body.model.geom_bodyid == body.model.body(name).id)
           for name in ['left_ankle_roll_link', 'right_ankle_roll_link']]
    env = SimpleNamespace(cpu_model=body.model, feet=[body.model.geom_bodyid[i[0]] for i in ids],
                          qpos=torch.zeros(1, body.model.nq), time=torch.tensor([.2]), gait_period_s=.8,
                          data=SimpleNamespace(geom_xpos=wp.array(body.data.geom_xpos[None].astype(np.float32),
                                                                 dtype=wp.vec3, device='cpu')))
    measured = torch.tensor([[min(body.data.geom_xpos[i, 2]-body.model.geom_size[i, 0]) for i in ids]],
                            dtype=torch.float32)
    torch.testing.assert_close(SoleClearanceReward(env)(), phase_clearance_reward(measured, torch.tensor([torch.pi/2])))
    env.dt = .02
    env.episode_steps = torch.ones(1)
    reward = SoleClearanceReward(env, 'phase_support')
    reward()
    assert reward.components[0, 2] == 0, 'Reset displacement was counted as slip'
    env.episode_steps[:] = 2
    reward()
    assert reward.components[0, 2] == 0, 'Stationary feet were counted as moving'
    reward.previous_xy -= .01
    reward()
    assert reward.components[0, 2] < 0
    print('PASS: alternating sole targets distinguish shuffle, hopping and reversed phase')


if __name__ == '__main__':
    main()
