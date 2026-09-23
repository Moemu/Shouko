"""Optional phase/sole-clearance experiment; does not change the body or policy."""
import numpy as np
import torch
import warp as wp


def phase_clearance_reward(clearance, phase):
    """Reward a 6 cm alternating sole trajectory, including the supported half-cycle."""
    wave = torch.stack((torch.sin(phase), -torch.sin(phase)), dim=1)
    target = 0.06 * wave.clamp_min(0)
    # Average rather than sum: the maximum added reward is one per transition.
    return torch.exp(-((clearance - target) / 0.02).square()).mean(dim=1)


def phase_support_components(clearance, phase, foot_speed):
    """Flat-ground clearance, support timing and support-motion terms, kept separate."""
    wave = torch.stack((torch.sin(phase), -torch.sin(phase)), dim=1)
    desired_air = wave > .2
    air = torch.sigmoid((clearance-.006)/.002)
    support_timing = 2 * torch.where(desired_air, air, 1-air).mean(dim=1)
    supported = clearance < .006
    support_motion = -.2 * (foot_speed.square().clamp(max=4)*supported).mean(dim=1)
    return torch.stack((phase_clearance_reward(clearance, phase), support_timing, support_motion), dim=1)


class SoleClearanceReward:
    def __init__(self, env, profile='phase_clearance'):
        if profile not in ('phase_clearance', 'phase_support'):
            raise ValueError('Unknown sole reward profile')
        self.env = env
        self.profile = profile
        model = env.cpu_model
        ids = [np.flatnonzero(model.geom_bodyid == foot) for foot in env.feet]
        if len(ids[0]) != len(ids[1]) or any(len(side) == 0 for side in ids):
            raise ValueError('Expected matching nonempty foot geometry groups')
        self.ids = torch.tensor(np.stack(ids), device=env.qpos.device)
        self.radii = torch.tensor(model.geom_size[np.stack(ids), 0],
                                  dtype=env.qpos.dtype, device=env.qpos.device)
        # This diagnostic supports the Yumi spherical sole geometry only.
        import mujoco
        if not np.all(model.geom_type[np.stack(ids)] == mujoco.mjtGeom.mjGEOM_SPHERE):
            raise ValueError('Clearance reward requires spherical sole geometries')
        self.positions = wp.to_torch(env.data.geom_xpos)
        self.previous_xy = torch.zeros((env.qpos.shape[0], 2, 2), device=env.qpos.device)
        self.components = torch.zeros((env.qpos.shape[0], 3), device=env.qpos.device)

    def __call__(self):
        clearance = (self.positions[:, self.ids, 2] - self.radii).min(dim=2).values
        phase = self.env.time * (2 * torch.pi / self.env.gait_period_s)
        if self.profile == 'phase_clearance':
            bonus = phase_clearance_reward(clearance, phase)
            self.components[:, 0] = bonus
            return bonus
        feet_xy = self.positions[:, self.ids, :2].mean(dim=2)
        speed = torch.linalg.vector_norm(feet_xy-self.previous_xy, dim=2)/self.env.dt
        # Reset jumps and evaluation transitions are not stance-foot motion.
        speed = torch.where(self.env.episode_steps[:, None] > 1, speed, 0)
        self.previous_xy.copy_(feet_xy)
        self.components = phase_support_components(clearance, phase, speed)
        return self.components.sum(dim=1)
