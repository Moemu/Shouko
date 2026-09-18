"""Batched G1 physics with direct joint actions; the environment has no motor policy."""
from pathlib import Path

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp
import yaml

ROOT = Path(__file__).resolve().parents[1]
UNITREE = ROOT / 'vendor/unitree_rl_gym'

from .sim import ROBOTS


@wp.kernel
def pd_torque(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
              action: wp.array2d(dtype=float), home: wp.array(dtype=float),
              kp: wp.array(dtype=float), kd: wp.array(dtype=float), limit: wp.array(dtype=float),
              ctrl: wp.array2d(dtype=float)):
    world, joint = wp.tid()
    torque = kp[joint] * (home[joint] + 0.25 * action[world, joint] - qpos[world, 7 + joint])
    torque -= kd[joint] * qvel[world, 6 + joint]
    ctrl[world, joint] = wp.clamp(torque, -limit[joint], limit[joint])


class GPUHumanoid:
    def __init__(self, worlds=64, robot='g1'):
        spec = ROBOTS[robot]
        self.robot = robot
        self.fall_height = spec['fall_height']
        wp.init()
        self.worlds = worlds
        self.dt = 0.02
        self.torch_stream = torch.cuda.Stream()
        self.stream = wp.stream_from_torch(self.torch_stream)
        cfg = yaml.safe_load(spec['cfg'].read_text(encoding='utf-8'))
        self.cpu_model = mujoco.MjModel.from_xml_path(str(spec['xml']))
        self.cpu_model.opt.timestep = 0.002
        self.cpu_model.opt.iterations = 50
        self.cpu_model.opt.ls_iterations = 50
        self.cpu_model.opt.tolerance = 1e-6
        # Train against the same inertias and foot contacts, with primitive collision geometry.
        # Mesh collision is unnecessary for flat-ground gait and allocates large CCD buffers.
        mesh = self.cpu_model.geom_type == mujoco.mjtGeom.mjGEOM_MESH
        self.cpu_model.geom_contype[mesh] = 0
        self.cpu_model.geom_conaffinity[mesh] = 0
        self.home = torch.tensor(cfg['default_angles'], device='cuda')
        self.initial_qpos = torch.tensor(self.cpu_model.qpos0, dtype=torch.float32, device='cuda')
        self.initial_qpos[2] = spec['initial_height']
        self.initial_qpos[7:] = self.home
        self.actions = torch.zeros(worlds, 12, device='cuda')
        self.command = torch.zeros(worlds, 3, device='cuda')
        self.command[:, 0] = 0.5
        self.episode_steps = torch.zeros(worlds, dtype=torch.long, device='cuda')
        self.kp = wp.array(cfg['kps'], dtype=float, device='cuda')
        self.kd = wp.array(cfg['kds'], dtype=float, device='cuda')
        # Unitree motors put force limits on joints rather than actuators.
        limits = self.cpu_model.jnt_actfrcrange[1:, 1]
        self.limit = wp.array(limits, dtype=float, device='cuda')
        self.wp_home = wp.from_torch(self.home)
        self.wp_actions = wp.from_torch(self.actions)
        self.torch_stream.wait_stream(torch.cuda.current_stream())
        with wp.ScopedStream(self.stream):
            self.model = mjw.put_model(self.cpu_model)
            self.data = mjw.make_data(self.cpu_model, nworld=worlds, nconmax=48, njmax=192)
        torch.cuda.current_stream().wait_stream(self.torch_stream)
        self.qpos = wp.to_torch(self.data.qpos)
        self.qvel = wp.to_torch(self.data.qvel)
        self.time = wp.to_torch(self.data.time)
        self.xpos = wp.to_torch(self.data.xpos)
        self.feet = [self.cpu_model.body(name).id for name in ['left_ankle_roll_link', 'right_ankle_roll_link']]
        self.reset(torch.ones(worlds, dtype=torch.bool, device='cuda'))
        self.torch_stream.wait_stream(torch.cuda.current_stream())
        with wp.ScopedStream(self.stream):
            self._substep()
            wp.synchronize_stream(self.stream)
            with wp.ScopedCapture(stream=self.stream) as capture:
                for _ in range(10):
                    self._substep()
        self.graph = capture.graph
        torch.cuda.current_stream().wait_stream(self.torch_stream)
        self.reset(torch.ones(worlds, dtype=torch.bool, device='cuda'))

    def _substep(self):
        wp.launch(pd_torque, dim=(self.worlds, 12), inputs=[self.data.qpos, self.data.qvel,
                  self.wp_actions, self.wp_home, self.kp, self.kd, self.limit], outputs=[self.data.ctrl])
        mjw.step(self.model, self.data)

    def reset(self, mask, randomize=False):
        self.qpos[mask] = self.initial_qpos
        self.qvel[mask] = 0
        self.time[mask] = 0
        self.actions[mask] = 0
        self.episode_steps[mask] = 0
        wp.to_torch(self.data.qacc_warmstart)[mask] = 0
        if randomize:
            count = int(mask.sum())
            self.qpos[mask, 7:] += torch.randn(count, 12, device='cuda') * 0.015
            self.qvel[mask, :2] = torch.randn(count, 2, device='cuda') * 0.025

    def gravity(self):
        w, x, y, z = self.qpos[:, 3:7].unbind(1)
        return torch.stack([2*(-z*x+w*y), -2*(z*y+w*x), 1-2*(w*w+z*z)], dim=1)

    def observation(self):
        phase = self.time * (2 * torch.pi / 0.8)
        return torch.cat([self.qvel[:, 3:6] * 0.25, self.gravity(),
                          self.command * torch.tensor([2, 2, 0.25], device='cuda'),
                          self.qpos[:, 7:] - self.home, self.qvel[:, 6:] * 0.05,
                          self.actions, torch.sin(phase)[:, None], torch.cos(phase)[:, None]], dim=1)

    def step(self, action):
        self.actions.copy_(action.clamp(-8, 8))
        self.torch_stream.wait_stream(torch.cuda.current_stream())
        with wp.ScopedStream(self.stream):
            wp.capture_launch(self.graph)
        torch.cuda.current_stream().wait_stream(self.torch_stream)
        self.episode_steps += 1
        fallen = (self.qpos[:, 2] < self.fall_height) | (-self.gravity()[:, 2] < 0.45)
        return fallen

    def snapshot(self, index=0):
        qpos = self.qpos[index].cpu().numpy()
        qvel = self.qvel[index].cpu().numpy()
        w, x, y, z = qpos[3:7]
        return dict(time=float(self.time[index]), qpos=qpos.tolist(), velocity=qvel[:3].tolist(),
                    yaw=float(np.arctan2(2*(w*z+x*y), 1-2*(y*y+z*z))),
                    upright=float(-self.gravity()[index, 2]), height=float(qpos[2]),
                    distance=float(np.linalg.norm(qpos[:2])),
                    fallen=bool(qpos[2] < self.fall_height or -self.gravity()[index, 2] < 0.45),
                    joints=self.xpos[index, 1:].cpu().tolist())

    def check_physics(self):
        overflow = wp.to_torch(self.data.overflow)
        if bool(overflow.any()):
            raise RuntimeError(f'MuJoCo Warp constraint/solver overflow: {overflow.cpu().tolist()}')
        if not bool(torch.isfinite(self.qpos).all() and torch.isfinite(self.qvel).all()):
            raise FloatingPointError('Non-finite physics state')
