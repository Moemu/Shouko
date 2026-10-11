"""Batched CUDA physics and goal semantics for the isolated 18-joint proxy."""
from dataclasses import dataclass
import math

import mujoco_warp as mjw
import torch
import warp as wp

from .free_body import FreeBody
from .goal_task import GoalTask


@wp.kernel
def goal_pd(qpos: wp.array2d(dtype=float), qvel: wp.array2d(dtype=float),
            action: wp.array2d(dtype=float), qadr: wp.array(dtype=int), vadr: wp.array(dtype=int),
            actuator: wp.array(dtype=int), home: wp.array(dtype=float), kp: wp.array(dtype=float),
            kd: wp.array(dtype=float), lower: wp.array(dtype=float), upper: wp.array(dtype=float),
            limit: wp.array(dtype=float), scale: float, ctrl: wp.array2d(dtype=float)):
    world, joint = wp.tid()
    target = wp.clamp(home[joint] + scale * action[world, joint], lower[joint], upper[joint])
    torque = kp[joint] * (target - qpos[world, qadr[joint]]) - kd[joint] * qvel[world, vadr[joint]]
    ctrl[world, actuator[joint]] = wp.clamp(torque, -limit[joint], limit[joint])


@wp.kernel
def accumulate_work(force: wp.array2d(dtype=float), velocity: wp.array2d(dtype=float),
                    vadr: wp.array(dtype=int), dt: float, work: wp.array(dtype=float)):
    world = wp.tid()
    power = float(0.0)
    for joint in range(18):
        power += wp.abs(force[world, vadr[joint]] * velocity[world, vadr[joint]])
    work[world] += dt * power


@wp.kernel
def contact_diagnostics(nacon: wp.array(dtype=int), nefc: wp.array(dtype=int),
                        worldid: wp.array(dtype=int), distance: wp.array(dtype=float),
                        constraints: wp.array(dtype=int), contacts: wp.array(dtype=int),
                        penetration: wp.array(dtype=float)):
    index = wp.tid()
    if index < nefc.shape[0]:
        wp.atomic_max(constraints, index, nefc[index])
    if index == 0:
        wp.atomic_max(contacts, 0, nacon[0])
    if index < nacon[0] and index < distance.shape[0]:
        world = worldid[index]
        if world >= 0 and world < penetration.shape[0]:
            wp.atomic_max(penetration, world, wp.max(-distance[index], 0.0))


@dataclass(frozen=True)
class BatchedGoalResult:
    reward: torch.Tensor
    progress: torch.Tensor
    work_j: torch.Tensor
    reached: torch.Tensor
    terminated: torch.Tensor
    truncated: torch.Tensor
    complete: torch.Tensor
    timeout: torch.Tensor
    arena: torch.Tensor
    numerical: torch.Tensor


class GPUGoalEnv:
    def __init__(self, worlds: int = 16, seed: int = 11) -> None:
        if worlds < 1:
            raise ValueError('World count must be positive')
        if not torch.cuda.is_available():
            raise RuntimeError('The batched goal environment requires CUDA')
        wp.init()
        self.body = FreeBody()
        self.worlds = worlds
        self.dt = self.body.dt
        self.device = torch.device('cuda:0')
        self.torch_stream = torch.cuda.Stream(device=self.device)
        self.stream = wp.stream_from_torch(self.torch_stream)
        self.generator = torch.Generator(device=self.device).manual_seed(seed)
        self.home = torch.tensor(self.body.home, device=self.device, dtype=torch.float32)
        self.qadr = torch.tensor(self.body.qadr, device=self.device, dtype=torch.long)
        self.vadr = torch.tensor(self.body.vadr, device=self.device, dtype=torch.long)
        self.initial_qpos = torch.tensor(self.body.data.qpos.copy(), device=self.device, dtype=torch.float32)
        self.actions = torch.zeros(worlds, 18, device=self.device)
        self.origin = torch.zeros(worlds, 2, device=self.device)
        self.target = torch.zeros_like(self.origin)
        self.previous_distance = torch.zeros(worlds, device=self.device)
        self.dwell_steps = torch.zeros(worlds, device=self.device, dtype=torch.long)
        self.target_steps = torch.zeros_like(self.dwell_steps)
        self.completed = torch.zeros_like(self.dwell_steps)
        self.done = torch.zeros(worlds, device=self.device, dtype=torch.bool)
        self.dwell_limit = math.ceil((GoalTask.dwell_seconds - 1e-9) / self.dt)
        self.target_limit = math.ceil((GoalTask.target_seconds - 1e-9) / self.dt)
        with wp.ScopedDevice('cuda:0'), wp.ScopedStream(self.stream):
            self.model = mjw.put_model(self.body.model)
            self.data = mjw.put_data(self.body.model, self.body.data, nworld=worlds, nconmax=48, njmax=192)
            self.work_wp = wp.zeros(worlds, dtype=float)
            self.constraints_wp = wp.zeros(worlds, dtype=int)
            self.contacts_wp = wp.zeros(1, dtype=int)
            self.penetration_wp = wp.zeros(worlds, dtype=float)
            self.wp_actions = wp.from_torch(self.actions)
            self.pd_arrays = [wp.array(values, dtype=kind) for values, kind in (
                (self.body.qadr, int), (self.body.vadr, int), (self.body.actuators, int),
                (self.body.home, float), (self.body.kp, float), (self.body.kd, float),
                (self.body.ranges[:, 0], float), (self.body.ranges[:, 1], float), (self.body.limits, float))]
        self.qpos = wp.to_torch(self.data.qpos)
        self.qvel = wp.to_torch(self.data.qvel)
        self.rotation = wp.to_torch(self.data.xmat)[:, self.body.pelvis]
        self.center = wp.to_torch(self.data.subtree_com)[:, self.body.pelvis]
        self.velocity = wp.to_torch(self.data.subtree_linvel)[:, self.body.pelvis]
        self.angular_world = wp.to_torch(self.data.cvel)[:, self.body.pelvis, :3]
        self.work = wp.to_torch(self.work_wp)
        self.peak_constraints = wp.to_torch(self.constraints_wp)
        self.peak_contacts_total = wp.to_torch(self.contacts_wp)
        self.peak_penetration = wp.to_torch(self.penetration_wp)
        mask = torch.ones(worlds, device=self.device, dtype=torch.bool)
        self.reset(mask)
        with wp.ScopedStream(self.stream):
            self._physics_step()
            wp.synchronize_stream(self.stream)
            with wp.ScopedCapture(stream=self.stream) as capture:
                self._physics_step()
        self.graph = capture.graph
        self.reset(mask)

    def refresh(self) -> None:
        self.torch_stream.wait_stream(torch.cuda.current_stream(self.device))
        with wp.ScopedStream(self.stream):
            self._refresh()
        torch.cuda.current_stream(self.device).wait_stream(self.torch_stream)

    def _refresh(self) -> None:
        mjw.kinematics(self.model, self.data)
        mjw.com_pos(self.model, self.data)
        mjw.com_vel(self.model, self.data)
        mjw.subtree_vel(self.model, self.data)

    def _physics_step(self) -> None:
        self.work_wp.zero_()
        self.constraints_wp.zero_()
        self.contacts_wp.zero_()
        self.penetration_wp.zero_()
        for _ in range(self.body.decimation):
            wp.launch(goal_pd, dim=(self.worlds, 18), inputs=[
                self.data.qpos, self.data.qvel, self.wp_actions, *self.pd_arrays,
                self.body.action_scale, self.data.ctrl])
            mjw.step1(self.model, self.data)
            mjw.fwd_actuation(self.model, self.data)
            wp.launch(accumulate_work, dim=self.worlds, inputs=[self.data.qfrc_actuator, self.data.qvel,
                      self.pd_arrays[1], self.body.model.opt.timestep, self.work_wp])
            wp.launch(contact_diagnostics, dim=self.data.naconmax, inputs=[self.data.nacon, self.data.nefc,
                      self.data.contact.worldid, self.data.contact.dist, self.constraints_wp,
                      self.contacts_wp, self.penetration_wp])
            mjw.step2(self.model, self.data)
        self._refresh()

    def _new_targets(self, mask: torch.Tensor, center: torch.Tensor) -> None:
        indices = mask.nonzero(as_tuple=True)[0]
        pending = indices
        while pending.numel():
            angle = torch.rand(len(pending), generator=self.generator, device=self.device) * (2 * math.pi) - math.pi
            candidates = self.origin[pending] + GoalTask.target_distance * torch.stack((angle.cos(), angle.sin()), 1)
            self.target[pending] = candidates
            near = torch.linalg.vector_norm(candidates - center[pending, :2], dim=1) < GoalTask.target_distance - 1e-6
            pending = pending[near]
        self.previous_distance[indices] = torch.linalg.vector_norm(self.target[indices] - center[indices, :2], dim=1)
        self.dwell_steps[indices] = 0
        self.target_steps[indices] = 0

    @torch.no_grad()
    def reset(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.shape != (self.worlds,) or mask.dtype != torch.bool or mask.device != self.device:
            raise ValueError('Reset requires a CUDA boolean mask for all worlds')
        self.torch_stream.wait_stream(torch.cuda.current_stream(self.device))
        with wp.ScopedStream(self.stream):
            mjw.reset_data(self.model, self.data, wp.from_torch(mask))
        torch.cuda.current_stream(self.device).wait_stream(self.torch_stream)
        self.qpos[mask] = self.initial_qpos
        self.actions[mask] = 0
        self.work[mask] = 0
        self.completed[mask] = 0
        self.done[mask] = False
        self.refresh()
        self.origin[mask] = self.center[mask, :2]
        self._new_targets(mask, self.center)
        return self.observation()

    def observation(self) -> torch.Tensor:
        world_target = torch.cat((self.target - self.center[:, :2], torch.zeros(self.worlds, 1, device=self.device)), 1)
        inverse = self.rotation.transpose(1, 2)
        rotate = lambda vector: torch.bmm(inverse, vector.unsqueeze(2)).squeeze(2)
        gravity = torch.zeros(self.worlds, 3, device=self.device)
        gravity[:, 2] = -1
        return torch.cat((rotate(world_target), rotate(self.velocity), rotate(self.angular_world), rotate(gravity),
                          self.qpos[:, self.qadr] - self.home, self.qvel[:, self.vadr], self.actions,
                          self.center[:, 2:3], (self.dwell_steps * self.dt / GoalTask.dwell_seconds)[:, None],
                          (1 - self.target_steps * self.dt / GoalTask.target_seconds).clamp_min(0)[:, None]), 1)

    @torch.no_grad()
    def evaluate(self, center: torch.Tensor, velocity: torch.Tensor, work: torch.Tensor) -> BatchedGoalResult:
        if bool(self.done.any()):
            raise RuntimeError('Reset completed worlds before stepping')
        self.target_steps += 1
        numerical = ~(torch.isfinite(center).all(1) & torch.isfinite(velocity).all(1) & torch.isfinite(work))
        distance = torch.linalg.vector_norm(self.target - center[:, :2], dim=1)
        progress = torch.where(numerical, 0., self.previous_distance - distance)
        self.previous_distance.copy_(distance)
        eligible = ((distance <= GoalTask.radius) & (torch.linalg.vector_norm(velocity[:, :2], dim=1) <= GoalTask.speed_limit)
                    & ~numerical)
        self.dwell_steps = torch.where(eligible, self.dwell_steps + 1, 0)
        reached = self.dwell_steps >= self.dwell_limit
        arena = (torch.linalg.vector_norm(center[:, :2] - self.origin, dim=1) > GoalTask.arena_radius) & ~numerical
        accepted = reached & ~arena & ~numerical
        self.completed += accepted.long()
        complete = accepted & (self.completed >= GoalTask.targets_per_episode)
        timeout = (self.target_steps >= self.target_limit) & ~reached & ~arena & ~numerical
        terminated, truncated = complete | numerical, timeout | arena
        self.done.copy_(terminated | truncated)
        self._new_targets(accepted & ~complete, center)
        reward = progress + GoalTask.arrival_reward * accepted - GoalTask.work_cost * work
        reward = torch.where(numerical, 0., reward)
        return BatchedGoalResult(reward, progress, torch.where(numerical, 0., work).clone(), reached, terminated, truncated,
                                 complete, timeout, arena, numerical)

    @torch.no_grad()
    def step(self, action: torch.Tensor) -> tuple[torch.Tensor, BatchedGoalResult]:
        if action.shape != (self.worlds, 18) or action.device != self.device or not bool(torch.isfinite(action).all()):
            raise ValueError('Expected a CUDA batch of 18 finite actions per world')
        if bool(self.done.any()):
            raise RuntimeError('Reset completed worlds before stepping')
        self.actions.copy_(action.clamp(-8, 8))
        self.torch_stream.wait_stream(torch.cuda.current_stream(self.device))
        with wp.ScopedStream(self.stream):
            wp.capture_launch(self.graph)
        torch.cuda.current_stream(self.device).wait_stream(self.torch_stream)
        if bool(wp.to_torch(self.data.overflow).any()):
            raise RuntimeError('MuJoCo Warp capacity/solver overflow')
        center = self.center.clone()
        invalid = ~(torch.isfinite(self.qpos).all(1) & torch.isfinite(self.qvel).all(1)
                    & torch.isfinite(wp.to_torch(self.data.qacc)).all(1))
        center[invalid] = float('nan')
        result = self.evaluate(center, self.velocity, self.work)
        return self.observation(), result
