"""Fallen-state local goal task and explicit autoregressive action contract."""
from dataclasses import dataclass
from pathlib import Path
import math
import hashlib

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from .evaluate_goal import file_sha256
from .free_body import FreeBody
from .goal_checkpoint import LocalGoalContext, goal_contract, read_goal_payload
from .goal_task import GoalTask, OBSERVATION_SIZE
from .gpu_goal import GPUGoalEnv, BatchedGoalResult
from .mlp_policy import MLPPolicy


@dataclass(frozen=True)
class LocalGoalSetup:
    state_path: Path
    rho: float

    def context(self) -> LocalGoalContext:
        return LocalGoalContext(file_sha256(self.state_path), self.rho)


@dataclass(frozen=True)
class LocalResetState:
    integration: np.ndarray
    qpos: np.ndarray
    qvel: np.ndarray
    qacc_warmstart: np.ndarray
    ctrl: np.ndarray
    qfrc_applied: np.ndarray
    xfrc_applied: np.ndarray
    previous_action: np.ndarray
    time: float

    @classmethod
    def load(cls, path: Path) -> 'LocalResetState':
        with np.load(path) as z:
            body = FreeBody()
            if str(z['body_sha256']) != hashlib.sha256(body.xml.encode()).hexdigest():
                raise ValueError('Local reset state body mismatch')
            result = cls(z['integration'].copy(), z['qpos'].copy(), z['qvel'].copy(), z['qacc_warmstart'].copy(),
                         z['ctrl'].copy(), z['qfrc_applied'].copy(), z['xfrc_applied'].copy(),
                         z['previous_action'].copy(), float(z['time']))
        if not all(np.isfinite(x).all() for x in (result.integration, result.qpos, result.qvel,
                   result.qacc_warmstart, result.ctrl, result.qfrc_applied, result.xfrc_applied, result.previous_action)):
            raise ValueError('Non-finite local reset state')
        return result

    def restore(self, body: FreeBody) -> None:
        body.reset()
        mujoco.mj_setState(body.model, body.data, self.integration, mujoco.mjtState.mjSTATE_INTEGRATION)
        body.action[:] = self.previous_action
        body.refresh()


def prepare_state(path: Path) -> str:
    if path.exists():
        raise FileExistsError(path)
    body = FreeBody()
    for _ in range(150):
        body.step(np.zeros(18))
        if body.last_contacts.penetration > .030 or body.last_contacts.limit_error > .15 or any(body.data.warning.number):
            raise RuntimeError('Natural fallen-state preparation exceeded physical guard')
    state = np.empty(mujoco.mj_stateSize(body.model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(body.model, body.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    np.savez_compressed(path, integration=state, qpos=body.data.qpos.copy(), qvel=body.data.qvel.copy(),
                        qacc_warmstart=body.data.qacc_warmstart.copy(), ctrl=body.data.ctrl.copy(),
                        qfrc_applied=body.data.qfrc_applied.copy(), xfrc_applied=body.data.xfrc_applied.copy(),
                        time=body.data.time, previous_action=body.action.copy(),
                        body_sha256=hashlib.sha256(body.xml.encode()).hexdigest())
    return file_sha256(path)


def action_distribution(mean: torch.Tensor, log_std: torch.Tensor, observation: torch.Tensor,
                        rho: float) -> torch.distributions.Normal:
    if not 0 <= rho < 1:
        raise ValueError('Action persistence must be in [0,1)')
    center = rho * observation[..., 48:66] + (1-rho) * mean if rho else mean
    std = log_std.exp().clamp(.05, 1.5) * math.sqrt(1-rho*rho)
    return torch.distributions.Normal(center, std)


class LocalGoalTask(GoalTask):
    target_seconds = 5.
    targets_per_episode = 1

    def __init__(self, body: FreeBody, state_path: Path) -> None:
        self.reset_state = LocalResetState.load(state_path)
        super().__init__(body)

    def reset(self) -> np.ndarray:
        self.reset_state.restore(self.body)
        self.origin = self.body.center[:2]
        self.target = self.origin + [2., 0.]
        self.previous_distance = 2.
        self.completed = 0
        self.done = False
        self.dwell = self.target_elapsed = 0.
        return self.observation()


class LocalGoalEnv(GPUGoalEnv):
    def __init__(self, worlds: int, seed: int, setup: LocalGoalSetup) -> None:
        self.local_context = setup.context()
        self.reset_state = LocalResetState.load(setup.state_path)
        super().__init__(worlds, seed)

    @torch.no_grad()
    def reset(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.shape != (self.worlds,) or mask.dtype != torch.bool or mask.device != self.device:
            raise ValueError('Reset requires a CUDA boolean mask for all worlds')
        self.torch_stream.wait_stream(torch.cuda.current_stream(self.device))
        with wp.ScopedStream(self.stream):
            mjw.reset_data(self.model, self.data, wp.from_torch(mask))
        torch.cuda.current_stream(self.device).wait_stream(self.torch_stream)
        state = self.reset_state
        self.qpos[mask] = torch.as_tensor(state.qpos, dtype=self.qpos.dtype, device=self.device)
        self.qvel[mask] = torch.as_tensor(state.qvel, dtype=self.qvel.dtype, device=self.device)
        wp.to_torch(self.data.qacc_warmstart)[mask] = torch.as_tensor(state.qacc_warmstart, dtype=torch.float32, device=self.device)
        wp.to_torch(self.data.ctrl)[mask] = torch.as_tensor(state.ctrl, dtype=torch.float32, device=self.device)
        wp.to_torch(self.data.qfrc_applied)[mask] = torch.as_tensor(state.qfrc_applied, dtype=torch.float32, device=self.device)
        wp.to_torch(self.data.xfrc_applied)[mask] = torch.as_tensor(state.xfrc_applied, dtype=torch.float32, device=self.device)
        wp.to_torch(self.data.time)[mask] = state.time
        self.actions[mask] = torch.as_tensor(state.previous_action, dtype=torch.float32, device=self.device)
        self.work[mask] = 0
        self.completed[mask] = self.dwell_steps[mask] = self.target_steps[mask] = 0
        self.done[mask] = False
        self.refresh()
        self.origin[mask] = self.center[mask, :2]
        self.target[mask] = self.origin[mask] + torch.tensor([2., 0.], device=self.device)
        self.previous_distance[mask] = 2.
        self.target_limit = round(5. / self.dt)
        return self.observation()

    def observation(self) -> torch.Tensor:
        obs = super().observation()
        obs[:, -1] = (1-self.target_steps*self.dt/5.).clamp_min(0)
        return obs

    @torch.no_grad()
    def evaluate(self, center: torch.Tensor, velocity: torch.Tensor, work: torch.Tensor) -> BatchedGoalResult:
        if bool(self.done.any()):
            raise RuntimeError('Reset completed local worlds before stepping')
        self.target_steps += 1
        numerical = ~(torch.isfinite(center).all(1) & torch.isfinite(velocity).all(1) & torch.isfinite(work))
        distance = torch.linalg.vector_norm(self.target-center[:, :2], dim=1)
        progress = torch.where(numerical, 0., self.previous_distance-distance)
        self.previous_distance.copy_(distance)
        eligible = (distance <= .3) & (torch.linalg.vector_norm(velocity[:, :2], dim=1) <= .2) & ~numerical
        self.dwell_steps = torch.where(eligible, self.dwell_steps+1, 0)
        arena = (torch.linalg.vector_norm(center[:, :2]-self.origin, dim=1) > 10) & ~numerical
        reached = (self.dwell_steps >= self.dwell_limit) & ~arena & ~numerical
        timeout = (self.target_steps >= self.target_limit) & ~reached & ~arena & ~numerical
        self.completed += reached.long()
        terminated, truncated = reached | numerical, timeout | arena
        self.done.copy_(terminated | truncated)
        reward = torch.where(numerical, 0., progress + reached.float() - .001*work)
        return BatchedGoalResult(reward, progress, torch.where(numerical, 0., work).clone(), reached,
                                 terminated, truncated, reached, timeout, arena, numerical)


def load_local_policy(checkpoint: Path, state_path: Path, device: str = 'cpu') -> tuple[MLPPolicy, LocalGoalContext]:
    raw = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if raw.get('schema') != 'yumi_local_goal_ppo':
        raise ValueError('Expected local goal checkpoint')
    context = LocalGoalContext(**raw['local_context'])
    if context.reset_sha256 != file_sha256(state_path):
        raise ValueError('Local reset state SHA-256 mismatch')
    payload = read_goal_payload(checkpoint, goal_contract(FreeBody()), device, local_context=context)
    policy = MLPPolicy(OBSERVATION_SIZE, 18, device=device)
    policy.load_state_dict(payload['policy'])
    return policy.eval(), context
