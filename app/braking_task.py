"""Finite-horizon task semantics and full native replay forks for braking diagnostics."""
from dataclasses import dataclass, replace
import hashlib
from pathlib import Path

import mujoco
import numpy as np
import torch

from .local_goal import LocalGoalTask, LocalGoalEnv, LocalGoalSetup
from .goal_checkpoint import HoldingGoalContext
from .evaluate_goal import file_sha256
from .goal_task import GoalResult
from .gpu_goal import BatchedGoalResult


@dataclass(frozen=True)
class FiniteTaskContract:
    protocol: str = 'fallen_finite_goal_v1'
    horizon_seconds: float = 5.
    timeout_is_terminal: bool = True
    radius_m: float = .3
    speed_limit_m_s: float = .2
    dwell_seconds: float = 1.


class FiniteLocalGoalTask(LocalGoalTask):
    def evaluate(self, center: np.ndarray, velocity: np.ndarray, work_j: float) -> GoalResult:
        result = super().evaluate(center, velocity, work_j)
        return replace(result, terminated=True, truncated=False) if result.reason == 'timeout' else result


class FiniteLocalGoalEnv(LocalGoalEnv):
    @torch.no_grad()
    def evaluate(self, center: torch.Tensor, velocity: torch.Tensor, work: torch.Tensor) -> BatchedGoalResult:
        result = super().evaluate(center, velocity, work)
        return replace(result, terminated=result.terminated | result.timeout,
                       truncated=result.truncated & ~result.timeout)


@dataclass(frozen=True)
class HoldingGoalSetup:
    state_path: Path
    potential_scale: float

    def context(self) -> HoldingGoalContext:
        return HoldingGoalContext(file_sha256(self.state_path), self.potential_scale)


class HoldingGoalTask(FiniteLocalGoalTask):
    def __init__(self, body, state_path: Path) -> None:
        self.fork_path = state_path
        super().__init__(body, state_path)

    def reset(self) -> np.ndarray:
        return restore_fork(self, self.fork_path)


class HoldingGoalEnv(FiniteLocalGoalEnv):
    def __init__(self, worlds: int, seed: int, setup: HoldingGoalSetup) -> None:
        with np.load(setup.state_path) as state:
            self.fork_origin = state['origin'].copy()
            self.fork_target = state['target'].copy()
            self.fork_elapsed = float(state['elapsed'])
            self.fork_dwell = float(state['dwell'])
            self.fork_completed = int(state['completed'])
        if not (0 <= self.fork_elapsed < 5. and 0 <= self.fork_dwell < 1. and self.fork_completed == 0):
            raise ValueError('Holding reset requires an unfinished finite episode')
        super().__init__(worlds, seed, LocalGoalSetup(setup.state_path, 0.))
        self.local_context = setup.context()

    @torch.no_grad()
    def reset(self, mask: torch.Tensor) -> torch.Tensor:
        super().reset(mask)
        self.origin[mask] = torch.as_tensor(self.fork_origin, dtype=self.origin.dtype, device=self.device)
        self.target[mask] = torch.as_tensor(self.fork_target, dtype=self.target.dtype, device=self.device)
        self.target_steps[mask] = round(self.fork_elapsed/self.dt)
        self.dwell_steps[mask] = round(self.fork_dwell/self.dt)
        self.completed[mask] = self.fork_completed
        self.previous_distance[mask] = torch.linalg.vector_norm(self.target[mask]-self.center[mask, :2], dim=1)
        return self.observation()


def dwell_potential_reward(observation: torch.Tensor, next_observation: torch.Tensor,
                           terminated: torch.Tensor, gamma: float, scale: float) -> torch.Tensor:
    current = scale*observation[..., 67]
    future = scale*next_observation[..., 67]*~terminated
    return gamma*future-current


def save_fork(task: LocalGoalTask, path: Path) -> None:
    if path.exists():
        raise FileExistsError(path)
    body = task.body
    state = np.empty(mujoco.mj_stateSize(body.model, mujoco.mjtState.mjSTATE_INTEGRATION))
    mujoco.mj_getState(body.model, body.data, state, mujoco.mjtState.mjSTATE_INTEGRATION)
    np.savez_compressed(path, integration=state, qpos=body.data.qpos.copy(), qvel=body.data.qvel.copy(),
                        qacc_warmstart=body.data.qacc_warmstart.copy(), ctrl=body.data.ctrl.copy(),
                        qfrc_applied=body.data.qfrc_applied.copy(), xfrc_applied=body.data.xfrc_applied.copy(),
                        time=body.data.time, previous_action=body.action.copy(),
                        body_sha256=hashlib.sha256(body.xml.encode()).hexdigest(),
                        target=task.target.copy(), origin=task.origin.copy(), elapsed=task.target_elapsed,
                        dwell=task.dwell, completed=task.completed)


def restore_fork(task: FiniteLocalGoalTask, path: Path, goal_y_shift: float = 0.) -> np.ndarray:
    task.reset_state.restore(task.body)
    with np.load(path) as state:
        task.target = state['target'].copy()
        task.origin = state['origin'].copy()
        task.target_elapsed = float(state['elapsed'])
        task.dwell = float(state['dwell'])
        task.completed = int(state['completed'])
    if goal_y_shift:
        task.target[1] += goal_y_shift
        task.dwell = 0.
    task.previous_distance = float(np.linalg.norm(task.target-task.body.center[:2]))
    task.done = False
    return task.observation()
