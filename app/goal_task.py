"""Goal reaching semantics for the isolated free-movement body; no gait reward."""
from dataclasses import dataclass

import numpy as np
import mujoco

from .free_body import FreeBody

OBSERVATION_FIELDS = (
    ('target_body_m', 3), ('com_velocity_body_m_s', 3), ('angular_velocity_body_rad_s', 3),
    ('gravity_body', 3), ('joint_offset_rad', 18), ('joint_velocity_rad_s', 18),
    ('previous_action', 18), ('com_height_m', 1), ('dwell_fraction', 1), ('target_time_remaining_fraction', 1),
)
OBSERVATION_SIZE = sum(size for _, size in OBSERVATION_FIELDS)


@dataclass(frozen=True)
class GoalResult:
    reward: float
    progress: float
    work_j: float
    reached: bool
    terminated: bool
    truncated: bool
    reason: str


class GoalTask:
    radius = .3
    speed_limit = .2
    dwell_seconds = 1.
    target_seconds = 30.
    arena_radius = 10.
    targets_per_episode = 3
    target_distance = 2.
    arrival_reward = 1.
    work_cost = .001

    @classmethod
    def parameters(cls) -> dict[str, float | int]:
        return dict(radius=cls.radius, speed_limit=cls.speed_limit, dwell_seconds=cls.dwell_seconds,
                    target_seconds=cls.target_seconds, arena_radius=cls.arena_radius,
                    targets_per_episode=cls.targets_per_episode, target_distance=cls.target_distance,
                    arrival_reward=cls.arrival_reward, work_cost=cls.work_cost)

    def __init__(self, body: FreeBody, seed: int = 0) -> None:
        self.body = body
        self.rng = np.random.default_rng(seed)
        self.reset()

    def _new_target(self, center: np.ndarray) -> None:
        # Two metres from the initial centre always remains inside the ten-metre arena.
        angle = self.rng.uniform(-np.pi, np.pi)
        self.target = self.origin + self.target_distance * np.array([np.cos(angle), np.sin(angle)])
        # Consecutive targets must also require movement rather than another dwell.
        while np.linalg.norm(self.target - center[:2]) < self.target_distance - 1e-9:
            angle = self.rng.uniform(-np.pi, np.pi)
            self.target = self.origin + self.target_distance * np.array([np.cos(angle), np.sin(angle)])
        self.previous_distance = float(np.linalg.norm(self.target - center[:2]))
        self.dwell = 0.
        self.target_elapsed = 0.

    def reset(self) -> np.ndarray:
        self.body.reset()
        self.origin = self.body.center[:2]
        self.completed = 0
        self.done = False
        self._new_target(self.body.center)
        return self.observation()

    def observation(self) -> np.ndarray:
        body = self.body
        center = body.center
        # Rotate the horizontal world vector in full 3D; this stays defined upside down.
        target_vector = np.r_[self.target - center[:2], 0.]
        rotation = body.rotation
        angular = np.zeros(6)
        mujoco.mj_objectVelocity(body.model, body.data, mujoco.mjtObj.mjOBJ_XBODY, body.pelvis, angular, 1)
        values = np.concatenate((rotation.T @ target_vector, rotation.T @ body.velocity,
                                 angular[:3], rotation.T @ [0., 0., -1.],
                                 body.data.qpos[body.qadr] - body.home,
                                 body.data.qvel[body.vadr], body.action,
                                 [center[2], self.dwell / self.dwell_seconds,
                                  max(0., 1 - self.target_elapsed / self.target_seconds)]))
        return values.astype(np.float32)

    def evaluate(self, center: np.ndarray, velocity: np.ndarray, work_j: float) -> GoalResult:
        """Consume exactly one body control step, also usable for boundary tests."""
        if self.done:
            raise RuntimeError('Reset the completed goal episode before stepping')
        self.target_elapsed += self.body.dt
        if not (np.isfinite(center).all() and np.isfinite(velocity).all() and np.isfinite(work_j)):
            self.done = True
            return GoalResult(0., 0., 0., False, True, False, 'nonfinite')
        distance = float(np.linalg.norm(self.target - center[:2]))
        progress = self.previous_distance - distance
        self.previous_distance = distance
        eligible = distance <= self.radius and np.linalg.norm(velocity[:2]) <= self.speed_limit
        self.dwell = self.dwell + self.body.dt if eligible else 0.
        reached = self.dwell + 1e-9 >= self.dwell_seconds
        terminated = False
        truncated = False
        reason = ''
        if np.linalg.norm(center[:2] - self.origin) > self.arena_radius:
            truncated, reason = True, 'arena'
        elif reached:
            self.completed += 1
            terminated = self.completed >= self.targets_per_episode
            reason = 'complete' if terminated else 'target'
            if not terminated:
                self._new_target(center)
        elif self.target_elapsed + 1e-9 >= self.target_seconds:
            truncated, reason = True, 'timeout'
        self.done = terminated or truncated
        # Tentative weights: signed progress (m), arrival bonus, absolute mechanical work (J).
        reward = progress + (self.arrival_reward if reached and not truncated else 0.) - self.work_cost * work_j
        return GoalResult(reward, progress, work_j, reached, terminated, truncated, reason)

    def step(self, action: np.ndarray) -> tuple[np.ndarray, GoalResult]:
        if self.done:
            raise RuntimeError('Reset the completed goal episode before stepping')
        work = self.body.step(action)
        center, velocity = self.body.center, self.body.velocity
        numerical_warnings = (mujoco.mjtWarning.mjWARN_BADQPOS, mujoco.mjtWarning.mjWARN_BADQVEL,
                              mujoco.mjtWarning.mjWARN_BADQACC, mujoco.mjtWarning.mjWARN_BADCTRL)
        if (not np.isfinite(self.body.data.qpos).all() or not np.isfinite(self.body.data.qvel).all()
                or any(self.body.data.warning[warning].number for warning in numerical_warnings)):
            center = np.full(3, np.nan)
        result = self.evaluate(center, velocity, work)
        return self.observation(), result
