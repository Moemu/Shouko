"""Finite goal states with fixed world categories and explicit actor initialization."""
from dataclasses import dataclass
import json
from pathlib import Path
import shutil

import numpy as np
import torch

from .braking_task import FiniteLocalGoalEnv
from .evaluate_goal import file_sha256
from .free_body import FreeBody
from .goal_checkpoint import CoverageGoalContext, LocalGoalContext, goal_contract, read_goal_payload
from .gpu_goal import GPUGoalEnv
from .local_goal import LocalResetState
from .mlp_policy import MLPPolicy
from .diagnose_goal_d0 import write_json


STATE_IDS = ('natural', '1401_100', '1401_125', '1402_100', '1402_125', '1403_100', '1403_125', '1402_145')
ELAPSED_STEPS = (0, 100, 125, 100, 125, 100, 125, 145)
CATEGORIES = ('original', 'middle', 'near')
CATEGORY_FIELDS = ('transitions', 'resets', 'arrivals', 'timeouts', 'arena', 'progress_sum_m', 'backtracking_m', 'work_j')


def prepare_pool(screen: Path, braking: Path, output: Path) -> None:
    forks = json.loads((braking/'diagnostic/forks/manifest.json').read_text(encoding='utf-8'))['forks']
    old_prereg = json.loads((braking/'transfer/preregistration.json').read_text(encoding='utf-8'))
    natural = braking/'transfer/reset.npz'
    if (file_sha256(natural) != old_prereg['target_state_sha256']
            or file_sha256(screen/'reset.npz') != old_prereg['original_reset_sha256']):
        raise ValueError('Original finite reset differs from its source evidence')
    with np.load(natural) as full, np.load(screen/'reset.npz') as original:
        for name in ('qpos', 'qvel', 'qacc_warmstart', 'ctrl', 'previous_action'):
            np.testing.assert_array_equal(full[name], original[name])
        assert float(full['elapsed']) == 0. and float(full['dwell']) == 0.
        np.testing.assert_allclose(full['target']-full['origin'], [2., 0.], atol=1e-12, rtol=0)
    output.mkdir(parents=True, exist_ok=False)
    entries = []
    for identity in STATE_IDS:
        path = natural if identity == 'natural' else braking/'diagnostic/forks'/next(f['filename'] for f in forks if f['id'] == identity)
        if identity != 'natural':
            assert file_sha256(path) == next(f['sha256'] for f in forks if f['id'] == identity)
        filename = identity+'.npz'
        shutil.copyfile(path, output/filename)
        entries.append(dict(id=identity, filename=filename, sha256=file_sha256(output/filename)))
    write_json(output/'manifest.json', dict(schema='approach_holding_pool_v1', states=entries,
                                           original_source_reset_sha256=file_sha256(screen/'reset.npz'),
                                           fork_manifest_sha256=file_sha256(braking/'diagnostic/forks/manifest.json')))


@dataclass(frozen=True)
class CoverageState:
    path: Path
    physical: LocalResetState
    origin: np.ndarray
    target: np.ndarray
    elapsed_steps: int
    dwell_steps: int


def load_pool(pool: Path) -> tuple[CoverageState, ...]:
    manifest = json.loads((pool/'manifest.json').read_text(encoding='utf-8'))
    if manifest['schema'] != 'approach_holding_pool_v1' or tuple(s['id'] for s in manifest['states']) != STATE_IDS:
        raise ValueError('Expected the frozen eight-state approach/holding pool')
    states = []
    for index, entry in enumerate(manifest['states']):
        if entry['filename'] != entry['id']+'.npz':
            raise ValueError('Pool state must use its canonical local filename')
        path = pool/entry['filename']
        if file_sha256(path) != entry['sha256']:
            raise ValueError('Pool state SHA-256 mismatch')
        with np.load(path) as z:
            elapsed, dwell = float(z['elapsed']), float(z['dwell'])
            if not (0 <= elapsed < 5. and 0 <= dwell < 1. and int(z['completed']) == 0):
                raise ValueError('Pool must contain unfinished finite episodes')
            if abs(elapsed-.02*ELAPSED_STEPS[index]) > 1e-9 or dwell != 0.:
                raise ValueError('Pool elapsed/dwell differs from the frozen schedule')
            if (not np.isfinite(z['origin']).all() or not np.isfinite(z['target']).all()
                    or not np.allclose(z['target']-z['origin'], [2., 0.], atol=1e-12, rtol=0)):
                raise ValueError('Pool must retain the original +x two-metre target')
            if index == 0:
                body = FreeBody()
                body.data.qpos[:], body.data.qvel[:] = z['qpos'], z['qvel']
                body.refresh()
                if not np.allclose(z['origin'], body.center[:2], atol=1e-12, rtol=0):
                    raise ValueError('Natural origin must match the original centre')
            if index and (not np.array_equal(z['origin'], states[0].origin) or not np.array_equal(z['target'], states[0].target)):
                raise ValueError('Pool states must share the original goal and origin')
            states.append(CoverageState(path, LocalResetState.load(path), z['origin'].copy(), z['target'].copy(),
                                        round(elapsed/.02), round(dwell/.02)))
    return tuple(states)


@dataclass(frozen=True)
class CoverageGoalSetup:
    pool: Path
    source_checkpoint: Path
    source_state: Path
    arm: str

    def context(self) -> CoverageGoalContext:
        manifest = json.loads((self.pool/'manifest.json').read_text(encoding='utf-8'))
        if manifest['original_source_reset_sha256'] != file_sha256(self.source_state):
            raise ValueError('Pool original source reset mismatch')
        with np.load(self.pool/'natural.npz') as natural, np.load(self.source_state) as source:
            for name in ('qpos', 'qvel', 'qacc_warmstart', 'ctrl', 'qfrc_applied', 'xfrc_applied', 'previous_action', 'time', 'body_sha256'):
                if not np.array_equal(natural[name], source[name]):
                    raise ValueError('Pool natural physics differs from its source reset')
        return CoverageGoalContext(file_sha256(self.pool/'manifest.json'), file_sha256(self.source_checkpoint),
                                   file_sha256(self.source_state), self.arm)

    def initialize_actor(self, policy: MLPPolicy, body: FreeBody) -> None:
        raw = torch.load(self.source_checkpoint, map_location='cpu', weights_only=True)
        if raw.get('schema') != 'yumi_local_goal_ppo':
            raise ValueError('Expected a source local-goal actor')
        context = LocalGoalContext(**raw['local_context'])
        if context.reset_sha256 != file_sha256(self.source_state) or context.rho != 0.:
            raise ValueError('Approach actor source state or rho mismatch')
        payload = read_goal_payload(self.source_checkpoint, goal_contract(body), 'cpu', local_context=context)
        if payload['settings'].get('training_work_cost', .001) != 0.:
            raise ValueError('Expected a zero-cost approach source')
        policy.load_state_dict(payload['policy'], strict=True)


class CoverageGoalEnv(FiniteLocalGoalEnv):
    def __init__(self, worlds: int, seed: int, setup: CoverageGoalSetup) -> None:
        self.local_context = setup.context()
        self.states = load_pool(setup.pool)
        if setup.arm == 'mixed' and (worlds < 4 or worlds % 4):
            raise ValueError('Mixed coverage needs a multiple of four worlds')
        device = torch.device('cuda:0')
        self.categories = torch.full((worlds,), 2, device=device, dtype=torch.long)
        if setup.arm == 'mixed':
            self.categories[:worlds//2] = 0
            self.categories[worlds//2:3*worlds//4] = 1
        self.middle_offset = torch.arange(worlds, device=device)-worlds//2
        self.reset_round = torch.zeros(worlds, device=device, dtype=torch.long)
        self.state_index = torch.full_like(self.reset_round, 7)
        self.initializing = True
        GPUGoalEnv.__init__(self, worlds, seed)
        self.initializing = False

    @torch.no_grad()
    def reset(self, mask: torch.Tensor) -> torch.Tensor:
        if mask.shape != (self.worlds,) or mask.dtype != torch.bool or mask.device != self.device:
            raise ValueError('Reset requires a CUDA boolean mask for all worlds')
        if not self.initializing:
            self.reset_round[mask] += 1
        selected = torch.where(self.categories == 0, 0,
                               torch.where(self.categories == 1, 1+(self.middle_offset+self.reset_round)%6, 7))
        self.state_index[mask] = selected[mask]
        for index, state in enumerate(self.states):
            subset = mask & (selected == index)
            if not bool(subset.any()):
                continue
            self.restore_physics(subset, state.physical)
            self.origin[subset] = torch.as_tensor(state.origin, dtype=self.origin.dtype, device=self.device)
            self.target[subset] = torch.as_tensor(state.target, dtype=self.target.dtype, device=self.device)
            self.target_steps[subset] = state.elapsed_steps
            self.dwell_steps[subset] = state.dwell_steps
            self.completed[subset] = 0
            self.done[subset] = False
            self.previous_distance[subset] = torch.linalg.vector_norm(self.target[subset]-self.center[subset, :2], dim=1)
        self.target_limit = 250
        return self.observation()
