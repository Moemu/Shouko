"""Evaluate a frozen near-field checkpoint against its original target and deadline."""
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .braking_task import HoldingGoalTask, HoldingGoalEnv, HoldingGoalSetup, FiniteTaskContract
from .diagnose_goal_d0 import measured_step, write_json
from .evaluate_goal import file_sha256
from .free_body import FreeBody, ROOT
from .goal_checkpoint import HoldingGoalContext, goal_contract, read_goal_payload
from .mlp_policy import MLPPolicy


@torch.inference_mode()
def evaluate(checkpoint: Path, state_path: Path, output: Path, backend: str, *,
             transfer_state: Path | None = None) -> dict:
    if backend not in ('native', 'cuda'):
        raise ValueError('Expected native or cuda backend')
    output.mkdir(parents=True, exist_ok=False)
    raw = torch.load(checkpoint, map_location='cpu', weights_only=True)
    if raw.get('schema') != 'yumi_holding_goal_ppo':
        raise ValueError('Expected a versioned holding checkpoint')
    context = HoldingGoalContext(**raw['local_context'])
    if context.reset_sha256 != file_sha256(state_path):
        raise ValueError('Holding reset state mismatch')
    device = 'cpu' if backend == 'native' else 'cuda'
    payload = read_goal_payload(checkpoint, goal_contract(FreeBody()), device, local_context=context)
    policy = MLPPolicy(69, 18, device=device).eval()
    policy.load_state_dict(payload['policy'])
    body = FreeBody()
    evaluation_state = transfer_state if transfer_state is not None else state_path
    with np.load(evaluation_state) as state:
        start = float(state['elapsed'])
        target, origin = state['target'].copy(), state['origin'].copy()
    if backend == 'native':
        task = HoldingGoalTask(body, evaluation_state)
        observation = torch.from_numpy(task.observation())[None]
        def physical():
            return body.data.qpos.copy(), body.data.qvel.copy(), body.center.copy(), body.velocity.copy()
    else:
        env = HoldingGoalEnv(1, 0, HoldingGoalSetup(evaluation_state, context.potential_scale))
        observation = env.observation()
        def physical():
            return tuple(x[0].cpu().numpy().copy() for x in (env.qpos, env.qvel, env.center, env.velocity))
    q, v, c, velocity = physical()
    rows = dict(qpos=[q], qvel=[v], center=[c], velocity=[velocity], time=[start],
                observation=[observation[0].cpu().numpy().copy()])
    steps = {k: [] for k in ('raw_action', 'executed_action', 'work_j', 'progress', 'reward', 'dwell',
                              'penetration', 'joint_limit_error', 'terminated', 'truncated', 'control_step_complete')}
    reason, reached = '', False
    for i in range(round((5.-start)/.02)):
        action, _ = policy(observation)
        if backend == 'native':
            meter = measured_step(body, action[0].numpy())
            complete = not bool(meter['reason'])
            elapsed = rows['time'][-1]+meter['substeps']*.002
            if complete:
                result = task.evaluate(body.center, body.velocity, meter['work_j'])
                work, progress, reward = result.work_j, result.progress, result.reward
                done, reached, reason = task.done, result.reached, result.reason
                terminated, truncated = result.terminated, result.truncated
            else:
                work = meter['work_j']
                progress = task.previous_distance-float(np.linalg.norm(task.target-body.center[:2]))
                reward = progress-.001*work
                done, reached, reason = True, False, meter['reason']
                terminated = truncated = False
            observation = torch.from_numpy(task.observation())[None]
            penetration, limit = meter['max_penetration_m'], meter['max_limit_error_rad']
            executed, dwell = body.action.copy(), task.dwell
        else:
            observation, result = env.step(action)
            work, progress, reward = (float(x[0]) for x in (result.work_j, result.progress, result.reward))
            penetration = float(env.peak_penetration[0])
            angles = env.qpos[0, env.qadr].cpu().numpy()
            limit = max(0., float(np.maximum(body.ranges[:, 0]-angles, angles-body.ranges[:, 1]).max()))
            executed, dwell = env.actions[0].cpu().numpy().copy(), float(env.dwell_steps[0])*.02
            done, reached = bool(env.done[0]), bool(result.reached[0])
            terminated, truncated = bool(result.terminated[0]), bool(result.truncated[0])
            reason = ('numerical' if bool(result.numerical[0]) else 'complete' if reached else
                      'timeout' if bool(result.timeout[0]) else 'arena' if bool(result.arena[0]) else '')
            complete, elapsed = True, start+(i+1)*.02
        q, v, c, velocity = physical()
        for key, val in dict(qpos=q, qvel=v, center=c, velocity=velocity, time=elapsed,
                             observation=observation[0].cpu().numpy().copy()).items():
            rows[key].append(val)
        for key, val in dict(raw_action=action[0].cpu().numpy().copy(), executed_action=executed,
                             work_j=work, progress=progress, reward=reward, dwell=dwell,
                             penetration=penetration, joint_limit_error=limit,
                             terminated=terminated, truncated=truncated, control_step_complete=complete).items():
            steps[key].append(val)
        if penetration > .030 or limit > .15:
            reason = 'physical_guard'
        if reason not in ('', 'complete', 'timeout', 'arena') or done:
            break
    trace = {k: np.asarray(v) for k, v in {**rows, **steps}.items()}
    np.savez_compressed(output/'trace.npz', **trace, target=target, origin=origin)
    distance = np.linalg.norm(trace['center'][:, :2]-target, axis=1)
    safe = reason in ('complete', 'timeout')
    report = dict(schema='holding_goal_evaluation', backend=backend, local_context=asdict(context),
                  checkpoint_sha256=file_sha256(checkpoint), state_sha256=file_sha256(evaluation_state),
                  trace_sha256=file_sha256(output/'trace.npz'), start_seconds=start,
                  end_seconds=float(trace['time'][-1]), reason=reason, safe=safe, reached=reached,
                  passed=bool(safe and reached and max(steps['dwell']) >= 1.-1e-9),
                  max_dwell_seconds=max(steps['dwell']), final_distance_m=float(distance[-1]),
                  final_speed_m_s=float(np.linalg.norm(velocity[:2])),
                  net_progress_m=float(distance[0]-distance[-1]), work_j=float(sum(steps['work_j'])),
                  max_penetration_m=max(steps['penetration']), max_joint_limit_error_rad=max(steps['joint_limit_error']),
                  source_sha256={name: file_sha256(ROOT/name) for name in (
                      'app/evaluate_holding_goal.py', 'app/braking_task.py', 'app/local_goal.py',
                      'app/free_body.py', 'app/gpu_goal.py', 'app/goal_task.py', 'app/goal_checkpoint.py',
                      'app/mlp_policy.py', 'app/diagnose_goal_d0.py')})
    if transfer_state is not None:
        report['schema'] = 'holding_transfer_evaluation'
        report['source_context'] = report.pop('local_context')
        report['source_state_sha256'] = file_sha256(state_path)
        report['transfer_contract'] = asdict(FiniteTaskContract())
    write_json(output/'summary.json', report)
    return report
