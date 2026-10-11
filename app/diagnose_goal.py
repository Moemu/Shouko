"""Bounded control mapping, reward accounting and exploration probes; no optimization."""
import argparse
import json
from pathlib import Path

import numpy as np

from .evaluate_goal import file_sha256
from .free_body import FreeBody, JOINT_NAMES, ROOT
from .goal_task import GoalTask


def joint_probes() -> list[dict]:
    rows = []
    for index, name in enumerate(JOINT_NAMES):
        for sign in (-1., 1.):
            body = FreeBody()
            # Isolate the control mapping from gravitational collapse in this diagnostic copy only.
            body.model.opt.gravity[:] = 0
            action = np.zeros(18)
            action[index] = sign
            before = body.data.qpos[body.qadr].copy()
            target = np.clip(body.home + body.action_scale * action, body.ranges[:, 0], body.ranges[:, 1])
            work = sum(body.step(action) for _ in range(10))
            displacement = body.data.qpos[body.qadr] - before
            target_delta = float(target[index] - before[index])
            actual = float(displacement[index])
            blocked = abs(target_delta) < 1e-12
            applied = body.data.qfrc_actuator[body.vadr]
            control = body.data.ctrl[body.actuators]
            force_error = float(np.max(np.abs(applied - control)))
            if force_error > 1e-9 or not np.isfinite(body.data.qpos).all():
                raise AssertionError('Named actuator force mapping or finite state failed')
            if not blocked and actual * target_delta <= 0:
                raise AssertionError(f'Joint moved against its target: {name}, {sign}')
            rows.append(dict(joint=name, action=sign, target_delta_rad=target_delta,
                             actual_delta_rad=actual, target_blocked_by_limit=blocked,
                             final_force_control_error_nm=force_error, work_j=work,
                             max_other_joint_displacement_rad=float(np.max(np.abs(np.delete(displacement, index))))))
    return rows


def noise_probe(group: str, std: float, seed: int) -> dict:
    body = FreeBody()
    task = GoalTask(body, seed=seed)
    rng = np.random.default_rng(seed)
    totals = dict(work_j=0., progress_m=0., after_3s_progress_m=0., reward=0.,
                  max_penetration_m=0., max_joint_limit_error_rad=0., max_reward_error=0.,
                  max_distance_progress_error_m=0., target_clipped_joint_steps=0,
                  force_saturated_joint_steps=0, action_clipped_joint_steps=0)
    steps = 0
    while not task.done:
        action = rng.normal(0., std, 18)
        if group == 'legs':
            action[12:] = 0.
        elif group == 'arms':
            action[:12] = 0.
        old_target, old_distance = task.target.copy(), task.previous_distance
        _, result = task.step(action)
        if result.reason == 'nonfinite' or any(warning.number for warning in body.data.warning):
            raise FloatingPointError(f'Numerical warning in noise probe: {group}, {std}, {seed}')
        expected = result.progress + float(result.reached) - GoalTask.work_cost * result.work_j
        distance = float(np.linalg.norm(old_target - body.center[:2]))
        totals['max_reward_error'] = max(totals['max_reward_error'], abs(result.reward - expected))
        totals['max_distance_progress_error_m'] = max(totals['max_distance_progress_error_m'], abs(result.progress - (old_distance - distance)))
        totals['reward'] += result.reward
        totals['progress_m'] += result.progress
        totals['work_j'] += result.work_j
        steps += 1
        if steps > 150:
            totals['after_3s_progress_m'] += result.progress
        contact = body.last_contacts
        totals['max_penetration_m'] = max(totals['max_penetration_m'], contact.penetration)
        totals['max_joint_limit_error_rad'] = max(totals['max_joint_limit_error_rad'], contact.limit_error)
        requested = body.home + body.action_scale * body.action
        totals['target_clipped_joint_steps'] += int(np.count_nonzero((requested < body.ranges[:, 0]) | (requested > body.ranges[:, 1])))
        totals['force_saturated_joint_steps'] += int(np.count_nonzero(np.abs(body.data.qfrc_actuator[body.vadr]) >= .999 * body.limits))
        totals['action_clipped_joint_steps'] += int(np.count_nonzero(np.abs(action) > 8))
    assert totals['max_reward_error'] < 1e-12 and totals['max_distance_progress_error_m'] < 1e-12
    return dict(group=group, std=std, seed=seed, steps=steps, reason=result.reason,
                arrivals=task.completed, finite=True, final_com_m=body.center.tolist(),
                saturation_sampling='Final physics substep of each control step', **totals)


def run(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    pulses = joint_probes()
    rows = [noise_probe('all', 0., 741)]
    for std in (.35, .70):
        for group in ('legs', 'arms', 'all'):
            for seed in (741, 742, 743):
                row = noise_probe(group, std, seed)
                rows.append(row)
                print(json.dumps(row), flush=True)

    def acceptable(std: float) -> bool:
        selected = [row for row in rows if row['group'] == 'all' and row['std'] == std]
        return len(selected) == 3 and all(row['max_penetration_m'] <= .030 and row['max_joint_limit_error_rad'] <= .15 for row in selected)

    candidate = .70
    if not acceptable(candidate):
        candidate = .50
        for seed in (741, 742, 743):
            rows.append(noise_probe('all', candidate, seed))
    report = dict(joint_probes=pulses, trajectory_probes=rows,
                  exploration_std=candidate, exploration_passed=acceptable(candidate),
                  body_xml_sha256=file_sha256(ROOT / 'app/yumi_description/yumi.xml'),
                  pulse_gravity_m_s2=[0., 0., 0.], trajectory_gravity_m_s2=[0., 0., -9.81])
    names = ('app/diagnose_goal.py', 'app/free_body.py', 'app/goal_task.py')
    report['source_sha256'] = {name: file_sha256(ROOT / name) for name in names}
    (output / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('PASS: control mapping and reward accounting')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
