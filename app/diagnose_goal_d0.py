"""Frozen normal-gravity control menu; no policy or body changes."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from .evaluate_goal import file_sha256
from .free_body import FreeBody, JOINT_NAMES, ROOT
from .goal_checkpoint import goal_contract

STATE = mujoco.mjtState.mjSTATE_INTEGRATION
SOURCES = ('app/diagnose_goal_d0.py', 'app/free_body.py', 'app/goal_task.py',
           'app/goal_checkpoint.py', 'app/diagnose_goal_distribution.py')


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def vector(entries: dict[str, float]) -> np.ndarray:
    result = np.zeros(18)
    for name, value in entries.items():
        result[JOINT_NAMES.index(name + '_joint')] = value
    return result


def frozen_menu() -> dict:
    fold = []
    for phase in (0, 1):
        fold.append(vector({f'{side}_{joint}': value for side in ('left', 'right')
                            for joint, value in (('hip_pitch', -1. if phase == 0 else 1.),
                                                  ('knee', 1. if phase == 0 else 0.),
                                                  ('shoulder_pitch', 1. if phase == 0 else -1.),
                                                  ('elbow', .5 if phase == 0 else 1.))}).tolist())
    alternate = []
    for phase in (0, 1):
        entries = {}
        for side, active in (('left', phase == 0), ('right', phase == 1)):
            for joint, val in (('hip_pitch', -1. if active else 1.), ('knee', 1. if active else 0.),
                               ('shoulder_pitch', 1. if active else -1.), ('elbow', 1. if active else 0.)):
                entries[f'{side}_{joint}'] = val
        alternate.append(vector(entries).tolist())
    histories = [dict(id='standing', seconds=0., pulse=[0.] * 18),
                 dict(id='zero_fall', seconds=3., pulse=[0.] * 18),
                 dict(id='roll_pulse', seconds=3., pulse=vector({'left_hip_roll': 4., 'right_hip_roll': -4.}).tolist()),
                 dict(id='pitch_pulse', seconds=3., pulse=vector({'left_hip_pitch': 4., 'right_hip_pitch': 4.}).tolist())]
    templates = dict(fold_extend=fold, alternate_sides=alternate)
    trials = []
    for history in histories:
        trials.append(dict(id=history['id'] + '_zero', state=history['id'], template='zero', amplitude=0., hold_seconds=.1))
        for name in templates:
            for amplitude in (1., 2., 4., 8.):
                for hold in (.1, .4, 1.):
                    trials.append(dict(id=f"{history['id']}_{name}_{int(amplitude)}_{hold}", state=history['id'],
                                       template=name, amplitude=amplitude, hold_seconds=hold))
    return dict(schema='goal_control_d0', version=1, histories=histories, templates=templates, trials=trials,
                duration_seconds=4., target_axis_world=[1., 0.], target_distance_m=2.,
                progress_margin_m=.1, penetration_limit_m=.030, joint_limit_error_rad=.15,
                simulation_dt=.002, control_dt=.02, joint_names=JOINT_NAMES)


def get_state(body: FreeBody) -> np.ndarray:
    result = np.empty(mujoco.mj_stateSize(body.model, STATE))
    mujoco.mj_getState(body.model, body.data, result, STATE)
    return result


def classify(body: FreeBody) -> str:
    up_world_in_body = body.rotation[2]
    if up_world_in_body[2] >= .7:
        return 'upright'
    if up_world_in_body[2] <= -.7:
        return 'inverted'
    axis = int(np.argmax(np.abs(up_world_in_body[:2])))
    return ('front' if axis == 0 else 'left') if up_world_in_body[axis] > 0 else ('back' if axis == 0 else 'right')


def measured_step(body: FreeBody, action: np.ndarray) -> dict:
    """Mirror FreeBody.step and measure each physical substep before clipping."""
    body.action = np.clip(np.asarray(action, dtype=float), -8., 8.)
    raw_target = body.home + body.action_scale * body.action
    target = np.clip(raw_target, body.ranges[:, 0], body.ranges[:, 1])
    saturated = np.zeros(18, dtype=int)
    ground = set()
    penetration = limit_error = work = 0.
    reason = ''
    substeps = 0
    for _ in range(body.decimation):
        torque = body.kp * (target - body.data.qpos[body.qadr]) - body.kd * body.data.qvel[body.vadr]
        saturated += np.abs(torque) >= body.limits - 1e-6
        body.data.ctrl[body.actuators] = np.clip(torque, -body.limits, body.limits)
        mujoco.mj_step1(body.model, body.data)
        contact = body.contacts()
        penetration = max(penetration, contact.penetration)
        limit_error = max(limit_error, contact.limit_error)
        ground.update(contact.ground_bodies)
        mujoco.mj_fwdActuation(body.model, body.data)
        work += float(np.abs(body.data.qfrc_actuator[body.vadr] * body.data.qvel[body.vadr]).sum()) * body.model.opt.timestep
        mujoco.mj_step2(body.model, body.data)
        substeps += 1
        post = body.contacts()
        limit_error = max(limit_error, post.limit_error)
        if (not np.isfinite(body.data.qpos).all() or not np.isfinite(body.data.qvel).all()
                or not np.isfinite(body.data.qacc).all() or any(body.data.warning.number)):
            reason = 'numerical_or_warning'
        elif penetration > .030:
            reason = 'penetration'
        elif limit_error > .15:
            reason = 'joint_limit'
        if reason:
            break
    body.refresh()
    final_contact = body.contacts()
    penetration = max(penetration, final_contact.penetration)
    limit_error = max(limit_error, final_contact.limit_error)
    ground.update(final_contact.ground_bodies)
    if not reason and penetration > .030:
        reason = 'penetration'
    if not reason and limit_error > .15:
        reason = 'joint_limit'
    body.work += work
    return dict(work_j=work, max_penetration_m=penetration, max_limit_error_rad=limit_error,
                saturated=saturated, substeps=substeps, ground=sorted(ground), reason=reason,
                target=target, clipped=np.abs(target - raw_target) > 1e-10)


def run_trajectory(body: FreeBody, actions: np.ndarray, deadline: float) -> tuple[dict, dict[str, np.ndarray]]:
    centers, qpos, qvel = [body.center], [body.data.qpos.copy()], [body.data.qvel.copy()]
    times = [0.]
    work, saturation, targets, clipped, contacts = [], [], [], [], []
    substeps = []
    peak_penetration = peak_limit = 0.
    reason = ''
    body_names = [mujoco.mj_id2name(body.model, mujoco.mjtObj.mjOBJ_BODY, i) or 'world' for i in range(body.model.nbody)]
    for action in actions:
        if time.time() >= deadline:
            reason = 'budget'
            break
        result = measured_step(body, action)
        centers.append(body.center); qpos.append(body.data.qpos.copy()); qvel.append(body.data.qvel.copy())
        times.append(times[-1] + result['substeps'] * body.model.opt.timestep)
        work.append(result['work_j']); saturation.append(result['saturated']); targets.append(result['target'])
        clipped.append(result['clipped']); contacts.append([name in result['ground'] for name in body_names])
        substeps.append(result['substeps'])
        peak_penetration = max(peak_penetration, result['max_penetration_m'])
        peak_limit = max(peak_limit, result['max_limit_error_rad'])
        if result['reason']:
            reason = result['reason']
            break
    arrays = dict(time=np.asarray(times), center=np.asarray(centers), qpos=np.asarray(qpos), qvel=np.asarray(qvel),
                  work_j=np.asarray(work), saturation_substeps=np.asarray(saturation), substeps=np.asarray(substeps),
                  target=np.asarray(targets), target_clipped=np.asarray(clipped), ground_contact=np.asarray(contacts),
                  command=actions[:len(work)])
    elapsed = float(times[-1])
    initial_center = centers[0]
    target = initial_center[:2] + [2., 0.]
    start_late = int(np.argmin(np.abs(arrays['time'] - 1.)))
    late_progress = (float(np.linalg.norm(target - centers[start_late][:2]) - np.linalg.norm(target - centers[-1][:2]))
                     if elapsed >= 4. - 1e-7 else None)
    row = dict(complete=not reason and len(work) == len(actions), reason=reason, duration_s=elapsed,
               work_j=float(sum(work)), max_penetration_m=peak_penetration, max_limit_error_rad=peak_limit,
               total_progress_m=float(np.linalg.norm(target - initial_center[:2]) - np.linalg.norm(target - centers[-1][:2])),
               late_progress_m=late_progress, signed_x_displacement_m=float(centers[-1][0] - initial_center[0]),
               path_m=float(np.linalg.norm(np.diff(arrays['center'][:, :2], axis=0), axis=1).sum()),
               ground_bodies=sorted({body_names[i] for mask in contacts for i, flag in enumerate(mask) if flag}),
               saturation_fraction=(np.sum(saturation, axis=0) / max(1, sum(substeps))).tolist() if work else [0.] * 18,
               body_names=body_names)
    return row, arrays


def run_controls(output: Path, deadline: float) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    menu = frozen_menu()
    write_json(output / 'menu.json', menu)
    body = FreeBody()
    write_json(output / 'contract.json', asdict(goal_contract(body)))
    source_hashes = {name: file_sha256(ROOT / name) for name in SOURCES}
    states = {}
    histories = []
    for history in menu['histories']:
        body.reset()
        actions = np.zeros((round(history['seconds'] / body.dt), 18))
        if len(actions):
            actions[:30] = history['pulse']
        row, trace = run_trajectory(body, actions, deadline)
        if history['seconds'] == 0:
            row['complete'] = True
        trace['integration_state'] = get_state(body)
        trace['previous_action'] = body.action.copy()
        name = history['id']
        path = output / f'state_{name}.npz'
        np.savez_compressed(path, **trace)
        row.update(id=name, state_sha256=file_sha256(path), category=classify(body), com_height_m=float(body.center[2]))
        row['low_posture'] = row['com_height_m'] < .35 and row['category'] != 'upright'
        states[name] = (trace['integration_state'], trace['previous_action'], row)
        histories.append(row)
    write_json(output / 'states.json', dict(states=histories, menu_sha256=file_sha256(output / 'menu.json'), source_sha256=source_hashes))
    rows = []
    for trial in menu['trials']:
        state, action, metadata = states[trial['state']]
        if not metadata['complete']:
            rows.append(dict(**trial, complete=False, reason='invalid_initial_state', late_progress_m=None))
            continue
        if time.time() >= deadline:
            break
        body.reset()
        mujoco.mj_setState(body.model, body.data, state, STATE)
        body.action[:] = action
        body.refresh()
        commands = np.zeros((200, 18))
        if trial['template'] != 'zero':
            phase = (np.arange(200) // round(trial['hold_seconds'] / body.dt)) % 2
            commands = np.asarray(menu['templates'][trial['template']])[phase] * trial['amplitude']
        row, trace = run_trajectory(body, commands, deadline)
        np.savez_compressed(output / (trial['id'] + '.npz'), **trace)
        row.update(trial)
        row['trace_sha256'] = file_sha256(output / (trial['id'] + '.npz'))
        rows.append(row)
        write_json(output / 'progress.json', dict(trials=rows, elapsed_trials=len(rows)))
        print(json.dumps({k: row[k] for k in ('id', 'complete', 'reason', 'late_progress_m')}, allow_nan=False), flush=True)
    references = {row['state']: row for row in rows if row['template'] == 'zero'}
    positive_categories = set()
    for row in rows:
        meta = states[row['state']][2]
        reference = references.get(row['state'])
        row['late_extra_progress_m'] = None
        row['positive'] = False
        if row['complete'] and reference and reference['complete']:
            row['late_extra_progress_m'] = row['late_progress_m'] - reference['late_progress_m']
            row['positive'] = bool(meta['low_posture'] and row['late_extra_progress_m'] >= .1)
            if row['positive']:
                positive_categories.add(meta['category'])
    categories = sorted({row['category'] for row in histories if row['complete'] and row['low_posture']})
    branch = ('budget_stop' if time.time() >= deadline or len(rows) != 100 else
              'local_control_positive' if len(positive_categories) >= 2 else
              'coverage_or_local_only' if len(categories) < 2 or positive_categories else 'route_uncertainty_stop')
    report = dict(branch=branch, trials=rows, states=histories, low_posture_categories=categories,
                  positive_categories=sorted(positive_categories), trial_count=len(rows),
                  menu_sha256=file_sha256(output / 'menu.json'), states_sha256=file_sha256(output / 'states.json'),
                  source_sha256=source_hashes)
    write_json(output / 'summary.json', report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-seconds', type=float, default=3600.)
    parser.add_argument('--checkpoint-root', type=Path,
                        default=ROOT / 'runs/goal_local_campaign_20261011/screen',
                        help='Directory containing baseline_404/505/606/last.pt from the matched screen')
    args = parser.parse_args()
    if not 0 < args.max_seconds <= 3600:
        raise ValueError('Use a remaining wall budget in (0, 3600] seconds')
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.time()
    deadline = started + args.max_seconds
    write_json(args.output / 'execution.json', dict(started_unix=started, deadline_unix=deadline, max_seconds=args.max_seconds))
    controls = run_controls(args.output / 'controls', deadline)
    from .diagnose_goal_distribution import run_distribution
    distribution = run_distribution(args.output / 'distribution', deadline, args.checkpoint_root)
    branch = controls['branch']
    if not distribution['complete']:
        branch = 'budget_stop' if time.time() >= deadline else 'engineering_block'
    write_json(args.output / 'summary.json', dict(branch=branch, controls_branch=controls['branch'],
               control_trials=controls['trial_count'], positive_categories=controls['positive_categories'],
               distribution_transitions=distribution['transitions'], distribution_complete=distribution['complete'],
               elapsed_seconds=time.time()-started, optimizer_updates=0,
               controls_summary_sha256=file_sha256(args.output / 'controls/summary.json'),
               distribution_summary_sha256=file_sha256(args.output / 'distribution/summary.json')))


if __name__ == '__main__':
    main()
