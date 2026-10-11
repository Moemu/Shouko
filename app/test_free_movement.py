"""Real local physics and task-contract checks; no training or released weights needed."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco
import numpy as np

from .free_body import FreeBody, JOINT_NAMES, ROOT, parallel_axis
from .goal_task import GoalTask, OBSERVATION_FIELDS, OBSERVATION_SIZE


def rigid_properties(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[float, np.ndarray, np.ndarray]:
    mass = float(model.body_mass.sum())
    center = (model.body_mass[:, None] * data.xipos).sum(0) / mass
    inertia = np.zeros((3, 3))
    for i in range(1, model.nbody):
        rotation = data.ximat[i].reshape(3, 3)
        inertia += rotation @ np.diag(model.body_inertia[i]) @ rotation.T
        inertia += model.body_mass[i] * parallel_axis(data.xipos[i] - center)
    return mass, center, inertia


def check_body(body: FreeBody) -> dict:
    baseline = mujoco.MjModel.from_xml_path(str(ROOT / 'app/yumi_description/scene.xml'))
    original = mujoco.MjData(baseline)
    original.qpos[2] = .962
    original.qpos[7:] = body.home[:12]
    mujoco.mj_forward(baseline, original)
    old = rigid_properties(baseline, original)
    new = rigid_properties(body.model, body.data)
    errors = [float(np.max(np.abs(a - b))) for a, b in zip(old, new)]
    assert errors[0] < 1e-9 and errors[1] < 1e-9 and errors[2] < 1e-8, errors
    for inertia in body.model.body_inertia[1:]:
        assert (inertia > 0).all() and max(inertia) <= sum(inertia) - max(inertia) + 1e-12
    assert body.model.nu == 18 and len(set(body.qadr)) == 18 and len(set(body.vadr)) == 18
    for joint, actuator in zip(body.joints, body.actuators):
        assert body.model.actuator_trnid[actuator, 0] == joint
    response = {}
    for index, name in enumerate(JOINT_NAMES):
        for sign in (-1, 1):
            body.reset()
            # Free-air one-substep response separates actuator direction from ground reactions.
            body.data.qpos[2] = 3.
            if name.endswith('elbow_joint'):
                body.data.qpos[body.qadr[index]] = .5
            body.refresh()
            body.data.ctrl.fill(0)
            body.data.ctrl[body.actuators[index]] = sign * 2.
            mujoco.mj_step(body.model, body.data)
            delta = float(body.data.qvel[body.vadr[index]])
            assert delta * sign > 0, (name, sign, delta)
            response[f'{name}:{sign}'] = delta
    for bad in (np.zeros(12), np.full(18, np.nan), np.full(18, np.inf)):
        try:
            body.step(bad)
        except ValueError:
            pass
        else:
            raise AssertionError('Invalid action accepted')
    body.reset()
    body.step(np.full(18, 1e6))
    assert (np.abs(body.data.qfrc_actuator[body.vadr]) <= body.limits + 1e-9).all()
    body.reset()
    return dict(total_mass_kg=new[0], mass_error_kg=errors[0], com_error_m=errors[1],
                inertia_error_kg_m2=errors[2], joint_responses_rad_s=response,
                torso_principal_inertia=body.model.body_inertia[body.pelvis].tolist())


def check_task(body: FreeBody) -> dict:
    task = GoalTask(body, seed=11)
    assert task.observation().shape == (OBSERVATION_SIZE,) == (69,)
    assert np.isfinite(task.observation()).all()
    center = body.center
    task.target = center[:2] + [2., 0.]
    task.previous_distance = 2.
    toward = center + [.1, 0., 0.]
    first = task.evaluate(toward, np.zeros(3), 0.)
    reverse = task.evaluate(center, np.zeros(3), 0.)
    assert np.isclose(first.progress, .1) and np.isclose(reverse.progress, -.1)
    assert np.isclose(first.reward + reverse.reward, 0.)
    energy = task.evaluate(center, np.zeros(3), 10.)
    assert np.isclose(energy.reward, -.01)
    # A transformed target must retain world distance in every posture, including upside down.
    for axis in np.eye(3):
        quaternion = np.r_[0., axis]
        body.data.qpos[3:7] = quaternion
        body.refresh()
        expected = np.r_[task.target - body.center[:2], 0.]
        assert np.allclose(body.rotation @ task.observation()[:3], expected, atol=1e-6)
    body.reset()
    task.reset()
    task.target = body.center[:2].copy()
    task.previous_distance = 0.
    slow = np.zeros(3)
    fast = np.array([.21, 0., 0.])
    for _ in range(30):
        assert not task.evaluate(body.center, slow, 0.).reached
    task.evaluate(body.center, fast, 0.)
    assert task.dwell == 0
    for _ in range(49):
        assert not task.evaluate(body.center, slow, 0.).reached
    arrived = task.evaluate(body.center, slow, 0.)
    assert arrived.reached and task.completed == 1 and not task.done
    switched = task.evaluate(body.center, slow, 0.)
    assert abs(switched.progress) < 1e-12 and not switched.reached
    assert np.linalg.norm(task.target - body.center[:2]) >= 2 - 1e-9
    # Repeat arrivals with synthetic measured states; no physics success is claimed.
    for expected in (2, 3):
        position = np.r_[task.target, body.center[2]]
        for _ in range(50):
            result = task.evaluate(position, slow, 0.)
        assert result.reached and task.completed == expected
    assert result.terminated and not result.truncated and result.reason == 'complete'
    task.reset()
    task.target_elapsed = 30 - body.dt
    timeout = task.evaluate(body.center, slow, 0.)
    assert timeout.truncated and not timeout.terminated and timeout.reason == 'timeout'
    task.reset()
    arena = task.evaluate(np.r_[task.origin + [10.1, 0], .1], slow, 0.)
    assert arena.truncated and arena.reason == 'arena'
    task.reset()
    invalid = task.evaluate(np.full(3, np.nan), slow, 0.)
    assert invalid.terminated and invalid.reason == 'nonfinite'
    task.reset()
    low = task.evaluate(np.r_[body.center[:2], .05], slow, 0.)
    assert not low.terminated and not low.truncated
    body.data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number = 1
    _, numerical = task.step(np.zeros(18))
    assert numerical.terminated and numerical.reason == 'nonfinite'
    return dict(observation_size=OBSERVATION_SIZE, observation_fields=OBSERVATION_FIELDS,
                forward_progress_m=first.progress, reverse_progress_m=reverse.progress,
                switch_progress_m=switched.progress, arrival_seconds=50 * body.dt,
                completed_targets=3, boundaries=['timeout', 'arena', 'nonfinite', 'solver_warning', 'low_pose_allowed'])


POSES = {'upright': np.array([1., 0., 0., 0.]),
         'front': np.array([np.sqrt(.5), 0., np.sqrt(.5), 0.]),
         'back': np.array([np.sqrt(.5), 0., -np.sqrt(.5), 0.]),
         'side': np.array([np.sqrt(.5), np.sqrt(.5), 0., 0.])}


def place_pose(body: FreeBody, quaternion: np.ndarray) -> None:
    body.reset()
    body.data.qpos[3:7] = quaternion
    body.data.qpos[2] = 3.
    body.refresh()
    # Use MuJoCo geom distance to put the lowest collidable geometry just above the floor.
    distances = [mujoco.mj_geomDistance(body.model, body.data, body.floor, geom, 10., None)
                 for geom in range(body.model.ngeom) if geom != body.floor]
    body.data.qpos[2] -= min(distances) - .002
    body.refresh()


def check_physics(body: FreeBody) -> dict:
    scenarios = {}
    rng = np.random.default_rng(123)
    for name, quaternion in POSES.items():
        place_pose(body, quaternion)
        task = GoalTask(body, seed=2)
        # GoalTask.reset intentionally resets the body; reapply the physical test posture.
        place_pose(body, quaternion)
        task.origin = body.center[:2]
        task._new_target(body.center)
        max_contacts = max_constraints = 0
        penetration = limit_error = 0.
        ground_bodies = set()
        # Ten seconds per posture, with moderate random joint targets in the last five seconds.
        started = time.perf_counter()
        for step in range(500):
            action = np.zeros(18) if step < 250 else rng.uniform(-1., 1., 18)
            observation, result = task.step(action)
            assert np.isfinite(observation).all() and np.isfinite(body.data.qacc).all()
            assert not result.terminated and not result.truncated, (name, step, result)
            metrics = body.last_contacts
            max_contacts = max(max_contacts, metrics.count)
            max_constraints = max(max_constraints, metrics.constraints)
            penetration = max(penetration, metrics.penetration)
            ground_bodies.update(metrics.ground_bodies)
            q = body.data.qpos[body.qadr]
            limit_error = max(limit_error, metrics.limit_error, float(np.max(np.maximum(body.ranges[:, 0] - q,
                                                                  q - body.ranges[:, 1]))))
        warnings = {mujoco.mjtWarning(i).name: int(w.number) for i, w in enumerate(body.data.warning) if w.number}
        assert not warnings, (name, warnings)
        assert penetration < .03, (name, penetration)
        assert limit_error < .1, (name, limit_error)
        assert max_contacts > 0 and body.work > 0
        if name != 'upright':
            assert any('ankle' not in part for part in ground_bodies), (name, ground_bodies)
        scenarios[name] = dict(simulated_seconds=500 * body.dt, wall_seconds=time.perf_counter() - started,
                               max_contacts=max_contacts, max_constraints=max_constraints,
                               max_penetration_m=penetration, max_joint_limit_error_rad=limit_error,
                               absolute_work_j=body.work, ground_bodies=sorted(ground_bodies), warnings=warnings)
    return scenarios


def check_policy(body: FreeBody) -> dict:
    import torch
    from .mlp_policy import MLPPolicy
    torch.manual_seed(11)
    task = GoalTask(body)
    policy = MLPPolicy(OBSERVATION_SIZE, 18, device='cpu')
    observation = torch.from_numpy(task.observation()).unsqueeze(0)
    actions, _ = policy(observation)
    actions.square().sum().backward()
    assert actions.shape == (1, 18) and torch.isfinite(actions).all()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in policy.parameters())
    task.step(actions.detach().numpy()[0])
    return dict(mlp_forward_backward=True, training=False)


def check_palms(body: FreeBody) -> dict:
    body.reset()
    body.data.qpos[3:7] = POSES['front']
    body.data.qpos[body.qadr[[12, 15]]] = -1.
    body.data.qpos[2] = 3.
    body.refresh()
    distances = [mujoco.mj_geomDistance(body.model, body.data, body.floor, geom, 10., None)
                 for geom in range(body.model.ngeom) if geom != body.floor]
    body.data.qpos[2] -= min(distances) + .0005
    body.refresh()
    palms = {body.model.geom(name).id for name in ('left_palm', 'right_palm')}
    actual = {int(c.geom2 if c.geom1 == body.floor else c.geom1) for c in body.data.contact
              if body.floor in (c.geom1, c.geom2)}
    assert palms <= actual
    assert body.contacts().penetration < .001
    return dict(both_palms_reach_floor=True, initial_penetration_m=body.contacts().penetration,
                learned_support=False)


def render_poses(body: FreeBody, output: Path) -> None:
    from PIL import Image, ImageDraw
    canvas = Image.new('RGB', (960, 600))
    with mujoco.Renderer(body.model, height=300, width=480) as renderer:
        for index, (name, quaternion) in enumerate(POSES.items()):
            place_pose(body, quaternion)
            camera = mujoco.MjvCamera()
            camera.lookat[:] = body.center
            camera.distance, camera.azimuth, camera.elevation = 2.4, 130, -20
            renderer.update_scene(body.data, camera)
            image = Image.fromarray(renderer.render())
            ImageDraw.Draw(image).text((12, 12), name + ' (initial pose)', fill='white')
            canvas.paste(image, ((index % 2) * 480, (index // 2) * 300))
    canvas.save(output / 'initial_poses.png')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional directory for raw local evidence')
    parser.add_argument('--render', action='store_true', help='Render initial poses; requires --output and Pillow')
    args = parser.parse_args()
    if args.render and args.output is None:
        parser.error('--render requires --output')
    body = FreeBody()
    results = dict(mujoco_version=mujoco.__version__, model=check_body(body),
                   task=check_task(body), physics=check_physics(body), policy=check_policy(body), palms=check_palms(body))
    sources = ['app/free_body.py', 'app/goal_task.py', 'app/test_free_movement.py',
               'app/mlp_policy.py', 'app/yumi_description/yumi.xml', 'app/yumi_description/yumi.yaml']
    results['source_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    results['generated_xml_sha256'] = hashlib.sha256(body.xml.encode()).hexdigest()
    if args.output:
        args.output.mkdir(parents=True, exist_ok=True)
        (args.output / 'native.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
        (args.output / 'body.xml').write_text(body.xml, encoding='utf-8')
        if args.render:
            render_poses(body, args.output)
    print(json.dumps(results, indent=2))
    print('PASS: local free-movement body and goal contracts (no training)')


if __name__ == '__main__':
    main()
