"""Bounded local CUDA smoke check, not a PPO runner or a performance benchmark."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import mujoco_warp as mjw
import numpy as np
import torch
import warp as wp

from .free_body import FreeBody, ROOT
from .goal_task import GoalTask, OBSERVATION_SIZE
from .gpu_body import pd_torque
from .test_free_movement import POSES, place_pose


def check_gpu_physics(body: FreeBody) -> dict:
    wp.init()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this optional local check')
    assert np.array_equal(body.qadr, np.arange(7, 25))
    assert np.array_equal(body.vadr, np.arange(6, 24))
    assert np.array_equal(body.actuators, np.arange(18))
    with wp.ScopedDevice('cuda:0'):
        model = mjw.put_model(body.model)
        data = mjw.put_data(body.model, body.data, nworld=4, nconmax=48, njmax=192)
        positions = []
        for quaternion in POSES.values():
            place_pose(body, quaternion)
            positions.append(body.data.qpos.copy())
        wp.copy(data.qpos, wp.array(np.asarray(positions, dtype=np.float32), dtype=wp.float32))
        data.qvel.zero_()
        home = wp.array(body.home.astype(np.float32), dtype=wp.float32)
        kp = wp.array(body.kp.astype(np.float32), dtype=wp.float32)
        kd = wp.array(body.kd.astype(np.float32), dtype=wp.float32)
        limit = wp.array(body.limits.astype(np.float32), dtype=wp.float32)
        # Positive elbow commands are inside its one-sided range; targets remain within all limits.
        actions = np.random.default_rng(11).uniform(-.5, .5, (4, 18)).astype(np.float32)
        actions[:, [14, 17]] = np.abs(actions[:, [14, 17]])
        action = wp.array(actions, dtype=wp.float32)
        max_contacts = max_constraints = 0
        penetration = 0.
        started = time.perf_counter()
        for _ in range(500):
            wp.launch(pd_torque, dim=(4, 18), inputs=[data.qpos, data.qvel, action, home, kp, kd, limit,
                                                     .25, data.ctrl])
            mjw.step(model, data)
            wp.synchronize()
            count = int(data.nacon.numpy()[0])
            constraints = int(data.nefc.numpy().max())
            assert count < data.naconmax and constraints < data.njmax, (count, constraints)
            max_contacts = max(max_contacts, count)
            max_constraints = max(max_constraints, constraints)
            if count:
                penetration = max(penetration, float(np.maximum(-data.contact.dist.numpy()[:count], 0).max()))
            assert np.isfinite(data.qpos.numpy()).all() and np.isfinite(data.qvel.numpy()).all()
        assert penetration < .03, penetration
        return dict(worlds=4, poses=list(POSES), simulated_seconds_per_world=1.,
                    wall_seconds_including_first_compile=time.perf_counter() - started,
                    contacts_capacity_total=data.naconmax, constraints_capacity_per_world=data.njmax,
                    max_contacts_total=max_contacts, max_constraints_per_world=max_constraints,
                    max_penetration_m=penetration)


def check_connectome(body: FreeBody) -> dict:
    from .full_brain import ConnectomePolicy
    torch.manual_seed(11)
    torch.cuda.reset_peak_memory_stats()
    policy = ConnectomePolicy(OBSERVATION_SIZE, 18, device='cuda')
    policy.edge_delta.requires_grad_(False)
    policy.neuron_bias.requires_grad_(False)
    task = GoalTask(body)
    observation = torch.from_numpy(task.observation()).unsqueeze(0).cuda()
    started = time.perf_counter()
    action, state = policy(observation)
    action.square().sum().backward()
    torch.cuda.synchronize()
    assert action.shape == (1, 18) and torch.isfinite(action).all() and torch.isfinite(state).all()
    assert torch.isfinite(policy.encoder.weight.grad).all() and torch.isfinite(policy.readout.weight.grad).all()
    assert policy.encoder.weight.grad.abs().max() > 0 and policy.readout.weight.grad.abs().max() > 0
    assert policy.edge_delta.grad is None and policy.neuron_bias.grad is None
    return dict(neurons=policy.n, edges=policy.weight.numel(), observation_size=OBSERVATION_SIZE,
                action_size=18, frozen_graph=True, optimizer_steps=0,
                wall_seconds_forward_backward=time.perf_counter() - started,
                peak_torch_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_torch_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                graph_sha256=policy.meta['graph_sha256'])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--connectome', action='store_true', help='Check the local full graph, without updating weights')
    args = parser.parse_args()
    body = FreeBody()
    results = dict(device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                   physics=check_gpu_physics(body))
    if args.connectome:
        results['connectome'] = check_connectome(body)
    sources = ('app/free_body.py', 'app/goal_task.py', 'app/test_free_movement.py',
               'app/test_free_movement_gpu.py', 'app/gpu_body.py', 'app/full_brain.py')
    results['source_sha256'] = {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in sources}
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'gpu.json').write_text(json.dumps(results, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(results, indent=2))
    print('PASS: local CUDA smoke checks (zero optimizer steps)')


if __name__ == '__main__':
    main()
