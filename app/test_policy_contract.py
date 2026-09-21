"""CPU checks with real MuJoCo bodies and a four-neuron policy fixture."""
from contextlib import contextmanager
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from app import full_brain
from app.full_brain import ConnectomePolicy, SparseMessage, checkpoint_configuration
from app.sim import Body


@contextmanager
def tiny_graph():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        (root/'data').mkdir()
        path = root/'data/full_graph.npz'
        np.savez(path, ids=np.arange(4), ptr=np.arange(5, dtype=np.int64),
                 pre=np.array([3, 0, 1, 2], dtype=np.int64), weight=np.ones(4, dtype=np.float32),
                 inputs=np.array([0, 1]), outputs=np.array([2, 3]), coords=np.zeros((4, 3)))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        (root/'data/full_graph.json').write_text(json.dumps(dict(graph_sha256=digest)))
        with patch.object(full_brain, 'ROOT', root):
            yield root


def check_observations():
    from app.gpu_body import GPUHumanoid
    old = Body(False, 'yumi')
    assert set(old.interface) == {'default_angles', 'action_scale', 'cmd_scale',
                                  'ang_vel_scale', 'dof_vel_scale', 'control_decimation'}
    assert old.motor_observation([0.5, 0, 0]).shape == (47,)
    old.data.time = 0.2
    np.testing.assert_allclose(old.motor_observation([0, 0, 0])[-2:],
                               [np.sin(0.4*np.pi), np.cos(0.4*np.pi)], atol=1e-6)
    for size in (47, 50):
        body = Body(False, 'yumi', observation_size=size,
                    interface=dict(observation_size=size, gait_period_s=0.8,
                                   control_decimation=10.0, action_scale=0.3))
        rng = np.random.default_rng(31)
        for _ in range(10):
            body.data.qpos[3:7] = rng.normal(size=4)
            body.data.qpos[3:7] /= np.linalg.norm(body.data.qpos[3:7])
            body.data.qpos[2] = 0.87
            body.data.qvel[:] = rng.normal(size=18)
            body.data.time = rng.uniform(0, 5)
            command = [0.35, 0, 0.1]
            fake = SimpleNamespace(qpos=torch.tensor(body.data.qpos[None], dtype=torch.float32),
                                   qvel=torch.tensor(body.data.qvel[None], dtype=torch.float32),
                                   time=torch.tensor([body.data.time], dtype=torch.float32),
                                   command=torch.tensor([command]), home=torch.tensor(body.home),
                                   actions=torch.tensor(body.action[None]),
                                   cmd_scale=torch.tensor(body.cfg['cmd_scale']))
            for key in ('observation_size', 'gait_period_s', 'linear_velocity_scale',
                        'height_reference', 'ang_vel_scale', 'dof_vel_scale'):
                setattr(fake, key, body.cfg[key])
            fake.gravity = lambda: GPUHumanoid.gravity(fake)
            # Call the actual GPU-path observation code on CPU tensors.
            np.testing.assert_allclose(GPUHumanoid.observation(fake).numpy()[0],
                                       body.motor_observation(command), atol=4e-6)
        body.reset()
        body.step_joints(np.zeros(12))  # Recorded 10.0 decimation must be executable.
        assert abs(body.data.time - 0.02) < 1e-8
    assert Body(False, 'yumi', observation_size=50).cfg['gait_period_s'] == 0.8
    for interface in (dict(gait_period_s=0), dict(control_decimation=1.5),
                      dict(observation_size=47)):
        try:
            Body(False, 'yumi', interface=interface, observation_size=50)
        except ValueError:
            pass
        else:
            raise AssertionError(interface)


def check_reload():
    from app import studio_server as studio
    from app.evaluate_full import episode
    with tiny_graph() as root, patch.object(studio, 'DEVICE', 'cpu'):
        instance = studio.Studio.__new__(studio.Studio)
        instance.robot = 'yumi'
        instance.brain = ConnectomePolicy(device='cpu')
        for size, period in ((50, 0.8), (47, 1.0), (50, 0.6)):
            policy = ConnectomePolicy(observation_size=size, device='cpu')
            body = Body(False, 'yumi', observation_size=size,
                        interface=dict(gait_period_s=period, observation_size=size))
            path = root/f'policy{size}.pt'
            policy.save(path, physics=body.interface)
            configuration = checkpoint_configuration(path)
            assert configuration['observation_size'] == size
            instance.load_checkpoint(path)
            assert instance.body.motor_observation([0, 0, 0]).shape == (size,)
            assert instance.body.cfg['gait_period_s'] == period
            result, _ = episode(instance.brain, instance.body, 1, 0.35, 0.04)
            assert result['seconds'] > 0
        # Switching back to a checkpoint without a recorded interface clears the old contract.
        path = root/'legacy.pt'
        ConnectomePolicy(device='cpu').save(path)
        instance.load_checkpoint(path)
        assert instance.body.cfg['gait_period_s'] == 1.0
        assert instance.body.observation_size == 47
        checkpoint = torch.load(path, weights_only=True)
        checkpoint['observation_size'] = 50
        torch.save(checkpoint, path)
        try:
            checkpoint_configuration(path)
        except ValueError:
            pass
        else:
            raise AssertionError('Mismatched encoder metadata accepted')


def check_evidence_binding():
    from fastapi import HTTPException
    from app import studio_server as studio
    body = Body(False, 'yumi', observation_size=50)
    with TemporaryDirectory() as directory:
        root = Path(directory)
        live = SimpleNamespace(robot='yumi', body=body, checkpoint_hash='fixture')
        record = dict(checkpoint_sha256='fixture', robot='yumi', kind='held_out_native_mujoco',
                      physics_interface_sha256='wrong-interface')
        path = root/'heldout_yumi.json'
        with patch.object(studio, 'studio', live), patch.object(studio, 'runs_for', return_value=root), \
                patch.object(studio, 'ROOT', root), patch.object(studio, 'evaluation_job', {}):
            path.write_text(json.dumps(record))
            try:
                studio.evaluation()
            except HTTPException as error:
                assert error.status_code == 404
            else:
                raise AssertionError('Evidence for a different physics interface was served')
            record['physics_interface_sha256'] = studio.interface_digest(body.interface)
            path.write_text(json.dumps(record))
            assert studio.evaluation().status_code == 200


def check_frozen_edges():
    from app.gpu_benchmark import sparse_check
    sparse_check('cpu')
    ptr = torch.tensor([0, 1, 3], dtype=torch.int32)
    pre = torch.tensor([1, 0, 1], dtype=torch.int32)
    values = torch.tensor([0.4, -0.2, 0.3])
    activity = torch.tensor([[0.2, -0.3], [0.7, 0.4]], requires_grad=True)
    reference_activity = activity.detach().clone().requires_grad_()
    reference = torch.tensor([[0., 0.4], [-0.2, 0.3]]) @ reference_activity
    with patch.object(torch.sparse, 'sampled_addmm', side_effect=AssertionError('Frozen edge gradient requested')):
        result = SparseMessage.apply(values, activity, ptr, pre)
        result.square().sum().backward()
    reference.square().sum().backward()
    torch.testing.assert_close(result, reference)
    torch.testing.assert_close(activity.grad, reference_activity.grad)
    from app.ppo_yumi import freeze_policy_core
    with tiny_graph():
        policy = ConnectomePolicy(device='cpu')
        freeze_policy_core(policy)
        expected = {'neuron_bias', 'encoder.weight', 'readout.weight', 'readout.bias'}
        assert {name for name, p in policy.named_parameters() if p.requires_grad} == expected
        optimizer = torch.optim.Adam([p for p in policy.parameters() if p.requires_grad])
        for _ in range(2):
            optimizer.zero_grad(set_to_none=True)
            policy(torch.randn(2, 47))[0].square().sum().backward()
            assert policy.normalizer.weight.grad is None and policy.normalizer.bias.grad is None


if __name__ == '__main__':
    check_observations()
    check_reload()
    check_evidence_binding()
    check_frozen_edges()
    print('Policy contract: legacy, 47/50 reload, native/batched observations, CPU episode OK')
