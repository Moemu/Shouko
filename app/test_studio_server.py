"""Regression checks for the unified studio server. No real training: placeholder
subprocesses and a booted app prove ownership, origin and capability behavior."""
import asyncio
import subprocess
import sys
import time
from unittest.mock import patch
from tempfile import TemporaryDirectory
from pathlib import Path

from app import studio_server as s
from starlette.requests import Request


def make_request(origin, host, method='POST', scheme='http'):
    headers = []
    if origin:
        headers.append((b'origin', origin.encode()))
    if host:
        headers.append((b'host', host.encode()))
    return Request({'type': 'http', 'method': method, 'path': '/api/train',
                    'scheme': scheme, 'headers': headers})


async def check_same_origin():
    async def ok(_request):
        return 'ok'
    # SSH tunnel: browser-facing port differs from the server bind, same Host header.
    assert await s.same_origin(make_request('http://127.0.0.1:8890', '127.0.0.1:8890'), ok) == 'ok'
    assert await s.same_origin(make_request('http://127.0.0.1:8740', '127.0.0.1:8740'), ok) == 'ok'
    # A missing Origin (CLI) is allowed; a foreign or https origin is not.
    assert await s.same_origin(make_request(None, '127.0.0.1:8740'), ok) == 'ok'
    for origin in ('http://evil.com', 'https://127.0.0.1:8740'):
        response = await s.same_origin(make_request(origin, '127.0.0.1:8740'), ok)
        assert response.status_code == 403, origin
    print('same-origin: tunnel passthrough + rejections OK')


def check_capabilities():
    with TemporaryDirectory() as directory, patch.object(s, 'runs_for', return_value=Path(directory)):
        with patch.object(s, 'training_dependencies', return_value='cuda_unavailable'):
            assert s.training_unavailable_reason('yumi') == 'cuda_unavailable'
        with patch.object(s, 'training_dependencies', return_value=None):
            assert s.training_unavailable_reason('yumi') == 'checkpoint_missing'
            (Path(directory)/'best.pt').write_bytes(b'test fixture, not a model')
            assert s.training_unavailable_reason('yumi') is None
            assert s.training_unavailable_reason('g1') is None
    print('capabilities: body-specific prerequisites OK')


def check_ownership():
    with TemporaryDirectory() as directory, patch.object(s, 'runs_for', return_value=Path(directory)):
        # A placeholder honoring the cooperative protocol; no model code runs here.
        child = subprocess.Popen(
            [sys.executable, '-c',
             "import os, sys, time, json\n"
             "from pathlib import Path\n"
             "path = Path(os.environ['TEST_TRAINING_JSON'])\n"
             "path.parent.mkdir(parents=True, exist_ok=True)\n"
             "path.write_text(json.dumps(dict(run_id=os.environ['FLYBODY_RUN_ID'], phase='training', active_elapsed=1.0)))\n"
             "print('ready', flush=True)\n"
             "while sys.stdin.readline().strip() != 'finish':\n"
             "    time.sleep(0.05)\n"
             "path.write_text(json.dumps(dict(run_id=os.environ['FLYBODY_RUN_ID'], phase='stopped', active_elapsed=2.0)))"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
            env={**__import__('os').environ, 'FLYBODY_RUN_ID': 'test-run',
                 'TEST_TRAINING_JSON': str(Path(directory)/'training.json')})
        child.stdout.readline()  # wait for the ready line before reading status
        with s.trainer_lock:
            s.trainer, s.trainer_robot, s.trainer_phase = child, 'g1', None
            s.trainer_run_id, s.trainer_request = 'test-run', None
            s.trainer_log_start = 0
        status = s.read_training('g1')
        assert status['active'] is True and status['can_stop'] is True and status['can_pause'] is True, status
        assert status['run_id'] == 'test-run' and status['phase'] == 'training', status
        # A copied training.json with a foreign run_id must not masquerade as the live run.
        (Path(directory)/'training.json').write_text(
            '{"run_id": "old", "phase": "training", "active_elapsed": 99}')
        status = s.read_training('g1')
        assert status['run_id'] == 'test-run' and status['phase'] == 'starting', status
        s.terminate_training(child)
        with s.trainer_lock:
            s.trainer = None
        status = s.read_training('g1')
        assert status['can_stop'] is False, status
        assert status['lifecycle'] in ('historical', 'external', 'not_started'), status
    print('ownership: own process stoppable, foreign run_id never is OK')
    with s.trainer_lock:
        s.trainer = None
    # A copied training.json pid is history, never ownership.
    status = s.read_training('g1')
    assert status['can_stop'] is False, status
    assert status['lifecycle'] in ('historical', 'external', 'not_started'), status
    print('ownership: own process stoppable, historical pid never is OK')


def check_timeout():
    # A placeholder that honors the cooperative finish command: no model code runs.
    child = subprocess.Popen(
        [sys.executable, '-c',
         "import sys; line = sys.stdin.readline(); sys.exit(0 if line.strip() == 'finish' else 9)"],
        stdin=subprocess.PIPE, text=True)
    with s.trainer_lock:
        s.trainer, s.trainer_robot, s.trainer_phase = child, 'g1', None
        s.trainer_run_id, s.trainer_request = 'test-run', None
        s.trainer_log_start = 0
    started = time.time()
    s.watch_training(child, 1, 'g1', 'test-run')
    assert child.poll() == 0 and time.time()-started < 30, (child.poll(), time.time()-started)
    assert s.trainer_phase == 'budget_finished', s.trainer_phase
    with s.trainer_lock:
        s.trainer = None
    print('timeout: cooperative finish finalizes an overrunning child OK')


async def check_boot():
    async with s.lifespan(s.app):
        meta = s.meta()
        assert meta['training_enabled'] == (s.training_unavailable_reason('g1') is None)
        assert meta['body_switch'] is True and 'execution_location' not in meta
        from fastapi import HTTPException
        try:
            s.evaluation()
            raise AssertionError('evaluation must 404 without matching evidence')
        except HTTPException as error:
            assert error.status_code == 404
    print('boot: meta capabilities + evaluation binding OK')


def check_physics_interface():
    """Round-trip the observation/action interface through save/load and fail closed on drift."""
    from app.full_brain import ConnectomePolicy
    from app.sim import Body, observation_interface
    live = observation_interface(Body(load_motor_policy=False, robot='yumi').cfg)
    policy = ConnectomePolicy(device='cpu')
    with TemporaryDirectory() as directory:
        path = Path(directory)/ 'roundtrip.pt'
        policy.save(path, physics=live)
        fresh = ConnectomePolicy(device='cpu')
        fresh.load(path, interface=live)
        assert fresh.physics_interface == live
        drifted = dict(live, default_angles=[0.0] * len(live['default_angles']))
        try:
            ConnectomePolicy(device='cpu').load(path, interface=drifted)
            raise AssertionError('home drift must be rejected')
        except ValueError as error:
            assert 'physics interface' in str(error)
        # 检查点驱动：记录值覆盖 yaml，两条世系可共存
        overridden = Body(load_motor_policy=False, robot='yumi', interface=drifted)
        assert overridden.interface['default_angles'] == [0.0] * 12
        assert overridden.home.tolist() == [0.0] * 12
    print('physics interface: save/load round-trip, mismatch rejection, checkpoint-driven home OK')


if __name__ == '__main__':
    check_capabilities()
    check_ownership()
    check_timeout()
    asyncio.run(check_same_origin())
    asyncio.run(check_boot())
    check_physics_interface()
