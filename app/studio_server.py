"""Flybody studio: one entry for the G1 and Yumi bodies, local inference, live training files.

Replaces the split server.py (8740) / cloud_server.py (8743) entries: the body is a
runtime choice, not a deployment location.
"""
import asyncio
import hashlib
import importlib
from functools import lru_cache
from urllib.parse import urlsplit
from contextlib import asynccontextmanager
import json
import os
import platform
import shutil
import uuid
import subprocess
import sys
import threading
import time
from pathlib import Path
from queue import SimpleQueue
from typing import Literal

import numpy as np
import psutil
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .full_brain import ConnectomePolicy, ROOT, checkpoint_configuration
from .sim import Body
from .evaluate_full import interface_digest

DEVICE = os.environ.get('FLYBODY_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')


def runs_for(robot):
    return ROOT/('runs/yumi' if robot == 'yumi' else 'runs/cloud')


def avatar_path(robot):
    if robot not in ('g1', 'yumi'):
        raise HTTPException(404, 'Unknown body')
    return ROOT/'web/public'/('yumi.vrm' if robot == 'yumi' else 'avatar.vrm')


@lru_cache(maxsize=8)
def avatar_digest(path, modified_ns, size):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def avatar_identity(robot):
    path = avatar_path(robot)
    try:
        stat = path.stat()
        digest = avatar_digest(path, stat.st_mtime_ns, stat.st_size)
    except FileNotFoundError:
        digest = None
    return dict(robot=robot, avatar_sha256=digest,
                avatar_url=f'/api/avatar?body={robot}&v={digest}' if digest else None,
                avatar_pending=digest is None)


class Control(BaseModel):
    action: Literal['play', 'pause', 'reset', 'target', 'lesion', 'push', 'checkpoint']
    speed: float = Field(default=0.5, ge=0, le=0.75, allow_inf_nan=False)
    yaw: float = Field(default=0, ge=-1.57, le=1.57, allow_inf_nan=False)
    lesion: Literal['intact', 'disconnected'] = 'intact'


class BodyChoice(BaseModel):
    robot: Literal['g1', 'yumi']


class TrainingOptions(BaseModel):
    batch: int = Field(default=64, ge=1, le=256, strict=True)
    worlds: int = Field(default=16, ge=1, le=64, strict=True)
    max_seconds: int = Field(default=1800, ge=1, le=86400, strict=True)
    minibatch: int = Field(default=128, ge=1, le=512, strict=True)
    steps: int = Field(default=128, ge=1, le=512, strict=True)
    warm_start: bool = True
    freeze_brain: bool = True


class EvaluationOptions(BaseModel):
    quick: bool = True
    seconds: int = Field(default=30, ge=1, le=120, strict=True)


@lru_cache(maxsize=2)
def training_dependencies(robot='g1'):
    if not torch.cuda.is_available():
        return 'cuda_unavailable'
    for module in ('warp', 'mujoco_warp'):
        try:
            importlib.import_module(module)
        except (ImportError, OSError, RuntimeError):
            return 'warp_unavailable'
    if not all((ROOT/f'data/full_graph.{suffix}').exists() for suffix in ('npz', 'json')):
        return 'graph_missing'
    if robot == 'g1' and not (ROOT/'vendor/unitree_rl_gym/deploy/pre_train/g1/motion.pt').exists():
        return 'teacher_missing'
    return None


def training_unavailable_reason(robot):
    if robot not in ('g1', 'yumi'):
        return 'unsupported_body'
    reason = training_dependencies(robot)
    if reason is None and robot == 'yumi' and not (runs_for(robot)/'best.pt').exists():
        return 'checkpoint_missing'
    return reason


class Studio:
    """One live body. Built with its robot; swap via replace_studio(), never in place."""

    def __init__(self, robot):
        self.robot = robot
        self.runs = runs_for(robot)
        self.body = Body(load_motor_policy=False, robot=robot)
        assert self.body.policy is None
        self.brain = ConnectomePolicy(device=DEVICE).eval()
        graph = np.load(ROOT/'data/full_graph.npz', allow_pickle=False)
        self.coords = graph['coords']
        finite = np.flatnonzero(np.isfinite(self.coords).all(axis=1))
        self.sample = finite[np.linspace(0, len(finite)-1, min(2048, len(finite)), dtype=int)]
        self.commands = SimpleQueue()
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.state = {}
        self.speed, self.yaw, self.episode = 0.5, 0.0, 0
        self.running, self.lesion = False, 'intact'
        self.checkpoint_mtime = 0
        self.checkpoint = None
        self.checkpoint_hash = None
        self.checkpoint_updates = 0
        self.activity = np.zeros(len(self.sample))
        self.neural_ms = 0
        self.output_rms = 0
        self.thread = threading.Thread(target=self.loop, daemon=True)
        self.thread.start()

    def checkpoint_path(self):
        return Path(os.environ.get('FLYBODY_CHECKPOINT', str(self.runs/'best.pt')))

    def load_checkpoint(self, checkpoint):
        configuration = checkpoint_configuration(checkpoint)
        body = Body(load_motor_policy=False, robot=self.robot,
                    interface=configuration['physics_interface'],
                    observation_size=configuration['observation_size'])
        architecture = {key: configuration[key] for key in
                        ('observation_size', 'action_size', 'neural_steps')}
        brain = self.brain
        if (brain.encoder.in_features != architecture['observation_size'] or
                brain.readout.out_features != architecture['action_size'] or
                brain.neural_steps != architecture['neural_steps']):
            brain = ConnectomePolicy(device=DEVICE, **architecture).eval()
        extra = brain.load(checkpoint, interface=body.interface)
        body.reset(4242)
        self.body, self.brain = body, brain
        return extra

    def loop(self):
        deadline = time.perf_counter()
        try:
            while not self.stop.is_set():
                checkpoint = self.checkpoint_path()
                if checkpoint.exists() and checkpoint.stat().st_mtime != self.checkpoint_mtime:
                    extra = self.load_checkpoint(checkpoint)
                    self.checkpoint_mtime = checkpoint.stat().st_mtime
                    self.checkpoint = f"u{extra.get('updates', 0):07d}"
                    self.checkpoint_updates = extra.get('updates', 0)
                    with checkpoint.open('rb') as handle:
                        self.checkpoint_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
                    self.episode += 1
                    self.running = False
                while not self.commands.empty():
                    control = self.commands.get()
                    if control.action == 'play': self.running = self.checkpoint is not None
                    elif control.action == 'pause': self.running = False
                    elif control.action == 'target': self.speed, self.yaw = control.speed, control.yaw
                    elif control.action == 'push': self.body.data.qvel[1] += 0.25
                    elif control.action == 'lesion': self.lesion = control.lesion
                    if control.action in ['reset', 'lesion', 'checkpoint']:
                        self.body.reset(4242)
                        self.episode += 1
                        self.running = self.checkpoint is not None
                start = time.perf_counter()
                if self.running:
                    gravity, yaw = self.body.observation()
                    error = np.arctan2(np.sin(self.yaw-yaw), np.cos(self.yaw-yaw))
                    obs = self.body.motor_observation([self.speed, 0, np.clip(error*1.4, -0.2, 0.2)])
                    neural_start = time.perf_counter()
                    with torch.inference_mode():
                        action, activity = self.brain(torch.tensor(obs, device=DEVICE)[None], lesion=self.lesion=='disconnected')
                        self.activity = activity[self.sample, 0].cpu().numpy()
                        self.output_rms = float(activity[self.brain.outputs].square().mean().sqrt())
                        joint_action = action[0].cpu().numpy()
                    self.neural_ms = (time.perf_counter()-neural_start)*1000
                    state = self.body.step_joints(joint_action)
                else:
                    state = self.body.snapshot()
                control_dt = self.body.cfg['simulation_dt'] * self.body.cfg['control_decimation']
                state.update(episode=self.episode, running=self.running, target_speed=self.speed, target_yaw=self.yaw,
                    lesion=self.lesion, brain_ms=self.neural_ms, activity=self.activity.tolist(), output_rms=self.output_rms,
                    real_time_factor=control_dt/max(time.perf_counter()-start, control_dt) if self.running else 0,
                    checkpoint=self.checkpoint_hash or self.checkpoint, checkpoint_updates=self.checkpoint_updates,
                    **avatar_identity(self.robot))
                with self.lock:
                    self.state = state
                deadline = max(deadline+control_dt, time.perf_counter()-0.1)
                self.stop.wait(max(0, deadline-time.perf_counter()))
        except Exception as exc:
            with self.lock:
                self.state.update(error=f'{type(exc).__name__}: {exc}', running=False)


ACTIVE_PHASES = ['starting', 'initializing', 'collecting', 'training', 'evaluating',
                 'pause_requested', 'paused', 'stopping']


def external_training(robot):
    modules = ('app.train_full',) if robot == 'g1' else ('app.train_yumi', 'app.ppo_yumi')
    for process in psutil.process_iter(['pid', 'cmdline']):
        command = process.info['cmdline'] or []
        if not any(command[i:i+2] == ['-m', module]
                   for module in modules for i in range(len(command)-1)):
            continue
        try:
            if Path(process.cwd()).resolve() == ROOT.resolve():
                return True
        except (psutil.Error, OSError):
            continue
    return False


def read_json(path, default=None):
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (FileNotFoundError, json.JSONDecodeError):
        return {} if default is None else default


def read_training(robot):
    path = runs_for(robot)/'training.json'
    status = read_json(path, dict(phase='not_started', history=[]))
    with trainer_lock:
        owned = trainer is not None and trainer_robot == robot
        active = owned and trainer.poll() is None
        if owned:
            # A per-launch token survives Windows' Python launcher/worker PID split.
            # Never use a PID from a copied status file as a process handle.
            if status.get('run_id') != trainer_run_id:
                status = dict(phase='starting', history=[], run_id=trainer_run_id)
            if not active:
                status['phase'] = trainer_phase or ('complete' if trainer.returncode == 0 else 'failed')
            elif trainer_phase:
                status['phase'] = trainer_phase
            elif trainer_request == 'pause' and status.get('phase') != 'paused':
                status['phase'] = 'pause_requested'
            status['lifecycle'] = status['phase']
        else:
            status['lifecycle'] = 'historical' if path.exists() else 'not_started'
        external = not active and external_training(robot)
        if external:
            status['lifecycle'] = 'external'
        elif not owned and status.get('phase') in ACTIVE_PHASES:
            status['phase'] = 'historical'
        controlled = active and not trainer_phase
        status.update(robot=robot, algorithm='ppo' if robot == 'yumi' else 'dagger',
                      active=bool(active or external), can_stop=bool(controlled),
                      can_pause=bool(controlled and trainer_request != 'pause' and status.get('phase') != 'paused'),
                      can_resume=bool(controlled and (trainer_request == 'pause' or status.get('phase') == 'paused')))
    return status


def training_active():
    return (trainer is not None and trainer.poll() is None) or external_training(studio.robot)


def evaluation_active():
    return evaluator is not None and evaluator.poll() is None


studio = None
trainer = None
trainer_robot = None
trainer_phase = None
trainer_request = None
trainer_run_id = None
trainer_started = 0
trainer_log_start = 0
evaluator = None
evaluation_job = {}
trainer_lock = threading.Lock()


def terminate_training(process):
    if process.poll() is not None:
        return
    # Only descendants of the Popen handle, never a PID supplied by a status file.
    try:
        children = psutil.Process(process.pid).children(recursive=True)
    except psutil.Error:
        children = []
    for child in reversed(children):
        try:
            child.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(children, timeout=3)
    for child in alive:
        try:
            child.kill()
        except psutil.Error:
            pass
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def send_training(process, command):
    try:
        process.stdin.write(command+'\n')
        process.stdin.flush()
    except (OSError, ValueError, AttributeError) as exc:
        raise HTTPException(409, 'Training control channel is unavailable') from exc


def finish_training(process, terminal='stopped'):
    global trainer_phase
    try:
        process.wait(timeout=120)
    except subprocess.TimeoutExpired:
        terminate_training(process)
        terminal = 'timed_out'
    with trainer_lock:
        if trainer is process:
            trainer_phase = terminal if process.returncode in (0, None) or terminal == 'timed_out' else 'failed'


def watch_training(process, seconds, robot, run_id):
    global trainer_phase
    started = time.monotonic()
    while process.poll() is None:
        status = read_json(runs_for(robot)/'training.json')
        wall = time.monotonic()-started
        if status.get('run_id') == run_id:
            active = status.get('active_elapsed', wall)
        else:
            active = wall
        # ponytail: one owned job; paused jobs have a visible 24-hour wall cap.
        if active >= seconds or wall >= seconds+86400:
            with trainer_lock:
                if trainer is not process or trainer_phase == 'stopping':
                    return
                trainer_phase = 'stopping'
                try:
                    send_training(process, 'finish')
                except HTTPException:
                    pass
            finish_training(process, 'budget_finished' if wall < seconds+86400 else 'timed_out')
            return
        try:
            process.wait(timeout=1)
        except subprocess.TimeoutExpired:
            pass


@asynccontextmanager
async def lifespan(app):
    global studio
    studio = Studio(os.environ.get('FLYBODY_BODY', 'g1'))
    try:
        yield
    finally:
        if trainer is not None and trainer.poll() is None:
            try:
                send_training(trainer, 'finish')
                trainer.wait(timeout=15)
            except (HTTPException, subprocess.TimeoutExpired):
                terminate_training(trainer)
        if evaluator is not None:
            terminate_training(evaluator)
        studio.stop.set()
        studio.thread.join(5)


app = FastAPI(lifespan=lifespan)


@app.middleware('http')
async def same_origin(request: Request, call_next):
    if request.method == 'POST':
        origin = request.headers.get('origin')
        if origin:
            # Compare the browser-facing origin with the request's own Host
            # (scheme, host, port): an SSH tunnel forwards to a different local
            # port than the server binds, so no port may be assumed here.
            try:
                parts = urlsplit(origin)
                expected = urlsplit(str(request.url))
                def address(value):
                    return (value.scheme, value.hostname, value.port or (443 if value.scheme == 'https' else 80))
                valid = (parts.scheme in ('http', 'https') and parts.hostname
                         and not parts.username and not parts.password
                         and not parts.path and not parts.query and not parts.fragment
                         and address(parts) == address(expected))
            except ValueError:
                valid = False
            if not valid:
                return JSONResponse(dict(detail='Use the studio address to control this instance'), status_code=403)
    return await call_next(request)


@app.get('/api/meta')
def meta():
    memory_limit = Path('/sys/fs/cgroup/memory.max')
    memory = psutil.virtual_memory()
    limit_text = memory_limit.read_text().strip() if memory_limit.exists() else 'max'
    limit = int(limit_text) if limit_text != 'max' else memory.total
    memory_current = Path('/sys/fs/cgroup/memory.current')
    available = limit-int(memory_current.read_text()) if memory_current.exists() and limit_text != 'max' else memory.available
    # The avatar follows the physics body. Tying it to the file alone meant `-Body g1`
    # rendered the G1 proxy in Yumi's skin and labelled it "Yumi".
    identity = avatar_identity(studio.robot)
    yumi_body = studio.robot == 'yumi'
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    reason = training_unavailable_reason(studio.robot)
    return dict(**studio.brain.meta, mode='full_connectome',
                observation_size=studio.body.observation_size,
                gait_period_s=studio.body.cfg['gait_period_s'],
                training_enabled=reason is None, training_unavailable_reason=reason,
                training_method='ppo' if yumi_body else 'dagger',
                training_checkpoint_available=(runs_for(studio.robot)/'best.pt').exists(),
                body_switch=True, sample_indices=studio.sample.tolist(),
                sample_coords=studio.coords[studio.sample].tolist(),
                hardware=dict(cpu=platform.processor(),
                              gpu=gpu_name, inference_device=DEVICE, ram_total_gb=round(limit/2**30,1),
                              ram_available_gb=round(available/2**30,2),
                              process_rss_mb=round(psutil.Process().memory_info().rss/2**20),
                              engine=f'{DEVICE.upper()} full-connectome inference + MuJoCo direct joint physics'),
                body=dict(**identity, hips_height=studio.body.hips_height,
                          avatar='Yumi' if yumi_body else 'pixiv VRM1 Constraint Twist Sample',
                          credit='原设：松酒 · 画师：7Apoi · 模型：星晨水影工作室 · 发布：墨海徽' if yumi_body else 'pixiv Inc.'))


@app.get('/api/avatar')
def avatar(body: str | None = None, v: str | None = None):
    robot = body if body is not None else studio.robot
    identity = avatar_identity(robot)
    if identity['avatar_pending']:
        raise HTTPException(404, 'Avatar asset is missing for this body')
    if v is not None and v != identity['avatar_sha256']:
        raise HTTPException(409, 'Avatar asset changed; refresh body metadata')
    return FileResponse(avatar_path(robot), media_type='model/gltf-binary',
                        headers={'Cache-Control': 'no-cache'})


@app.post('/api/body')
def switch_body(choice: BodyChoice):
    """Swap the live body. Builds the replacement first so a failed load keeps the old one serving."""
    global studio
    if choice.robot == studio.robot:
        return dict(swapped=False, robot=studio.robot)
    with trainer_lock:
        if training_active() or evaluation_active():
            raise HTTPException(409, 'Training is running; wait for it to finish or stop it first')
        replacement = Studio(choice.robot)
        old, studio = studio, replacement
    old.stop.set()
    old.thread.join(5)
    return dict(swapped=True, robot=studio.robot)


@app.get('/api/state')
def state():
    with studio.lock:
        return studio.state.copy()


@app.get('/api/events')
async def events(request: Request):
    async def stream():
        while not await request.is_disconnected():
            with studio.lock:
                state = studio.state.copy()
            yield 'data: '+json.dumps(state, separators=(',', ':'))+'\n\n'
            await asyncio.sleep(0.1)
    return StreamingResponse(stream(), media_type='text/event-stream', headers={'Cache-Control':'no-cache'})


@app.post('/api/control')
def control(value: Control):
    studio.commands.put(value)
    return dict(queued=True)


@app.get('/api/training')
def training():
    return read_training(studio.robot)


@app.post('/api/train')
def train(options: TrainingOptions | None = None):
    global trainer, trainer_robot, trainer_phase, trainer_run_id, trainer_request, trainer_started, trainer_log_start
    options = options or TrainingOptions()
    with trainer_lock:
        robot = studio.robot
        reason = training_unavailable_reason(robot)
        if reason is not None:
            raise HTTPException(409, f'training_unavailable_reason: {reason}')
        if training_active() or evaluation_active():
            raise HTTPException(409, 'A training or evaluation job is already active')
        ppo = robot == 'yumi'
        if ppo and not options.warm_start:
            raise HTTPException(409, 'Yumi PPO requires a Yumi checkpoint warm start')
        command = [sys.executable, '-u', '-m', 'app.ppo_yumi' if ppo else 'app.train_full',
                   '--worlds', str(options.worlds), '--max-seconds', str(options.max_seconds)]
        if ppo:
            command += ['--minibatch', str(options.minibatch), '--steps', str(options.steps)]
            if options.freeze_brain:
                command.append('--freeze-brain')
        else:
            command += ['--batch', str(options.batch)]
        runs = runs_for(robot)
        runs.mkdir(parents=True, exist_ok=True)
        if options.warm_start and (runs/'best.pt').exists():
            command += ['--resume', str(runs/'best.pt')]
        run_id = uuid.uuid4().hex
        log_path = runs/'train.log'
        log_start = log_path.stat().st_size if log_path.exists() else 0
        with log_path.open('ab') as log:
            process = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.PIPE, text=True,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
                                       env=dict(os.environ, FLYBODY_RUN_ID=run_id))
        trainer, trainer_robot, trainer_phase = process, robot, None
        trainer_run_id, trainer_request = run_id, None
        trainer_started, trainer_log_start = time.time(), log_start
        threading.Thread(target=watch_training,
                         args=(process, options.max_seconds, robot, run_id), daemon=True).start()
        return dict(started=True, pid=process.pid, run_id=run_id, robot=robot,
                    algorithm='ppo' if ppo else 'dagger', **options.model_dump())


@app.post('/api/train/stop')
def train_stop():
    """Request safe checkpoint finalization; force only after a bounded grace."""
    global trainer_phase, trainer_request
    with trainer_lock:
        if trainer is None or trainer.poll() is not None or trainer_robot != studio.robot:
            raise HTTPException(409, 'No owned training run is active')
        if trainer_phase != 'stopping':
            send_training(trainer, 'finish')
            trainer_phase, trainer_request = 'stopping', 'finish'
            threading.Thread(target=finish_training, args=(trainer,), daemon=True).start()
        return dict(phase='stopping', run_id=trainer_run_id)


def training_command(command):
    global trainer_request
    with trainer_lock:
        if trainer is None or trainer.poll() is not None or trainer_robot != studio.robot or trainer_phase:
            raise HTTPException(409, 'No controllable training run is active')
        send_training(trainer, command)
        trainer_request = command
        return dict(phase='pause_requested' if command == 'pause' else 'training', run_id=trainer_run_id)


@app.post('/api/train/pause')
def train_pause():
    return training_command('pause')


@app.post('/api/train/resume')
def train_resume():
    return training_command('resume')


@app.get('/api/train/logs')
def training_logs(cursor: int = 0, run_id: str = ''):
    if cursor < 0:
        raise HTTPException(422, 'Cursor must be nonnegative')
    with trainer_lock:
        robot = studio.robot
        owned = trainer_robot == robot and trainer is not None
        current_id = trainer_run_id if owned else read_json(runs_for(robot)/'training.json').get('run_id', '')
        start = trainer_log_start if owned else 0
    path = runs_for(robot)/'train.log'
    if not path.exists():
        return dict(run_id=current_id, cursor=0, text='', reset=True)
    with path.open('rb') as handle:
        size = path.stat().st_size
        reset = run_id != current_id or cursor < start or cursor > size
        offset = max(start, size-65536) if reset or cursor == 0 else cursor
        handle.seek(offset)
        if offset > start and (reset or cursor == 0):
            handle.readline()  # discard a truncated first line when tailing
        data = handle.read(65536)
        end = handle.tell()
    return dict(run_id=current_id, cursor=end, text=data.decode('utf-8', errors='replace'), reset=reset)


@app.get('/api/evaluation')
def evaluation():
    """Serve held-out evidence only when it is bound to the loaded checkpoint.
    candidates: body dir first, then the legacy runs/local files (kept for the
    recorded 2026-09-15 acceptance). The training-time screening evaluation.json
    is never passed off as independent acceptance."""
    robot = studio.robot
    candidates = [runs_for(robot)/f'heldout{"_" + robot if robot != "g1" else ""}.json']
    if robot == 'g1':
        candidates.append(ROOT/'runs/local/heldout.json')
    else:
        candidates.append(ROOT/'runs/local/heldout_yumi.json')
    if evaluation_job.get('robot') == robot and evaluation_job.get('output'):
        candidates.insert(0, Path(evaluation_job['output']))
    for path in candidates:
        record = read_json(path)
        if (studio.checkpoint_hash and record.get('checkpoint_sha256') == studio.checkpoint_hash
                and record.get('robot') == robot and record.get('kind') == 'held_out_native_mujoco'
                and record.get('physics_interface_sha256', interface_digest(studio.body.interface))
                    == interface_digest(studio.body.interface)):
            return JSONResponse(record)
    raise HTTPException(404, '本机当前检查点的验证尚未完成' if robot == 'yumi' else
                        'Independent evaluation for the loaded checkpoint has not finished')


@app.post('/api/evaluate')
def start_evaluation(options: EvaluationOptions | None = None):
    global evaluator, evaluation_job
    options = options or EvaluationOptions()
    with trainer_lock:
        if training_active() or evaluation_active():
            raise HTTPException(409, 'A training or evaluation job is already active')
        robot, expected_hash = studio.robot, studio.checkpoint_hash
        source = studio.checkpoint_path()
        if not expected_hash or not source.exists():
            raise HTTPException(409, 'Load a checkpoint before independent evaluation')
        run_id = uuid.uuid4().hex
        directory = runs_for(robot)/'evaluations'/run_id
        directory.mkdir(parents=True)
        snapshot = directory/'checkpoint.pt'
        shutil.copyfile(source, snapshot)
        with snapshot.open('rb') as handle:
            actual_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
        if actual_hash != expected_hash:
            snapshot.unlink()
            raise HTTPException(409, 'Checkpoint changed; wait for the preview to reload it')
        output = directory/'heldout.json'
        command = [sys.executable, '-u', '-m', 'app.evaluate_full', '--robot', robot,
                   '--checkpoint', str(snapshot), '--output', str(output),
                   '--seconds', str(options.seconds), '--device', DEVICE]
        if options.quick:
            command.append('--quick')
        with (directory/'evaluate.log').open('ab') as log:
            evaluator = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        evaluation_job = dict(run_id=run_id, robot=robot, checkpoint_sha256=expected_hash,
                              output=str(output), started_at=time.time(), total=9 if options.quick else 16,
                              options=options.model_dump())
        return dict(started=True, **evaluation_job)


@app.get('/api/evaluate')
def evaluation_status():
    with trainer_lock:
        if not evaluation_job or evaluation_job.get('robot') != studio.robot:
            return dict(active=False, phase='not_started', completed=0, total=0)
        record = read_json(Path(evaluation_job['output']))
        completed = sum(len(record.get(key, [])) for key in ('tests', 'long_walks', 'perturbations', 'controls'))
        active = evaluation_active()
        phase = 'evaluating' if active else 'complete' if evaluator.returncode == 0 and record.get('complete') else 'failed'
        return dict(**evaluation_job, active=active, phase=phase, completed=completed,
                    error='Evaluation failed; see evaluate.log in its run directory' if phase == 'failed' else None)


@app.get('/api/report')
def report():
    return FileResponse(ROOT/'research/REPORT.md', media_type='text/plain; charset=utf-8')


app.mount('/', StaticFiles(directory=ROOT/'web/dist', html=True), name='studio')

