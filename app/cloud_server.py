"""Private cloud preview: direct connectome actions, real physics, live training files."""
import asyncio
import hashlib
from contextlib import asynccontextmanager
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from queue import SimpleQueue
import threading
import time
from typing import Literal

import numpy as np
import psutil
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .full_brain import ConnectomePolicy, ROOT
from .sim import Body

LOCAL = os.environ.get('FLYBODY_LOCAL') == '1'
DEVICE = os.environ.get('FLYBODY_DEVICE', 'cuda' if torch.cuda.is_available() else 'cpu')
BODY_ROBOT = os.environ.get('FLYBODY_BODY', 'g1')
RUNS = ROOT/('runs/yumi' if BODY_ROBOT == 'yumi' else 'runs/cloud')
PORT = 8743 if LOCAL else 8742


class Control(BaseModel):
    action: Literal['play', 'pause', 'reset', 'target', 'lesion', 'push', 'checkpoint']
    speed: float = Field(default=0.5, ge=0, le=0.75, allow_inf_nan=False)
    yaw: float = Field(default=0, ge=-1.57, le=1.57, allow_inf_nan=False)
    lesion: Literal['intact', 'disconnected'] = 'intact'


class CloudStudio:
    def __init__(self):
        self.body = Body(load_motor_policy=False, robot=BODY_ROBOT)
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

    def loop(self):
        deadline = time.perf_counter()
        try:
            while not self.stop.is_set():
                checkpoint = Path(os.environ.get('FLYBODY_CHECKPOINT', str(RUNS/'best.pt')))
                if checkpoint.exists() and checkpoint.stat().st_mtime != self.checkpoint_mtime:
                    extra = self.brain.load(checkpoint)
                    self.checkpoint_mtime = checkpoint.stat().st_mtime
                    self.checkpoint = f"u{extra.get('updates', 0):07d}"
                    self.checkpoint_updates = extra.get('updates', 0)
                    with checkpoint.open('rb') as handle:
                        self.checkpoint_hash = hashlib.file_digest(handle, 'sha256').hexdigest()
                    self.body.reset(4242)
                    self.episode += 1
                    self.running = not LOCAL
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
                state.update(episode=self.episode, running=self.running, target_speed=self.speed, target_yaw=self.yaw,
                    lesion=self.lesion, brain_ms=self.neural_ms, activity=self.activity.tolist(), output_rms=self.output_rms,
                    real_time_factor=0.02/max(time.perf_counter()-start, 0.02) if self.running else 0,
                    checkpoint=self.checkpoint_hash or self.checkpoint, checkpoint_updates=self.checkpoint_updates)
                with self.lock:
                    self.state = state
                deadline = max(deadline+0.02, time.perf_counter()-0.1)
                self.stop.wait(max(0, deadline-time.perf_counter()))
        except Exception as exc:
            with self.lock:
                self.state.update(error=f'{type(exc).__name__}: {exc}', running=False)


def read_training():
    path = RUNS/'training.json'
    status = json.loads(path.read_text()) if path.exists() else dict(phase='not_started', history=[])
    if not LOCAL and status['phase'] in ['initializing', 'collecting', 'training', 'evaluating'] and status.get('pid'):
        if not psutil.pid_exists(status['pid']):
            status.update(phase='failed', error=f'Training process exited. Read {RUNS}/train.log.')
    return status


studio = None
trainer = None
trainer_lock = threading.Lock()


@asynccontextmanager
async def lifespan(app):
    global studio
    studio = CloudStudio()
    yield
    studio.stop.set()
    studio.thread.join(5)


app = FastAPI(lifespan=lifespan)


@app.middleware('http')
async def local_only(request: Request, call_next):
    if request.method == 'POST' and request.headers.get('origin') not in [None, f'http://127.0.0.1:{PORT}', f'http://localhost:{PORT}']:
        return JSONResponse(dict(detail='Use the local preview address to control this instance'), status_code=403)
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
    yumi_ready = (ROOT/'web/public/yumi.vrm').exists()
    yumi_body = BODY_ROBOT == 'yumi'
    return dict(**studio.brain.meta, mode='full_connectome', execution_location='local' if LOCAL else 'cloud',
                training_enabled=not LOCAL, sample_indices=studio.sample.tolist(),
                sample_coords=studio.coords[studio.sample].tolist(),
                hardware=dict(cpu=platform.processor() if LOCAL else 'AutoDL · 16 CPU cores',
                              gpu=torch.cuda.get_device_name() if DEVICE == 'cuda' else 'CPU inference', ram_total_gb=round(limit/2**30,1),
                              ram_available_gb=round(available/2**30,2),
                              process_rss_mb=round(psutil.Process().memory_info().rss/2**20),
                              engine=f'{DEVICE.upper()} full-connectome inference + MuJoCo direct joint physics'),
                body=dict(robot=BODY_ROBOT, hips_height=studio.body.hips_height,
                          avatar='Yumi' if yumi_body else 'pixiv VRM1 Constraint Twist Sample',
                          avatar_url='/api/avatar', avatar_pending=yumi_body and not yumi_ready,
                          credit='原设：松酒 · 画师：7Apoi · 模型：星晨水影工作室 · 发布：墨海徽' if yumi_body else 'pixiv Inc.'))


@app.get('/api/avatar')
def avatar():
    yumi = ROOT/'web/public/yumi.vrm'
    if BODY_ROBOT == 'yumi' and yumi.exists():
        return FileResponse(yumi, media_type='model/gltf-binary')
    return FileResponse(ROOT/'web/dist/avatar.vrm', media_type='model/gltf-binary')


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
    return read_training()


@app.post('/api/train')
def train():
    global trainer
    if LOCAL:
        raise HTTPException(409, '本地页面用于推理验证。这里显示的是已备份的云端训练记录。')
    with trainer_lock:
        if (trainer is not None and trainer.poll() is None) or read_training()['phase'] in ['initializing', 'collecting', 'training', 'evaluating']:
            raise HTTPException(409, 'A cloud training run is already active')
        command = ['timeout', '--signal=TERM', '--kill-after=30', '1920', sys.executable,
                   '-u', '-m', 'app.train_full', '--batch', '256', '--max-seconds', '1800']
        if (RUNS/'best.pt').exists():
            command += ['--resume', str(RUNS/'best.pt')]
        with (RUNS/'train.log').open('a') as log:
            trainer = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                       stderr=subprocess.STDOUT, start_new_session=True)
        return dict(started=True, pid=trainer.pid, maximum_training_seconds=1800)


@app.get('/api/evaluation')
def evaluation():
    path = ROOT/'runs/local/heldout.json' if LOCAL else RUNS/'heldout.json'
    if BODY_ROBOT != 'g1':
        path = path.with_name(f'heldout_{BODY_ROBOT}.json')
    if LOCAL:
        if not path.exists() or json.loads(path.read_text()).get('checkpoint_sha256') != studio.checkpoint_hash:
            raise HTTPException(404, '本机当前检查点的验证尚未完成')
        return FileResponse(path, media_type='application/json')
    if not path.exists() or json.loads(path.read_text()).get('checkpoint_sha256') != studio.checkpoint_hash:
        path = RUNS/'evaluation.json'
    if not path.exists(): raise HTTPException(404, 'Independent evaluation has not finished')
    return FileResponse(path, media_type='application/json')


@app.get('/api/report')
def report():
    return FileResponse(ROOT/('research/guides/LOCAL.md' if LOCAL else 'research/guides/CLOUD.md'), media_type='text/plain; charset=utf-8')


app.mount('/', StaticFiles(directory=ROOT/'web/dist', html=True), name='cloud-studio')
