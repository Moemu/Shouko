"""Legacy subgraph prototype (8,192 neurons + frozen G1 gait policy), kept for reference.

No start script since the unified studio entry (app/studio_server.py, port 8740).
Run manually if needed: python -m uvicorn app.server:app --port 8741
"""
import asyncio
from contextlib import asynccontextmanager
import json
from pathlib import Path
import os
import platform
from queue import SimpleQueue
import shutil
import subprocess
import sys
import threading
import time
from typing import Literal

import numpy as np
import psutil
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from threadpoolctl import threadpool_limits

from .brain import FlyBrain, sensation, ROOT
from .sim import Body


class Control(BaseModel):
    action: Literal["play","pause","reset","target","lesion","push","checkpoint"]
    speed: float = Field(default=0.5,ge=0,le=0.75,allow_inf_nan=False)
    yaw: float = Field(default=0,ge=-1.57,le=1.57,allow_inf_nan=False)
    lesion: Literal["intact","disconnected","rewired"] = "intact"


class Studio:
    def __init__(self):
        threadpool_limits(1)
        self.body=Body()
        self.brain=FlyBrain(ROOT/"runs/best.npz")
        finite=np.flatnonzero(np.isfinite(self.brain.coords).all(axis=1))
        self.sample=finite[np.linspace(0,len(finite)-1,min(2048,len(finite)),dtype=int)]
        self.commands=SimpleQueue()
        self.state={}; self.lock=threading.Lock(); self.stop=threading.Event()
        self.running=True; self.speed=0.5; self.yaw=0.0; self.episode=1
        self.trainer=None; self.error=None
        self.thread=threading.Thread(target=self.loop,daemon=True)
        self.thread.start()

    def loop(self):
        command=np.zeros(3)
        tick=0; wall=time.perf_counter(); sim_start=0.0; deadline=wall
        try:
            while not self.stop.is_set():
                start=time.perf_counter()
                while not self.commands.empty():
                    c=self.commands.get()
                    if c.action=="play": self.running=True
                    elif c.action=="pause": self.running=False
                    elif c.action=="target": self.speed=c.speed; self.yaw=c.yaw
                    elif c.action=="push": self.body.data.qvel[1]+=0.25
                    elif c.action in ("lesion","checkpoint"):
                        self.brain=FlyBrain(ROOT/"runs/best.npz",lesion=c.lesion)
                        tick=0
                    if c.action in ("reset","lesion","checkpoint"):
                        self.body.reset(42); self.episode+=1; command=np.zeros(3); tick=0
                        wall=time.perf_counter(); sim_start=0.0
                if self.running:
                    if tick%5==0:
                        command=self.brain.command(sensation(self.body,self.speed,self.yaw))
                    snapshot=self.body.step(command)
                    tick+=1
                    if snapshot["fallen"]: self.running=False
                else:
                    snapshot=self.body.snapshot()
                    wall=time.perf_counter(); sim_start=snapshot["time"]
                snapshot.update({"episode":self.episode,"running":self.running,
                    "target_speed":self.speed,"target_yaw":self.yaw,
                    "lesion":self.brain.lesion,"brain_ms":self.brain.ms,
                    "activity":np.round(self.brain.activity[self.sample],5).tolist(),
                    "output_rms":float(np.sqrt(np.mean(self.brain.activity[self.brain.outputs]**2))),
                    "real_time_factor":(snapshot["time"]-sim_start)/max(time.perf_counter()-wall,0.001),
                    "checkpoint":self.brain.checkpoint_hash})
                with self.lock: self.state=snapshot
                deadline=max(deadline+0.02,time.perf_counter()-0.1)
                self.stop.wait(max(0,deadline-time.perf_counter()))
        except Exception as exc:
            self.error=f"{type(exc).__name__}: {exc}"
            with self.lock: self.state={**self.state,"error":self.error,"running":False}


studio: Studio | None=None


@asynccontextmanager
async def lifespan(app):
    global studio
    studio=Studio()
    yield
    studio.stop.set()
    studio.thread.join(timeout=3)


app=FastAPI(lifespan=lifespan)


@app.middleware("http")
async def local_only(request: Request,call_next):
    if request.method=="POST":
        origin=request.headers.get("origin")
        if origin and origin not in ["http://127.0.0.1:8740","http://localhost:8740"]:
            from fastapi.responses import JSONResponse
            return JSONResponse({"detail":"Only the local studio may control this simulation"},status_code=403)
    return await call_next(request)


def gpu_name():
    return torch.cuda.get_device_name() if torch.cuda.is_available() else ""


def cpu_name():
    # Measured, not assumed: the old hard-coded string was wrong on any other machine.
    return f"{platform.processor() or platform.machine()} · {os.cpu_count()} cores"


@app.get("/api/meta")
def meta():
    return {**studio.brain.meta,"sample_indices":studio.sample.tolist(),
        "sample_coords":studio.brain.coords[studio.sample].tolist(),
        "hardware":{"cpu":cpu_name(),"gpu":gpu_name(),
            "ram_total_gb":round(psutil.virtual_memory().total/2**30,1),
            "ram_available_gb":round(psutil.virtual_memory().available/2**30,2),
            "engine":"CPU · MuJoCo + SciPy + PyTorch","process_rss_mb":round(psutil.Process().memory_info().rss/2**20)},
        "body":{"robot":"g1","avatar":"pixiv VRM1 Constraint Twist Sample"}}


@app.get("/api/events")
async def events(request: Request):
    async def stream():
        while not await request.is_disconnected():
            with studio.lock: state=studio.state.copy()
            yield "data: "+json.dumps(state,separators=(",",":"))+"\n\n"
            await asyncio.sleep(0.05)
    return StreamingResponse(stream(),media_type="text/event-stream",headers={"Cache-Control":"no-cache"})


@app.post("/api/control")
def control(c: Control):
    if studio.error: raise HTTPException(500,studio.error)
    studio.commands.put(c)
    return {"queued":True}


@app.get("/api/training")
def training():
    path=ROOT/"runs/training.json"
    status=json.loads(path.read_text()) if path.exists() else {"phase":"not_started","history":[]}
    if studio.trainer is not None and studio.trainer.poll() not in (None,0):
        status.update(phase="failed",error="Training process failed. See runs/train.log.")
    return status


@app.post("/api/train")
def train():
    if studio.trainer is not None and studio.trainer.poll() is None:
        raise HTTPException(409,"Training is already running")
    archive=ROOT/"runs/archive"/time.strftime("%Y%m%d-%H%M%S")
    archive.mkdir(parents=True,exist_ok=False)
    for name in ["best.npz","evaluation.json","training.json","replay.json"]:
        source=ROOT/"runs"/name
        if source.exists(): shutil.copy2(source,archive/name)
    log=open(ROOT/"runs/train.log","w")
    flags=subprocess.CREATE_NO_WINDOW if sys.platform=="win32" else 0
    studio.trainer=subprocess.Popen([sys.executable,"-u","-m","app.train"],cwd=ROOT,
        stdout=log,stderr=subprocess.STDOUT,creationflags=flags)
    log.close()
    return {"started":True,"previous_run":str(archive.relative_to(ROOT))}


@app.get("/api/evaluation")
def evaluation():
    path=ROOT/"runs/evaluation.json"
    if not path.exists(): raise HTTPException(404,"Evaluation has not completed")
    return FileResponse(path,media_type="application/json")


@app.get("/api/report")
def report():
    return FileResponse(ROOT/"research/REPORT.md",media_type="text/plain; charset=utf-8")


app.mount("/",StaticFiles(directory=ROOT/"web/dist",html=True),name="studio")
