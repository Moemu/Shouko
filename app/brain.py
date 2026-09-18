"""Measured edges carry every signal from engineered sensation to learned intention."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
FEATURE_VERSION = "malecns-rate-v1-12steps"


class FlyBrain:
    def __init__(self, checkpoint=None, lesion="intact"):
        self.meta = json.loads((ROOT / "data/graph.json").read_text())
        graph_path = ROOT / "data/graph.npz"
        with graph_path.open("rb") as f:
            actual = hashlib.file_digest(f,"sha256").hexdigest()
        if actual != self.meta["graph_sha256"]:
            raise ValueError("Connectome hash mismatch")
        z = np.load(graph_path,allow_pickle=False)
        self.ids = z["ids"]
        self.coords = z["coords"]
        self.inputs = z["input_idx"]
        self.outputs = z["output_idx"]
        self.n = len(self.ids)
        pre = np.repeat(np.arange(self.n),np.diff(z["ptr"]))
        post = z["post"].copy()
        raw = np.sqrt(z["contact"].astype(np.float32))*z["signs"][pre]
        incoming = np.bincount(post,weights=np.abs(raw),minlength=self.n)
        raw *= (0.95 / np.maximum(incoming[post],1)).astype(np.float32)
        if lesion == "disconnected":
            raw[:] = 0
        elif lesion == "rewired":
            post = np.random.default_rng(909).permutation(post)
        elif lesion != "intact":
            raise ValueError("Unknown lesion")
        self.W = csr_matrix((raw,(post,pre)),shape=(self.n,self.n),dtype=np.float32)
        self.encoder = np.random.default_rng(17).normal(0,0.65,(len(self.inputs),8)).astype(np.float32)
        self.activity = np.zeros(self.n,np.float32)
        self.readout = np.zeros((len(self.outputs),3),np.float32)
        self.scale = np.ones(len(self.outputs),np.float32)
        self.ms = 0.0
        self.checkpoint_hash = None
        self.lesion = lesion
        if checkpoint is not None:
            self.load(checkpoint)

    def encode(self, sensory):
        t = time.perf_counter()
        signal = np.zeros(self.n,np.float32)
        signal[self.inputs] = np.tanh(self.encoder @ np.asarray(sensory,np.float32))
        r = np.zeros(self.n,np.float32)
        for _ in range(12):
            r = 0.35*r + 0.65*np.tanh(self.W @ r + signal)
        self.activity = r
        self.ms = (time.perf_counter()-t)*1000
        return r[self.outputs].copy()

    def command(self, sensory):
        features = self.encode(sensory)
        command = (features/self.scale) @ self.readout
        return np.clip(command,[-0.1,-0.15,-0.65],[0.9,0.15,0.65])

    def save(self, path):
        target=Path(path)
        temporary=target.with_suffix(".tmp")
        with temporary.open("wb") as f:
            np.savez_compressed(f,readout=self.readout,scale=self.scale,
                graph_hash=self.meta["graph_sha256"],feature_version=FEATURE_VERSION)
        temporary.replace(target)

    def load(self, path):
        z = np.load(path,allow_pickle=False)
        if str(z["graph_hash"]) != self.meta["graph_sha256"] or str(z["feature_version"]) != FEATURE_VERSION:
            raise ValueError("Checkpoint does not match graph or neural dynamics")
        if z["readout"].shape != self.readout.shape or not np.all(np.isfinite(z["readout"])):
            raise ValueError("Invalid readout weights")
        if z["scale"].shape != self.scale.shape or not np.all(z["scale"]>0):
            raise ValueError("Invalid feature scale")
        self.readout = z["readout"].copy()
        self.scale = z["scale"].copy()
        with open(path,"rb") as f:
            self.checkpoint_hash = hashlib.file_digest(f,"sha256").hexdigest()


def sensation(body, target_speed=0.5, target_yaw=0.0):
    gravity, yaw = body.observation()
    heading_error = np.arctan2(np.sin(target_yaw-yaw),np.cos(target_yaw-yaw))
    forward_speed = np.cos(yaw)*body.data.qvel[0]+np.sin(yaw)*body.data.qvel[1]
    return np.array([target_speed,forward_speed,heading_error,
        body.data.qvel[5],gravity[0],gravity[1],body.data.qpos[2]-0.76,1.0],np.float32)
