"""Render a recorded native CPU trajectory; does not resimulate or drive the body."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.full_brain import checkpoint_configuration
from app.sim import Body

parser = argparse.ArgumentParser()
parser.add_argument('--trace', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--max-frames', type=int, default=750)
args = parser.parse_args()
if args.output.exists():
    raise FileExistsError(args.output)
meta = json.loads(args.trace.with_suffix('.json').read_text())
cfg = checkpoint_configuration(meta['checkpoint'])
body = Body(False, 'yumi', cfg['physics_interface'], 50)
trace = np.load(args.trace)
camera = mujoco.MjvCamera()
camera.type = mujoco.mjtCamera.mjCAMERA_FREE
camera.distance = 2.5
camera.azimuth = 125
camera.elevation = -12
renderer = mujoco.Renderer(body.model, height=480, width=640)
args.output.parent.mkdir(parents=True, exist_ok=True)
command = ['ffmpeg', '-hide_banner', '-loglevel', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24',
           '-s', '640x480', '-r', '25', '-i', '-', '-an', '-c:v', 'libx264', '-crf', '22',
           '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(args.output)]
process = subprocess.Popen(command, stdin=subprocess.PIPE,
                           creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
try:
    for index in range(min(len(trace['qpos']), args.max_frames*2))[::2]:
        body.data.qpos[:] = trace['qpos'][index]
        mujoco.mj_forward(body.model, body.data)
        camera.lookat[:] = [body.data.qpos[0], body.data.qpos[1], .55]
        renderer.update_scene(body.data, camera=camera)
        process.stdin.write(renderer.render().tobytes())
finally:
    process.stdin.close()
    renderer.close()
if process.wait() != 0:
    raise RuntimeError('Video encoder failed')
print(json.dumps(dict(output=str(args.output), frames=min(len(trace['qpos'])//2, args.max_frames),
                      note='Supplemental CPU trajectory, physical proxy only; not a separate validation.')))
