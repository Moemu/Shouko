"""Remote launcher over SSH: starts the unified studio, or training through it.

Training launched here goes through the running studio's /api/train so the page
can see and stop it; a trainer spawned directly by this script would be invisible
to the server's ownership model.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import psutil

ROOT = Path(__file__).resolve().parents[1]
PORT = 8742


def studio_running():
    for process in psutil.process_iter(['pid', 'cmdline']):
        if process.pid != psutil.Process().pid and 'app.studio_server:app' in (process.info['cmdline'] or []):
            return process.pid
    return None


def launch(action, seconds=1800):
    runs = ROOT/'runs/cloud'
    runs.mkdir(parents=True, exist_ok=True)
    server = studio_running()
    if server is not None:
        print(json.dumps(dict(already_running=True, pid=server)))
        return
    if action == 'train':
        # No studio to own the trainer; start one first. A later HTTP start
        # through the page remains the supported way to stop it.
        command = [sys.executable, '-m', 'uvicorn', 'app.studio_server:app', '--host', '127.0.0.1',
                   '--port', str(PORT), '--timeout-graceful-shutdown', '3']
        with (runs/'preview.log').open('a') as log:
            subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                             stderr=subprocess.STDOUT, start_new_session=True)
        print(json.dumps(dict(started=True, action='preview', port=PORT,
                              note='start training via POST http://127.0.0.1:%d/api/train' % PORT)))
        return
    command = [sys.executable, '-m', 'uvicorn', 'app.studio_server:app', '--host', '127.0.0.1',
               '--port', str(PORT), '--timeout-graceful-shutdown', '3']
    with (runs/'preview.log').open('a') as log:
        child = subprocess.Popen(command, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log,
                                 stderr=subprocess.STDOUT, start_new_session=True)
    (runs/'preview.pid').write_text(str(child.pid))
    print(json.dumps(dict(started=True, pid=child.pid, action=action, port=PORT)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['train', 'preview'])
    parser.add_argument('--seconds', type=int, default=1800)
    args = parser.parse_args()
    if not 60 <= args.seconds <= 7200:
        parser.error('Training duration must be between 60 and 7200 seconds')
    launch(args.action, args.seconds)
