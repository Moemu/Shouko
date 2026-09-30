"""Wait for the fixed teacher test, then add the preregistered matched source control."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path('runs/direction_control_20260929')
deadline=time.monotonic()+2400
while not json.loads((root/'teacher_repro_execution.json').read_text())['completed']:
    if time.monotonic()>deadline:
        raise TimeoutError('Teacher replication still running')
    time.sleep(10)
command=[sys.executable,'-u','-m','app.evaluate_direction','--checkpoints','runs/source_main/last.pt',
    '--seed-base','310201','--seconds','120','--trace','--output',str(root/'main_source_matched_development_long.json')]
with (root/'main_source_matched_development_long.log').open('w') as log:
    subprocess.run(command,check=True,stdout=log,stderr=subprocess.STDOUT,timeout=1200,
        env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8'))
subprocess.run([sys.executable,'-u',str(root/'archive_phase_closeout.py')],check=True,timeout=1200)
