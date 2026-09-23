import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path('runs/connectome_transfer_20260923')
deadline=time.monotonic()+3600
while not json.loads((root/'verification_execution.json').read_text())['completed']:
    if time.monotonic()>deadline: raise TimeoutError('Verification incomplete')
    time.sleep(10)
for name,args in [
    ('student_trace',[str(root/'record_student_trace.py')]),
    ('final_freeze_audit',[str(root/'audit_data_and_freeze.py'),'--output',str(root/'final_data_freeze_audit.json')]),
    ('archive',[str(root/'archive_results.py')])]:
    with (root/(name+'.log')).open('w') as handle:
        subprocess.run([sys.executable,'-u',*args],stdout=handle,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8'),check=True,timeout=1200)
