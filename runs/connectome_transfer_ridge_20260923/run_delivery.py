import json
import os
from pathlib import Path
import subprocess
import sys
import time
root=Path('runs/connectome_transfer_ridge_20260923');deadline=time.monotonic()+5400
while not json.loads((root/'execution.json').read_text())['completed']:
    if time.monotonic()>deadline:raise TimeoutError('Experiment incomplete')
    time.sleep(10)
if 'candidate' not in json.loads((root/'execution.json').read_text()):raise RuntimeError('No candidate')
for name,args in [
    ('matched_lambda_control',['-m','app.evaluate_locomotion','--checkpoints','runs/connectome_transfer_push_20260923/push2/last.pt',
        '--seed-base','115001','--cohorts','1','--seconds','120','--output',str(root/'matched_lambda_control.json')]),
    ('candidate_trace',[str(root/'record_candidate.py')]),('cache_audit',[str(root/'audit_cached.py')]),('archive',[str(root/'archive_results.py')])]:
    with (root/(name+'.log')).open('w') as handle:
        subprocess.run([sys.executable,'-u',*args],stdout=handle,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8'),check=True,timeout=1200)
