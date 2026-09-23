import json
import os
from pathlib import Path
import subprocess
import sys
import time
root=Path('runs/connectome_transfer_push_20260923');deadline=time.monotonic()+5400
while not json.loads((root/'execution.json').read_text())['completed']:
    if time.monotonic()>deadline:raise TimeoutError('Experiment incomplete')
    time.sleep(10)
for name,args in [('candidate_trace',[str(root/'record_candidate.py')]),('chain_audit',[str(root/'audit_chain.py')]),('archive',[str(root/'archive_results.py')])]:
    with (root/(name+'.log')).open('w') as handle:
        subprocess.run([sys.executable,'-u',*args],stdout=handle,stderr=subprocess.STDOUT,
            env=dict(os.environ,OMP_NUM_THREADS='8',MKL_NUM_THREADS='8'),check=True,timeout=1200)
