import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

root = Path('runs/connectome_transfer_replica_20260923')
deadline = time.monotonic() + 1800
while not json.loads((root/'execution.json').read_text())['completed']:
    if time.monotonic() > deadline:
        raise TimeoutError('Replica did not finish')
    time.sleep(10)
for name, script in [
    ('feature_row_audit', 'runs/connectome_transfer_ridge_20260923/audit_feature_rows.py'),
    ('chain_audit', str(root/'audit_chain.py')),
]:
    with (root/(name+'.log')).open('w') as handle:
        subprocess.run([sys.executable, '-u', script], stdout=handle, stderr=subprocess.STDOUT, check=True)
for name in ['audit_feature_rows.py', 'feature_row_audit.json']:
    shutil.copy2(Path('runs/connectome_transfer_ridge_20260923')/name, root/name)
with (root/'archive.log').open('w') as handle:
    subprocess.run([sys.executable, '-u', str(root/'archive_results.py')], stdout=handle, stderr=subprocess.STDOUT, check=True)
