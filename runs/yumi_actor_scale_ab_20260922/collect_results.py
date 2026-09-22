"""Seal cloud evidence and record idle resources, without releasing the instance."""
import datetime
import hashlib
import json
from pathlib import Path
import subprocess

root = Path('runs/yumi_actor_scale_ab_20260922')
execution = json.loads((root/'execution.json').read_text())
assert 'completed_utc' in execution
assert all(job['returncode'] == 0 for job in execution['jobs'])
assert json.loads((root/'verification.json').read_text())['source_unchanged']
resource = dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                instance_policy='keep_running',
                gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=name,utilization.gpu,memory.used,power.draw',
                                             '--format=csv,noheader'], text=True).strip(),
                gpu_processes=subprocess.check_output(['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory',
                                                       '--format=csv,noheader'], text=True).strip())
(root/'resource_finish.json').write_text(json.dumps(resource, indent=2))
files = {}
for path in sorted(root.rglob('*')):
    if path.is_file() and path.name != 'result_manifest.json':
        with path.open('rb') as handle:
            files[str(path.relative_to(root))] = dict(bytes=path.stat().st_size,
                                                     sha256=hashlib.file_digest(handle, 'sha256').hexdigest())
sources = {}
for path in [*sorted(Path('app').rglob('*.py')), Path('run_experiment.py'), Path('precheck.py'),
             Path('verify_results.py'), Path('collect_results.py')]:
    with path.open('rb') as handle:
        sources[str(path)] = hashlib.file_digest(handle, 'sha256').hexdigest()
manifest = dict(utc=resource['utc'], files=files, source_files=sources,
                instance_policy='keep_running', evaluation_scope='paired development screen only')
(root/'result_manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(dict(files=len(files), resource=resource)))
