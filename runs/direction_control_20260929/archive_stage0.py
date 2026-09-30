"""Archive completed diagnostics independently of concurrent teacher training."""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path('runs/direction_control_20260929')
archive = root/'stage0_complete.tgz'
manifest_path = root/'stage0_manifest.json'
if archive.exists() or manifest_path.exists():
    raise FileExistsError(archive)
paths = set()
for label in ['stage0_execution', 'probe_validation_execution']:
    p = root/(label+'.json'); data = json.loads(p.read_text()); assert data['completed']; paths.add(p)
    for job in data['jobs']:
        result = root/(job['name']+'.json'); paths.add(result)
        paths.add(root/(job['name']+'.log'))
        for c in json.loads(result.read_text())['conditions']:
            for row in c['results']:
                if 'trace' in row:
                    paths.add(root/row['trace'])
for name in ['run_stage0.py', 'run_probe_validation.py', 'analyze_stage0.py', 'stage0_analysis.json']:
    paths.add(root/name)
files = {str(p): dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size) for p in sorted(paths)}
initial = Path('../direction-source-20260929.tgz')
files['source_initial.tgz'] = dict(sha256=hashlib.sha256(initial.read_bytes()).hexdigest(), bytes=initial.stat().st_size)
with tarfile.open(archive, 'w:gz') as bundle:
    for p in sorted(paths):
        bundle.add(p, arcname=str(p))
    bundle.add(initial, arcname='source_initial.tgz')
manifest = dict(files=files, archive=archive.name, archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
manifest_path.write_text(json.dumps(manifest, indent=2))
print(json.dumps(dict(files=len(files), archive_sha256=manifest['archive_sha256'], bytes=archive.stat().st_size)))
