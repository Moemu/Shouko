"""Archive completed teacher comparison and the exact training-source snapshot."""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path('runs/direction_control_20260929')
report = json.loads((root/'teacher_execution.json').read_text())
assert report['completed']
archive, manifest_path = root/'teachers_complete.tgz', root/'teachers_manifest.json'
if archive.exists() or manifest_path.exists():
    raise FileExistsError(archive)
paths = {root/name for name in ['teacher_execution.json', 'teacher_preregistration.json', 'teacher_selection.json',
                               'teacher_source.tgz', 'run_teacher_pair.py']}
for branch in report['branches']:
    paths.update(p for p in (root/branch).rglob('*') if p.is_file())
for job in report['jobs']:
    for suffix in ['.json', '.log']:
        p = root/(job['name']+suffix)
        if p.exists():
            paths.add(p)
            if suffix == '.json':
                data = json.loads(p.read_text())
                for c in data.get('conditions', []):
                    for row in c['results']:
                        if 'trace' in row:
                            paths.add(root/row['trace'])
files = {str(p): dict(sha256=hashlib.sha256(p.read_bytes()).hexdigest(), bytes=p.stat().st_size) for p in sorted(paths)}
with tarfile.open(archive, 'w:gz') as bundle:
    for p in sorted(paths):
        bundle.add(p, arcname=str(p))
manifest = dict(files=files, archive=archive.name, archive_sha256=hashlib.sha256(archive.read_bytes()).hexdigest())
manifest_path.write_text(json.dumps(manifest, indent=2))
print(json.dumps(dict(files=len(files), archive_sha256=manifest['archive_sha256'], bytes=archive.stat().st_size)))
