"""Preserve final teacher checks and direction traces without duplicating model archives."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path('runs/direction_control_20260929')
assert json.loads((root/'teacher_repro_execution.json').read_text())['completed']
assert json.loads((root/'yaw_gain28_execution.json').read_text())['completed']
paths=set(p for p in root.iterdir() if p.is_file() and p.suffix in {'.py','.json','.npz','.log','.md'})
paths.update(p for p in (root/'teacher_control3030').rglob('*') if p.is_file())
paths.update(Path('app').rglob('*.py'))
paths.update(Path('app/yumi_description').glob('*.xml'))
archive=root/'phase_closeout_complete.tgz'
if archive.exists():
    raise FileExistsError(archive)
def sha(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()
files={str(p):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(paths)}
with tarfile.open(archive,'w:gz',compresslevel=1) as bundle:
    for p in sorted(paths):
        bundle.add(p,arcname=str(p))
manifest=dict(files=files,archive=archive.name,archive_sha256=sha(archive))
(root/'phase_closeout_manifest.json').write_text(json.dumps(manifest,indent=2))
print(json.dumps(dict(files=len(files),archive_sha256=manifest['archive_sha256'],bytes=archive.stat().st_size)))
