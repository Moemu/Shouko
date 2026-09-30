"""Archive bounded PPO attempts and their exact source after completion."""
import hashlib
import json
from pathlib import Path
import tarfile

root = Path('runs/direction_control_20260929')
for name in ['readout_ppo_execution.json', 'readout_ppo_warm_execution.json']:
    assert json.loads((root/name).read_text())['completed']
paths = set(Path('app').rglob('*.py'))
paths.update(root.glob('*.py'))
paths.update(root.glob('readout*.json'))
paths.update(root.glob('readout*.log'))
for directory in ['readout_ppo3029', 'readout_warm_initial', 'readout_ppo_warm3029']:
    paths.update(p for p in (root/directory).rglob('*') if p.is_file())
def sha(p):
    with p.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()
archive = root/'readout_attempts_complete.tgz'
if archive.exists():
    raise FileExistsError(archive)
files = {str(p): dict(sha256=sha(p), bytes=p.stat().st_size) for p in sorted(paths)}
with tarfile.open(archive, 'w:gz', compresslevel=1) as bundle:
    for p in sorted(paths):
        bundle.add(p, arcname=str(p))
manifest = dict(files=files, archive=archive.name, archive_sha256=sha(archive))
(root/'readout_attempts_manifest.json').write_text(json.dumps(manifest, indent=2))
print(json.dumps(dict(files=len(files), archive_sha256=manifest['archive_sha256'], bytes=archive.stat().st_size)))
