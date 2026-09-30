"""Package exact source and fixed training inputs, never private connection settings."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path('runs/sensory_access_20260930')
old=Path('runs/direction_control_20260929')
fit=json.loads((old/'transfer_main/round4/fit.json').read_text(encoding='utf-8'))
paths=set(p for p in Path('app').rglob('*') if p.is_file() and '__pycache__' not in p.parts)
paths.update(Path(p) for p in fit['data_sha256'])
paths.update(p.with_suffix('.json') for p in list(paths) if p.suffix=='.pt' and p.with_suffix('.json').exists())
paths.update(old/p for p in ['teacher_control3029/last.pt','transfer_main/round4/last.pt',
    'transfer_main/round4/features.pt','transfer_main/round4/fit.json'])
def sha(path):
    with path.open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
for p,h in fit['data_sha256'].items():assert sha(Path(p))==h
files={str(p.as_posix()):dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(paths)}
archive=root/'bootstrap.tgz'
with tarfile.open(archive,'w:gz',compresslevel=1) as bundle:
    for p in sorted(paths):bundle.add(p,arcname=p.as_posix())
report=dict(files=files,archive_sha256=sha(archive),archive_bytes=archive.stat().st_size)
(root/'bootstrap_manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(dict(files=len(files),archive_bytes=archive.stat().st_size,archive_sha256=report['archive_sha256'])))
