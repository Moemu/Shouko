"""Verify each bootstrap member before admitting the new isolated experiment."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path.cwd().resolve();manifest=json.loads(Path('../sensory-bootstrap-manifest.json').read_text())
archive=Path('../sensory-bootstrap.tgz')
with archive.open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==manifest['archive_sha256']
seen=set()
with tarfile.open(archive,'r:gz') as bundle:
    for member in bundle:
        assert member.isfile() and member.name in manifest['files']
        destination=(root/member.name).resolve();assert destination.is_relative_to(root)
        destination.parent.mkdir(parents=True,exist_ok=True)
        payload=bundle.extractfile(member).read();meta=manifest['files'][member.name]
        assert len(payload)==meta['bytes'] and hashlib.sha256(payload).hexdigest()==meta['sha256']
        if destination.exists():assert destination.read_bytes()==payload
        else:destination.write_bytes(payload)
        seen.add(member.name)
assert seen==set(manifest['files'])
base=root.parent.parent
for name in ['data','vendor']:
    (root/name).symlink_to(base/name,target_is_directory=True)
Path('runs/sensory_access_20260930').mkdir(exist_ok=True)
Path('runs/sensory_access_20260930/bootstrap_verified.json').write_text(json.dumps(dict(completed=True,files=len(seen),archive_sha256=manifest['archive_sha256']),indent=2))
print(json.dumps(dict(verified_files=len(seen))))
