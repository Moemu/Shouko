"""Verify and recover the closing evidence archive within its experiment directory."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile

root = Path('runs/direction_control_20260929').resolve()
project = root.parent.parent
manifest = json.loads((root/'phase_closeout_manifest.json').read_text())
archive = root/manifest['archive']
with archive.open('rb') as f:
    digest = hashlib.file_digest(f, 'sha256').hexdigest()
assert digest == manifest['archive_sha256']
seen = set()
with tarfile.open(archive, 'r:gz') as bundle:
    for member in bundle:
        assert member.isfile() and member.name in manifest['files']
        target = ((root/'phase_closeout_source'/member.name) if member.name.startswith('app/') else (project/member.name)).resolve()
        assert target.is_relative_to(root)
        target.parent.mkdir(parents=True, exist_ok=True)
        expected = manifest['files'][member.name]
        same = False
        if target.exists() and target.stat().st_size == expected['bytes']:
            with target.open('rb') as f:
                same = hashlib.file_digest(f, 'sha256').hexdigest() == expected['sha256']
        if not same:
            with bundle.extractfile(member) as source, target.open('wb') as destination:
                shutil.copyfileobj(source, destination)
            with target.open('rb') as f:
                actual = hashlib.file_digest(f, 'sha256').hexdigest()
            assert actual == expected['sha256'] and target.stat().st_size == expected['bytes']
        seen.add(member.name)
assert seen == set(manifest['files'])
result = dict(verified_files=len(seen), archive_sha256=digest, completed=True)
(root/'phase_closeout_backup_verification.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result))
