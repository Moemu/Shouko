"""Extract the bounded cloud archive and verify every artifact against its manifest."""
import datetime
import hashlib
import json
from pathlib import Path
import tarfile

root = Path(__file__).resolve().parent
workspace = root.parents[1]


def sha256(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


digest = sha256(root/'results.tgz')
assert digest == '225d716b310837fee9ee4b3bc4b8f3fe08e2d0b461830c509153d6f907bf4b60'
with tarfile.open(root/'results.tgz', 'r:gz') as archive:
    for member in archive.getmembers():
        target = (workspace/member.name).resolve()
        assert target.is_relative_to(root) and (member.isfile() or member.isdir()), member.name
    archive.extractall(workspace, filter='data')
manifest = json.loads((root/'result_manifest.json').read_text())
for name, expected in manifest['files'].items():
    path = root/name
    assert path.stat().st_size == expected['bytes'], name
    assert sha256(path) == expected['sha256'], name
for name, expected in manifest['source_files'].items():
    path = workspace/name if name.startswith('app/') else root/name
    assert sha256(path) == expected, name
record = dict(verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
               archive_sha256=digest, files_verified=len(manifest['files']),
               source_files_verified=len(manifest['source_files']), instance_policy='keep_running')
(root/'backup_verification.json').write_text(json.dumps(record, indent=2), encoding='utf-8')
print(json.dumps(record))
