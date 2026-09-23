"""Verify cloud artifacts, preserve source snapshots inside their archive, extract results."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import tarfile

parser = argparse.ArgumentParser()
parser.add_argument('archive')
parser.add_argument('--manifest', required=True)
args = parser.parse_args()
path = Path(args.archive)
root = path.parent.resolve()
with tarfile.open(path, 'r:gz') as archive:
    manifest_bytes = archive.extractfile(args.manifest).read()
    manifest = json.loads(manifest_bytes)
    for name, expected in manifest.items():
        relative = PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe archive path: '+name)
        member = archive.getmember(name)
        if not member.isfile():
            raise ValueError('Expected regular artifact: '+name)
        data = archive.extractfile(member).read()
        if hashlib.sha256(data).hexdigest() != expected:
            raise ValueError('Hash mismatch: '+name)
        if name.startswith('source_'):
            continue
        target = root.joinpath(*relative.parts).resolve()
        if not target.is_relative_to(root):
            raise ValueError('Artifact outside evidence directory')
        if target.exists():
            if hashlib.sha256(target.read_bytes()).hexdigest() != expected:
                raise ValueError('Different existing evidence: '+name)
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
    target = root/args.manifest
    if target.exists() and target.read_bytes() != manifest_bytes:
        raise ValueError('Different existing manifest')
    target.write_bytes(manifest_bytes)
print(json.dumps(dict(verified=len(manifest), manifest=args.manifest,
                      source_snapshots='retained in archive',
                      archive_sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
