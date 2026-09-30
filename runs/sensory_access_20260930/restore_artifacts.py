"""Verify and recover completed experiment artifacts without replacing conflicting files."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('archive');parser.add_argument('sha256');args=parser.parse_args()
archive=Path(args.archive)
def sha(path):
    with Path(path).open('rb') as f:return hashlib.file_digest(f,'sha256').hexdigest()
assert sha(archive)==args.sha256
workspace=Path.cwd().resolve();allowed=(workspace/'runs/sensory_access_20260930').resolve();rows=[]
with tarfile.open(archive) as tar:
    for member in tar:
        path=(workspace/member.name).resolve()
        if not path.is_relative_to(allowed):raise ValueError(member.name)
        if member.isdir():continue
        if not member.isfile():raise ValueError(member.name)
        payload=tar.extractfile(member).read();digest=hashlib.sha256(payload).hexdigest()
        if path.exists():
            if sha(path)!=digest:raise ValueError(f'Conflicting local artifact: {member.name}')
        else:
            path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payload)
        rows.append(dict(path=member.name,sha256=digest,bytes=len(payload)))
archive.with_suffix('.verified.json').write_text(json.dumps(dict(archive_sha256=args.sha256,files=rows),indent=2))
print(json.dumps(dict(verified_files=len(rows),bytes=sum(r['bytes'] for r in rows))))
