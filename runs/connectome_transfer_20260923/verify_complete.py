import argparse
import hashlib
import json
from pathlib import Path,PurePosixPath
import shutil
import tarfile
import time

p=argparse.ArgumentParser();p.add_argument('archive');p.add_argument('--manifest',required=True);a=p.parse_args()
archive_path=Path(a.archive);root=archive_path.parent.resolve();manifest_path=Path(a.manifest)
manifest=json.loads(manifest_path.read_text());expected={row['name']:row for row in manifest['files']};seen=set()
prefix='runs/'+root.name+'/'
with tarfile.open(archive_path,'r|gz') as archive:
    for member in archive:
        name=member.name;relative=PurePosixPath(name)
        if relative.is_absolute() or '..' in relative.parts or not member.isfile():raise ValueError('Unsafe member '+name)
        stream=archive.extractfile(member)
        if name not in expected:
            if relative.name!='complete_manifest.json':raise ValueError('Unlisted member '+name)
            if json.loads(stream.read())!=manifest:raise ValueError('Manifest mismatch')
            continue
        row=expected[name];seen.add(name)
        if name.startswith('actual_source/'):
            if hashlib.file_digest(stream,'sha256').hexdigest()!=row['sha256']:raise ValueError(name)
            continue
        if not name.startswith(prefix):raise ValueError('Wrong artifact root '+name)
        target=root.joinpath(*PurePosixPath(name[len(prefix):]).parts).resolve()
        if not target.is_relative_to(root):raise ValueError('Outside artifact directory')
        target.parent.mkdir(parents=True,exist_ok=True)
        tmp=target.with_name(target.name+'.verify-tmp')
        with tmp.open('wb') as handle:shutil.copyfileobj(stream,handle,1024*1024)
        with tmp.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
        if digest!=row['sha256']:raise ValueError('Hash mismatch '+name)
        if target.exists():
            with target.open('rb') as handle:old=hashlib.file_digest(handle,'sha256').hexdigest()
            if old==digest:tmp.unlink();continue
            if target.name.endswith('_execution.json') or target.name=='execution.json':
                snapshot=target.with_name(target.stem+'.snapshot-'+old[:12]+'.json')
                if snapshot.exists():raise FileExistsError(snapshot)
                target.rename(snapshot)
            else:
                raise ValueError('Different immutable local artifact '+str(target))
        for attempt in range(6):
            try:
                tmp.replace(target)
                break
            except PermissionError:
                if attempt == 5:
                    raise
                time.sleep(1)
if seen!=set(expected):raise ValueError('Missing members')
with archive_path.open('rb') as handle:digest=hashlib.file_digest(handle,'sha256').hexdigest()
print(json.dumps(dict(verified=len(seen),archive_sha256=digest,source_snapshots='retained in archive')))
