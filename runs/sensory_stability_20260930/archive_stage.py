"""Archive a completed stage with a dictionary large enough to reuse fixed graph tensors."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('stage', choices=['stage1', 'stage2', 'final'])
parser.add_argument('--dataset', choices=['A', 'B'])
args = parser.parse_args()
assert not args.dataset or args.stage == 'stage2'
root = Path('runs/sensory_stability_20260930')
folder = root/args.stage
executions = sorted(folder.glob((args.dataset+'_*' if args.dataset else '*')+'/execution.json'))
assert len(executions) == (4 if args.dataset else {'stage1':9, 'stage2':8, 'final':2}[args.stage])
assert all(json.loads(path.read_text())['completed'] for path in executions)
archive = root/(args.stage+('_'+args.dataset if args.dataset else '')+'.tar.xz')
assert not archive.exists()
started = time.monotonic()
with archive.open('xb') as stream:
    compressor = subprocess.Popen(['xz', '-T1', '-1', '--lzma2=dict=512MiB,depth=1,nice=16', '-c'], stdin=subprocess.PIPE, stdout=stream)
    with tarfile.open(fileobj=compressor.stdin, mode='w|') as tar:
        for path in executions:
            tar.add(path.parent, arcname=path.parent.as_posix())
    compressor.stdin.close()
    assert compressor.wait() == 0
with archive.open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
report = dict(archive=str(archive), sha256=digest, bytes=archive.stat().st_size,
              wall_seconds=time.monotonic()-started)
archive.with_suffix('.json').write_text(json.dumps(report, indent=2))
print(json.dumps(report), flush=True)
