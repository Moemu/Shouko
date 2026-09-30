"""Preserve supervised migration evidence, excluding duplicate regenerable feature caches."""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--branch', choices=['main', 'replica'], required=True)
args = parser.parse_args()
root = Path('runs/direction_control_20260929'); folder = root/('transfer_'+args.branch)
assert json.loads((folder/'execution.json').read_text())['completed']
archive = root/f'supervised_{args.branch}_complete.tgz'
manifest_path = root/f'supervised_{args.branch}_manifest.json'
if archive.exists() or manifest_path.exists():
    raise FileExistsError(archive)


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


paths, excluded = [], {}
for path in sorted(folder.rglob('*')):
    if not path.is_file():
        continue
    if path.parent.name == 'feature_cache' and path.suffix == '.pt':
        excluded[str(path)] = dict(sha256=sha(path), bytes=path.stat().st_size)
    else:
        paths.append(path)
paths.extend(root/name for name in ['migration_source.tgz', 'run_refit_probe.py', 'refit_cached.py', 'teacher_selection.json'])
files = {str(p): dict(sha256=sha(p), bytes=p.stat().st_size) for p in paths}
with tarfile.open(archive, 'w:gz', compresslevel=3) as bundle:
    for p in paths:
        bundle.add(p, arcname=str(p))
manifest = dict(files=files, archive=archive.name, archive_sha256=sha(archive),
                excluded_regenerable_feature_cache=excluded,
                exclusion_reason='Per-dataset acceleration copies remain on the cloud. Every round features.pt, raw dataset, weight and fitting source is archived.')
manifest_path.write_text(json.dumps(manifest, indent=2))
print(json.dumps(dict(files=len(files), excluded_cache_files=len(excluded), archive_sha256=manifest['archive_sha256'], bytes=archive.stat().st_size)))
