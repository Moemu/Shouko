"""Hash completed artifacts and the actual source, then create a self-contained audit archive."""
import hashlib
import json
from pathlib import Path
import tarfile

root=Path('runs/connectome_transfer_balance_20260923')
output=root/'complete.tgz'
manifest=root/'complete_manifest.json'
if output.exists() or manifest.exists(): raise FileExistsError(output)
assert json.loads((root/'execution.json').read_text())['completed']
files=[p for p in root.rglob('*') if p.is_file() and p.suffix not in ['.tgz','.pid']
       and p.name not in ['archive.log','complete_manifest.json','archive_results.py','features.pt']]
sources=[p for p in Path('app').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
entries=[]
with tarfile.open(output,'w:gz',compresslevel=1) as archive:
    for p in sorted(files+sources):
        name=str(p) if p in files else 'actual_source/'+str(p)
        with p.open('rb') as handle: digest=hashlib.file_digest(handle,'sha256').hexdigest()
        entries.append(dict(name=name,sha256=digest,bytes=p.stat().st_size))
        archive.add(p,arcname=name,recursive=False)
    payload=dict(files=entries,source_checkpoint_sha256='458fc465a6fb802a338c218fc4be414fc9153376c5d2f76f116518e6cc5aa475',
        teacher_checkpoint_sha256='9c5ca9339b26f5af2e8acf94e7edd3b9371ef5efb41d647bcbd7904243ac50e1')
    manifest.write_text(json.dumps(payload,indent=2))
    archive.add(manifest,arcname=str(manifest))
with output.open('rb') as handle: digest=hashlib.file_digest(handle,'sha256').hexdigest()
print(json.dumps(dict(archive=str(output),sha256=digest,files=len(entries),bytes=output.stat().st_size)),flush=True)
