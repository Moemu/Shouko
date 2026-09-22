"""Capture completed route-diagnostic evidence and exact experimental scripts."""
import hashlib
import json
from pathlib import Path
import shutil
import tarfile
from datetime import datetime, timezone

workspace = Path.cwd()
root = workspace / 'runs/yumi_route_decision_20260922'
required = ['feedback.json', 'repeatability.json', 'ordered_csr.json', 'native_feedback27.json',
            'command_response.json', 'command_factorial.json', 'heading_limits.json']
for name in required:
    result = json.loads((root/name).read_text())
    # Older repeatability probe writes its complete result only once.
    if 'completed' in result:
        assert result['completed'], name
snapshot = root/'scripts'
snapshot.mkdir(exist_ok=True)
for name in ['ordered_csr.py', 'native_feedback.py', 'probe_repeatability.py', 'probe_segment_inference.py',
             'probe_commands.py', 'probe_command_factorial.py', 'probe_heading_limits.py', 'test_feedback.py']:
    shutil.copyfile(workspace/name, snapshot/name)
shutil.copyfile(workspace/'app/probe_feedback_control.py', snapshot/'probe_feedback_control.py')
files = {str(path.relative_to(root)).replace('\\','/'): hashlib.sha256(path.read_bytes()).hexdigest()
         for path in sorted(root.rglob('*')) if path.is_file() and path.suffix not in ['.tgz','.pyc']
         and path.name not in ['manifest.json','archive_sha256.txt'] and '__pycache__' not in path.parts}
manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), files=files,
                source_checkpoint_sha256=hashlib.sha256((workspace/'runs/yumi_obs50/best.pt').read_bytes()).hexdigest(),
                instance_policy='Keep running; no power or GPU release action',
                base_app_snapshot='See ../yumi_actor_scale_ab_20260922/deployment.json and verified results archive')
(root/'manifest.json').write_text(json.dumps(manifest,indent=2))
archive = workspace/'runs/yumi_route_decision_20260922.tgz'
with tarfile.open(archive,'w:gz') as handle:
    for name in [*files,'manifest.json']:
        handle.add(root/name,arcname=name)
digest=hashlib.sha256(archive.read_bytes()).hexdigest()
(root/'archive_sha256.txt').write_text(digest+'\n')
print(json.dumps(dict(files=len(files),archive_bytes=archive.stat().st_size,archive_sha256=digest)))
