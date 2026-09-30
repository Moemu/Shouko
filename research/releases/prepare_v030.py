"""Build the allowlisted v0.3.0 runtime; never replace an existing release directory."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from app.release_evidence import build_direction_report, body_xml_hashes

MODELS = {
    'primary': ('runs/sensory_stability_20260930/stage2/A_3031_s1/train/last.pt',
                '1e1c3160448f6acaddd28b1a8782dc75b037fed3e296525aa52ae060926801e6'),
    'replica': ('runs/sensory_access_20260930/replica/dagger/round1/last.pt',
                'fe386851ed9aa910aa837767ca647967388fba26b10c1c13784d82f996d2c916'),
}
GRAPH = '72a6a9ade0117063d3b2515b12ceec3633d9f429ae43f0f8b73b2066a9e51b89'
ARCHIVE = 'shouko-v0.3.0.tgz'


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def inventory():
    files = {}
    for role, (path, digest) in MODELS.items():
        if sha(ROOT/path) != digest:
            raise ValueError('Changed model: '+role)
        files['models/'+role+'.pt'] = ROOT/path
        directory = ROOT/'research/releases/v0.3.0/evidence'/role
        report = build_direction_report(directory, digest)
        if report['successes'] != 72 or report['long_corridor_passes'] != 9:
            raise ValueError('Acceptance failed: '+role)
        if report['body_xml_sha256'] != body_xml_hashes(ROOT):
            raise ValueError('Body differs from evidence')
        for name, expected in report['source_sha256'].items():
            if sha(ROOT/'app'/name) != expected:
                raise ValueError('Runtime differs from evidence: '+name)
        for name in ['normal', 'yaw', 'long', 'push_positive', 'push_negative', 'release_evaluation']:
            files[f'evidence/{role}/{name}.json'] = directory/(name+'.json')
        if json.loads((directory/'release_evaluation.json').read_text(encoding='utf-8')) != report:
            raise ValueError('Derived report differs: '+role)
        long = json.loads((directory/'long.json').read_text(encoding='utf-8'))
        source = 'main' if role == 'primary' else 'verification'
        for cohort in long['conditions'][0]['results']:
            trace = ROOT/'runs/sensory_stability_20260930/final'/source/cohort['trace']
            if sha(trace) != cohort['trace_sha256']:
                raise ValueError('Trajectory differs: '+role)
            files[f'evidence/{role}/{trace.name}'] = trace
    if sha(ROOT/'data/full_graph.npz') != GRAPH:
        raise ValueError('Graph differs')
    for name in ['full_graph.npz', 'full_graph.json']:
        files['data/'+name] = ROOT/'data'/name
    modules = ['__init__', 'full_brain', 'sim', 'evaluate_locomotion', 'evaluate_direction',
               'direction_metrics', 'ordered_inference', 'mlp_policy', 'train_full',
               'gpu_body', 'teacher', 'training_control', 'release_evidence']
    for name in modules:
        files['app/'+name+'.py'] = ROOT/'app'/(name+'.py')
    for name in ['scene.xml', 'yumi.xml', 'yumi.yaml']:
        files['app/yumi_description/'+name] = ROOT/'app/yumi_description'/name
    for name in ['LICENSE', 'THIRD_PARTY.md', 'requirements.cloud.lock.txt']:
        files[name] = ROOT/name
    for path in (ROOT/'research/provenance/licenses').glob('*'):
        if path.is_file():
            files[path.relative_to(ROOT).as_posix()] = path
    for name in ['MODEL_CARD.md', 'REPRODUCE.md', 'DECISION.md', 'RELEASE.md', 'VALIDATION.md']:
        files[name] = ROOT/'research/releases/v0.3.0'/name
    files['smoke.py'] = ROOT/'research/releases/smoke_v030.py'
    for name in ['matrix.json', 'analysis.json', 'final.json', 'closeout.json', 'preregistration.json', 'protocol_snapshot.md']:
        files['provenance/'+name] = ROOT/'runs/sensory_stability_20260930'/name
    for name in ['SENSORY_ACCESS_20260930.md', 'SENSORY_STABILITY_20260930.md', 'DIRECTION_CONTROL_20260929.md']:
        files['provenance/'+name] = ROOT/'research/experiments'/name
    return files


def build(out, candidate):
    if out.exists():
        raise FileExistsError(out)
    files = inventory()
    if not candidate:
        for source in files.values():
            if source.suffix in ['.pt', '.npz'] or source.parent == ROOT/'data':
                continue
            relative = source.relative_to(ROOT).as_posix()
            committed = subprocess.check_output(['git', 'show', 'HEAD:'+relative], cwd=ROOT)
            if hashlib.sha256(committed).hexdigest() != sha(source):
                raise ValueError('Release source differs from HEAD: '+relative)
    manifest = dict(version='v0.3.0', status='candidate' if candidate else 'release',
                    source_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    source_commit_is_release=not candidate, graph_sha256=GRAPH,
                    models={role: dict(path='models/'+role+'.pt', sha256=value[1]) for role, value in MODELS.items()},
                    files=[dict(path=name, bytes=path.stat().st_size, sha256=sha(path)) for name, path in sorted(files.items())],
                    exclusions=['VRM/avatar', 'private endpoints', 'historical training datasets', 'teacher/ancestor checkpoints', 'optimizer states'],
                    reproduction='Inference and evaluation; not byte-identical historical training.')
    out.mkdir(parents=True)
    (out/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    with tarfile.open(out/ARCHIVE, 'w:gz', compresslevel=1) as handle:
        for name, source in sorted(files.items()):
            handle.add(source, arcname=name, recursive=False)
        handle.add(out/'manifest.json', arcname='manifest.json')
    (out/'SHA256SUMS.txt').write_text(''.join(sha(out/name)+'  '+name+'\n' for name in [ARCHIVE, 'manifest.json']), encoding='utf-8')
    verify(out)


def verify(out):
    manifest = json.loads((out/'manifest.json').read_text(encoding='utf-8'))
    expected = {row['path']: row for row in manifest['files']}
    seen = set()
    with tarfile.open(out/ARCHIVE, 'r|gz') as handle:
        for member in handle:
            if not member.isfile() or member.name in seen:
                raise ValueError('Unexpected member: '+member.name)
            seen.add(member.name)
            stream = handle.extractfile(member)
            if member.name == 'manifest.json':
                if json.loads(stream.read()) != manifest:
                    raise ValueError('Embedded manifest differs')
            else:
                row = expected[member.name]
                if member.size != row['bytes'] or hashlib.file_digest(stream, 'sha256').hexdigest() != row['sha256']:
                    raise ValueError('Archive mismatch: '+member.name)
    if seen != set(expected) | {'manifest.json'}:
        raise ValueError('Incomplete archive')
    for line in (out/'SHA256SUMS.txt').read_text().splitlines():
        digest, name = line.split('  ', 1)
        if sha(out/name) != digest:
            raise ValueError('Checksum mismatch: '+name)
    print(json.dumps(dict(verified_files=len(expected), archive_sha256=sha(out/ARCHIVE))))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['candidate', 'build', 'verify'])
    parser.add_argument('--out', type=Path, default=ROOT/'runs/releases/v0.3.0')
    args = parser.parse_args()
    if args.command == 'verify':
        verify(args.out)
    else:
        build(args.out, args.command == 'candidate')
