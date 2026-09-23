"""Build and verify the allowlisted v0.2.0 model release."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'runs/releases/v0.2.0'
MODELS = {
    'primary': ('runs/connectome_transfer_ridge_20260923/ridge1e4/last.pt',
                'f1a200718ebfec845cbf9c02ba743e937ea94c747fba53e7b12dfd12f8d3e1f9'),
    'replica': ('runs/connectome_transfer_replica_20260923/extra1/last.pt',
                '8841868cf1cbbbd01be19585bc18e751c11901b3e806556705d124bb207616cf'),
    'upright-source': ('runs/yumi/upright-source.pt',
                '458fc465a6fb802a338c218fc4be414fc9153376c5d2f76f116518e6cc5aa475'),
    'mlp-teacher': ('runs/yumi_route_20260923/support_refine2027/last.pt',
                '9c5ca9339b26f5af2e8acf94e7edd3b9371ef5efb41d647bcbd7904243ac50e1'),
}
GRAPH = '72a6a9ade0117063d3b2515b12ceec3633d9f429ae43f0f8b73b2066a9e51b89'


def sha(path):
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def inventory():
    files = {}
    for role, (path, digest) in MODELS.items():
        source = ROOT/path
        if sha(source) != digest:
            raise ValueError('Changed source model: '+role)
        files['models/'+role+'.pt'] = source
    if sha(ROOT/'data/full_graph.npz') != GRAPH:
        raise ValueError('Changed graph')
    for name in ['full_graph.npz', 'full_graph.json']:
        files['data/'+name] = ROOT/'data'/name
    modules = ['__init__', 'full_brain', 'sim', 'evaluate_locomotion', 'ordered_inference',
               'mlp_policy', 'train_full', 'gpu_body', 'teacher', 'training_control', 'imitate_yumi', 'release_evidence']
    for name in modules:
        files['app/'+name+'.py'] = ROOT/'app'/(name+'.py')
    for name in ['scene.xml', 'yumi.xml', 'yumi.yaml']:
        files['app/yumi_description/'+name] = ROOT/'app/yumi_description'/name
    for name in ['LICENSE', 'THIRD_PARTY.md', 'requirements.cloud.lock.txt']:
        files[name] = ROOT/name
    for source in sorted((ROOT/'research/provenance/licenses').glob('*')):
        if source.is_file():
            files[source.relative_to(ROOT).as_posix()] = source
    for role, directory in [('primary', 'connectome_transfer_ridge_20260923'),
                             ('replica', 'connectome_transfer_replica_20260923')]:
        for name in ['holdout', 'yaw_holdout', 'long_holdout', 'push_holdout', 'native_csr']:
            source = ROOT/'runs'/directory/(name+'.json')
            data = json.loads(source.read_text(encoding='utf-8'))
            if not data['completed'] or any(c['checkpoint_sha256'] != MODELS[role][1] for c in data['conditions']):
                raise ValueError('Unbound evaluation: '+str(source))
            runtime_hashes = {'app/'+name: digest for name, digest in data['source_sha256'].items()}
            runtime_hashes.update(data['body_xml_sha256'])
            for path, digest in runtime_hashes.items():
                if sha(ROOT/path) != digest:
                    raise ValueError('Runtime differs from evaluated source: '+path)
            files['evidence/'+role+'/'+name+'.json'] = source
    extras = {
        'evidence/primary/lesion.json': 'runs/connectome_transfer_ridge_20260923/lesion.json',
        'evidence/primary/cache_audit.json': 'runs/connectome_transfer_ridge_20260923/cache_and_freeze_audit.json',
        'evidence/feature_row_audit.json': 'runs/connectome_transfer_replica_20260923/feature_row_audit.json',
        'evidence/replica/chain_audit.json': 'runs/connectome_transfer_replica_20260923/chain_audit.json',
        'evidence/final_summary.json': 'runs/connectome_transfer_replica_20260923/final_summary.json',
        'MODEL_CARD.md': 'research/releases/v0.2.0/MODEL_CARD.md',
        'REPRODUCE.md': 'research/releases/v0.2.0/REPRODUCE.md',
        'STUDIO.md': 'research/releases/v0.2.0/STUDIO.md',
        'smoke.py': 'research/releases/smoke_v020.py',
    }
    for name, source in extras.items():
        files[name] = ROOT/source
    return files


def build():
    if OUT.exists():
        raise FileExistsError('Release directory exists; preserve it and choose another output directory')
    files = inventory()
    tracked = subprocess.check_output(['git', 'ls-files', '-z'], cwd=ROOT).decode().split('\0')
    for source in files.values():
        relative = source.relative_to(ROOT).as_posix()
        if relative.startswith(('app/', 'research/releases/')) or relative in ['LICENSE', 'THIRD_PARTY.md', 'requirements.cloud.lock.txt']:
            if relative not in tracked:
                raise ValueError('Uncommitted release source: '+relative)
            committed = subprocess.check_output(['git', 'show', 'HEAD:'+relative], cwd=ROOT)
            if hashlib.sha256(committed).hexdigest() != sha(source):
                raise ValueError('Release source differs from HEAD: '+relative)
    entries = [dict(path=name, bytes=source.stat().st_size, sha256=sha(source))
               for name, source in sorted(files.items())]
    OUT.mkdir(parents=True)
    manifest = dict(status='release', version='v0.2.0',
        source_head=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
        source_commit_is_release=True, source_note='Runtime and release documents match source_head; publish tag v0.2.0 at this commit.',
        graph_sha256=GRAPH, models={role:dict(path='models/'+role+'.pt',sha256=values[1]) for role,values in MODELS.items()},
        files=entries, exclusions=['Yumi VRM', 'private cloud config', 'optimizer states', 'private logs'])
    (OUT/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    archive = OUT/'shouko-v0.2.0.tgz'
    with tarfile.open(archive,'w:gz',compresslevel=1) as handle:
        for name, source in sorted(files.items()):
            handle.add(source, arcname=name, recursive=False)
        handle.add(OUT/'manifest.json',arcname='manifest.json')
    (OUT/'SHA256SUMS.txt').write_text(sha(archive)+'  '+archive.name+'\n'+sha(OUT/'manifest.json')+'  manifest.json\n',encoding='utf-8')
    verify()


def verify():
    manifest=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    expected={row['path']:row for row in manifest['files']}
    seen=set()
    with tarfile.open(OUT/'shouko-v0.2.0.tgz','r|gz') as handle:
        for member in handle:
            if not member.isfile() or member.name in seen:
                raise ValueError('Unexpected archive member: '+member.name)
            seen.add(member.name)
            stream=handle.extractfile(member)
            if member.name=='manifest.json':
                if json.loads(stream.read()) != manifest:
                    raise ValueError('Embedded manifest differs')
            else:
                row=expected[member.name]
                if member.size!=row['bytes'] or hashlib.file_digest(stream,'sha256').hexdigest()!=row['sha256']:
                    raise ValueError('Archive mismatch: '+member.name)
    if seen != set(expected)|{'manifest.json'}:
        raise ValueError('Archive inventory differs')
    for line in (OUT/'SHA256SUMS.txt').read_text().splitlines():
        digest,name=line.split('  ',1)
        if sha(OUT/name)!=digest:
            raise ValueError('Asset checksum differs: '+name)
    print(json.dumps(dict(verified_files=len(expected),archive_sha256=sha(OUT/'shouko-v0.2.0.tgz'))))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['build','verify'])
    args=parser.parse_args()
    (build if args.command=='build' else verify)()
