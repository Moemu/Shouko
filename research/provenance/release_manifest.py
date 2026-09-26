# Release manifest 生成与校验。
# 用法：
#   python research/provenance/release_manifest.py generate   # 重新计算全部 SHA-256 并写 release.json
#   python research/provenance/release_manifest.py validate   # 校验当前文件与 release.json 一致
# 只依赖标准库。发布通道（GitHub Release 等）建立后，本清单随附件一起发布，
# 社区用户据此做“下载 -> 哈希校验 -> 页面可用”的一步校验。

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = Path(__file__).with_name('release.json')

# (发布单元, 相对路径或 glob, 角色, 是否必需)
# required=False 表示该文件缺失不视为发布阻塞（如可选的 Yumi 角色）。
ARTIFACTS = [
    # L0：8192 神经元子图读出，setup.ps1 可自动复训（分钟级），权重可不发
    ('L0_subgraph', 'runs/best.npz', 'weight', True),
    ('L0_subgraph', 'runs/readout_*.npz', 'weight', True),
    ('L0_subgraph', 'runs/training_features.npz', 'feature', True),
    # L1：全图 + G1 身体，云 GPU 数小时训练，必须发布权重
    ('L1_full_g1', 'runs/cloud/best.pt', 'weight', True),
    ('L1_full_g1', 'runs/cloud/demonstrations.pt', 'dagger_buffer', True),
    ('L1_full_g1', 'runs/cloud/training.json', 'training_record', True),
    ('L1_full_g1', 'runs/cloud/evaluation.json', 'screening_eval', True),
    ('L1_full_g1', 'runs/cloud/heldout.json', 'acceptance_legacy', True),
    ('L1_full_g1', 'runs/local/heldout.json', 'acceptance_legacy', True),
    ('L1_full_g1', 'runs/cloud/evaluations/*/heldout.json', 'acceptance', True),
    # L2：Yumi PPO（配置 yumi.yaml/xml 已入库，不在此清单）
    ('L2_yumi', 'runs/yumi/best.pt', 'weight', True),
    ('L2_yumi', 'runs/yumi/ppo_state_best.pt', 'optimizer_state', True),
    ('L2_yumi', 'runs/yumi/evaluation.json', 'screening_eval', True),
    ('L2_yumi', 'runs/local/heldout_yumi.json', 'acceptance', True),
    # L3：Yumi VRM 许可禁止再分发，仅记录哈希供自行获取者校验
    ('L3_yumi_vrm', 'web/public/yumi.vrm', 'avatar_reference_only', False),
    # 默认角色（three-vrm 示例模型，随库可分发）
    ('avatar', 'web/public/avatar.vrm', 'avatar', True),
]

UNIT_NOTES = {
    'L0_subgraph': 'setup.ps1 自动训练；复训命令与种子见 README「复现训练过程」。',
    'L1_full_g1': '训练与验收经过见 docs/CLOUD.md 与 research/experiments/；'
                  '2026-09-18 独立验收绑定 best.pt（runs/cloud/evaluations/）。',
    'L2_yumi': 'PPO 编年与验收见 research/experiments/PPO_POSTURE_20260916.md；'
               'best.pt 为 9/9 验收世系（a7a4281f…）。',
    'L3_yumi_vrm': '禁止再分发；获取指引与许可见 research/avatar/YUMI.md。缺失时页面回退默认角色。',
    'avatar': 'pixiv/three-vrm 示例模型，哈希同源记录于 provenance.json。',
}


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b''):
            digest.update(chunk)
    return digest.hexdigest()


def collect():
    entries = []
    for unit, pattern, role, required in ARTIFACTS:
        matches = sorted(ROOT.glob(pattern)) if any(c in pattern for c in '*?[') else [ROOT / pattern]
        for path in matches:
            entry = {
                'unit': unit,
                'path': path.relative_to(ROOT).as_posix(),
                'role': role,
                'bytes': path.stat().st_size,
                'sha256': sha256_of(path),
            }
            if role == 'acceptance':
                record = json.loads(path.read_text(encoding='utf-8'))
                entry['bound_checkpoint_sha256'] = record.get('checkpoint_sha256')
                entry['successes'] = record.get('successes')
                entry['attempts'] = record.get('attempts')
            entries.append(entry)
        if required and not matches:
            raise SystemExit(f'缺少必需产物：{unit} {pattern}')
    return entries


def generate():
    provenance = json.loads((ROOT / 'research/provenance/provenance.json').read_text(encoding='utf-8'))
    manifest = {
        'generated_by': 'research/provenance/release_manifest.py',
        'graph_sha256': provenance['graph']['graph_sha256'],
        'code_revision': None,
        'code_revision_note': '仓库尚无 git 提交；建立远程仓库后应填首个发布的 commit SHA。',
        'units': {unit: {'note': note} for unit, note in UNIT_NOTES.items()},
        'artifacts': collect(),
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'已写入 {MANIFEST.relative_to(ROOT)}（{len(manifest["artifacts"])} 个产物）')


def validate():
    manifest = json.loads(MANIFEST.read_text(encoding='utf-8'))
    failures = []
    for entry in manifest['artifacts']:
        path = ROOT / entry['path']
        if not path.exists():
            failures.append(f'缺失：{entry["path"]}')
            continue
        if path.stat().st_size != entry['bytes']:
            failures.append(f'大小变更：{entry["path"]}')
            continue
        if sha256_of(path) != entry['sha256']:
            failures.append(f'哈希不一致：{entry["path"]}')
    if failures:
        print('校验失败：')
        for line in failures:
            print(' -', line)
        raise SystemExit(1)
    print(f'校验通过：{len(manifest["artifacts"])} 个产物与 release.json 一致。')


if __name__ == '__main__':
    command = sys.argv[1] if len(sys.argv) > 1 else 'validate'
    if command == 'generate':
        generate()
    elif command == 'validate':
        validate()
    else:
        raise SystemExit('用法：release_manifest.py [generate|validate]')
