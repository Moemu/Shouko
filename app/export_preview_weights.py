"""Export the browser preview package from a trained checkpoint.

The preview page (web/preview.html) runs the connectome and the physics inside
the visitor's browser, so everything it needs must ship as plain files under one
base URL. This module is the single source of that package: every artifact is
derived from runs/yumi/best.pt plus data/full_graph.npz, and meta.json records
the checkpoint and graph hashes so the page shows and verifies what it is
running (page numbers stay bound to the checkpoint hash).

Design notes:

- Files are content-addressed (`<key>.<sha8>.<ext>`) so a static host can serve
  them `immutable` and a returning visitor never re-downloads 98MB.
- The somata sample indices and their coordinates come from one call, so the
  panel cannot drift from the computation the way two hand-copied formulas can.
- `physics_interface` from the checkpoint travels with the package, and
  `body_config.json` is regenerated from that same interface, so the browser
  body cannot silently diverge from the interface the policy was trained
  against.

Run: .venv-gpu/Scripts/python.exe -m app.export_preview_weights
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.full_brain import ConnectomePolicy, checkpoint_configuration  # noqa: E402
from app.release_evidence import matches_checkpoint  # noqa: E402
from app.sim import Body  # noqa: E402

SAMPLE_COUNT = 2048
RESET_SEED = 4242
MUJOCO_PACKAGE = Path('bench_mujoco/node_modules/@mujoco/mujoco')


def sha256_hex(path):
    with Path(path).open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def clear_previous(out):
    """Drop the artifacts of an earlier export before writing a new one.

    Names are content-addressed, so exporting a different checkpoint would leave
    the previous hashed files behind — bloating the package and letting a deployer
    ship a mix of two revisions. Only extensions this script writes are removed.
    """
    # Sidecars (.zst/.gz) are derived from the .bin files, so a re-export invalidates
    # them and they must not linger from the previous revision.
    for pattern in ('*.bin', '*.bin.zst', '*.bin.gz', '*.wasm', '*.js', '*.xml', '*.json'):
        for stale in out.glob(pattern):
            stale.unlink()
    for name in ('meta.json', 'body_config.json'):
        (out / name).unlink(missing_ok=True)


def somata_sample(coords):
    """Indices of the neurons the brain panel draws.

    Mirrors `studio_server.py`: only neurons with finite measured coordinates are
    candidates, then a fixed 2048-point linear sample over them.
    """
    finite = np.flatnonzero(np.isfinite(coords).all(axis=1))
    picked = finite[np.linspace(0, len(finite) - 1, min(SAMPLE_COUNT, len(finite)), dtype=int)]
    return finite, picked.astype(np.int32)


def copy_mujoco(out):
    """Copy the MuJoCo-WASM build the browser body runs on into the package.

    `bench_mujoco/public/` used to hold hand-copied glue and wasm with no recorded
    provenance. Taking them from the pinned npm package instead keeps the package
    self-contained and its origin reproducible, and lets the page verify both
    files like any other artifact.
    """
    source = ROOT / MUJOCO_PACKAGE
    version = json.loads((source / 'package.json').read_text())['version']
    assets = {'version': version, 'source': str(MUJOCO_PACKAGE)}
    for key, filename in (('glue', 'mujoco.js'), ('wasm', 'mujoco.wasm')):
        raw = (source / filename).read_bytes()
        name = f'mujoco-{key}.{hashlib.sha256(raw).hexdigest()[:8]}{Path(filename).suffix}'
        (out / name).write_bytes(raw)
        assets[key] = dict(path=name, sha256=sha256_hex(out / name), bytes=len(raw))
    return assets


def body_configuration(body):
    """The numeric body contract the browser needs, from the checkpoint-driven body.

    `hips_height` is a declared field of the body spec (app/sim.py), not something
    to re-derive: it is the skeleton measurement the avatar is scaled against, and
    the studio server publishes the same value.
    """
    cfg = body.cfg
    return dict(
        robot='yumi', initial_height=body.initial_height, fall_height=body.fall_height,
        hips_height=float(body.hips_height),
        simulation_dt=cfg['simulation_dt'], control_decimation=cfg['control_decimation'],
        kps=[float(v) for v in body.kp], kds=[float(v) for v in body.kd],
        home=[float(v) for v in body.home], action_scale=float(cfg['action_scale']),
        cmd_scale=[float(v) for v in cfg['cmd_scale']],
        ang_vel_scale=float(cfg['ang_vel_scale']), dof_vel_scale=float(cfg['dof_vel_scale']),
        gait_period_s=float(cfg['gait_period_s']),
        linear_velocity_scale=float(cfg['linear_velocity_scale']),
        height_reference=float(cfg['height_reference']),
        observation_size=body.observation_size)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', default=str(ROOT / 'runs/yumi/best.pt'))
    parser.add_argument('--graph', default=str(ROOT / 'data/full_graph.npz'))
    parser.add_argument('--out', default=str(ROOT / 'artifacts/preview'))
    parser.add_argument('--evaluation', default=str(ROOT / 'runs/yumi/release_evaluation.json'),
                        help='held-out evidence to show on the static page; refused unless it '
                             'is bound to this checkpoint')
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    graph_path = Path(args.graph)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clear_previous(out)

    checkpoint_sha256 = sha256_hex(checkpoint)
    graph_sha256 = json.loads(graph_path.with_suffix('.json').read_text())['graph_sha256']
    if sha256_hex(graph_path) != graph_sha256:
        raise SystemExit(f'{graph_path} does not match its recorded graph_sha256')

    configuration = checkpoint_configuration(checkpoint)
    brain = ConnectomePolicy(device='cpu',
                             observation_size=configuration['observation_size']).eval()
    brain.load(str(checkpoint))
    n = int(brain.n)

    # The policy's effective weights fold the learned edge delta into the graph.
    # Kept in torch (not numpy) so the bytes match the WebGPU parity fixtures.
    with torch.no_grad():
        values = (brain.weight * (1 + 0.9 * torch.tanh(brain.edge_delta))) \
            .detach().to(torch.float32).numpy()

    coords = np.load(graph_path, allow_pickle=False)['coords'].astype(np.float32)
    finite, sample = somata_sample(coords)

    tensors = {
        'values': values.astype(np.float32),
        'pre': brain.pre.numpy().astype(np.int32),
        'ptr': brain.ptr.numpy().astype(np.int32),
        'inputs': brain.inputs.numpy().astype(np.int32),
        'outputs': brain.outputs.numpy().astype(np.int32),
        'bias': brain.neuron_bias.detach().numpy().astype(np.float32),
        'encoder': brain.encoder.weight.detach().numpy().astype(np.float32),
        'readout': brain.readout.weight.detach().numpy().astype(np.float32),
        'readout_bias': brain.readout.bias.detach().numpy().astype(np.float32),
        'norm_w': brain.normalizer.weight.detach().numpy().astype(np.float32),
        'norm_b': brain.normalizer.bias.detach().numpy().astype(np.float32),
        'obs_mean': brain.obs_mean.numpy().astype(np.float32),
        'obs_std': brain.obs_std.numpy().astype(np.float32),
        'sample': sample,
        'coords': coords[sample].reshape(-1),
    }

    files = {}
    for key, tensor in tensors.items():
        name = f'{key}.{hashlib.sha256(tensor.tobytes()).hexdigest()[:8]}.bin'
        tensor.tofile(out / name)
        files[key] = dict(path=name, sha256=sha256_hex(out / name), kind=key)

    # The browser body must be built from the same interface the policy trained
    # against, and the page re-checks this pair at load time.
    body = Body(load_motor_policy=False, robot='yumi',
                interface=configuration['physics_interface'],
                observation_size=configuration['observation_size'])
    body.reset(RESET_SEED)
    config = body_configuration(body)
    (out / 'body_config.json').write_text(json.dumps(config, indent=1), encoding='utf-8')

    # The static page shows held-out evidence, so the package must carry evidence for
    # THIS checkpoint. The studio applies matches_checkpoint() before serving any
    # record; running the same function here means the two cannot disagree, and a
    # stale or foreign evidence file is refused at export rather than published.
    evidence_path = Path(args.evaluation)
    evidence = json.loads(evidence_path.read_text(encoding='utf-8'))
    if not matches_checkpoint(evidence, checkpoint_sha256, 'yumi', body.interface):
        raise SystemExit(f'{evidence_path} is not held-out evidence for checkpoint '
                         f'{checkpoint_sha256[:8]} (robot/interfaces/protocol must all match)')
    evidence_name = f'evaluation.{sha256_hex(evidence_path)[:8]}.json'
    (out / evidence_name).write_bytes(evidence_path.read_bytes())

    # The scene the export was validated against travels with the package, so the
    # browser cannot load a different XML revision than the one behind these numbers.
    scene_dir = ROOT / 'app/yumi_description'
    scene = dict(
        scene=dict(path='scene.xml', sha256=sha256_hex(scene_dir / 'scene.xml')),
        include=dict(path='yumi.xml', sha256=sha256_hex(scene_dir / 'yumi.xml')),
    )
    for entry in scene.values():
        (out / entry['path']).write_bytes((scene_dir / entry['path']).read_bytes())

    meta = dict(
        n=n, nnz=int(values.size), neural_steps=int(brain.neural_steps),
        observation_size=int(brain.encoder.in_features),
        action_size=int(brain.readout.out_features),
        inputs_count=int(len(brain.inputs)), outputs_count=int(len(brain.outputs)),
        sample_count=int(sample.size), somata_with_coords=int(finite.size),
        reset_seed=RESET_SEED,
        initial_qvel=[float(v) for v in np.random.default_rng(RESET_SEED).normal(0, 0.015, 2)],
        graph_sha256=graph_sha256, checkpoint_sha256=checkpoint_sha256,
        physics_interface=configuration['physics_interface'],
        precision=dict(values='f32', pre='i32'),
        files=files,
        scene=scene,
        mujoco=copy_mujoco(out),
        # ConnectomePolicy always loads data/full_graph, which is the mode the studio
        # reports as full_connectome; the results copy keys its wording off this.
        mode='full_connectome',
        evaluation=dict(path=evidence_name, sha256=sha256_hex(out / evidence_name),
                        kind=evidence['kind'], checkpoint_sha256=evidence['checkpoint_sha256']),
    )
    (out / 'meta.json').write_text(json.dumps(meta, indent=1), encoding='utf-8')

    total = sum(item.stat().st_size for item in out.iterdir() if item.is_file())
    print(f'checkpoint {checkpoint_sha256[:8]}  graph {graph_sha256[:8]}')
    print(f'n={n} nnz={meta["nnz"]} sample={meta["sample_count"]}/{finite.size} with coords')
    print(f'somata sample head={sample[:4].tolist()} tail={sample[-2:].tolist()}')
    print(f'hips_height={config["hips_height"]} (from the body spec, not re-derived)')
    print(f'mujoco @mujoco/mujoco@{meta["mujoco"]["version"]} '
          f'glue={meta["mujoco"]["glue"]["path"]} wasm={meta["mujoco"]["wasm"]["path"]}')
    print(f'evidence {evidence_path.name} -> {evidence_name} '
          f'({evidence["kind"]}, {evidence.get("successes")}/{evidence.get("attempts")} walks)')
    print(f'wrote {len(files)} tensors + meta.json + body_config.json = {total / 2**20:.1f} MiB -> {out}')


if __name__ == '__main__':
    main()
