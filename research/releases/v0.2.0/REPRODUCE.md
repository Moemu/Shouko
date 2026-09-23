# Run Shouko v0.2.0

This release is an independent headless runtime package, not the full web studio. It includes all required model, graph and Yumi physics files. No SSH endpoint, VRM, vendor checkout, teacher policy or original private workspace is required for inference. The MLP teacher and ancestor are included for provenance and optional future training.

## Linux / NVIDIA GPU

Use Python 3.12 and a CUDA-capable PyTorch environment. The tested experiment environment used PyTorch 2.12.1+cu130, MuJoCo 3.13.0 and Warp 1.17.0. GPU driver compatibility is required. The dependency file pins the direct packages used by the experiment; it is not a complete container image or a promise of identical behavior on every GPU.

```bash
sha256sum -c SHA256SUMS.txt
mkdir shouko-v020
tar -xzf shouko-v0.2.0.tgz -C shouko-v020
cd shouko-v020
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.cloud.lock.txt
.venv/bin/python smoke.py
```

`smoke.py` verifies every packaged file, then runs the primary checkpoint at batch one with native CSR for 30 seconds. It writes `package_smoke.json` and fails if the gait screen fails. This verifies the package and single-world execution, not the browser controls or general command behavior.

Run the original acceptance configurations from the extracted directory:

```bash
.venv/bin/python -m app.evaluate_locomotion --checkpoints models/primary.pt --seed-base 140001 --output verification/normal.json
.venv/bin/python -m app.evaluate_locomotion --checkpoints models/primary.pt --seed-base 141001 --cohorts 1 --yaws -.2 .2 --output verification/yaw.json
.venv/bin/python -m app.evaluate_locomotion --checkpoints models/primary.pt --seed-base 142001 --cohorts 1 --seconds 120 --output verification/long.json
.venv/bin/python -m app.evaluate_locomotion --checkpoints models/primary.pt --seed-base 143001 --cohorts 1 --push --output verification/push.json
```

These rerun known evaluation seeds; they are reproducibility checks, not new held-out evidence. The scripts refuse to overwrite outputs. Use fresh output paths for later runs. For the replica use `models/replica.pt` and seed bases 240001, 241001, 242001 and 243001.

The JSON `success` field retains the six historical criteria. Strict results additionally require every foot's `qualifying_swings_per_second` to be at least 1. Do not report the old field alone as strict gait acceptance.

## Training boundary

The package includes `app.imitate_yumi` for native student-state collection and frozen-core readout fitting. It does not contain the complete historical DAgger datasets or an optimizer state. It therefore supports inference/evaluation reproduction, not byte-identical reconstruction of all training history. The full experiment archive hashes and protocol are recorded in `research/experiments/CONNECTOME_TRANSFER_20260923.md` in the source repository.

Extraction alone does not change the studio. For the browser preview, check out tag `v0.2.0` and follow [STUDIO.md](STUDIO.md). The installer selects the primary model and installs hash-bound strict evaluation data.
