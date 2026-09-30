# Run Shouko v0.3.0

This is a self-contained headless inference/evaluation package. No private workspace, SSH service, teacher weights, vendor checkout or VRM is needed. Use Linux, Python 3.12 and an NVIDIA CUDA GPU. The pinned direct dependencies include PyTorch 2.12.1+cu130, MuJoCo 3.13.0 and Warp 1.17.0. This is not a full container or a guarantee of bit-identical trajectories on every GPU.

Download all five assets from the GitHub Release into one directory: `shouko-v0.3.0.tgz`, `manifest.json`, `primary_native.json`, `replica_native.json` and `SHA256SUMS.txt`. The checksum command also verifies both native runtime reports:

```bash
sha256sum -c SHA256SUMS.txt
mkdir shouko-v030
tar -xzf shouko-v0.3.0.tgz -C shouko-v030
cd shouko-v030
python3.12 -m venv .venv
.venv/bin/python -m pip install -r requirements.cloud.lock.txt
.venv/bin/python smoke.py --role primary --output verification/primary_native.json
.venv/bin/python smoke.py --role replica --output verification/replica_native.json
```

Each smoke verifies every packaged file, then runs three 30-second speed trials and one 120-second high-speed trial with native CSR at batch one. It requires strict gait in all four and the 2 m corridor for the long trial. Output paths must not exist. These rerun known seeds; they are package checks, not new independent holdouts or browser validation.

Rerun the primary's full five-group protocol from the extracted directory:

```bash
.venv/bin/python -m app.evaluate_direction --checkpoints models/primary.pt --seed-base 9000001 --cohorts 3 --output verification/normal.json
.venv/bin/python -m app.evaluate_direction --checkpoints models/primary.pt --seed-base 9001001 --yaws -.2 .2 --output verification/yaw.json
.venv/bin/python -m app.evaluate_direction --checkpoints models/primary.pt --seed-base 9002001 --seconds 120 --trace --output verification/long.json
.venv/bin/python -m app.evaluate_direction --checkpoints models/primary.pt --seed-base 9003001 --push-velocity .25 --output verification/push_positive.json
.venv/bin/python -m app.evaluate_direction --checkpoints models/primary.pt --seed-base 9004001 --push-velocity -.25 --output verification/push_negative.json
.venv/bin/python -c "import json; from pathlib import Path; from app.release_evidence import build_direction_report; print(json.dumps(build_direction_report(Path('verification'), '1e1c3160448f6acaddd28b1a8782dc75b037fed3e296525aa52ae060926801e6'), indent=2))"
```

For replica, use `models/replica.pt`, its hash from MODEL_CARD.md, seed bases 9100001/9101001/9102001/9103001/9104001 and a different output directory. Backend defaults to ordered CSR with nine simultaneous worlds. `build_direction_report` checks configuration, cohorts, hashes and full-trajectory corridor. It deliberately requires all five groups. It does not treat cached summary scores as acceptance.

Extraction does not replace the studio or online preview. The v0.2.0 preview remains the default. To explore integration, use source tag v0.3.0 and docs/LOCAL.md; do not copy these weights over an old evaluation file. The new protocol binds the checkpoint, physics interface and body XML. Browser parity and preview deployment require separate validation.

For historical training methods and current scripts, see docs/TRAINING.md and the included experiment records. Full historical datasets are not in this inference package. The archive manifest binds each included file to its SHA-256 and the release source commit.

## Windows

The independent package was also tested on Windows with the same Python/CUDA dependencies. After extraction, create the environment and run from that directory in PowerShell:

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.cloud.lock.txt
.venv\Scripts\python.exe smoke.py --role primary --output verification/primary_native.json
.venv\Scripts\python.exe smoke.py --role replica --output verification/replica_native.json
```

Use `Get-FileHash -Algorithm SHA256` to compare the downloaded files with SHA256SUMS.txt before extraction. A fresh dependency installation was not tested during this release; the smoke runs reused an existing environment.
