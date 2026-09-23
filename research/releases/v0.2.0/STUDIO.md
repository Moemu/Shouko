# v0.2.0 Windows studio installation

Use Windows PowerShell, Git, Python 3.12, uv, Node.js 20.19+ or 22.12+, and an NVIDIA GPU with a driver compatible with CUDA 13.0. The tested local preview used an RTX 4070 Laptop 8 GB. The release does not require SSH or access to the original workspace.

## Download and verify

```powershell
git -c core.autocrlf=false clone --branch v0.2.0 https://github.com/Moemu/Shouko.git
cd Shouko
$releaseDir = 'runs/releases/v0.2.0'
New-Item -ItemType Directory -Force $releaseDir | Out-Null
$releaseUrl = 'https://github.com/Moemu/Shouko/releases/download/v0.2.0'
foreach ($asset in @('shouko-v0.2.0.tgz','manifest.json','SHA256SUMS.txt')) {
    Invoke-WebRequest "$releaseUrl/$asset" -OutFile "$releaseDir/$asset"
}
foreach ($line in Get-Content "$releaseDir/SHA256SUMS.txt") {
    $parts = $line -split '  ', 2
    $actual = (Get-FileHash "$releaseDir/$($parts[1])" -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($actual -ne $parts[0]) { throw "Checksum mismatch: $($parts[1])" }
}
New-Item -ItemType Directory -Force "$releaseDir/extracted" | Out-Null
tar -xzf "$releaseDir/shouko-v0.2.0.tgz" -C "$releaseDir/extracted"
if ($LASTEXITCODE -ne 0) { throw 'Extraction failed' }
```

The archive includes the full graph, primary and replication models, upright source, MLP teacher and acceptance evidence. The teacher is not used at runtime. The external manifest records the source commit and each payload hash. The installer verifies all payload files and rejects mismatching checkout runtime files.

## Install the default model

```powershell
uv venv .venv-gpu --python 3.12
uv pip install --python .venv-gpu/Scripts/python.exe -r requirements.cloud.lock.txt
.\stop.ps1
.venv-gpu\Scripts\python.exe research/releases/install_v020.py --package runs/releases/v0.2.0/extracted
if ($LASTEXITCODE -ne 0) { throw 'Model installation failed' }
```

Stop an existing studio before installation. The installer puts the primary model at `runs/yumi/best.pt`, keeps `runs/yumi/upright-source.pt`, installs the graph and writes hash-bound strict evaluation data. An old default model and its training/optimizer records are copied to `runs/yumi/archive/` before replacement. Old optimizer states are removed from the active paths. No training starts. The new endpoint has no matching PPO optimizer state.

The dependency versions match the tested experiment environment. An installation into an entirely fresh Windows environment has not been independently tested. GPU driver and wheel availability may affect setup; the package smoke test and local studio tests used existing dependency environments.

## Public avatar and browser

Yumi's VRM cannot be redistributed. Without it the studio automatically displays the physics skeleton. This is sufficient to preview and control the released model; no private asset or avatar download is required.

```powershell
npm ci
npm run build
.\start.ps1
```

Open <http://127.0.0.1:8740>. Default body is Yumi. The checkpoint footer must show `f1a200718e`. Training and evaluation displays normal 27/27, yaw 18/18, long 9/9 and impulse 9/9. A graph-disconnection control shows 9/9 expected falls. Evidence disappears if the loaded model or physics interface differs. Training history is empty because this endpoint was fitted by ridge regression, not by the old PPO run.

The skeleton displays the Yumi simulation body. Switching to G1 requires its separate model and vendor assets from the earlier setup. The current package does not install those. Use the Yumi body for this release's checks. If you already have a permitted local Yumi VRM, place it at `web/public/yumi.vrm` before building the frontend. Its attribution remains visible.

Stopping, turning and arbitrary target changes remain unvalidated capabilities, even though the interface exposes these controls. The published scores concern only the configurations in [MODEL_CARD.md](MODEL_CARD.md). For headless reproduction use [REPRODUCE.md](REPRODUCE.md).
