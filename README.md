# Shouko · ショウコ

**English** · [简体中文](README.zh.md)

**We trained a fruit fly connectome network to control Yumi's legs for flat-ground walking. Here is the training process, and how to run and train it locally.**

Her full name is ショウジョウバエ. A public male *Drosophila* connectome dataset is her brain; the girl is a virtual character in VRM format, [Yumi](research/avatar/YUMI.md). Everything runs in a simulation built on MuJoCo, an open-source physics engine.

WebGPU online preview version: [Shouko · ショウコ · Studio](https://shouko.snowy.moe/)

## Overview

We set out to validate one engineering chain: **turn a fruit fly's real neural wiring into a control network, and let it drive a walking body.** A fly's **connectome** records which neuron connects to which. It carries no brain function by itself. This project imports that wiring list as a neural network, trains it to output 12 lower-limb joint targets, hands them to MuJoCo for gravity and contact, and lets the character skeleton follow the same physical state.

The data is [MaleCNS v1.0](https://male-cns.janelia.org/): **166,700 neurons and 25,582,938 connections**.

The body has moved twice. The first two generations used **G1**, Unitree's humanoid robot model, as the physical proxy: the 8,192-neuron subgraph prototype handed the connectome's velocity commands to a frozen official gait policy, while the full-connectome cloud generation trained end-to-end, outputting joint targets straight from the connectome — both passed their own acceptance. The current mainline moved to the **Yumi** lower-limb body rebuilt from the VRM character's measured skeleton: joint order, axes and zero positions match, so the network interface is unchanged, but height, mass and torque differ, and weights do not transfer (see [Historical results](#historical-results)).

### Motivation and inspirations

When the fruit fly connectome was released, many interesting projects appeared around it. The two that surprised me most came from X users [@yakshawan](https://x.com/yakshawan) and @satorunet, who each put the fly brain into a 3D body: [FFREP](https://heavyrain39.github.io/ffrep/) trains MaleCNS to drive the SHOKI quadruped and the YUMEKA humanoid, and [hae](https://hae.satoru.net/) ([source](https://github.com/satorunet/hae)) runs a FlyWire connectome as a spiking simulation in the browser. That is where Shouko started — I wanted to build one of these myself.

Shouko differs from both in what is open: the training method, the weights and the records are public, and everything can be reproduced locally. It also differs in scope, and you should know this up front — unlike their full-limb implementations, Shouko currently controls only the lower limbs, with the upper body as one fixed mass, and achieves controlled stable flat-ground walking under speed and heading commands (no goal points). The result is less fun to watch than the two above, but full limbs are already on the roadmap under [Future work](#future-work). Our pinned-version survey of both projects lives in [research/references/fly-embodiment/](research/references/fly-embodiment/README.md).

I am not an expert on fly simulation — my background is some NLP studied years ago — so this project doubles as a multi-agent collaboration experiment: several independent frontier models (GPT 6 Astra, Kimi K3, Qwen 3.8 Max and others) run in parallel with harnesses to discuss proposals and training, GPT 6 Astra acts as the main coordinator for concrete training and goal setting, and training runs on an AutoDL RTX 4090D instance.

Community reproduction is welcome — see [Running Locally](#running-locally) and the [training reproduction guide](docs/TRAINING.md) — and future directions (full limbs, running and dopamine-based goal rewards) are listed under [Future work](#future-work). Feedback and suggestions are very welcome.

### Historical results and current status

**Current milestone: Yumi lower-limb walking on flat ground.** The policy controls 12 hip, knee and ankle joints, six per leg. The upper body is fixed to the pelvis in the physics model. Visible arm swing is display animation; the policy does not control the arms or use them for balance.

| Body / stage | Model during walking | Neural activity at the same instant |
|---|---|---|
| **G1 · earlier stage**<br>`ffcf9a97` · 8.28 s | <img src="assets/preview-g1-model.png" alt="G1 walking at 0.5 m/s, rendered with the pixiv sample avatar" width="300"> | <img src="assets/preview-g1-neural.png" alt="G1 connectome activity at the same simulation time, fixed scale minus one to plus one" width="300"> |
| **Yumi · current v0.2.0**<br>`f1a20071` · 8.26 s | <img src="assets/preview-yumi-model.png" alt="Current v0.2.0 Yumi model during a left-foot swing" width="300"> | <img src="assets/preview-yumi-neural.png" alt="Current Yumi connectome activity at the same simulation time, fixed scale minus one to plus one" width="300"> |

Each pair freezes one real walking frame at a commanded 0.50 m/s and 0° heading; model and neural images share the same simulation time. Neural views show **2,048 sampled somata out of 166,700 neurons**, using signed model activity at a fixed ±1 scale (blue: negative; orange: positive). These are not biological spikes or new acceptance tests. [Capture details and hashes](assets/preview-metadata.json). The G1 image uses the later `ffcf9a97` checkpoint ([6/9 held-out record](research/experiments/G1_CHECKPOINT_REBIND_20260918.md)); the 9/9 in the [historical results](#historical-results) table below belongs to `f1d5147c`. G1 avatar: © 2022 pixiv Inc. Yumi: concept 松酒, artist 7Apoi, model 星晨水影工作室, publisher 墨海徽. [Asset attribution](THIRD_PARTY.md).

| Version | Checkpoint / stage | Scope | Status |
|---|---|---|---|
| Unreleased | `f1d5147c` · G1 cloud full connectome | End-to-end full-connectome training, G1 proxy | Accepted |
| v0.1.0 | `a7a4281f` · Yumi squat lineage | The same chain on the Yumi body | Accepted |
| Bundled in v0.2.0 | `458fc465` · Yumi upright source | 50-dim observations, physics interface embedded in the checkpoint | Retained baseline: upright survival and alternating rhythm hold; directed tracking did not |
| **v0.2.0 (current)** | `f1a20071` · primary model | Full-connectome readout transfer, Yumi lower limbs | Four strict held-out groups passed with zero falls; precise heading and lateral drift remain open |
| Reserved | Next stage | Solve the issues left open by the previous stage's holdouts (precise heading and lateral drift) | Scope and acceptance criteria to be confirmed |
| Reserved | Later stages | Everything in [Future work](#future-work) | Pending |

Future rows reserve space for confirmed stages. They do not commit to a schedule or mark candidate research as implemented.

#### Historical results

| Check | Full-graph G1 body `f1d5147c` | Yumi squat lineage `a7a4281f` |
|---|---|---|
| 9 fresh initial states × 30 s | 9/9 | 9/9 |
| 120 s continuous walk | 55.40 m | 65.55 m (0.13 m lateral drift) |
| Lateral push recovery | 3/3 | 3/3 |
| Brain connections severed | falls in ≈1.4 s | falls in ≈1 s |
| Record | `runs/local/heldout.json` | `runs/local/heldout_yumi.json` |

Sever the brain connections, keep the body and output layer intact, and the character falls in about a second — walking depends on this wiring. Older results use criteria different from the current model — no matched comparison.

#### Current results (v0.2.0 primary model `f1a20071`)

The default model freezes (fixes after training) the previously trained full-connectome core and fits only the 9,792 readout parameters — the readout is the network's output layer, translating neuron activity into 12 joint targets. Training data is labeled by an MLP (multi-layer perceptron, an ordinary neural network) teacher on the Yumi body; runtime control uses the connectome alone. The explicit phase input (a gait clock cycling every 0.8 s) remains.

Held-out groups are starting states never seen during training — the final exam. The primary model and the independent-data branch both passed all four strict held-out groups — 27 normal-start episodes, 18 initial-yaw ±0.2 rad episodes, 9 episodes each at 120 s continuous walk and a specified lateral impulse (one sideways shove at second 10) — with zero falls. Strict acceptance includes at least one qualifying swing per foot per second (airborne at least 0.12 s and at least 2.5 cm off the ground); re-running both models on a different sparse backend (native CSR) also passed 27/27. The independent branch needed an extra DAgger round (the teacher correcting the student online — see [How it trains](#how-it-trains)); it is not equal-budget replication.

The screen permits some drift (how far the walker strays sideways): primary mean lateral drift at 120 s is **5.47 m**, and precise yaw recovery is **9/18** (9 of 18 yaw starts returned to facing forward within about 3° and held for half a second). Passing does not establish precise straight-line walking, natural gait, an autonomous CPG (a rhythm generated without the external clock) or topology superiority. See the [model card](research/releases/v0.2.0/MODEL_CARD.md) and the [training record](research/experiments/CONNECTOME_TRANSFER_20260923.md).

### Not attempted yet

A same-scale random-network control, rough terrain, starting and stopping, upper-limb balance, natural human gait. The full list is in section 5 of the [research report](research/REPORT.md).

### Future work

**Upper limbs.** Above the pelvis the body is currently one ~25.7 kg capsule welded to the pelvis, and arm swing is display animation only. Without counter-swinging arms acting as a momentum flywheel, the in-air attitude control running needs is not achievable — so upper-limb control precedes running.

**Running.** The gait clock is hardcoded to a 0.8 s cycle, sized for G1's short legs; Yumi's nearly 1 m legs only produce cramped small steps at walking speed. Running needs a retuned clock, a flight phase and speeds above 1.2 m/s.

**Dopamine-driven learning.** Dynamics are a rate approximation — no STDP, no dopamine reward learning — and connection gains have to stay frozen to remain stable. The fly's mushroom body (sparse Kenyon-cell encoding + MBON value readout + dopaminergic TD error) can be rewritten as a three-factor local plasticity rule, letting the network learn online without global backpropagation.

**Visual obstacle avoidance.** The optic lobe (the fly's visual center) holds over 60% of the brain's neurons. Feeding depth or optical flow straight into the visual projection neurons should produce a steering reflex on its own, with no separate CNN to train.

## Training Approach & Details
### Data flow

```text
User target speed / heading + MuJoCo body observation (50 dims)
    ↓ sensory encoder
Data-labelled sensory neurons
    ↓ 4 rate updates per decision, propagating only along measured pre → post edges
166,700 neurons / 25,582,938 connections
    ↓
Data-labelled VNC / CB motor neurons (readout)
    ↓
12 lower-limb joint targets → PD torque → MuJoCo gravity and contact
    ↓
Pose feedback; the VRM skeleton displays the same physical state
```

The 815 output neurons are annotated VNC (ventral nerve cord) and CB (central brain) motor neurons. Edge weights come from `sign × sqrt(contact count)`, normalized per target node. Acetylcholine is treated as excitatory, GABA, glutamate and histamine as inhibitory, neuromodulators and unknown classes default to positive — a simplification policy, not a per-synapse receptor measurement. Dynamics use a rate approximation (`r ← a·r + b·tanh(W·r + sensory drive)`) with no neural memory across decisions, no STDP and no dopamine reward learning. Sensory drive enters the input group only, the output group does not overlap it, and no path bypasses the network to reach the joints.

### How it trains

The v0.2.0 final stage: a Yumi MLP teacher labels actions for the training data; the student then executes on its own, and the states it actually visits (including failures) go back to the teacher for relabeling over several rounds (**DAgger**); ridge regression analytically fits the 9,792 readout parameters on features cached from the frozen core. The core (connections, sensory encoder, neuron biases, normalization) stays bit-identical to its source model. The teacher takes part in training only; evaluation and preview drive the joints straight from the connectome policy.

A new body means retraining. G1 is Unitree's humanoid robot model; Yumi is a 12-joint body rebuilt from the VRM character's measured skeleton (hip height 0.97231 m, roughly 1.55 m overall, 38 kg assumed from Dempster segment ratios). Joint order, axes and zero positions match, so the network interface is unchanged; height, mass, torque limits and fall thresholds differ, and weights do not transfer. The earlier G1 and Yumi lineages were trained end-to-end with **PPO** (at that time the sensory encoder, neuron biases, connection gains and readout trained together, about 26.6 M parameters); the process and tuning notes live in `research/experiments/` and the [training reproduction guide](docs/TRAINING.md).

### Compute and cost

- Local: Ryzen 9 7940HX, ~16 GB RAM, RTX 4070 Laptop 8 GB. The full graph runs inference only on this machine, with a median 10.04 ms per decision.
- Cloud: training runs on an AutoDL RTX 4090D instance, billed on total powered-on time (idle included); instance configuration, benchmarks and cost records are in the [cloud measurements and reproduction record](docs/CLOUD.md).

## Running Locally
For a clean checkout, use the [v0.2.0 studio installation guide](research/releases/v0.2.0/STUDIO.md). It downloads the public weights and graph, verifies checksums and installs the default model. Existing prepared environments can use:

```powershell
cd D:\Project\Neuromechfly
.\start.ps1            # defaults to Yumi + CUDA
```

Open the [studio](http://127.0.0.1:8740). The VRM character on the page walks in real time, with every action computed on your machine. Both panels are the same live studio: the left pane displays lower-limb walking and gait stats; the right pane shows 2,048 sampled neuron activity rates of the 166,700-neuron connectome.

- Defaults to the Yumi body and CUDA. The page's top bar switches between G1 and Yumi live; training must not be running.
- `.\stop.ps1` stops it. The startup log is `runs/cloud/server-error.log` (or `runs/yumi/server-error.log`).
- On an RTX 4070 Laptop 8 GB the median decision is 10.04 ms and the Yumi page measured 1.00× real time; heavy system load drops it to 0.7–1.0×.

The UI supports pause, resume, reset, target speed and heading, lateral pushes, character/skeleton toggle, connection severing (cutting the brain's wiring), retraining, training curves and raw evaluation. Retraining archives the previous round's core results. Hair and upper-limb display following stay out of the physics evaluation.

### From a clean environment

Needs Python 3.12, uv, Node.js 20.19+ or 22.12+, and Git. `setup.ps1` downloads about 1.1 GB of raw connectome data and builds `.venv` (the legacy subgraph-prototype environment; the v0.2.0 weights and graph are installed through [STUDIO.md](research/releases/v0.2.0/STUDIO.md) instead):

```powershell
.\setup.ps1
.\start.ps1
```

Vendor repositories and large data stay out of normal Git history.

### Installing and accepting the v0.2.0 model

Follow [STUDIO.md](research/releases/v0.2.0/STUDIO.md) for the browser or [REPRODUCE.md](research/releases/v0.2.0/REPRODUCE.md) for headless verification. Local acceptance command:

```powershell
.venv-gpu\Scripts\python.exe -m app.evaluate_locomotion --checkpoints runs/yumi/best.pt --seed-base 140001 --output runs/local/v020_normal.json
```

The acceptance script neither trains nor selects checkpoints. Measured Yumi skeleton dimensions, physics body construction and the screen-consistency check are in the [Yumi body adaptation record](research/avatar/YUMI_BODY.md); training and tuning notes live in `research/experiments/`.

### Run modes

| Mode | Entry | Address | Notes |
|---|---|---|---|
| Local studio (mainline) | `.\start.ps1 [-Body g1\|yumi]` | 8740 | Full connectome + MuJoCo locally; body switchable in the page |
| Cloud full graph | `.\cloud.ps1 Preview` | 8742 | AutoDL instance over an SSH tunnel, bound to localhost |
| Local static preview | see [PREVIEW.md](docs/PREVIEW.md) (export the preview package, then `python -m app.serve_preview`) | 8741 | Serves the released weight package without a server |

The 8,192-neuron subgraph prototype (`app/server.py`) remains as legacy code without a start script.

On the cloud side, `./cloud.ps1 Status` reports state, `./cloud.ps1 Train -Seconds 1800` resumes training from a checkpoint, and `./cloud.ps1 Sync` syncs code and UI. The SSH target is machine-specific: pass `-CloudHost user@host -CloudPort 12345`, or write it once to `runs/cloud/target.json` (git-ignored). Training does not shut down the instance; the preview keeps running and keeps billing.

### Weights and checkpoints

`runs/yumi/best.pt` is the v0.2.0 primary model `f1a20071`; its upright source model `458fc465` is retained as `runs/yumi/upright-source.pt` and shipped in the release package as `models/upright-source.pt`. The installer archives the previous default, training records and optimizer states under `runs/yumi/archive/` before switching. The older squat model `a7a4281f` remains available in the public v0.1.0 Release, with its historical results bound to that checkpoint. The page serves release evidence only when checkpoint hash, Yumi body and physics interface match.

### Publishing and verifying artifacts

Download [v0.2.0](https://github.com/Moemu/Shouko/releases/tag/v0.2.0): `shouko-v0.2.0.tgz`, `manifest.json` and `SHA256SUMS.txt`. The package includes four model roles, full graph, evaluation evidence, minimal runtime and attribution. The manifest binds exact runtime files to the tagged source commit. Yumi VRM and old PPO states are excluded.

The old `research/provenance/release.json` describes an earlier local inventory, not the v0.2.0 manifest.

## Key files

- `app/prepare.py`: validates and streams the dataset, builds a traceable subgraph.
- `app/full_brain.py`: sensory encoding, measured sparse connections and trained readout for the full connectome.
- `app/yumi_skeleton.py`, `app/yumi_description/`: parse the skeleton from the VRM and build the Yumi physics body.
- `app/sim.py`: MuJoCo physics and G1 / Yumi body switching.
- `app/train_full.py`, `app/ppo_yumi.py`: full-graph training, DAgger and PPO fine-tuning.
- `app/export_preview_weights.py`, `app/serve_preview.py`: preview weight export and the local static preview server.
- `app/studio_server.py`: mainline studio service, training control and hash-bound evaluation display.
- `web/`: Three.js / VRM visualization and the experiment UI.
- `research/REPORT.md`: literature review, method, results, compute and unfinished scope.

## About
### License

The project's own code and documentation are released under the [MIT License](LICENSE). MIT is compatible with every bundled upstream license: the permissive ones (FlyCube, DOOMFLY, three-vrm, Three.js, Vite, FastAPI — MIT; Unitree RL Gym, NumPy, SciPy, Uvicorn — BSD-3-Clause; MuJoCo — Apache-2.0; PyTorch — BSD-style) only require retaining their notices, which stay in `vendor/` checkouts and installed packages; `app/sim.py` keeps its Unitree adaptation note in the file header.

MIT covers this repository's code only and does not override the data and asset terms: the MaleCNS dataset stays CC BY 4.0 (attribution required), the pixiv VRM sample stays under VRM Public License 1.0, and the Yumi VRM is not redistributed at all under its author's terms. Third-party components are listed in [THIRD_PARTY.md](THIRD_PARTY.md), with license copies under `research/provenance/licenses/`:

- **MaleCNS v1.0** (connectome data): FlyEM / HHMI Janelia and collaborators, CC BY 4.0. We normalize records, select a subgraph, convert contact counts into signed weights and define new dynamics; these transformations are not physiological measurements.
- **Unitree RL Gym**: BSD-3-Clause, providing the G1 model, pretrained checkpoint and simulation conventions.
- **VRM1_Constraint_Twist_Sample** v1.0.1: pixiv, VRM Public License 1.0, redistribution and modification permitted ("Hikari" is a UI nickname).
- **FlyCube**, **DOOMFLY**: MIT, source of the MaleCNS downloader, importer and transmitter sign policy.
- **NeuroMechFly / FlyGym**: research reference for hierarchical control; no code bundled.
- **MuJoCo** (Apache-2.0), **PyTorch**, **NumPy / SciPy**, **Three.js / three-vrm**, **Vite**, **FastAPI / Uvicorn** and other dependencies keep their notices in the installed packages.

This experiment is not affiliated with or endorsed by the cited researchers or organizations.

### Assets and models

**Yumi** (1.3.0 / 20240715) — concept: 松酒; artist: 7Apoi; model: 星晨水影工作室; publisher: 墨海徽 ([author's release page](https://www.bilibili.com/video/BV1uG411f72b/)). File `web/public/yumi.vrm`, 25,754,408 bytes, SHA-256 `3f1eaf57…`, path listed in `.gitignore`. The author permits non-profit use, monetized livestreams and derivative works, prohibits redistribution and resale, and the embedded metadata prohibits sexual and violent use. Details in the [Yumi provenance record](research/avatar/YUMI.md).

**Weights and data**: `runs/yumi/best.pt` (`f1a20071`, v0.2.0 primary model); historical checkpoints are described under [Weights and checkpoints](#weights-and-checkpoints) and in the GitHub Releases. `data/full_graph.npz` enforces SHA-256 validation on load, with the provenance manifest in `research/provenance/provenance.json`.

### Disclaimer

Research experiment code. Results cover flat-ground walking only, with no validation on a real robot. The developers accept no responsibility for any direct or indirect consequences of using this project.
