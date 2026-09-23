# Shouko v0.2.0 — full-connectome Yumi gait

Release version: v0.2.0, 2026-09-23. Source and artifact hashes are bound in the release manifest.

This package contains a full-connectome controller for the project's 12-joint Yumi MuJoCo body. It is an experimental flat-ground controller with explicit phase input. It does not demonstrate autonomous biological rhythm, natural human gait, general recovery, or biological topology superiority.

## Model roles

| File | Role | SHA-256 |
|---|---|---|
| `models/primary.pt` | Default release model, fixed before final acceptance | `f1a200718ebfec845cbf9c02ba743e937ea94c747fba53e7b12dfd12f8d3e1f9` |
| `models/replica.pt` | Independent-data replication, with one additional DAgger round | `8841868cf1cbbbd01be19585bc18e751c11901b3e806556705d124bb207616cf` |
| `models/upright-source.pt` | Previous unpublished v0.2.0 plan; preserved ancestor, not recommended walking model | `458fc465a6fb802a338c218fc4be414fc9153376c5d2f76f116518e6cc5aa475` |
| `models/mlp-teacher.pt` | Training reference; never used during connectome evaluation | `9c5ca9339b26f5af2e8acf94e7edd3b9371ef5efb41d647bcbd7904243ac50e1` |

The public v0.1.0 squat checkpoint `a7a4281f…` remains unchanged in its original Release. Results under its older criteria are historical evidence, not a matched comparison against this package.

## Training and interface

The controller retains 166,700 neurons and 25,582,938 measured connections. It uses 50 observations, 12 actions, four neural updates and reset neural activity at every control step. The checkpoint embeds its physics interface. Only the 9,792 readout parameters changed relative to the previously trained upright source. The core is frozen relative to that source, not untrained biological tissue.

Teacher-state regression alone failed closed-loop control. Student-state DAgger, lateral-impulse data and ridge regularization produced the final controllers. The primary used three initial DAgger rounds, two impulse-data rounds and a cached ridge refit from 0.001 to 0.0001. The independent-data branch needed one further impulse-data round. This is not equal-budget replication or replication from a different graph initialization.

The runtime uses the existing PD actuator and heading-command feedback. It does not load or mix teacher actions. The 0.8-second phase input remains; measured swing rate is 1.25 per foot per second.

## Acceptance

| Held-out test | Primary | Replica |
|---|---:|---:|
| Normal starts, 30 seconds | 27/27 | 27/27 |
| Initial yaw ±0.2 radians, 30 seconds | 18/18 | 18/18 |
| 120 seconds | 9/9 | 9/9 |
| Lateral velocity +0.25 m/s at 10 seconds | 9/9 | 9/9 |

All listed episodes had zero falls. Tests use commanded speeds 0.35, 0.50 and 0.65 m/s, native MuJoCo and ordered CSR. Separate native CSR checks passed 27/27 for both models. Primary graph disconnection failed 9/9. Disconnection shows dependence on graph computation, not an advantage over random graphs.

Strict acceptance combines six existing criteria with at least one qualifying swing per foot per second. A qualifying swing is airborne for at least 0.12 seconds and reaches 0.025 m peak minimum-sole clearance. The direction threshold permits lateral displacement up to max(1 m, 20% of commanded distance). Passing does not mean precise straight-line walking.

At 120 seconds, mean lateral drift was 5.47345 m / 4.89646 m for primary / replica. Precise yaw recovery was 9/18 / 6/18. Precise recovery means |yaw|≤0.05 radians for 0.5 seconds, not sustained straight walking thereafter. Stopping, turning, variable speeds, uneven terrain and arbitrary pushes remain unvalidated.

Each evaluation JSON embeds its checkpoint hash. Historical filenames inside audit reports refer to the original experiment layout; packaged evidence is organized under `evidence/`. The four model files and actual runtime source files are independently hashed in `manifest.json`. The recorded source HEAD identifies the release commit. Packaged runtime files must match that commit byte for byte.

## Attribution and scope

Project code: MIT, see `LICENSE`. Imported notices remain under `research/provenance/licenses/`; see `THIRD_PARTY.md`.

MaleCNS v1.0: FlyEM / HHMI Janelia, University of Cambridge, MRC LMB, Google Research and collaborators. [Official data and CC-BY attribution](https://male-cns.janelia.org/download/). The packaged graph is a transformed data product: contact counts become signed normalized weights, sensory/motor groups define interfaces, and engineered rate dynamics replace biological spiking. See `data/full_graph.json` for source URLs, checksums and transformations; [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) applies as documented in this project's attribution. Project MIT does not replace upstream terms.

The Yumi VRM is NOT included. Yumi credits: concept 松酒; artist 7Apoi; model 星晨水影工作室; publisher 墨海徽. Its original distribution terms prohibit redistribution. The package uses the repository's geometric simulation body and needs no VRM or private download to run the headless evaluation. No affiliation or endorsement is implied.

No PPO optimizer state is supplied for these ridge-fitted endpoints. Do not pair the new weights with the old source's PPO state. Inference and readout refitting are supported by the packaged code; exact historical end-to-end retraining requires the separately archived datasets and protocol.
