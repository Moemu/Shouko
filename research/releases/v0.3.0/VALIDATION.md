# v0.3.0 release validation

Date: 2026-09-30. No new training was run for release preparation.

## Evidence and source contract

Both fixed candidates passed 72/72 frozen ordered-CSR trials, including 9/9 whole-trajectory corridor trials. Training closeout verified 34 data files and all 17 matrix cells. Original checkpoint, runtime and body hashes and the long-walk trajectories are included. `provenance/closeout.json` records the historical checks.

The v2 adapter recomputes strict swing and long-walk corridor passes. Regression checks reject wrong checkpoint, body binding, changed protocol, incomplete groups and a crossing trajectory whose cached success fields still say true. v0.2.0 compatibility remains covered. An invalid preview export was also tested to leave an existing output directory intact.

## Independent package execution

The candidate archive was extracted into an isolated directory. All 55 payload files were verified by SHA-256. The runtime confirms that its imported model module is inside that directory. It has no vendor checkout, VRM, teacher checkpoint or private service configuration.

Both models passed native CSR at batch one on Windows, Python 3.12, PyTorch 2.12.1+cu130, MuJoCo 3.13.0 and an RTX 4070 Laptop GPU. Each model ran 30 seconds at 0.35/0.50/0.65 m/s and 120 seconds at 0.65 m/s. All eight episodes passed strict gait; both long episodes stayed in the corridor.

| Role | 30 s maximum lateral displacement at each speed (m) | 120 s maximum lateral displacement (m) |
|---|---|---:|
| Primary | 0.210394 / 0.041312 / 0.187656 | 0.821339 |
| Replica | 0.078393 / 0.060186 / 0.210890 | 0.932537 |

These are known-seed deployment checks, not additional independent holdouts. Small differences from ordered-CSR trajectories are expected; no bitwise cross-backend claim is made. The local dependency environment was reused. A fresh pip installation was not tested. Historical frozen acceptance ran on Linux/RTX 4090D; the cloud SSH connection was unavailable during packaging, so no new Linux package run is claimed.

The final archive must be built from committed source and checked again after extraction. Its release assets include `primary_native.json` and `replica_native.json`, recording the final source commit and actual final-package results. `SHA256SUMS.txt` binds these reports, the archive and manifest. Candidate manifests are never published as final releases.

## Regression and publication checks

Passed: release-evidence semantics, studio boot and checkpoint/interface binding, training control, direction metrics, sensory-column gradients and freeze boundary, imitation feature-cache invalidation, and readout-only PPO equivalence. All five frontend test suites and the Vite production build passed. The existing v0.2.0 preview package still passes its contract test.

All proposed tracked files, both checkpoint metadata trees and graph metadata passed private-endpoint/user-path checks. Weights, traces and release archives are release assets, not Git blobs. The package excludes the Yumi VRM and retains third-party notices. Both READMEs, training/installation guidance and model limitations were updated together.

Browser/WebGPU execution of v0.3.0 is not part of this validation. The default model, default preview package and published site remain v0.2.0. Upper-body integration remains separate work.
