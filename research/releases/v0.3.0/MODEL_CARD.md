# Shouko v0.3.0 — bounded directional walking

Version: v0.3.0, 2026-09-30. This is an experimental flat-ground controller for the 12-joint Yumi lower body. The independent headless package includes two fixed checkpoints, the complete graph, physics, evaluation code and evidence. The online preview remains on v0.2.0.

## Model roles and training

| File | Role | SHA-256 |
|---|---|---|
| `models/primary.pt` | Stage 2 A/3031/S1, fixed before final holdouts | `1e1c3160448f6acaddd28b1a8782dc75b037fed3e296525aa52ae060926801e6` |
| `models/replica.pt` | Earlier independent-data verification branch, one extra DAgger round | `fe386851ed9aa910aa837767ca647967388fba26b10c1c13784d82f996d2c916` |

The primary source is `runs/sensory_stability_20260930/stage2/A_3031_s1/train/last.pt`. The replica source is `runs/sensory_access_20260930/replica/dagger/round1/last.pt`. Role selection was fixed before final holdouts; do not rank them by final means from different seeds.

Both controllers retain 166,700 neurons and 25,582,938 connections. They have 50 observations, 12 actions and four neural updates. Neural state resets each control step. Training changed 17,336 parameters in the lateral-velocity encoder column (index 48) and 9,792 readout parameters: 27,128 total. The graph, neuron bias, other encoder columns and LayerNorm remained frozen relative to the trained source, not an untrained biological brain.

The source checkpoint SHA-256 is `e550239157c4ed2bf1b67914377e4e77c7d38c9d29a64066097abe9369072a10`; the MLP teacher is `56bf1dfac5fad0459bd184ccd96e96b09b5407fadb071330f7ae365f4f4bd4a7`. Teacher actions are not used during inference. Training provenance is in the three included experiment records and `provenance/` manifests.

The primary used 500 Adam updates on dataset A, then 500 further updates with fresh Adam state and a shared 70,200-row student-state pool. In the second-stage 2-dataset × 2-seed matrix, both readout-only C0 and S1 passed 4/4 configurations. Mean 120-second endpoint drift was 0.795107 m for C0 and 0.550951 m for S1. These are development results. The extra pool was shared, not independently recollected for every cell. The earlier same-budget replication failed. There is no equal-update/no-extra-data control, so the gain cannot be attributed solely to the new encoder column or solely to extra data.

## Frozen acceptance

| Test | Primary | Replica |
|---|---:|---:|
| Normal starts, 30 s | 27/27 | 27/27 |
| Initial yaw ±0.2 rad, 30 s | 18/18 | 18/18 |
| Long walks, 120 s, whole-trajectory corridor | 9/9 | 9/9 |
| Lateral velocity +0.25 m/s at 10 s | 9/9 | 9/9 |
| Lateral velocity −0.25 m/s at 10 s | 9/9 | 9/9 |
| Total, zero falls | 72/72 | 72/72 |
| Mean absolute endpoint drift, 120 s | 0.488854 m | 0.447332 m |
| Largest whole-trajectory lateral displacement, 120 s | 0.837310 m | 0.965464 m |

Evidence: `evidence/{primary,replica}/{normal,yaw,long,push_positive,push_negative}.json`. These are native MuJoCo, ordered-CSR, batch-nine evaluations. Long-walk NPZ trajectories are included and hash-bound. `release_evaluation.json` derives the display report from each raw group. Native CSR package checks are described in VALIDATION.md and have a separate status from the frozen holdouts.

Protocol `strict_gait_v2_corridor2` retains the six historical walking criteria plus ≥1 qualifying swing per foot per second (airborne ≥0.12 s, peak minimum sole clearance ≥0.025 m). Only the 120-second group adds the whole-trajectory |y| ≤ 2 m gate. Cached `success` or `strict_success` alone is not v2 acceptance. Speeds are 0.35, 0.50 and 0.65 m/s. Final seed bases are 9000001/9001001/9002001/9003001/9004001 for primary, and 9100001/9101001/9102001/9103001/9104001 for replica. They do not overlap the 693 previously used unique seeds checked before evaluation.

On the same development long-walk seed base 310201, the v0.2.0 primary had 5.426609 m mean endpoint drift and 0/9 corridor passes; this primary had 0.474983 m and 9/9. This is a matched development comparison, not an independent held-out comparison between releases. Source: `runs/direction_control_20260929/main_source_matched_development_long.json` and the stability matrix in the source repository.

## Limits

The 0.8-second phase input and external yaw feedback (gain 1.4, clamp ±0.2) remain. The lateral command is zero. The existing PD controller and body are unchanged. The policy has velocity feedback but no absolute lateral-position target. Residual drift depends on speed; passing 120 seconds does not establish indefinite position keeping.

This release does not establish autonomous CPG, biological topology superiority, natural human gait, general terrain recovery, arbitrary turning, start/stop control, or upper-body control. No new checkpoint-bound graph lesion experiment is claimed. The verification branch and stability matrix have different training histories; they are not fully independent end-to-end replications. Browser/WebGPU behavior of these new weights is not certified by native MuJoCo results.

Inference and evaluation are reproducible from the package. Complete historical training datasets, teacher/ancestor checkpoints and optimizer states are excluded, so byte-identical training reconstruction is not promised. Code uses MIT; graph and third-party terms and notices are in THIRD_PARTY.md and the included license files. No Yumi VRM is redistributed.
