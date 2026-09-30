# Shouko v0.3.0 — directional walking checkpoints

Two full-connectome Yumi lower-body checkpoints, a self-contained native MuJoCo runtime, graph, physics, hash-bound evidence and long-walk trajectories.

- Primary and verification model each pass 72/72 frozen trials with zero falls.
- All nine 120-second trials per model stay within |y| ≤ 2 m throughout.
- Mean absolute endpoint drift: 0.489 m primary / 0.447 m verification. Maximum whole-trajectory drift: 0.837 m / 0.965 m.
- Protocol `strict_gait_v2_corridor2` checks strict gait and the long-walk corridor separately. It rejects mismatched checkpoints, body files and incomplete cohorts.
- Includes SHA-256 inventory, model card, evaluation commands and native-CSR batch-one smoke checks for both models.

The primary was fixed before holdouts. Different seeds and training histories mean the two final means are not a ranking. Explicit phase and external yaw feedback remain. This is limited flat-ground lower-body control, not autonomous CPG, topology superiority, general navigation or upper-body control. The shared-data matrix does not isolate the cause of improvement.

Download the archive, manifest and SHA256SUMS.txt together, verify them, then follow REPRODUCE.md inside the archive. The package supports inference/evaluation reproduction; complete historical training datasets and VRM assets are excluded.

The online preview and default installed model remain on v0.2.0. Existing v0.2.0 assets and tag are preserved. See MODEL_CARD.md for hashes, training provenance and limitations, and VALIDATION.md for release checks.
