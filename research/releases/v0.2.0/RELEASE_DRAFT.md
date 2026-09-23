# v0.2.0 — full-connectome Yumi gait transfer

This release supersedes the unpublished upright-baseline v0.2.0 plan. It does not replace any files in the public v0.1.0 release.

The new primary model preserves the full connectome core and changes only 9,792 readout parameters. It performs closed-loop Yumi walking without a teacher at runtime. Both the primary and an independent-data branch passed normal starts, yaw starts, 120-second walking and the specified lateral impulse. All final groups had zero falls.

The independent branch required an additional DAgger round. This is not equal-budget replication. Explicit phase input remains. Precision heading, straight-line walking, general disturbances, autonomous biological rhythm and topology superiority are not established.

The model package includes the primary, replication model, upright source, MLP teacher, full graph, checkpoint-bound evidence, runtime source snapshot, attribution and checksum manifest. The Yumi VRM and old optimizer states are excluded.

See `MODEL_CARD.md` for scope and `REPRODUCE.md` for package verification and headless evaluation. The previous upright source `458fc465…` remains an ancestor and diagnostic baseline. The published v0.1.0 squat model remains accessible in its original release.

Download `shouko-v0.2.0.tgz`, `manifest.json` and `SHA256SUMS.txt` from this Release. The archive contains four weights, the full graph and independently verified evidence. Source is tagged `v0.2.0`; `manifest.json` records the exact commit and per-file SHA-256 values.

The studio now defaults to Yumi and the primary checkpoint. [Windows studio installation](https://github.com/Moemu/Shouko/blob/v0.2.0/research/releases/v0.2.0/STUDIO.md) includes graph installation, checksum verification and automatic physics-skeleton display when the avatar is absent. [Headless evaluation](https://github.com/Moemu/Shouko/blob/v0.2.0/research/releases/v0.2.0/REPRODUCE.md) needs no avatar.

Validation: all four held-out groups and native CSR checks passed before release. An isolated package runtime passed a 30-second batch-one smoke test. Studio resume, pause, reset and bilingual evaluation display were checked locally. Unit tests and frontend build passed. A fresh dependency installation on every supported machine is not claimed.
