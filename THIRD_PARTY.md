# Third-party attribution

This experiment is not affiliated with or endorsed by the cited researchers or organizations.

The project's own code and documentation are under the MIT License (see `LICENSE`); the components below keep their own licenses, which MIT does not override.

- **Yumi**, version 1.3.0 / 20240715: original concept 松酒, artist 7Apoi, model 星晨水影工作室, publisher 墨海徽. [Author's release](https://www.bilibili.com/video/BV1uG411f72b/). The author allows non-profit use, monetized livestreams and derivative creation, and prohibits redistribution and resale. Embedded metadata also prohibits sexual and violent use. Keep attribution. Source and hash: `research/avatar/YUMI.md`; original metadata: `research/provenance/yumi-meta.json`. The local VRM file is excluded from Git.

- **MaleCNS v1.0**: FlyEM / HHMI Janelia, University of Cambridge, MRC LMB, Google Research and collaborators. [Official data](https://male-cns.janelia.org/download/), [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). We normalize records, select a subgraph, transform contact counts into signed weights and specify new dynamics. These transformations are not physiological measurements.
- **FlyCube**, MIT: [source](https://github.com/lntegrals/flycube-public), revision `c14b3adcbccbd15e3036bbc502af1a77bb50fe92`. Imported modules: MaleCNS downloader, importer, transmitter sign policy. Original notices in `vendor/flycube/LICENSE` and `vendor/flycube/THIRD_PARTY.md`; copies in `research/provenance/licenses/`.
- **DOOMFLY**, MIT: nftechie and contributors, upstream revision `71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33`. The reused FlyCube importer and sign policy derive from this project. No Doom assets are used.
- **Unitree RL Gym**, BSD-3-Clause: [source](https://github.com/unitreerobotics/unitree_rl_gym), revision `276801e46c5d433564f24658bac64f254b7d2d4b`. G1 model, pretrained checkpoint and adapted simulation conventions. Copyright (c) 2016–2023 HangZhou YuShu TECHNOLOGY CO., LTD. License retained in the checkout and `research/provenance/licenses/unitree.txt`.
- **VRM1_Constraint_Twist_Sample**, v1.0.1: copyright (c) 2022 pixiv Inc. [Official three-vrm sample](https://github.com/pixiv/three-vrm/tree/dev/packages/three-vrm/examples/models), [VRM Public License 1.0](https://vrm.dev/licenses/1.0/). Metadata permits everyone, redistribution, modification and modification redistribution. Model metadata is preserved in `research/provenance/avatar-meta.json`. “Hikari” is only a studio nickname, not the original asset name.
- **NeuroMechFly / FlyGym**: EPFL NeLy. Research reference for hierarchical sensorimotor control; no FlyGym code is bundled into the humanoid runtime.
- **MuJoCo**, Apache-2.0; **PyTorch**, BSD-style; **NumPy**, BSD-3-Clause; **SciPy**, BSD-3-Clause; **Three.js**, MIT; **three-vrm**, MIT; **Vite**, MIT; **FastAPI**, MIT; **Uvicorn**, BSD-3-Clause. Dependency license notices remain in the installed packages.
