# Attribution and license scope

FlyCube is an independent research experiment. It is not affiliated with or
endorsed by the MaleCNS collaboration, HHMI Janelia, the DOOMFLY authors, or
the cited researchers. Dataset names and marks belong to their owners.

## Upstream fork: DOOMFLY

- Source: https://github.com/nftechie/doomfly
- Revision used: `71ecf53d78eaffaf1a57ed7b0ccf5d458abc9f33`
- License: MIT (root [`LICENSE`](LICENSE), copyright 2026 nftechie and DOOMFLY
  contributors)
- Adapted into FlyCube:
  - `flycube/connectome/import_malecns.py` — loss-accounted exact-ID import
    (adapted from `doom/connectome.py`);
  - `flycube/connectome/datasets.json` — MaleCNS source registry
    (from `doom/datasets.json`);
  - `flycube/connectome/transmitters.py` — transmitter-to-fast-sign policy
    (from `doom/transmitters.py`).
- The upstream Doom/ViZDoom/Freedoom runtime, assets, WADs and training loops
  are **not** part of this repository and are not required to run FlyCube.
  Upstream license notices for those excluded components are preserved for
  provenance in
  [`data-provenance/upstream-doomfly-THIRD_PARTY.md`](data-provenance/upstream-doomfly-THIRD_PARTY.md)
  and
  [`data-provenance/upstream-doomfly-THIRD_PARTY_NOTICES.md`](data-provenance/upstream-doomfly-THIRD_PARTY_NOTICES.md).

## MaleCNS v1.0 connectome

Credit: the MaleCNS collaboration, including FlyEM at HHMI Janelia, the
University of Cambridge Department of Zoology, the MRC Laboratory of Molecular
Biology, and Google Research, with the authors and contributors identified by
the release.

- Dataset and release: https://male-cns.janelia.org/download/
- Paper: https://doi.org/10.1016/j.cell.2026.08.015
- License: [CC BY 4.0](licenses/CC-BY-4.0.txt)
- Scope: source-derived annotations, connectivity summaries, soma voxel
  coordinates and rendered dataset information. The large original files are
  downloaded separately and are not bundled.
- Changes: entries are normalized; explicit non-neuronal/unresolved objects are
  accounted for; all released edges among retained neurons are preserved.
  Weight scaling, transmitter-sign defaults, dynamics and trained weights are
  project transformations, not measurements supplied or validated by the
  dataset creators.
- Source URLs, immutable hashes and retention rules:
  [`data-provenance/malecns_v1/`](data-provenance/malecns_v1/).

## Reference repositories (inspected, not incorporated)

- `ornata/fly` (Fly64) and `dzhng/fly-escape`: no license was detected at the
  revisions inspected. No code or assets are copied; only independently
  reimplemented interaction ideas are used.
- `eonsystemspbc/fly-brain`: GPL-2.0; reference only. No code copied, and no
  FlyWire IDs, connectivity or geometry are mixed with MaleCNS.
- `forestagostinelli/DeepCubeA`: MIT; reference for learned cube value
  functions and search. No DeepCubeA code is vendored; its pretrained network
  is not used.

## Tooling dependencies (not redistributed)

- **kociemba** (python-kociemba wrapper): GPLv2. Optional, lazily imported,
  training/evaluation-only teacher. It is never imported by `flycube.runtime`,
  `flycube.search`, `flycube.train.evaluate`, or the live server, and it is not
  a runtime dependency of the demo.
- **pycuber**: MIT. Independent cube implementation used for cross-checks.
- **Zig** (MIT) was used locally only as a C compiler to build the GPL
  kociemba extension; no Zig code is linked into or distributed with FlyCube.
- **three.js** (MIT), **Vite** (MIT), **TypeScript** (Apache-2.0): frontend
  build/runtime dependencies installed from npm, not vendored into git.

## Asset and artifact policy

Committed binaries are limited to the small legacy checkpoints, verified
episode logs, and demo screenshots/recording. Datasets, prepared graphs, run
outputs and installed dependencies live outside git history.
