# TemporalMapLoc

**BEV-to-Vector-Map Localization with Motion-Guided Temporal Fusion**

[![CI](https://github.com/Zelong-G/visual-map-localization/actions/workflows/ci.yml/badge.svg)](https://github.com/Zelong-G/visual-map-localization/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1%2B-EE4C2C)

TemporalMapLoc is a compact PyTorch research implementation for local vehicle pose correction from bird's-eye-view (BEV) features and vectorized maps. It produces a probabilistic local SE(2) posterior and can propagate that posterior through time using relative vehicle motion and uncertainty.

This repository is a cleaned public implementation of ideas explored in my master's research on precise visual localization for autonomous driving. It is intentionally structured for **inspection, reproducibility, and experimentation** rather than as a dump of an internal training environment.

> **Public-release scope.** Proprietary datasets, sensor logs, HD-map assets, checkpoints, cached features, private launchers, and internal experiment tooling are not included. The core localization, posterior transport, fusion logic, synthetic demonstrations, and unit tests are standalone.

## Research idea

A single frame can be ambiguous: several nearby map alignments may explain the current BEV observation. Instead of immediately collapsing the estimate to one pose, TemporalMapLoc keeps a discrete probability distribution over local SE(2) corrections and carries that uncertainty into the next frame.

```mermaid
flowchart LR
    A[Camera / LiDAR] --> B[BEV encoder or external adapter]
    C[Vector map] --> D[Type-aware map encoder]
    B --> E[Cross-modal refinement]
    D --> E
    E --> F[Chunked SE(2) hypothesis scoring]
    F --> G[Observation posterior]
    H[Previous posterior] --> I[Motion transport + uncertainty diffusion]
    J[Relative vehicle motion] --> I
    I --> K[Robust probabilistic fusion]
    G --> K
    K --> L[Current pose posterior]
```

The public implementation focuses on two questions:

1. **Single-frame localization:** how to score a local grid of translation/yaw hypotheses by matching vector-map queries against BEV evidence.
2. **Temporal localization:** how to transport a previous pose posterior with relative motion, account for motion uncertainty, and fuse it causally with the current observation.

## What is implemented

| Component | Public implementation |
| --- | --- |
| BEV interface | Validated `[B,C,H,W]` tensors plus a pluggable external-backend adapter |
| Vector map | Typed line segments with padding/masks and learned map-query encoding |
| Cross-modal model | BEV-conditioned refinement of vector-map queries |
| Pose estimation | Exhaustive local SE(2) grid, chunked scoring, normalized posterior, circular yaw expectation |
| Temporal model | SE(2) posterior transport, uncertainty diffusion, reliability-aware product-of-experts fusion |
| Engineering | YAML configs, train/evaluate CLIs, synthetic benchmark/demo, data-free tests |

For the mathematical details, see [docs/METHOD.md](docs/METHOD.md).

## Synthetic temporal demo

The included demo is dataset-free. It constructs a broad current-frame observation, propagates a previous posterior with uncertain relative motion, fuses both distributions, and can write a before/after heatmap:

```bash
python scripts/demo.py --output outputs/temporal_demo.png
```

The generated visualization is a functional illustration only, not a real-data benchmark result.

## Quick start

```bash
git clone https://github.com/Zelong-G/visual-map-localization.git
cd visual-map-localization

python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev,visualization]"
```

Run the public checks:

```bash
python scripts/demo.py --output outputs/temporal_demo.png
python scripts/evaluate_temporal.py --config configs/temporal.yaml --synthetic
python scripts/benchmark.py --config configs/single_frame.yaml --synthetic
pytest -q
```

The current test suite contains **9 unit tests** covering SE(2) geometry, pose-grid construction, posterior normalization, pose-solver contracts, and temporal probability-mass propagation.

## Public data interface

The single-frame training/evaluation scripts consume directories of `.pt` samples. Each sample is a dictionary containing:

```text
bev            [C, H, W]   BEV feature tensor
segments       [N, 4]      vector-map segment endpoints
type_ids       [N]         semantic/type IDs
map_mask       [N]         valid-segment mask
target_offset  [3]         [dx, dy, dyaw] supervision
```

Segment endpoints are expressed in the initial ego frame in meters. Local poses use `[x, y, yaw]`, with `x` forward, `y` left, and positive yaw counter-clockwise.

A real camera/LiDAR system can be connected through `BEVFusionAdapter` (or another callable backend) as long as the backend returns a BEV tensor with shape `[B,C,H,W]`. No third-party perception backbone is vendored here.

## Training and evaluation

```bash
python scripts/train.py \
  --config configs/single_frame.yaml \
  --data /path/to/train_samples \
  --output outputs/run

python scripts/evaluate.py \
  --config configs/single_frame.yaml \
  --data /path/to/validation_samples \
  --checkpoint outputs/run/best.pt
```

The temporal CLI currently exposes a synthetic sequence contract rather than a dataset-specific sequence loader. This keeps the public repository independent of non-redistributable scene manifests, sensor logs, map assets, and calibration files.

## Repository layout

```text
temporal_maploc/
  bev/          BEV validation, encoder, preprocessing, external-backend adapter
  geometry/     SE(2) transforms and metric BEV geometry
  maps/         vector-map structures and query encoder
  models/       cross-modal matching, pose grid/solver, single-frame localizer
  temporal/     motion model, posterior transport, fusion, temporal localizer
  evaluation/   metrics and optional visualization
configs/        compact single-frame and temporal reference configs
scripts/        demo, train, evaluate, temporal evaluation, benchmark
tests/          data-free geometry/model/temporal tests
docs/           method and evaluation/reproducibility notes
```

## Reproducibility and release boundaries

The repository deliberately separates **public algorithmic code** from **project-specific assets**. A real-data benchmark should state the dataset version/split, sensor and map preprocessing, model configuration, checkpoint-selection rule, metric aggregation, and runtime protocol. See [docs/RESULTS.md](docs/RESULTS.md) for the reporting checklist.

Project-specific numerical results and checkpoints are not mirrored here until their release terms and evaluation protocol can be made public and independently interpretable. The synthetic demo and tests are therefore used as functional verification, not as performance claims.

## Third-party components

No third-party source code, pretrained weights, datasets, or generated features are vendored. Optional external backends remain under their own licenses and installation requirements. See [THIRD_PARTY.md](THIRD_PARTY.md).

## Citation

If this repository is useful in academic work, please cite the software metadata in [CITATION.cff](CITATION.cff). A paper-specific citation can be added when a corresponding public manuscript is available.

## Author

**Zelong Zheng**  
Technical University of Munich (TUM)  
Research interests: 3D computer vision, autonomous driving, multimodal perception, BEV perception, and visual localization.

GitHub: [Zelong-G](https://github.com/Zelong-G)

## License and reuse

No open-source license is granted for this repository at this time. The source is publicly visible for academic inspection, research discussion, and portfolio evaluation. Please contact the author before copying, redistributing, or incorporating substantial portions into another project. Third-party software remains governed by its respective license.
