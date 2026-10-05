# TemporalMapLoc

**BEV-to-Vector-Map Localization with Motion-Guided Temporal Fusion**

TemporalMapLoc is a PyTorch research framework for vehicle localization by
matching bird's-eye-view (BEV) features against vectorized maps, estimating a
probabilistic SE(2) pose correction, and fusing successive frames with a
motion-guided posterior.

```mermaid
flowchart LR
    A[Camera / LiDAR] --> B[BEV encoder or adapter]
    C[Vector map] --> D[Map-query encoder]
    B --> E[Cross-modal matching]
    D --> E
    E --> F[Pose score volume]
    F --> G[SE(2) posterior]
    H[Previous posterior] --> I[Motion propagation]
    J[Vehicle motion] --> I
    I --> K[Probabilistic fusion]
    G --> K
    K --> L[Refined pose]
```

## Overview

The single-frame localizer scores a discrete grid of local pose corrections.
Each hypothesis transforms vector-map segments into the initial ego frame,
samples BEV evidence along the segments, and produces a normalized posterior.
The temporal localizer transports the preceding posterior under a relative
SE(2) motion estimate, diffuses it according to motion uncertainty, and
combines it with the current-frame likelihood in log space.

This is a clean research release: proprietary data, checkpoints, cached
features, operational launchers, and exploratory analysis are excluded.

## Key features

- Explicit BEV coordinate convention and shape validation.
- Vector-map segments with typed, padded query encoding.
- Exhaustive, chunked SE(2) pose scoring and posterior expectation.
- Causal motion-prior propagation with trilinear mass transport.
- Stable product-of-experts temporal fusion.
- A BEVFusion-compatible adapter without vendored third-party source.
- Synthetic demo, benchmark, and data-free unit tests.

## Coordinate convention

All local poses are `[x, y, yaw]` where `x` is forward, `y` is left, and yaw
is counter-clockwise in radians. A pose maps coordinates from its local ego
frame into the map frame. BEV tensors are `[B, C, H, W]`; rows represent
decreasing `x` and columns represent increasing `y`. The supplied BEV range
defines the physical cell centers. Motion is a relative pose in the preceding
ego frame.

## Repository structure

```text
temporal_maploc/
  bev/          BEV validation, encoder, and external-backend adapter
  geometry/     SE(2) and metric BEV geometry
  maps/         vector-map data structures and query encoder
  models/       matching, pose grid, solver, and single-frame localizer
  temporal/     motion, posterior propagation, fusion, and temporal localizer
  evaluation/   metrics and optional visualization
scripts/        concise demo, train, evaluation, and benchmark entry points
tests/          data-free mathematical and model-contract tests
```

## Installation

```bash
git clone <your-fork-url> TemporalMapLoc
cd TemporalMapLoc
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev,visualization]'
```

For a production camera–LiDAR backbone, install the chosen backend in its own
environment. This package does not bundle backend code or weights.

## Quick start

The synthetic demo requires no dataset or checkpoint and writes an optional
before/after plot.

```bash
python scripts/demo.py --output outputs/demo.png
python scripts/benchmark.py --config configs/single_frame.yaml --synthetic
python scripts/evaluate_temporal.py --config configs/temporal.yaml --synthetic
pytest
```

## Data preparation

Training and evaluation accept a directory of `.pt` samples. Each sample is a
dictionary with `bev` `[C,H,W]`, `segments` `[N,4]`, `type_ids` `[N]`,
`map_mask` `[N]`, and `target_offset` `[3]`. Segment endpoints are expressed
in the initial ego frame in meters. `type_ids` must be integers in
`[0, map_type_count)`; padding is controlled by `map_mask`.

The data owner is responsible for checking dataset terms, map terms, sensor
calibration, and de-identification before preparing a release.

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

python scripts/evaluate_temporal.py \
  --config configs/temporal.yaml \
  --synthetic
```

The temporal evaluation CLI currently exposes a synthetic, data-free contract
test. Integrating a sequence loader is intentionally left to each dataset's
release policy; no private scene manifests or sensor logs are assumed here.

## Results

No numerical claim is included in this release. See
[docs/RESULTS.md](docs/RESULTS.md) for the reproducibility policy and the
required protocol before adding a benchmark table.

## Tests

The test suite covers SE(2) composition and inversion, yaw wrapping, pose-grid
construction, posterior normalization, pose-solver contracts, and temporal
probability-mass behavior.

## Third-party components

TemporalMapLoc vendors no third-party source files. The optional BEVFusion
adapter invokes a separately installed backend through a small callable
interface. See [THIRD_PARTY.md](THIRD_PARTY.md) and
[LICENSE_REVIEW_REQUIRED.md](LICENSE_REVIEW_REQUIRED.md) before publishing.

## Citation and contact

If you use this code, please replace this section with the authors' final
citation and contact information before publication.
