# Evaluation and Reproducibility

This public repository separates functional verification from project-specific benchmark claims.

## Publicly verifiable checks

The release includes the following dataset-free checks:

```bash
pytest -q
python scripts/demo.py --output outputs/temporal_demo.png
python scripts/evaluate_temporal.py --config configs/temporal.yaml --synthetic
python scripts/benchmark.py --config configs/single_frame.yaml --synthetic
```

The current unit-test suite contains 9 tests covering:

- SE(2) composition, inversion, point transformation, and yaw wrapping;
- discrete pose-grid construction;
- posterior normalization and decoding contracts;
- single-frame pose-solver behavior;
- temporal transport and probability-mass behavior.

The synthetic demo visualizes observation and temporally fused posteriors without requiring a dataset, map, checkpoint, or sensor calibration. The synthetic benchmark is a software/runtime smoke test only; it is **not** a research-performance result.

## Why real-data numbers are not mirrored here

The public release intentionally does not bundle project-specific datasets, HD-map assets, sensor logs, checkpoints, cached features, or private split manifests. Publishing an isolated metric without the corresponding release conditions would make the number difficult to interpret or reproduce.

A real-data benchmark should therefore be added only when the following can be stated clearly:

- dataset version and evaluation split construction;
- camera/LiDAR calibration and BEV preprocessing;
- vector-map preprocessing and semantic classes;
- pose perturbation/search range and grid resolution;
- model configuration and checkpoint-selection rule;
- random seeds and aggregation policy;
- metrics and units (`dx`, `dy`, yaw, `dxy`, percentile/tail statistics as applicable);
- runtime hardware, precision, batch size, warm-up, synchronization, and excluded I/O costs.

When multiple runs are performed, report all planned seeds or a pre-defined aggregate rather than selecting a favorable run after evaluation.

## Intended use of this repository

The public code is meant to make the algorithmic structure inspectable and executable while keeping non-redistributable project assets out of the repository. Once a benchmark can be released with the information above, a results table and corresponding configuration can be added here without changing the public method implementation.
