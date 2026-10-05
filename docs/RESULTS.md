# Results Policy

This release intentionally contains no benchmark numbers, checkpoints, cached
features, per-scene records, or generated tables. A public result should be
added only after the following information can be released and independently
checked:

- dataset version and its redistribution terms;
- exact split construction without scene identifiers that cannot be published;
- sensor and map preprocessing;
- configuration, seed set, and checkpoint-selection rule;
- single-frame and temporal metrics, including units and aggregation;
- runtime hardware, precision, batch size, and warm-up protocol.

Report all planned seeds or an explicitly defined aggregate. Do not select a
single favorable run after evaluation.

The supplied `scripts/benchmark.py --synthetic` is a functional smoke
benchmark, not a research-performance claim.
