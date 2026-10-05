# Third-Party Components

No third-party source code, checkpoints, datasets, or generated features are
vendored in this repository.

| Component | Source repository | License observed in source review | Use here | Vendored? |
| --- | --- | --- | --- | --- |
| BEVFusion | `https://github.com/mit-han-lab/bevfusion` | Apache-2.0 (local source review) | Optional external camera–LiDAR BEV backend; accessed only through `BEVFusionAdapter` | No |
| PyTorch | `https://github.com/pytorch/pytorch` | BSD-style license; verify the selected wheel's notices | Tensor operations and neural-network layers | No |
| PyYAML | `https://pyyaml.org/` | MIT license; verify selected distribution | YAML configuration parsing in CLI scripts | No |

The adapter does not copy, import by a fixed module path, modify, or redistribute
BEVFusion. A user who chooses that backend must install it separately and comply
with its license, notices, build requirements, and the licenses of its
dependencies.

nuScenes-related tooling, map SDKs, and map data are not included. Their use is
outside this repository's distribution scope and requires independent review.
