# Third-Party Components

This repository does **not** vendor third-party source code, checkpoints, datasets, map assets, or generated features.

| Component | Upstream | Role in this repository | Bundled here? |
| --- | --- | --- | --- |
| BEVFusion | https://github.com/mit-han-lab/bevfusion | Optional external camera/LiDAR BEV backend through `BEVFusionAdapter` | No |
| PyTorch | https://github.com/pytorch/pytorch | Tensor operations and neural-network layers | No |
| PyYAML | https://pyyaml.org/ | YAML configuration parsing | No |
| Matplotlib | https://matplotlib.org/ | Optional synthetic posterior visualization | No |

`BEVFusionAdapter` defines a small callable boundary and does not copy a BEVFusion implementation, checkpoint, or configuration into this project. Users who connect an external backend are responsible for installing it separately and complying with its upstream license and dependency notices.

The same applies to dataset SDKs and map tooling (for example, nuScenes-related tooling): they are outside this repository's distribution scope and must be installed and used under their own terms.

Upstream licenses can change; verify the version you actually install rather than relying on this document as legal advice.
