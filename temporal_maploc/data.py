"""Public, dataset-agnostic tensor sample loader for training and evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import Tensor
from torch.utils.data import Dataset

from .maps.vector_map import VectorMapBatch


REQUIRED_SAMPLE_KEYS = frozenset({"bev", "segments", "type_ids", "map_mask", "target_offset"})


class TensorSampleDataset(Dataset[dict[str, Tensor]]):
    """Load independent public tensor samples from a directory of `.pt` files.

    Files are intentionally not included in this repository. Each file must
    contain only the documented tensor dictionary from the README.
    """

    def __init__(self, directory: str | Path) -> None:
        root = Path(directory)
        if not root.is_dir():
            raise FileNotFoundError(f"sample directory does not exist: {root}")
        self.paths = sorted(root.glob("*.pt"))
        if not self.paths:
            raise FileNotFoundError(f"no .pt samples found in {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, Tensor]:
        item = torch.load(self.paths[index], map_location="cpu")
        if not isinstance(item, Mapping) or not REQUIRED_SAMPLE_KEYS.issubset(item):
            raise ValueError(f"{self.paths[index]} is not a valid TemporalMapLoc sample")
        return {name: torch.as_tensor(item[name]) for name in REQUIRED_SAMPLE_KEYS}


def collate_samples(samples: Sequence[Mapping[str, Tensor]]) -> dict[str, Any]:
    """Pad variable-length map segments and return a batch dictionary."""

    if not samples:
        raise ValueError("cannot collate an empty batch")
    bev = torch.stack([torch.as_tensor(sample["bev"]) for sample in samples])
    targets = torch.stack([torch.as_tensor(sample["target_offset"], dtype=bev.dtype) for sample in samples])
    max_elements = max(int(torch.as_tensor(sample["segments"]).shape[0]) for sample in samples)
    segments = torch.zeros(len(samples), max_elements, 4, dtype=bev.dtype)
    type_ids = torch.zeros(len(samples), max_elements, dtype=torch.long)
    mask = torch.zeros(len(samples), max_elements, dtype=torch.bool)
    for index, sample in enumerate(samples):
        current_segments = torch.as_tensor(sample["segments"], dtype=bev.dtype)
        current_types = torch.as_tensor(sample["type_ids"], dtype=torch.long)
        current_mask = torch.as_tensor(sample["map_mask"], dtype=torch.bool)
        count = current_segments.shape[0]
        if current_segments.shape != (count, 4) or current_types.shape != (count,) or current_mask.shape != (count,):
            raise ValueError("each map sample needs segments [N,4], type_ids [N], and map_mask [N]")
        segments[index, :count] = current_segments
        type_ids[index, :count] = current_types
        mask[index, :count] = current_mask
    return {"bev": bev, "vector_map": VectorMapBatch(segments, type_ids, mask), "target_offset": targets}
