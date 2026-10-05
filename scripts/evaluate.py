#!/usr/bin/env python3
"""Evaluate a single-frame checkpoint on documented public tensor samples."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
from torch.utils.data import DataLoader

from temporal_maploc.config import build_single_frame_localizer, load_yaml
from temporal_maploc.data import TensorSampleDataset, collate_samples
from temporal_maploc.evaluation import pose_metrics
from temporal_maploc.maps import VectorMapBatch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Single-frame YAML configuration.")
    parser.add_argument("--data", type=Path, required=True, help="Directory containing public .pt samples.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint written by scripts/train.py.")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = load_yaml(args.config)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    model = build_single_frame_localizer(config).to(device)
    payload = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(payload["model"] if isinstance(payload, dict) and "model" in payload else payload)
    model.eval()
    predictions, targets = [], []
    loader = DataLoader(TensorSampleDataset(args.data), batch_size=args.batch_size, shuffle=False, collate_fn=collate_samples)
    with torch.no_grad():
        for batch in loader:
            vector_map = batch["vector_map"]
            assert isinstance(vector_map, VectorMapBatch)
            vector_map = VectorMapBatch(vector_map.segments.to(device), vector_map.type_ids.to(device), vector_map.mask.to(device))
            result = model(batch["bev"].to(device), vector_map)
            predictions.append(result.pose.estimate.cpu())
            targets.append(batch["target_offset"].cpu())
    metrics = pose_metrics(torch.cat(predictions), torch.cat(targets))
    print(metrics.as_dict())


if __name__ == "__main__":
    main()
