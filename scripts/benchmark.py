#!/usr/bin/env python3
"""Benchmark the public single-frame model on synthetic tensors."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from temporal_maploc.config import build_single_frame_localizer, load_yaml
from temporal_maploc.maps import VectorMapBatch


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Single-frame YAML configuration.")
    parser.add_argument("--synthetic", action="store_true", help="Benchmark generated tensors only.")
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--size", type=int, default=64, help="Synthetic square BEV side length.")
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    if not args.synthetic:
        raise SystemExit("This release exposes the safe --synthetic benchmark only.")
    if args.iterations < 1 or args.size < 8:
        raise SystemExit("iterations must be positive and size must be at least 8")
    config = load_yaml(args.config)
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    model = build_single_frame_localizer(config).to(device).eval()
    channels = int(config["model"]["bev_channels"])
    elements = 24
    bev = torch.randn(1, channels, args.size, args.size, device=device)
    segments = torch.empty(1, elements, 4, device=device).uniform_(-20.0, 20.0)
    vector_map = VectorMapBatch(segments, torch.randint(0, int(config["model"]["map_type_count"]), (1, elements), device=device), torch.ones(1, elements, device=device, dtype=torch.bool))
    with torch.no_grad():
        model(bev, vector_map)
        _sync(device)
        started = time.perf_counter()
        for _ in range(args.iterations):
            output = model(bev, vector_map)
        _sync(device)
    elapsed = (time.perf_counter() - started) / args.iterations
    print({"device": str(device), "iterations": args.iterations, "seconds_per_frame": round(elapsed, 6), "candidate_count": output.pose.grid.count})


if __name__ == "__main__":
    main()
