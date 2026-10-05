#!/usr/bin/env python3
"""Train the single-frame localizer on documented public tensor samples."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn.functional as functional
from torch.utils.data import DataLoader

from temporal_maploc.config import build_single_frame_localizer, load_yaml
from temporal_maploc.data import TensorSampleDataset, collate_samples
from temporal_maploc.geometry.se2 import wrap_angle
from temporal_maploc.maps import VectorMapBatch


def _to_device(batch: dict, device: torch.device) -> tuple[torch.Tensor, VectorMapBatch, torch.Tensor]:
    vector_map = batch["vector_map"]
    assert isinstance(vector_map, VectorMapBatch)
    return (
        batch["bev"].to(device),
        VectorMapBatch(vector_map.segments.to(device), vector_map.type_ids.to(device), vector_map.mask.to(device)),
        batch["target_offset"].to(device),
    )


def _target_indices(target: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
    delta = offsets.unsqueeze(0) - target.unsqueeze(1)
    delta[..., 2] = torch.as_tensor(wrap_angle(delta[..., 2]))
    return delta.square().sum(dim=-1).argmin(dim=-1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True, help="Single-frame YAML configuration.")
    parser.add_argument("--data", type=Path, required=True, help="Directory containing public .pt samples.")
    parser.add_argument("--output", type=Path, required=True, help="Directory for locally generated checkpoints.")
    parser.add_argument("--device", default="auto", help="PyTorch device, or 'auto'.")
    parser.add_argument("--epochs", type=int, default=None, help="Optional override for train.epochs in the YAML file.")
    args = parser.parse_args()

    config = load_yaml(args.config)
    training = config.get("train", {})
    device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else "cpu" if args.device == "auto" else args.device)
    dataset = TensorSampleDataset(args.data)
    loader = DataLoader(dataset, batch_size=int(training["batch_size"]), shuffle=True, collate_fn=collate_samples)
    model = build_single_frame_localizer(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(training["learning_rate"]), weight_decay=float(training["weight_decay"]))
    args.output.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    offsets = model.pose_solver.candidate_offsets.to(device)
    epochs = int(training["epochs"]) if args.epochs is None else args.epochs
    if epochs < 1:
        raise SystemExit("--epochs must be positive")
    for epoch in range(1, epochs + 1):
        model.train()
        total, count = 0.0, 0
        for batch in loader:
            bev, vector_map, target = _to_device(batch, device)
            result = model(bev, vector_map)
            loss = functional.nll_loss(result.pose.log_probabilities, _target_indices(target, offsets))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.detach()) * bev.shape[0]
            count += int(bev.shape[0])
        mean_loss = total / max(count, 1)
        payload = {"model": model.state_dict(), "config": config, "epoch": epoch, "loss": mean_loss}
        torch.save(payload, args.output / "last.pt")
        if mean_loss < best_loss:
            best_loss = mean_loss
            torch.save(payload, args.output / "best.pt")
        print(f"epoch={epoch:03d} loss={mean_loss:.6f}")


if __name__ == "__main__":
    main()
