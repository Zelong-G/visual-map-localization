"""Small configuration loader and model constructor used by public CLIs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import yaml

from temporal_maploc.bev import BEVEncoder, BEVSpec
from temporal_maploc.geometry import BEVGeometry
from temporal_maploc.maps import MapQueryEncoder
from temporal_maploc.models import CrossModalDecoder, PoseGrid, PoseSolver, SingleFrameLocalizer


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML mapping without injecting environment-specific defaults."""

    config_path = Path(path)
    with config_path.open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ValueError("configuration root must be a mapping")
    return config


def _range(mapping: Mapping[str, Any], name: str) -> tuple[float, float, float]:
    value = mapping.get(name)
    if not isinstance(value, Mapping) or set(value) != {"min", "max", "step"}:
        raise ValueError(f"pose_grid.{name} must contain min, max, and step")
    return (float(value["min"]), float(value["max"]), float(value["step"]))


def build_single_frame_localizer(config: Mapping[str, Any]) -> SingleFrameLocalizer:
    """Construct the compact single-frame model described by a YAML mapping."""

    model = config.get("model")
    bev = config.get("bev")
    pose_grid = config.get("pose_grid")
    if not isinstance(model, Mapping) or not isinstance(bev, Mapping) or not isinstance(pose_grid, Mapping):
        raise ValueError("configuration requires model, bev, and pose_grid mappings")
    geometry = BEVGeometry(tuple(bev["x_range_m"]), tuple(bev["y_range_m"]))
    hidden = int(model["hidden_dim"])
    grid = PoseGrid.from_ranges(
        x_m=_range(pose_grid, "x_m"),
        y_m=_range(pose_grid, "y_m"),
        yaw_deg=_range(pose_grid, "yaw_deg"),
    )
    bev_encoder = BEVEncoder(BEVSpec(int(model["bev_channels"]), geometry), hidden)
    map_encoder = MapQueryEncoder(int(model["map_type_count"]), hidden)
    decoder = CrossModalDecoder(hidden, int(model["attention_heads"]), int(model["decoder_layers"]), geometry)
    solver = PoseSolver(
        grid,
        geometry,
        samples_per_segment=int(model["samples_per_segment"]),
        temperature=float(model["temperature"]),
    )
    return SingleFrameLocalizer(bev_encoder, map_encoder, decoder, solver)
