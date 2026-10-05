"""BEV encoders, preprocessing contracts, and external-backend adapters."""

from .bev_encoder import BEVEncoder
from .bevfusion_adapter import BEVFusionAdapter
from .preprocessing import BEVSpec, normalize_bev, validate_bev

__all__ = ["BEVEncoder", "BEVFusionAdapter", "BEVSpec", "normalize_bev", "validate_bev"]
