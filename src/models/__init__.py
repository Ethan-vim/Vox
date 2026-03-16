"""Unified model factory dispatching on cfg.approach."""
from typing import Any


def build_model(cfg: Any):
    if getattr(cfg, "approach", "stgcn_proto") == "stgcn_ce":
        from src.models.classifier import build_classifier
        return build_classifier(cfg)
    else:
        from src.models.prototypical import build_model as _build_proto
        return _build_proto(cfg)
