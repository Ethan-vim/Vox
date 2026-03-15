"""Tests for src.models — ST-GCN encoder, prototypical network, classifier, and unified build_model."""

import pytest
import torch

from src.models.stgcn import STGCNEncoder, build_stgcn_encoder
from src.models.prototypical import PrototypicalNetwork
from src.models.classifier import STGCNClassifier, build_classifier
from src.models import build_model
from src.training.config import Config


NUM_KP = 543
B, T = 2, 16


# ---------------------------------------------------------------------------
# STGCNClassifier
# ---------------------------------------------------------------------------


class TestSTGCNClassifier:
    def _make_cfg(self, **overrides):
        """Create a minimal Config for ST-GCN classifier tests."""
        defaults = dict(
            approach="stgcn_ce",
            num_keypoints=NUM_KP,
            num_classes=10,
            wlasl_variant=10,
            d_model=32,
            dropout=0.1,
            head_dropout=0.1,
            T=T,
            use_motion=False,
        )
        defaults.update(overrides)
        return Config(**defaults)

    def test_forward_shape(self):
        """STGCNClassifier output has shape (B, num_classes)."""
        cfg = self._make_cfg()
        encoder = build_stgcn_encoder(cfg)
        model = STGCNClassifier(encoder, num_classes=10, dropout=0.1)
        x = torch.randn(B, T, NUM_KP * 3)
        out = model(x)
        assert out.shape == (B, 10)

    def test_classify_matches_forward(self):
        """classify() returns same as forward()."""
        cfg = self._make_cfg()
        encoder = build_stgcn_encoder(cfg)
        model = STGCNClassifier(encoder, num_classes=10, dropout=0.1)
        model.eval()
        x = torch.randn(B, T, NUM_KP * 3)
        with torch.no_grad():
            out_fwd = model(x)
            out_cls = model.classify(x)
        torch.testing.assert_close(out_fwd, out_cls)

    def test_build_classifier_from_config(self):
        """build_classifier creates model from config."""
        cfg = self._make_cfg()
        model = build_classifier(cfg)
        assert isinstance(model, STGCNClassifier)
        assert model.num_classes == 10
        x = torch.randn(1, T, NUM_KP * 3)
        out = model(x)
        assert out.shape == (1, 10)

    def test_unified_build_model_ce(self):
        """build_model dispatches to classifier for stgcn_ce."""
        cfg = self._make_cfg(approach="stgcn_ce")
        model = build_model(cfg)
        assert isinstance(model, STGCNClassifier)

    def test_unified_build_model_proto(self):
        """build_model dispatches to prototypical for stgcn_proto."""
        cfg = self._make_cfg(approach="stgcn_proto")
        model = build_model(cfg)
        assert isinstance(model, PrototypicalNetwork)

    def test_model_is_differentiable(self):
        """Classifier model supports backprop."""
        cfg = self._make_cfg()
        model = build_classifier(cfg)
        x = torch.randn(1, T, NUM_KP * 3)
        out = model(x)
        loss = out.sum()
        loss.backward()
        has_grad = any(p.grad is not None for p in model.parameters())
        assert has_grad

    def test_forward_with_motion(self):
        """STGCNClassifier works with motion features (6 channels per keypoint)."""
        cfg = self._make_cfg(use_motion=True)
        model = build_classifier(cfg)
        x = torch.randn(B, T, NUM_KP * 6)
        out = model(x)
        assert out.shape == (B, 10)
