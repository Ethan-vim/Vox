"""Tests for src.inference.live_demo — FrameBuffer, MotionDetector, prediction smoothing."""

from dataclasses import dataclass

import numpy as np
import pytest

from src.inference.live_demo import FrameBuffer, LivePredictor, MotionDetector

NUM_KP = 543


# ---------------------------------------------------------------------------
# FrameBuffer
# ---------------------------------------------------------------------------


class TestFrameBuffer:
    def test_push_and_len(self):
        buf = FrameBuffer(max_size=10)
        assert len(buf) == 0
        frame = np.random.rand(NUM_KP, 3).astype(np.float32)
        buf.push(frame)
        assert len(buf) == 1

    def test_max_size(self):
        buf = FrameBuffer(max_size=5)
        for _ in range(10):
            buf.push(np.random.rand(NUM_KP, 3).astype(np.float32))
        assert len(buf) == 5

    def test_get_all_shape(self):
        buf = FrameBuffer(max_size=10)
        for _ in range(7):
            buf.push(np.random.rand(NUM_KP, 3).astype(np.float32))
        result = buf.get_all()
        assert result.shape == (7, NUM_KP, 3)

    def test_get_all_empty(self):
        buf = FrameBuffer(max_size=10)
        result = buf.get_all()
        assert result.shape == (0, NUM_KP, 3)

    def test_clear(self):
        buf = FrameBuffer(max_size=10)
        for _ in range(5):
            buf.push(np.random.rand(NUM_KP, 3).astype(np.float32))
        buf.clear()
        assert len(buf) == 0
        assert buf.get_all().shape[0] == 0

    def test_fifo_order(self):
        """Oldest frames should be dropped first."""
        buf = FrameBuffer(max_size=3)
        for i in range(5):
            frame = np.full((NUM_KP, 3), float(i), dtype=np.float32)
            buf.push(frame)
        result = buf.get_all()
        # Should contain frames 2, 3, 4
        assert result[0, 0, 0] == 2.0
        assert result[-1, 0, 0] == 4.0


# ---------------------------------------------------------------------------
# LivePredictor.smooth_predictions (static method)
# ---------------------------------------------------------------------------


class TestSmoothPredictions:
    def _pred(self, gloss, confidence, label_idx=0):
        return {
            "gloss": gloss,
            "confidence": confidence,
            "label_idx": label_idx,
            "top5": [(gloss, confidence)],
        }

    def test_empty_returns_none(self):
        result = LivePredictor.smooth_predictions([], mode="avg")
        assert result is None

    def test_majority_mode(self):
        preds = [
            self._pred("hello", 0.9, 0),
            self._pred("hello", 0.8, 0),
            self._pred("world", 0.7, 1),
        ]
        result = LivePredictor.smooth_predictions(preds, mode="majority")
        assert result["gloss"] == "hello"

    def test_avg_mode_picks_highest(self):
        preds = [
            self._pred("hello", 0.9, 0),
            self._pred("hello", 0.8, 0),
            self._pred("world", 0.7, 1),
        ]
        result = LivePredictor.smooth_predictions(preds, mode="avg")
        # "hello" has total prob 1.7 (avg 0.567) vs "world" 0.7 (avg 0.233)
        assert result["gloss"] == "hello"

    def test_single_prediction(self):
        preds = [self._pred("test", 0.95, 5)]
        result = LivePredictor.smooth_predictions(preds, mode="avg")
        assert result["gloss"] == "test"
        assert abs(result["confidence"] - 0.95) < 1e-5


# ---------------------------------------------------------------------------
# MotionDetector
# ---------------------------------------------------------------------------


@dataclass
class _MockCfg:
    """Minimal config for MotionDetector tests."""
    motion_start_threshold: float = 0.005
    motion_end_threshold: float = 0.003
    motion_settle_frames: int = 8
    max_sign_duration: int = 90
    static_sign_timeout: int = 45


def _still_frame():
    """Return a keypoint frame with hands present but no motion.

    Hand keypoints are non-zero so the motion detector treats them
    as valid detections (zero hand keypoints are skipped as detection
    failures).
    """
    kps = np.zeros((NUM_KP, 3), dtype=np.float32)
    # Set hand keypoints to a small constant so they pass the zero check
    kps[33:75, :] = 0.001
    return kps


def _moving_frame(magnitude: float = 0.05):
    """Return a keypoint frame with hand keypoints shifted by magnitude."""
    kps = np.zeros((NUM_KP, 3), dtype=np.float32)
    # Set hand keypoints (33-74) to a non-zero value
    kps[33:75, :] = magnitude
    return kps


class TestMotionDetector:
    def test_initial_state_is_idle(self):
        md = MotionDetector(_MockCfg())
        assert md.state == "IDLE"

    def test_idle_duration_increments(self):
        md = MotionDetector(_MockCfg())
        frame = _still_frame()
        for _ in range(10):
            md.update(frame)
        assert md.idle_duration >= 9  # first frame sets prev, subsequent ones count

    def test_detects_sign_start(self):
        md = MotionDetector(_MockCfg())
        # First frame: set baseline
        md.update(_still_frame())
        # Second frame: large hand displacement triggers SIGNING
        md.update(_moving_frame(0.05))
        assert md.state == "SIGNING"

    def test_detects_sign_end(self):
        cfg = _MockCfg(motion_settle_frames=3)
        md = MotionDetector(cfg)
        # Start signing
        md.update(_still_frame())
        md.update(_moving_frame(0.05))
        assert md.state == "SIGNING"

        # Settle: send identical still frames (zero velocity)
        still = _still_frame()
        for _ in range(4):
            md.update(still)
        assert md.state == "COMPLETED"

    def test_max_sign_duration_forces_completed(self):
        cfg = _MockCfg(max_sign_duration=5)
        md = MotionDetector(cfg)
        # Start signing
        md.update(_still_frame())
        md.update(_moving_frame(0.05))
        assert md.state == "SIGNING"

        # Keep signing for max_duration frames with continuous motion
        for i in range(5):
            md.update(_moving_frame(0.05 + i * 0.01))
        assert md.state == "COMPLETED"

    def test_reset_returns_to_idle(self):
        md = MotionDetector(_MockCfg())
        md.update(_still_frame())
        md.update(_moving_frame(0.05))
        assert md.state == "SIGNING"
        md.reset()
        assert md.state == "IDLE"
        assert md.idle_duration == 0

    def test_velocity_uses_hand_keypoints(self):
        """Only hand keypoints (33-74) should contribute to velocity."""
        md = MotionDetector(_MockCfg())
        # First frame: all zeros
        md.update(_still_frame())

        # Second frame: only body keypoints (0-32) move, hands stay still
        frame = _still_frame()
        frame[:33, :] = 0.1  # body moves
        state = md.update(frame)
        # Should remain IDLE since hands didn't move
        assert state == "IDLE"

    def test_stays_idle_with_no_motion(self):
        md = MotionDetector(_MockCfg())
        still = _still_frame()
        for _ in range(20):
            md.update(still)
        assert md.state == "IDLE"
