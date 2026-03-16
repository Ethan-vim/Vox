# CLAUDE.md

> **STRICT RULE: Read and follow `EXPECTATIONS.md` at all times. Every response must comply with the rules defined there. This is non-negotiable.**

## Project Overview

**Vox** -- Live American Sign Language Recognition using WLASL. A real-time ASL word-level recognition system that captures webcam video, extracts hand/body keypoints via MediaPipe Holistic (543 landmarks per frame), and predicts the signed word using deep learning models trained on the [WLASL dataset](https://github.com/dxli94/WLASL).

Two training approaches are supported:

| Approach | Description | Best For |
|----------|-------------|----------|
| `stgcn_ce` | ST-GCN encoder + LayerNorm classification head with class-weighted CrossEntropyLoss | **Recommended default.** Best accuracy on all WLASL variants. |
| `stgcn_proto` | ST-GCN encoder + Prototypical Network (episodic metric learning) | Few-shot scenarios (~3-8 samples/class) |

## Environment

- **Python 3.12** (virtual env in `.venv`)
- Activate: `source .venv/bin/activate` (macOS/Linux) or `.venv\Scripts\Activate.ps1` (Windows)
- Install: `pip install -r requirements.txt`
- Tests: `python -m pytest` (293 tests across 12 test files, fully isolated via `tmp_path`)

## Key Files to Read Before Editing

| File | Why |
|------|-----|
| `README.md` | Full user guide, configuration reference, troubleshooting, recommended configs |
| `STRUCTURE.md` | File dependency graph, data flow diagrams, every CLI entry point |
| `EXPECTATIONS.md` | **Mandatory behavioral rules. Always follow.** |
| `src/training/config.py` | `Config` dataclass. `num_classes` is always auto-derived from `wlasl_variant`. Never set `num_classes` manually. |

## Project Structure

```
configs/                  YAML configs (stgcn_ce.yaml, stgcn_proto.yaml)
data/
  raw/                    Downloaded .mp4 video files (shared across variants)
  processed/              Extracted .npy keypoints (shared across variants)
  annotations/            WLASL_v0.3.json
  splits/WLASL{N}/        train.csv, val.csv, test.csv per variant
src/
  data/
    preprocess.py         Download annotations, extract keypoints, normalize, create splits
    augment.py            Temporal + spatial augmentations (rotation, dropout, flip, noise, etc.)
    dataset.py            PyTorch Dataset (WLASLKeypointDataset) + motion feature computation
    episode_sampler.py    EpisodicBatchSampler for N-way K-shot episodes
  models/
    __init__.py           Unified build_model() factory dispatching on cfg.approach
    stgcn.py              STGCNEncoder with body/hand graph branches
    classifier.py         STGCNClassifier: ST-GCN encoder + LayerNorm classification head
    prototypical.py       PrototypicalNetwork wrapper, build_model()
  training/
    config.py             Config dataclass + YAML load/save + auto-scaling per variant
    train.py              CLI dispatcher -> train_ce or train_prototypical
    train_ce.py           Cross-entropy training loop (class-weighted loss, augmentation)
    train_prototypical.py Episodic prototypical training loop
    evaluate.py           Metrics, TTA, confusion matrix, hard negatives, latency benchmark
  inference/
    predict.py            Single-video prediction (SignPredictor)
    live_demo.py          Real-time webcam demo (MotionDetector + threaded capture + display)
    export_onnx.py        ONNX export, verification, benchmarking
scripts/
    download_wlasl.py     Download annotations from GitHub
    download_kaggle.py    Download full video archive from Kaggle (~12K videos)
    validate_videos.py    Detect/remove HTML files masquerading as .mp4
    auto_config.py        Auto-detect hardware, generate optimized YAML config
    reset_configs.py      Reset configs/ to README defaults
    check_mediapipe.py    Diagnose MediaPipe installation issues
tests/                    293 tests across 12 test files + conftest.py (all isolated)
.github/workflows/
    claude.yml            Claude Code action (issue/PR comments)
    claude-code-review.yml  Claude Code review (PR test gate + review)
```

## Pipeline Summary

```
download (Kaggle/URL) -> validate_videos -> preprocess (keypoints/frames)
    -> train -> evaluate -> predict / live_demo / export_onnx
```

## Module Details

### Models (`src/models/`)

| Module | Key Classes | Purpose |
|--------|-------------|---------|
| `stgcn.py` | `STGCNEncoder` | ST-GCN encoder with body (33kp), left hand (21kp), right hand (21kp) graph branches, conditional L2 normalization |
| `classifier.py` | `STGCNClassifier` | ST-GCN encoder + two-layer head: `Linear -> LayerNorm -> ReLU -> Dropout -> Linear`. Uses LayerNorm (not BatchNorm) to prevent class collapse with small per-class batch counts. |
| `prototypical.py` | `PrototypicalNetwork` | Prototypical network wrapper for few-shot episodic classification |
| `__init__.py` | `build_model()` | Unified factory dispatching on `cfg.approach` (`stgcn_ce` or `stgcn_proto`) |

### Training (`src/training/`)

| Module | Purpose |
|--------|---------|
| `config.py` | `Config` dataclass with all hyperparameters. `num_classes` auto-derived from `wlasl_variant`. Architecture auto-scales per variant. |
| `train.py` | CLI dispatcher -> `train_ce` or `train_prototypical` based on `cfg.approach` |
| `train_ce.py` | Cross-entropy training with: class-weighted loss (inverse-frequency), label smoothing, mixup, weighted sampling, cosine scheduler, early stopping |
| `train_prototypical.py` | Episodic N-way K-shot training with support/query splits |
| `evaluate.py` | Metrics (top-1, top-5), TTA, confusion matrix, hard negatives, latency benchmark |

### Inference (`src/inference/`)

| Module | Purpose |
|--------|---------|
| `predict.py` | `SignPredictor` -- single video/keypoint prediction with augmentation |
| `live_demo.py` | Real-time webcam demo with `MotionDetector` (IDLE/SIGNING/COMPLETED state machine), `FrameBuffer`, confidence scaling by buffer fill ratio, prediction cooldown |
| `export_onnx.py` | ONNX export, verification, benchmarking |

### Data (`src/data/`)

| Module | Purpose |
|--------|---------|
| `preprocess.py` | Download annotations, extract keypoints via MediaPipe, normalize (shoulder-center + hand-relative), create train/val/test splits |
| `augment.py` | Temporal + spatial augmentations. `get_ce_train_transforms()` (milder) vs `get_train_transforms()` (proto, more aggressive) |
| `dataset.py` | `WLASLKeypointDataset` -- loads .npy, pads/crops to T frames, computes motion (velocity) features, applies augmentations |
| `episode_sampler.py` | `EpisodicBatchSampler` for N-way K-shot episodes |

## Important Implementation Details

- **Classification head uses LayerNorm** (not BatchNorm). BatchNorm caused class collapse because with batch_size=32 and 100 classes, each class gets ~0.32 samples/batch on average, making running statistics unreliable.
- **Class-weighted CrossEntropyLoss** (`class_weighted_loss: true`). Computes inverse-frequency weights from training labels so rare classes are penalized more. Works across all WLASL variants.
- **MediaPipe import** has a fallback path (`mediapipe.python.solutions`) for Windows/Python 3.12 compatibility -- see `_import_mediapipe_holistic()` in `preprocess.py`
- **Normalization** centers on shoulder midpoint (not nose) and scales by shoulder width -- see `normalize_keypoints()` in `preprocess.py`
- **Motion features** (`use_motion: true`) concatenate frame-to-frame velocity with position, producing 6 features per keypoint -- computed in `dataset.py`
- **CE augmentation** uses milder `get_ce_train_transforms()` (no TemporalFlip, narrower ranges) vs proto's `get_train_transforms()`
- **normalize_embeddings** -- `False` for CE (unconstrained logits), `True` for proto (distance-based classification)
- **MotionDetector** in `live_demo.py` tracks hand keypoint velocity (indices 33-74) through IDLE -> SIGNING -> COMPLETED states to detect sign boundaries. Prevents premature predictions.
- **Confidence scaling** multiplies raw softmax by `buffer_fill_ratio = min(1.0, N / T)` to penalize partial sequences
- **Episodic training** samples N-way K-shot episodes from the training set -- see `train_prototypical.py`
- **TTA** (`use_tta: true`) averages predictions over original + horizontally flipped input -- see `_flip_keypoints_tensor()` in `evaluate.py`
- **Multiprocessing** uses `spawn` context (not `fork`) to avoid MediaPipe crashes on macOS
- All split CSVs live under `data/splits/WLASL{N}/` -- multiple variants coexist without conflict
- **Config auto-scaling**: `Config.__post_init__()` auto-derives `num_classes`, `d_model`, `nhead`, `num_layers`, and `dropout` from `wlasl_variant`. Never set `num_classes` manually.

## Config Auto-Scaling (per variant)

| Variant | num_classes | d_model | nhead | num_layers | dropout |
|---------|-------------|---------|-------|------------|---------|
| 100     | 100         | 128     | 4     | 2          | 0.1     |
| 300     | 300         | 192     | 6     | 4          | 0.3     |
| 1000    | 1000        | 256     | 8     | 5          | 0.4     |
| 2000    | 2000        | 384     | 8     | 6          | 0.5     |

## CI/CD (GitHub Actions)

| Workflow | Trigger | What It Does |
|----------|---------|--------------|
| `claude.yml` | `@claude` mentions in issues/PRs | Runs Claude Code action with Python 3.12 environment |
| `claude-code-review.yml` | PR opened/synced | Runs pytest (skipping `TestKaggle`), then Claude code review |

**Note**: CI does not have `kaggle.json` credentials. Tests requiring Kaggle are skipped with `-k "not TestKaggle"`.

## CLI Entry Points

| Command | What It Does |
|---------|-------------|
| `python scripts/download_wlasl.py` | Download annotations, print video download instructions |
| `python scripts/download_kaggle.py` | Download all videos from Kaggle |
| `python scripts/validate_videos.py` | Scan and clean fake video files |
| `python scripts/auto_config.py` | Auto-detect hardware and generate optimized config |
| `python scripts/reset_configs.py` | Reset configs to recommended defaults |
| `python scripts/check_mediapipe.py` | Verify MediaPipe installation |
| `python -m src.data.preprocess` | Extract keypoints from videos, create splits |
| `python -m src.training.train --config configs/stgcn_ce.yaml` | Train a model |
| `python -m src.training.evaluate --config configs/stgcn_ce.yaml` | Evaluate a trained model |
| `python -m src.inference.predict --video path/to/video.mp4` | Predict on a single video |
| `python -m src.inference.live_demo --config configs/stgcn_ce.yaml` | Run real-time webcam demo |
| `python -m src.inference.export_onnx` | Export model to ONNX format |
| `python -m pytest` | Run all 293 tests |
