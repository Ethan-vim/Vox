# CLAUDE.md

> **STRICT RULE: Read and follow `EXPECTATIONS.md` at all times. Every response must comply with the rules defined there. This is non-negotiable.**

## Project Overview

**Live American Sign Language Recognition using WLASL** — a real-time ASL word-level recognition system that captures webcam video, extracts hand/body keypoints via MediaPipe Holistic (543 landmarks per frame), and predicts the signed word using deep learning models trained on the [WLASL dataset](https://github.com/dxli94/WLASL).

Uses **ST-GCN** as the default encoder with two training approaches: **cross-entropy classification** (default, `stgcn_ce`) and **prototypical network** for few-shot learning (`stgcn_proto`).

| Module | Purpose |
|--------|---------|
| `src/models/stgcn.py` | ST-GCN encoder with body/hand graph branches |
| `src/models/prototypical.py` | Prototypical network wrapper for few-shot training |
| `src/models/classifier.py` | Linear classification head for cross-entropy training |
| `src/models/__init__.py` | Unified build_model() factory |

Additional approaches are also implemented:

| Approach | Module | Input | Description |
|----------|--------|-------|-------------|
| A — Pose Transformer | `src/models/pose_transformer.py` | MediaPipe keypoints (T, 543, 6) with velocity | Lightweight, fast. |
| B — Video Classifier | `src/models/video_i3d.py` | RGB frames (3, T, 224, 224) | Pretrained 3D CNNs (R(2+1)D, R3D, SlowFast). |
| C — Hybrid Fusion | `src/models/fusion.py` | Both streams | Concat or cross-attention fusion of A + B. |

## Environment

- **Python 3.12** (virtual env in `.venv`)
- Activate: `source .venv/bin/activate` (Linux/macOS) or `.venv\Scripts\Activate.ps1` (Windows)
- Install: `pip install -r requirements.txt`
- Tests: `python -m pytest` (136 tests, fully isolated via `tmp_path`)

## Key Files to Read Before Editing

- `README.md` — Full user guide, configuration reference, troubleshooting, recommended configs
- `STRUCTURE.md` — File dependency graph, data flow diagrams, every CLI entry point
- `EXPECTATIONS.md` — **Mandatory behavioral rules. Always follow.**
- `src/training/config.py` — `Config` dataclass. `num_classes` is always auto-derived from `wlasl_variant`. Never set `num_classes` manually.

## Project Structure

```
configs/                  YAML configs (stgcn_ce.yaml, stgcn_proto.yaml, pose_transformer, video_classifier, fusion)
data/
  raw/                    Downloaded .mp4 video files (shared across variants)
  processed/              Extracted .npy keypoints (shared across variants)
  annotations/            WLASL_v0.3.json
  splits/WLASL{N}/        train.csv, val.csv, test.csv per variant
src/
  data/
    preprocess.py         Download annotations, extract keypoints, normalize, create splits
    augment.py            Temporal + spatial augmentations (rotation, dropout, flip, noise, etc.)
    dataset.py            PyTorch Datasets (Keypoint, Video, Fusion) + motion feature computation
    episode_sampler.py    EpisodicBatchSampler for N-way K-shot episodes
  models/
    __init__.py           Unified build_model() factory dispatching on cfg.approach
    stgcn.py              STGCNEncoder with body/hand graph branches
    classifier.py         STGCNClassifier: ST-GCN encoder + linear classification head
    prototypical.py       PrototypicalNetwork wrapper, build_model()
    pose_transformer.py   PoseTransformer, PoseBiLSTM, build_pose_model()
    video_i3d.py          VideoClassifier with torchvision/pytorchvideo backbones
    fusion.py             FusionModel, CrossAttentionFusion
  training/
    config.py             Config dataclass + YAML load/save
    train.py              CLI dispatcher -> train_ce or train_prototypical
    train_ce.py           Standard cross-entropy training loop with label smoothing + mixup
    train_prototypical.py Episodic prototypical training loop
    evaluate.py           Metrics, TTA, confusion matrix, hard negatives, latency benchmark
  inference/
    predict.py            Single-video prediction (SignPredictor)
    live_demo.py          Real-time webcam demo (threaded capture + inference + display)
    export_onnx.py        ONNX export, verification, benchmarking
scripts/
    download_wlasl.py     Download annotations from GitHub
    download_kaggle.py    Download full video archive from Kaggle (~12K videos)
    validate_videos.py    Detect/remove HTML files masquerading as .mp4
    auto_config.py        Auto-detect hardware, generate optimized YAML config
    reset_configs.py      Reset configs/ to README defaults
    check_mediapipe.py    Diagnose MediaPipe installation issues
tests/                    136 tests across 10 test files + conftest.py (all isolated)
```

## Pipeline Summary

```
download (Kaggle/URL) -> validate_videos -> preprocess (keypoints/frames)
    -> train -> evaluate -> predict / live_demo / export_onnx
```

## Important Implementation Details

- **MediaPipe import** has a fallback path (`mediapipe.python.solutions`) for Windows/Python 3.12 compatibility — see `_import_mediapipe_holistic()` in `preprocess.py`
- **Normalization** centers on shoulder midpoint (not nose) and scales by shoulder width — see `normalize_keypoints()` in `preprocess.py`
- **Motion features** (`use_motion: true`) concatenate frame-to-frame velocity with position, producing 6 features per keypoint — computed in `dataset.py`
- **Mixup** (`mixup_alpha: 0.2`) interpolates random training pairs — see `_mixup_data()` in `train_ce.py`
- **Episodic training** samples N-way K-shot episodes from the training set — see `train_prototypical.py`
- **TTA** (`use_tta: true`) averages predictions over original + horizontally flipped input — see `_flip_keypoints_tensor()` in `evaluate.py`
- **Multiprocessing** uses `spawn` context (not `fork`) to avoid MediaPipe crashes on macOS
- All split CSVs live under `data/splits/WLASL{N}/` — multiple variants coexist without conflict
