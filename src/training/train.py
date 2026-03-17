"""
CLI entry point for training WLASL sign language recognition models.

Dispatches to the appropriate training loop based on cfg.approach:
- stgcn_ce: Standard cross-entropy training (src.training.train_ce)
- stgcn_proto: Episodic prototypical training (src.training.train_prototypical)
"""

import argparse
import logging

from src.training.config import load_config

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train WLASL recognition model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config file",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps"],
        help="Force device (default: auto-detect)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    cfg = load_config(args.config)

    if cfg.approach == "stgcn_ce":
        from src.training.train_ce import main
        main(cfg, device_override=args.device)
    elif cfg.approach == "stgcn_proto":
        from src.training.train_prototypical import main
        main(cfg, device_override=args.device)
    else:
        raise ValueError(
            f"Unknown approach: '{cfg.approach}'. "
            f"Use 'stgcn_ce' or 'stgcn_proto'."
        )
