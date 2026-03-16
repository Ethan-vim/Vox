## Pretrained models (best checkpoints)

This folder contains **the best pretrained model checkpoint for each model variant** used in this project.

**What “best” means**

- **Selection rule**: for each variant, we keep the checkpoint with the **best evaluation performance** among the trained candidates (e.g., highest Top-1 accuracy on the evaluation set for that variant).
- **One per variant**: if you train multiple runs/checkpoints for the same variant, only the **single best-performing** one should be copied into this folder.

### Available best checkpoints


| Variant  | Best checkpoint file |
| -------- | -------------------- |
| WLASL100 | `WLASL100.pt`        |


