"""ST-GCN encoder + linear classification head for standard cross-entropy training."""
import logging
import torch
import torch.nn as nn
from src.models.stgcn import STGCNEncoder, build_stgcn_encoder

logger = logging.getLogger(__name__)


class STGCNClassifier(nn.Module):
    def __init__(self, encoder: STGCNEncoder, num_classes: int, dropout: float = 0.3):
        super().__init__()
        self.encoder = encoder
        self.num_classes = num_classes
        self.head = nn.Sequential(
            nn.Linear(encoder.embedding_dim, encoder.embedding_dim),
            nn.BatchNorm1d(encoder.embedding_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(encoder.embedding_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, T, F) -> (B, num_classes) logits"""
        emb = self.encoder(x)
        return self.head(emb)

    def classify(self, x: torch.Tensor) -> torch.Tensor:
        """API compatibility with PrototypicalNetwork.classify()"""
        return self.forward(x)


def build_classifier(cfg) -> STGCNClassifier:
    encoder = build_stgcn_encoder(cfg)
    model = STGCNClassifier(
        encoder=encoder,
        num_classes=cfg.num_classes,
        dropout=getattr(cfg, "head_dropout", 0.3),
    )
    total_params = sum(p.numel() for p in model.parameters())
    logger.info("STGCNClassifier: %.2fM trainable parameters", total_params / 1e6)
    return model
