"""Diagnostic model (image-only-with-target): shared ResNet-18 over three MRI views."""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn as nn
from torchvision import models


class TripleImageEncoder(nn.Module):
    """Shared ResNet-18 backbone applied to axial / coronal / sagittal."""

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        aggregation: Literal["concat", "mean"] = "concat",
    ):
        super().__init__()
        self.aggregation = aggregation
        weights = None
        if pretrained:
            try:
                weights = models.ResNet18_Weights.DEFAULT
            except Exception:
                weights = None
        backbone = models.resnet18(weights=weights)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.single_embedding_dim = in_features
        self.output_dim = in_features * 3 if aggregation == "concat" else in_features

        if freeze_backbone:
            self.freeze_backbone()

    def freeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = False

    def unfreeze_backbone(self) -> None:
        for p in self.backbone.parameters():
            p.requires_grad = True

    def unfreeze_layer4_only(self) -> None:
        """Keep early ResNet blocks frozen; train only layer4 (+ already-trainable head)."""
        self.freeze_backbone()
        for p in self.backbone.layer4.parameters():
            p.requires_grad = True

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # images: [B, 3, C, H, W]
        b, n, c, h, w = images.shape
        x = images.view(b * n, c, h, w)
        emb = self.backbone(x)
        emb = emb.view(b, n, -1)
        if self.aggregation == "mean":
            return emb.mean(dim=1)
        if self.aggregation == "concat":
            return emb.reshape(b, -1)
        raise ValueError(f"Unknown aggregation: {self.aggregation}")


class DiagnosticNet(nn.Module):
    """
    diagnostic classifier.

    Head: LayerNorm -> Linear(1536->256) -> ReLU -> Dropout -> Linear(256->1) logit.
    """

    def __init__(
        self,
        pretrained: bool = True,
        freeze_backbone: bool = True,
        aggregation: Literal["concat", "mean"] = "concat",
        hidden_dim: int = 256,
        dropout: float = 0.35,
    ):
        super().__init__()
        self.encoder = TripleImageEncoder(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            aggregation=aggregation,
        )
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.head = nn.Sequential(
            nn.LayerNorm(self.encoder.output_dim),
            nn.Linear(self.encoder.output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def freeze_backbone(self) -> None:
        self.encoder.freeze_backbone()

    def unfreeze_layer4_only(self) -> None:
        self.encoder.unfreeze_layer4_only()

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        emb = self.encoder(images)
        return self.head(emb).squeeze(1)

    def architecture_dict(self) -> dict:
        return {
            "name": "DiagnosticNet",
            "model": "diagnostic",
            "backbone": "resnet18",
            "aggregation": self.encoder.aggregation,
            "embedding_dim": self.encoder.output_dim,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "head": [
                "LayerNorm",
                f"Linear({self.encoder.output_dim}->{self.hidden_dim})",
                "ReLU",
                "Dropout",
                "Linear->1",
            ],
        }
