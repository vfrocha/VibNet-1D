"""Interfaces e arquiteturas base em PyTorch para modelos 1D no domínio do tempo.

Módulo dedicado à fase Time-Domain (janelas de 1 segundo reamostradas e
normalizadas), separado do pipeline tabular (`tabvibnet`).
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn


class Conv1DFeatureExtractor(nn.Module):
    """Backbone convolucional 1D base com pooling global adaptativo.

    Aceita tensores no formato `(batch_size, 1, seq_len)` ou `(batch_size, seq_len)`.
    """

    def __init__(self, in_channels: int = 1, embedding_dim: int = 128) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, 32, kernel_size=64, stride=8, padding=28),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(32, 64, kernel_size=16, stride=2, padding=7),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=2, stride=2),
            nn.Conv1d(64, 128, kernel_size=8, stride=2, padding=3),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, embedding_dim, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(embedding_dim),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool1d(1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        out = self.features(x)
        return out.flatten(start_dim=1)


class TimeDomain1DClassifier(nn.Module):
    """Classificador 1D no domínio do tempo com suporte a única ou múltiplas cabeças."""

    def __init__(
        self,
        num_classes: int,
        in_channels: int = 1,
        embedding_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.encoder = Conv1DFeatureExtractor(
            in_channels=in_channels, embedding_dim=embedding_dim
        )
        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(embedding_dim, num_classes),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.encode(x)
        return self.head(latent)


def build_1d_time_model(
    num_classes: int,
    *,
    in_channels: int = 1,
    embedding_dim: int = 128,
    dropout: float = 0.3,
    device: Optional[str] = None,
) -> TimeDomain1DClassifier:
    """Instancia o modelo 1D no domínio do tempo e move para o dispositivo alvo."""
    target_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = TimeDomain1DClassifier(
        num_classes=num_classes,
        in_channels=in_channels,
        embedding_dim=embedding_dim,
        dropout=dropout,
    )
    return model.to(target_device)


__all__ = [
    "Conv1DFeatureExtractor",
    "TimeDomain1DClassifier",
    "build_1d_time_model",
]

