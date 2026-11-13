"""Arquitetura ResNet18-1D e treinamento From Scratch no domínio do tempo (VibNet1DTime).

Implementa a rede residual 1D `ResNet18_1D` operando diretamente sobre janelas
temporais `(Batch, 1, L)` e a rotina `train_and_eval_resnet1d_scratch`.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset


class BasicBlock1D(nn.Module):
    """Bloco residual básico 1D com duas convoluções `kernel_size=3`."""

    expansion: int = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


class ResNet18_1D(nn.Module):
    """ResNet-18 adaptada para sinais 1D no domínio do tempo `(Batch, 1, L)`.

    Estrutura:
    - Camada inicial: `Conv1d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)`,
      `BatchNorm1d(64)`, `ReLU()`, `MaxPool1d(kernel_size=3, stride=2, padding=1)`.
    - 4 estágios residuais (`[64, 128, 256, 512]`, 2 blocos cada, `kernel_size=3`,
      `stride=2` no início dos estágios 2, 3 e 4).
    - `AdaptiveAvgPool1d(1)` + `Flatten()` -> vetor latente de 512 dimensões ->
      `Linear(512, num_classes)`.
    """

    def __init__(self, num_classes: int, in_channels: int = 1) -> None:
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv1d(
            in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, num_blocks=2, stride=1)
        self.layer2 = self._make_layer(128, num_blocks=2, stride=2)
        self.layer3 = self._make_layer(256, num_blocks=2, stride=2)
        self.layer4 = self._make_layer(512, num_blocks=2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(start_dim=1)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(
        self, out_channels: int, num_blocks: int, stride: int = 1
    ) -> nn.Sequential:
        downsample = None
        if stride != 1 or self.in_planes != out_channels:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_planes,
                    out_channels,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm1d(out_channels),
            )

        layers = [
            BasicBlock1D(
                self.in_planes,
                out_channels,
                stride=stride,
                downsample=downsample,
            )
        ]
        self.in_planes = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock1D(self.in_planes, out_channels, stride=1))

        return nn.Sequential(*layers)

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = self.flatten(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = self.extract_features(x)
        return self.fc(latent)


def train_and_eval_resnet1d_scratch(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    task: str = "diagnosis",
    epochs: int = 30,
    batch_size: int = 64,
    lr: float = 1e-3,
) -> Tuple[float, float, float]:
    """Treina e avalia a `ResNet18_1D` do zero (From Scratch) no domínio do tempo.

    Utiliza otimizador `Adam(lr=lr)`, escalonador `StepLR(step_size=10, gamma=0.5)` e
    `CrossEntropyLoss`.

    Retorna
    -------
    Tuple[float, float, float]
        `(bal_acc, macro_f1, roc_auc)`
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    X_tr_arr = np.asarray(X_train, dtype=np.float32)
    X_te_arr = np.asarray(X_test, dtype=np.float32)
    if X_tr_arr.ndim == 2:
        X_tr_arr = np.expand_dims(X_tr_arr, axis=1)
    if X_te_arr.ndim == 2:
        X_te_arr = np.expand_dims(X_te_arr, axis=1)

    y_tr_arr = np.asarray(y_train, dtype=np.int64)
    y_te_arr = np.asarray(y_test, dtype=np.int64)

    X_tr_t = torch.from_numpy(X_tr_arr)
    y_tr_t = torch.from_numpy(y_tr_arr)
    X_te_t = torch.from_numpy(X_te_arr)
    y_te_t = torch.from_numpy(y_te_arr)

    train_loader = DataLoader(
        TensorDataset(X_tr_t, y_tr_t),
        batch_size=batch_size,
        shuffle=True,
        drop_last=(len(X_tr_t) > batch_size and len(X_tr_t) % batch_size == 1),
    )

    num_classes = (
        2 if task.lower() == "detection" else int(max(len(np.unique(y_tr_arr)), int(np.max(y_tr_arr)) + 1))
    )

    model = ResNet18_1D(num_classes=num_classes, in_channels=X_tr_arr.shape[1]).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    model.train()
    for _ in range(epochs):
        for bx, by in train_loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    all_logits = []
    test_loader = DataLoader(
        TensorDataset(X_te_t, y_te_t),
        batch_size=max(batch_size, 128),
        shuffle=False,
    )
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device, non_blocking=True)
            logits = model(bx)
            all_logits.append(logits.cpu())

    final_logits = torch.cat(all_logits, dim=0)
    probs = torch.softmax(final_logits, dim=1).numpy()
    preds = np.argmax(probs, axis=1)

    bal_acc = float(balanced_accuracy_score(y_te_arr, preds))

    if task.lower() == "detection":
        macro_f1 = float(f1_score(y_te_arr, preds, average="binary", zero_division=0))
        try:
            roc_auc = float(roc_auc_score(y_te_arr, probs[:, 1]))
        except ValueError:
            roc_auc = 0.0
    else:
        macro_f1 = float(f1_score(y_te_arr, preds, average="macro", zero_division=0))
        try:
            roc_auc = float(roc_auc_score(y_te_arr, probs, multi_class="ovr"))
        except ValueError:
            roc_auc = 0.0

    return bal_acc, macro_f1, roc_auc


__all__ = [
    "BasicBlock1D",
    "ResNet18_1D",
    "train_and_eval_resnet1d_scratch",
]

