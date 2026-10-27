"""MLP classifiers and supervised contrastive training for VibNet-1D."""

from __future__ import annotations

import random
from typing import Dict, Tuple

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


class MLPClassifier(nn.Module):
    """An MLP encoder and post-latent classification head (no decoder)."""

    def __init__(
        self,
        input_dim: int,
        n_classes: Dict[str, int],
        encoder_profile: str = "baseline",
        head_shape: str = "asymmetric",
        latent_dim: int = 64,
        dropout: float = 0.2,
    ):
        super().__init__()
        profiles = {
            "baseline": (256, 128),
            "expanded": (512, 256, 128),
        }
        if encoder_profile not in profiles:
            raise ValueError(f"Unknown encoder profile: {encoder_profile}")
        widths = profiles[encoder_profile]
        blocks = []
        previous = input_dim
        for width in widths:
            blocks.extend((nn.Linear(previous, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(dropout)))
            previous = width
        blocks.append(nn.Linear(previous, latent_dim))
        self.encoder = nn.Sequential(*blocks)

        if head_shape == "asymmetric":
            head_widths = (128, 64)
        elif head_shape == "symmetric":
            head_widths = tuple(reversed(widths))
        elif head_shape == "linear":
            head_widths = ()
        else:
            raise ValueError(f"Unknown head shape: {head_shape}")
        # Every domain has its own output vocabulary and classification head.
        self.classifiers = nn.ModuleDict()
        for domain, count in n_classes.items():
            domain_head = []
            previous = latent_dim
            for width in head_widths:
                domain_head.extend((nn.Linear(previous, width), nn.BatchNorm1d(width), nn.ReLU(), nn.Dropout(dropout)))
                previous = width
            domain_head.append(nn.Linear(previous, count))
            self.classifiers[domain] = nn.Sequential(*domain_head)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor, domain: str, return_embedding: bool = False):
        embedding = self.encode(x)
        logits = self.classifiers[domain](embedding)
        if return_embedding:
            return logits, embedding
        return logits


class ProjectionHead(nn.Module):
    def __init__(self, latent_dim: int, projection_dim: int = 64):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(latent_dim, latent_dim), nn.ReLU(), nn.Linear(latent_dim, projection_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.normalize(self.layers(x), dim=1)


def supervised_contrastive_loss(
    embeddings: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.1,
 ) -> torch.Tensor:
    """SupCon loss; labels passed here must belong to exactly one domain."""
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    if embeddings.ndim != 2 or len(embeddings) != len(labels):
        raise ValueError("Expected 2D embeddings and one label per embedding")
    if len(embeddings) < 2:
        return embeddings.sum() * 0.0

    z = nn.functional.normalize(embeddings, dim=1)
    logits = torch.matmul(z, z.T) / temperature
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    labels = labels.reshape(-1, 1)
    positive_mask = torch.eq(labels, labels.T).to(logits.dtype)
    positive_mask.fill_diagonal_(0.0)
    valid = positive_mask.sum(dim=1) > 0
    if not torch.any(valid):
        return embeddings.sum() * 0.0
    logits_mask = torch.ones_like(positive_mask)
    logits_mask.fill_diagonal_(0.0)
    exp_logits = torch.exp(logits) * logits_mask
    log_prob = logits - torch.log(exp_logits.sum(dim=1, keepdim=True).clamp_min(1e-12))
    mean_positive = (positive_mask * log_prob).sum(dim=1) / positive_mask.sum(dim=1).clamp_min(1.0)
    return -mean_positive[valid].mean()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _prepare_domains(
    train_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    target_name: str,
    X_test: np.ndarray,
    balance_strategy: str,
    seed: int,
):
    if target_name not in train_data:
        raise ValueError(f"Target {target_name!r} missing from training domains")
    from src.models.build_tabnet_resnet_balance import _balance_target_training

    domains = {}
    target_scaler = None
    for name, (X, y) in train_data.items():
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)
        if X.ndim != 2 or len(X) != len(y) or not len(X):
            raise ValueError(f"Invalid training arrays for domain {name}")
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X).astype(np.float32)
        if name == target_name:
            target_scaler = scaler
            X_scaled, y, counts_before, counts_after = _balance_target_training(
                X_scaled, y, balance_strategy, seed
            )
        domains[name] = (X_scaled, y)
    assert target_scaler is not None
    X_test_scaled = target_scaler.transform(np.asarray(X_test, dtype=np.float32)).astype(np.float32)
    return domains, X_test_scaled, counts_before, counts_after


def _loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    return DataLoader(
        TensorDataset(torch.from_numpy(X), torch.from_numpy(y)),
        batch_size=max(2, min(batch_size, len(X))),
        shuffle=shuffle,
        drop_last=False,
    )


def _step_ce(model, projection, optimizer, batch, domain, mode, weight, temperature, device):
    X, y = batch[0].to(device), batch[1].to(device)
    optimizer.zero_grad(set_to_none=True)
    if len(y) < 2 and model.training:
        return None
    logits, latent = model(X, domain=domain, return_embedding=True)
    loss = nn.functional.cross_entropy(logits, y)
    if mode == "joint" and weight and projection is not None:
        loss = loss + weight * supervised_contrastive_loss(projection(latent), y, temperature)
    loss.backward()
    optimizer.step()
    return float(loss.detach().cpu())


def train_and_evaluate(
    train_data: Dict[str, Tuple[np.ndarray, np.ndarray]],
    target_name: str,
    X_test: np.ndarray,
    y_test: np.ndarray,
    *,
    mode: str = "baseline",
    encoder_profile: str = "baseline",
    head_shape: str = "asymmetric",
    balance_strategy: str = "none",
    epochs: int = 20,
    finetune_epochs: int = 10,
    batch_size: int = 256,
    learning_rate: float = 1e-3,
    contrastive_weight: float = 0.1,
    temperature: float = 0.1,
    seed: int = 42,
) -> dict:
    """Train without using X_test/y_test until final evaluation."""
    if mode not in {"baseline", "joint", "pretrain"}:
        raise ValueError(f"Unknown mode: {mode}")
    if epochs < 1 or finetune_epochs < 0:
        raise ValueError("epochs must be positive and finetune_epochs nonnegative")
    seed_everything(seed)
    domains, X_test_scaled, before, after = _prepare_domains(
        train_data, target_name, X_test, balance_strategy, seed
    )
    class_counts = {}
    for name, (_, y) in domains.items():
        classes = np.unique(y)
        if not np.array_equal(classes, np.arange(len(classes))):
            raise ValueError(f"Labels for {name} must be contiguous integers starting at zero")
        class_counts[name] = len(classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MLPClassifier(
        domains[target_name][0].shape[1], class_counts, encoder_profile, head_shape
    ).to(device)
    loaders = {name: _loader(X, y, batch_size, True) for name, (X, y) in domains.items()}

    if mode == "pretrain":
        seed_everything(seed + 1)
        pretrain_projection = ProjectionHead(64).to(device)
        pretrain_optimizer = torch.optim.Adam(
            list(model.encoder.parameters()) + list(pretrain_projection.parameters()),
            lr=learning_rate,
            weight_decay=1e-5,
        )
        source_batches = {name: iter(loader) for name, loader in loaders.items() if name != target_name}
        steps = max((len(loader) for name, loader in loaders.items() if name != target_name), default=0)
        for _ in range(epochs):
            source_batches = {name: iter(loaders[name]) for name in source_batches}
            for _step in range(steps):
                for name, iterator in source_batches.items():
                    try:
                        X, y = next(iterator)
                    except StopIteration:
                        continue
                    if len(y) < 2:
                        continue
                    X, y = X.to(device), y.to(device)
                    pretrain_optimizer.zero_grad(set_to_none=True)
                    latent = model.encode(X)
                    loss = supervised_contrastive_loss(pretrain_projection(latent), y, temperature)
                    loss.backward()
                    pretrain_optimizer.step()
        # Discard contrastive projection head and pretraining optimizer; preserve encoder only.
        del pretrain_projection, pretrain_optimizer

    # Supervised source+target training. SupCon is evaluated independently per domain.
    if mode == "joint":
        seed_everything(seed + 1)
        projection = ProjectionHead(64).to(device)
        supervised_optimizer = torch.optim.Adam(
            list(model.encoder.parameters())
            + list(model.classifiers.parameters())
            + list(projection.parameters()),
            lr=learning_rate,
            weight_decay=1e-5,
        )
    else:
        projection = None
        supervised_optimizer = torch.optim.Adam(
            list(model.encoder.parameters()) + list(model.classifiers.parameters()),
            lr=learning_rate,
            weight_decay=1e-5,
        )

    seed_everything(seed + 2)
    for _ in range(epochs):
        for name, loader in loaders.items():
            for batch in loader:
                _step_ce(
                    model, projection, supervised_optimizer, batch, name,
                    mode, contrastive_weight, temperature, device,
                )

    # Discard joint projection head and supervised optimizer before target-only fine-tuning.
    projection = None
    del supervised_optimizer

    # Fixed-epoch target-only fine-tuning; no test-fold checkpoint selection.
    finetune_optimizer = torch.optim.Adam(
        list(model.encoder.parameters()) + list(model.classifiers[target_name].parameters()),
        lr=learning_rate,
        weight_decay=1e-5,
    )
    seed_everything(seed + 3)
    target_loader = loaders[target_name]
    for _ in range(finetune_epochs):
        for batch in target_loader:
            _step_ce(model, None, finetune_optimizer, batch, target_name, "baseline", 0.0, temperature, device)

    model.eval()
    with torch.no_grad():
        logits = model(torch.from_numpy(X_test_scaled).to(device), domain=target_name).cpu().numpy()
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted)
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    predictions = probabilities.argmax(axis=1)
    y_test = np.asarray(y_test, dtype=np.int64)
    present = np.unique(y_test)
    auc = float("nan")
    if len(present) == 2:
        try:
            present_probabilities = probabilities[:, present]
            row_sums = present_probabilities.sum(axis=1, keepdims=True)
            present_probabilities = np.divide(
                present_probabilities,
                row_sums,
                out=np.full_like(present_probabilities, 0.5),
                where=row_sums > 0,
            )
            y_binary = (y_test == present[1]).astype(np.int64)
            auc = float(roc_auc_score(y_binary, present_probabilities[:, 1]))
        except ValueError:
            pass
    elif len(present) > 2:
        try:
            present_probabilities = probabilities[:, present]
            row_sums = present_probabilities.sum(axis=1, keepdims=True)
            present_probabilities = np.divide(
                present_probabilities,
                row_sums,
                out=np.full_like(present_probabilities, 1.0 / len(present)),
                where=row_sums > 0,
            )
            auc = float(roc_auc_score(y_test, present_probabilities, labels=present, multi_class="ovr", average="macro"))
        except ValueError:
            pass
    return {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "macro_f1": float(f1_score(y_test, predictions, labels=present, average="macro", zero_division=0)),
        "roc_auc_ovr_macro": auc,
        "y_pred": predictions,
        "probabilities": probabilities,
        "counts_before": before,
        "counts_after": after,
    }
