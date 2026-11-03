"""
Unified facades for TabVibNet Transfer Learning and Contrastive models
(`src.vibnet.tabular.models`).

Addresses two critical findings from the codebase and results audit
(`reports/reproducibility_audit/EXPERIMENTAL_RESULTS_REPORT.md` & `ADR-003`):
1. Normalizes model return tuples so that both 4-tuple legacy baselines
   (`PureMLPBaseline.train_and_eval` returning `(acc, f1, prec, rec)`) and
   5-tuple transfer models (`(acc, f1, prec, rec, roc_auc)`) produce a uniform
   `TabularModelOutput` structure, preventing the `SHIFTED_COLUMN_CORRUPTION`
   bug observed in `results/exp3_1_transfer_learning.csv`.
2. Applies `trim_single_remainder_sample` (`8,577 -> 8,576` when `batch_size=64`)
   prior to `TabNetClassifier.fit()`, preventing the `BatchNorm1d` remainder-1
   crash on `PU` fold `C4_1500rpm_0.7Nm_400N`.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional, Sequence, Tuple

from src.vibnet.evaluation.balancing import trim_single_remainder_sample


@dataclass(frozen=True)
class TabularModelOutput:
    """Normalized return contract for all tabular models in VibNet-1D."""

    accuracy: float
    macro_f1: float
    precision_macro: float
    recall_macro: float
    roc_auc_ovr: Optional[float]
    roc_auc_valid: bool


def normalize_model_return_tuple(
    raw_result: Sequence[Any],
    all_classes_present_in_test: bool = True,
) -> TabularModelOutput:
    """
    Normalize a 4-tuple `(acc, f1, prec, rec)` or 5-tuple
    `(acc, f1, prec, rec, roc_auc)` returned by legacy `src/models/*` classes
    into a type-safe `TabularModelOutput`.

    Never shifts `recall` into `roc_auc` or `seed` into `recall` when a 4-tuple
    is returned (`SHIFTED_COLUMN_CORRUPTION` fix).
    """
    if len(raw_result) not in (4, 5):
        raise ValueError(
            f"Expected model return tuple of length 4 or 5, got length {len(raw_result)}."
        )

    acc = float(raw_result[0])
    f1 = float(raw_result[1])
    prec = float(raw_result[2])
    rec = float(raw_result[3])

    if len(raw_result) == 4:
        return TabularModelOutput(
            accuracy=acc,
            macro_f1=f1,
            precision_macro=prec,
            recall_macro=rec,
            roc_auc_ovr=None,
            roc_auc_valid=False,
        )

    raw_auc = raw_result[4]
    if raw_auc is None:
        return TabularModelOutput(
            accuracy=acc,
            macro_f1=f1,
            precision_macro=prec,
            recall_macro=rec,
            roc_auc_ovr=None,
            roc_auc_valid=False,
        )

    auc_val = float(raw_auc)
    if (
        math.isnan(auc_val)
        or math.isinf(auc_val)
        or not all_classes_present_in_test
        or auc_val <= 0.0
    ):
        return TabularModelOutput(
            accuracy=acc,
            macro_f1=f1,
            precision_macro=prec,
            recall_macro=rec,
            roc_auc_ovr=None,
            roc_auc_valid=False,
        )

    return TabularModelOutput(
        accuracy=acc,
        macro_f1=f1,
        precision_macro=prec,
        recall_macro=rec,
        roc_auc_ovr=auc_val,
        roc_auc_valid=True,
    )


def run_multihead_mlp_transfer(
    source_datasets_data: Dict[str, Tuple[Any, Any, int]],
    X_target_train: Any,
    y_target_train: Any,
    X_target_test: Any,
    y_target_test: Any,
    target_num_classes: int,
    class_weights: Optional[Dict[int, float]] = None,
    epochs_joint: int = 30,
    epochs_finetune: int = 20,
    batch_size: int = 64,
) -> Any:
    """
    Delegate Multi-Head MLP Transfer Learning (`141 -> 128 -> 64` shared encoder
    with domain-specific heads `H_s` and `H_T`) to `src.models.build_tabnet_resnet_balance`.
    """
    from src.models.build_tabnet_resnet_balance import MultiHeadMLPTransfer

    trainer = MultiHeadMLPTransfer(input_dim=X_target_train.shape[1])
    return trainer.train_and_eval(
        source_datasets_data=source_datasets_data,
        X_target_train=X_target_train,
        y_target_train=y_target_train,
        X_target_test=X_target_test,
        y_target_test=y_target_test,
        target_num_classes=target_num_classes,
        class_weights=class_weights,
        epochs_joint=epochs_joint,
        epochs_finetune=epochs_finetune,
        batch_size=batch_size,
    )


def run_contrastive_mlp_transfer(
    source_datasets_data: Dict[str, Tuple[Any, Any, int]],
    X_target_train: Any,
    y_target_train: Any,
    X_target_test: Any,
    y_target_test: Any,
    target_num_classes: int,
    contrastive_mode: str = "joint",
    encoder_type: str = "expanded",
    head_type: str = "asymmetric",
    temperature: float = 0.07,
    lambda_supcon: float = 0.5,
    proj_dim: int = 32,
    class_weights: Optional[Dict[int, float]] = None,
    seed: int = 42,
) -> Any:
    """
    Delegate Supervised Contrastive MLP Transfer Learning (`SupConLoss` with
    domain-prefixed labels `(domain_id, class_id)`) to `src.models.build_mlp_contrastive`.
    """
    from src.models.build_mlp_contrastive import ContrastiveMultiHeadMLPTransfer

    trainer = ContrastiveMultiHeadMLPTransfer(
        input_dim=X_target_train.shape[1],
        contrastive_mode=contrastive_mode,
        encoder_type=encoder_type,
        head_type=head_type,
        temperature=temperature,
        lambda_supcon=lambda_supcon,
        proj_dim=proj_dim,
        seed=seed,
    )
    return trainer.train_and_eval(
        source_datasets_data=source_datasets_data,
        X_target_train=X_target_train,
        y_target_train=y_target_train,
        X_target_test=X_target_test,
        y_target_test=y_target_test,
        target_num_classes=target_num_classes,
        class_weights=class_weights,
    )


def run_tabnet_transfer(
    X_source_combined: Any,
    X_target_train: Any,
    y_target_train: Any,
    X_target_test: Any,
    y_target_test: Any,
    class_weights: Optional[Dict[int, float]] = None,
    batch_size: int = 64,
    seed: int = 42,
) -> Any:
    """
    Delegate TabNet Transfer (`TabNetPretrainer` -> `TabNetClassifier`) to
    `src.models.build_tabnet_resnet_balance.TabNetTransfer`, applying
    `trim_single_remainder_sample` when `len(X_target_train) % batch_size == 1`
    so `BatchNorm1d` never receives a 1-sample final batch.
    """
    import numpy as np  # type: ignore
    from src.models.build_tabnet_resnet_balance import TabNetTransfer

    if len(X_target_train) > batch_size and (len(X_target_train) % batch_size == 1):
        X_trim, y_trim, _ = trim_single_remainder_sample(
            X_target_train.tolist(),
            y_target_train.tolist(),
            batch_size=batch_size,
            seed=seed,
        )
        X_target_train = np.asarray(X_trim, dtype=X_target_train.dtype)
        y_target_train = np.asarray(y_trim, dtype=y_target_train.dtype)

    trainer = TabNetTransfer()
    return trainer.train_and_eval(
        X_source=X_source_combined,
        X_target_train=X_target_train,
        y_target_train=y_target_train,
        X_target_test=X_target_test,
        y_target_test=y_target_test,
        class_weights=class_weights,
    )
