"""
Standardized classification metrics and selection score for VibNet-1D
(`src.vibnet.evaluation.metrics`).

Implements the scientific protocol requirements:
- `Macro_F1` as the primary metric.
- `Macro_F1_Selection_Score = 0.5 * Macro_F1(MLP) + 0.5 * Macro_F1(TABNET)` for
  preprocessing and feature-representation selection.
- Explicit `ROC_AUC_Valid` boolean flag (`True` only when all training classes
  are present in `y_true` of the test fold and probability predictions are valid;
  never substitutes `0.0` when `ROC-AUC` is undefined, fixing the historical
  `ROC_AUC_ZERO_FALLBACK` anomaly documented in `EXPERIMENTAL_RESULTS_REPORT.md`).
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict, Mapping, Optional, Sequence


@dataclass(frozen=True)
class FoldMetricsResult:
    """Standardized fold-level evaluation metrics."""

    accuracy: float
    macro_f1: float
    weighted_f1: float
    precision_macro: float
    recall_macro: float
    roc_auc_ovr: Optional[float]
    roc_auc_valid: bool


def _binary_roc_auc(y_true_bin: Sequence[int], y_score: Sequence[float]) -> Optional[float]:
    """
    Compute exact binary ROC-AUC via the Mann-Whitney U / Wilcoxon rank-sum
    statistic with average tie ranks in pure Python stdlib.
    """
    n_pos = sum(1 for y in y_true_bin if y == 1)
    n_neg = sum(1 for y in y_true_bin if y == 0)
    if n_pos == 0 or n_neg == 0:
        return None

    paired = sorted(zip(y_score, y_true_bin), key=lambda item: item[0])
    n = len(paired)
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j < n and paired[j][0] == paired[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0
        for k in range(i, j):
            ranks[k] = avg_rank
        i = j

    sum_pos_ranks = sum(
        ranks[idx] for idx in range(n) if paired[idx][1] == 1
    )
    auc = (sum_pos_ranks - (n_pos * (n_pos + 1)) / 2.0) / float(n_pos * n_neg)
    return float(auc)


def compute_fold_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_proba: Optional[Sequence[Sequence[float]]] = None,
    expected_classes: Optional[Sequence[int]] = None,
) -> FoldMetricsResult:
    """
    Compute `Accuracy`, `Macro_F1`, `Weighted_F1`, `Precision_Macro`, `Recall_Macro`,
    and macro One-vs-Rest `ROC_AUC` (with `roc_auc_valid`).

    When a class from `expected_classes` is absent from `y_true` in a LOCO test
    condition (e.g., `UOEMD` conditions `5..8`), `roc_auc_ovr` is set to `None`
    (`NaN` in CSV exports) and `roc_auc_valid=False` — never `0.0`.
    """
    if len(y_true) != len(y_pred) or len(y_true) == 0:
        raise ValueError("y_true and y_pred must have the same non-zero length.")

    y_t = [int(y) for y in y_true]
    y_p = [int(y) for y in y_pred]
    n_total = len(y_t)

    classes = (
        sorted({int(c) for c in expected_classes})
        if expected_classes is not None
        else sorted(set(y_t) | set(y_p))
    )

    correct = sum(1 for yt, yp in zip(y_t, y_p) if yt == yp)
    accuracy = float(correct) / float(n_total)

    precisions = []
    recalls = []
    f1s = []
    supports = []

    for cls in classes:
        tp = sum(1 for yt, yp in zip(y_t, y_p) if yt == cls and yp == cls)
        fp = sum(1 for yt, yp in zip(y_t, y_p) if yt != cls and yp == cls)
        fn = sum(1 for yt, yp in zip(y_t, y_p) if yt == cls and yp != cls)
        sup = tp + fn

        prec = float(tp) / float(tp + fp) if (tp + fp) > 0 else 0.0
        rec = float(tp) / float(tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2.0 * prec * rec) / (prec + rec) if (prec + rec) > 0.0 else 0.0

        precisions.append(prec)
        recalls.append(rec)
        f1s.append(f1)
        supports.append(sup)

    n_cls = max(len(classes), 1)
    precision_macro = sum(precisions) / float(n_cls)
    recall_macro = sum(recalls) / float(n_cls)
    macro_f1 = sum(f1s) / float(n_cls)
    total_support = sum(supports)
    weighted_f1 = (
        sum(f * s for f, s in zip(f1s, supports)) / float(total_support)
        if total_support > 0
        else 0.0
    )

    roc_auc_ovr: Optional[float] = None
    roc_auc_valid = False

    present_true_classes = set(y_t)
    if (
        y_proba is not None
        and len(y_proba) == n_total
        and len(classes) >= 2
        and all(cls in present_true_classes for cls in classes)
    ):
        per_class_aucs = []
        valid_all = True
        for col_idx, cls in enumerate(classes):
            if any(len(row) <= col_idx for row in y_proba):
                valid_all = False
                break
            y_bin = [1 if yt == cls else 0 for yt in y_t]
            scores = [float(row[col_idx]) for row in y_proba]
            if any(math.isnan(s) or math.isinf(s) for s in scores):
                valid_all = False
                break
            auc_c = _binary_roc_auc(y_bin, scores)
            if auc_c is None:
                valid_all = False
                break
            per_class_aucs.append(auc_c)
        if valid_all and len(per_class_aucs) == len(classes):
            roc_auc_ovr = sum(per_class_aucs) / float(len(per_class_aucs))
            roc_auc_valid = True

    return FoldMetricsResult(
        accuracy=accuracy,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        precision_macro=precision_macro,
        recall_macro=recall_macro,
        roc_auc_ovr=roc_auc_ovr,
        roc_auc_valid=roc_auc_valid,
    )


def compute_selection_score(
    model_macro_f1_means: Mapping[str, float],
    models: Sequence[str] = ("MLP", "TABNET"),
) -> float:
    """
    Compute `Macro_F1_Selection_Score` as the arithmetic mean of `Macro_F1` across
    the specified deep models (default `MLP` and `TABNET`).
    """
    if not models:
        raise ValueError("models sequence cannot be empty.")
    values = []
    for m in models:
        if m not in model_macro_f1_means:
            raise KeyError(f"Missing model '{m}' in model_macro_f1_means.")
        values.append(float(model_macro_f1_means[m]))
    return sum(values) / float(len(values))
