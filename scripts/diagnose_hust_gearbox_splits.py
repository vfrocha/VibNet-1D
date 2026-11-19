#!/usr/bin/env python3
"""
Quick leak-free diagnostic comparing `by_speed` (6 folds) vs `by_load` (5 folds)
LOCO groupings on `HUST_Gearbox` using `RandomForest`, `SVM`, and `TabNet Scratch`
(`seed=42`), with exact fold-local RMS normalization + StandardScaler matching the
official TL protocol.

Optionally updates `protocols.TL.target_loco_grouping.HUST_Gearbox` in
`configs/da_dg_tl_matrix.yaml` to the hardest (lowest mean Macro-F1) grouping.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sys
import time
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.vibnet.data.registry import load_yaml_config
from src.vibnet.evaluation.da_dg_tl_runner import (
    MatrixExecutionContext,
    _aggregate_loco_fold_metrics,
    _fit_eval_tabnet_tl_classifier,
    atomic_write_json,
    build_tl_loco_folds,
)
from src.vibnet.evaluation.metrics import compute_fold_metrics

DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "da_dg_tl_matrix.yaml"
DEFAULT_OUT_DIR = PROJECT_ROOT / "results" / "da_dg_tl_matrix" / "hust_gearbox_split_diagnostic"


def evaluate_grouping(
    ctx: MatrixExecutionContext,
    grouping_mode: str,
    seed: int,
) -> Dict[str, Any]:
    t0 = time.time()
    tgt_recs, _ = ctx.get_native_tl_dataset("HUST_Gearbox")
    native_classes = sorted({r.raw_class for r in tgt_recs})
    cls_to_id = {c: idx for idx, c in enumerate(native_classes)}
    expected_ids = list(range(len(native_classes)))
    y_all = np.asarray([cls_to_id[r.raw_class] for r in tgt_recs], dtype=np.int64)

    folds = build_tl_loco_folds(tgt_recs, "HUST_Gearbox", grouping_mode=grouping_mode)
    per_model: Dict[str, List[Dict[str, Any]]] = {
        "RandomForest": [],
        "SVM": [],
        "TabNet_Scratch": [],
    }

    for fold in folds:
        tr_idx = list(fold.train_indices)
        te_idx = list(fold.test_indices)
        y_tr = y_all[tr_idx]
        y_te = y_all[te_idx]

        fold_feats = ctx.get_or_extract_tl_fold_features(
            target_dataset="HUST_Gearbox",
            fold_index=fold.fold_index,
            fold_train_indices=tr_idx,
            fold_test_indices=te_idx,
            grouping_mode=grouping_mode,
        )
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(fold_feats["X_train_raw"]).astype(np.float32)
        X_te = scaler.transform(fold_feats["X_test_raw"]).astype(np.float32)

        # 1. RandomForest
        rf = RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=int(seed),
            n_jobs=-1,
        )
        rf.fit(X_tr, y_tr)
        rf_fm = compute_fold_metrics(
            y_te.tolist(), rf.predict(X_te).tolist(), rf.predict_proba(X_te).tolist(), expected_ids
        )
        per_model["RandomForest"].append(
            {
                "test_condition": fold.test_condition,
                "accuracy": rf_fm.accuracy,
                "macro_f1": rf_fm.macro_f1,
                "weighted_f1": rf_fm.weighted_f1,
                "precision_macro": rf_fm.precision_macro,
                "recall_macro": rf_fm.recall_macro,
                "roc_auc_ovr": rf_fm.roc_auc_ovr,
                "roc_auc_valid": rf_fm.roc_auc_valid,
            }
        )

        # 2. SVM RBF
        svm = SVC(
            kernel="rbf",
            C=10.0,
            gamma="scale",
            probability=True,
            class_weight="balanced",
            random_state=int(seed),
        )
        svm.fit(X_tr, y_tr)
        svm_fm = compute_fold_metrics(
            y_te.tolist(), svm.predict(X_te).tolist(), svm.predict_proba(X_te).tolist(), expected_ids
        )
        per_model["SVM"].append(
            {
                "test_condition": fold.test_condition,
                "accuracy": svm_fm.accuracy,
                "macro_f1": svm_fm.macro_f1,
                "weighted_f1": svm_fm.weighted_f1,
                "precision_macro": svm_fm.precision_macro,
                "recall_macro": svm_fm.recall_macro,
                "roc_auc_ovr": svm_fm.roc_auc_ovr,
                "roc_auc_valid": svm_fm.roc_auc_valid,
            }
        )

        # 3. TabNet Scratch
        tb_res = _fit_eval_tabnet_tl_classifier(
            X_tr, y_tr, X_te, y_te, expected_ids, int(seed), ctx.hp, pretrainer=None
        )
        tb_res["test_condition"] = fold.test_condition
        per_model["TabNet_Scratch"].append(tb_res)

    summaries = {
        m_name: {
            "summary": _aggregate_loco_fold_metrics(rows),
            "per_fold": rows,
        }
        for m_name, rows in per_model.items()
    }
    overall_macro_f1 = float(
        np.mean([summaries[m]["summary"]["macro_f1_mean"] for m in ("RandomForest", "SVM", "TabNet_Scratch")])
    )
    return {
        "grouping_mode": grouping_mode,
        "n_folds": len(folds),
        "fold_names": [f.test_condition for f in folds],
        "overall_mean_macro_f1_3models": overall_macro_f1,
        "elapsed_seconds": round(time.time() - t0, 2),
        "models": summaries,
    }


def update_yaml_hust_gearbox_grouping(config_path: Path, selected_grouping: str) -> None:
    text = config_path.read_text(encoding="utf-8")
    pattern = re.compile(r'^(\s*HUST_Gearbox:\s*)"(by_speed|by_load|native_condition)"(.*)$', re.MULTILINE)
    new_text, count = pattern.subn(rf'\1"{selected_grouping}"\3', text, count=1)
    if count != 1:
        raise RuntimeError(
            f"Could not uniquely update HUST_Gearbox grouping line in {config_path} (matches={count})."
        )
    config_path.write_text(new_text, encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Diagnostica by_speed vs by_load na HUST_Gearbox (RF, SVM, TabNet Scratch, seed=42)."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--update-yaml",
        action="store_true",
        help="Atualiza configs/da_dg_tl_matrix.yaml com a divisao de pior Macro-F1 medio.",
    )
    args = parser.parse_args(argv)

    cfg = load_yaml_config(args.config)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ctx = MatrixExecutionContext(run_dir=args.out_dir, config=cfg, project_root=PROJECT_ROOT)

    results: Dict[str, Any] = {}
    for mode in ("by_speed", "by_load"):
        print(f"[DIAGNOSTIC] Running HUST_Gearbox grouping='{mode}' (seed={args.seed})...")
        results[mode] = evaluate_grouping(ctx=ctx, grouping_mode=mode, seed=int(args.seed))

    hardest_mode = min(
        ("by_speed", "by_load"),
        key=lambda m: results[m]["overall_mean_macro_f1_3models"],
    )

    if args.update_yaml:
        update_yaml_hust_gearbox_grouping(args.config, hardest_mode)
        print(f"[YAML UPDATED] HUST_Gearbox target_loco_grouping fixed to '{hardest_mode}' in {args.config}")

    payload = {
        "dataset": "HUST_Gearbox",
        "seed": int(args.seed),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "hardest_grouping_selected": hardest_mode,
        "yaml_updated": bool(args.update_yaml),
        "results": results,
    }
    atomic_write_json(args.out_dir / f"hust_gearbox_split_diagnostic_seed{args.seed}.json", payload)

    print("\n=== RESUMO HUST_Gearbox (by_speed vs by_load, seed=%d) ===" % args.seed)
    for mode in ("by_speed", "by_load"):
        r = results[mode]
        rf_f1 = r["models"]["RandomForest"]["summary"]["macro_f1_mean"]
        svm_f1 = r["models"]["SVM"]["summary"]["macro_f1_mean"]
        tb_f1 = r["models"]["TabNet_Scratch"]["summary"]["macro_f1_mean"]
        avg_f1 = r["overall_mean_macro_f1_3models"]
        print(
            f"  {mode:8s} ({r['n_folds']} folds) | RF={rf_f1:.4f} | SVM={svm_f1:.4f} | "
            f"TabNet_Scratch={tb_f1:.4f} | Media 3 modelos={avg_f1:.4f} ({r['elapsed_seconds']}s)"
        )
    print(f"  -> Divisao mais dificil selecionada: {hardest_mode}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
