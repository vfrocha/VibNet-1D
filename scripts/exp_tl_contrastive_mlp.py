"""LOCO MLP transfer experiments with domain-local supervised contrastive loss."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
import os
import sys
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from scripts.exp_tl_balance_strategies import (  # noqa: E402
    SOURCE_DATASETS,
    TARGET_DATASET,
    build_feature_cache,
)
from src.models.build_mlp_contrastive import train_and_evaluate  # noqa: E402


RESULTS_ROOT = os.path.join(PROJECT_ROOT, "results", "tl_contrastive_mlp")


class _Tee:
    def __init__(self, terminal, log):
        self.terminal = terminal
        self.log = log

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def build_experiment_summaries(df: pd.DataFrame):
    """Build per-seed and across-seed summaries from fold-level results."""
    if df.empty:
        return None, None
    successful = df[df.Status == "OK"]
    if successful.empty:
        return None, None

    summary = successful.groupby(
        ["Mode", "Encoder Profile", "Head Shape", "Balance Strategy", "Seed"], dropna=False
    ).agg(
        N_Folds=("Test Condition", "count"),
        N_ROC_AUC_Folds=("ROC-AUC OvR Macro", "count"),
        Macro_F1_Mean=("Macro F1", "mean"),
        Macro_F1_Std=("Macro F1", "std"),
        ACC_Mean=("ACC", "mean"),
        Bal_Acc_Mean=("Bal Acc", "mean"),
        ROC_AUC_Mean=("ROC-AUC OvR Macro", "mean"),
    ).reset_index()

    summary_across_seeds = summary.groupby(
        ["Mode", "Encoder Profile", "Head Shape", "Balance Strategy"], dropna=False
    ).agg(
        N_Seeds=("Seed", "nunique"),
        N_Folds_Total=("N_Folds", "sum"),
        N_ROC_AUC_Folds_Total=("N_ROC_AUC_Folds", "sum"),
        Macro_F1_Mean=("Macro_F1_Mean", "mean"),
        Macro_F1_Seed_Std=("Macro_F1_Mean", "std"),
        ACC_Mean=("ACC_Mean", "mean"),
        ACC_Seed_Std=("ACC_Mean", "std"),
        Bal_Acc_Mean=("Bal_Acc_Mean", "mean"),
        Bal_Acc_Seed_Std=("Bal_Acc_Mean", "std"),
        ROC_AUC_Mean=("ROC_AUC_Mean", "mean"),
        ROC_AUC_Seed_Std=("ROC_AUC_Mean", "std"),
    ).reset_index()

    return summary, summary_across_seeds


def run_experiment(args):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(RESULTS_ROOT, f"run_{timestamp}")
    os.makedirs(output_dir, exist_ok=False)
    results_path = os.path.join(output_dir, "results.csv")
    log_path = os.path.join(output_dir, "experiment.log")
    config_path = os.path.join(output_dir, "config.json")

    config = vars(args).copy()
    config.update({
        "source_datasets": SOURCE_DATASETS,
        "target_dataset": TARGET_DATASET,
        "primary_metric": "Macro F1",
        "output_dir": output_dir,
        "selection_note": "Fixed epochs; LOCO test folds are evaluation only, never checkpoint or hyperparameter selection.",
    })
    with open(config_path, "w", encoding="utf-8") as stream:
        json.dump(config, stream, indent=2, ensure_ascii=False)

    results = []
    with open(log_path, "w", encoding="utf-8") as log:
        def report(message):
            print(message)
            log.write(message + "\n")
            log.flush()

        report(f"Output: {output_dir}")
        report("Loading features with the existing detrended_rms pipeline...")
        try:
            with redirect_stdout(_Tee(sys.stdout, log)):
                feature_cache, labels_cache, conditions_cache, n_features = build_feature_cache(
                    variants=["detrended_rms"],
                    datasets=SOURCE_DATASETS + [TARGET_DATASET],
                )
        except Exception as exc:
            report(f"[FATAL] Could not load/extract datasets: {exc}")
            raise

        target_labels = labels_cache[TARGET_DATASET]
        target_conditions = np.unique(conditions_cache[TARGET_DATASET])
        configurations = product(args.modes, args.encoder_profiles, args.head_shapes)

        for mode, encoder_profile, head_shape in configurations:
            for seed in args.seeds:
                for condition in target_conditions:
                    test_mask = conditions_cache[TARGET_DATASET] == condition
                    train_mask = ~test_mask
                    X_test = feature_cache["detrended_rms"][TARGET_DATASET][test_mask]
                    y_test_raw = target_labels[test_mask]
                    X_target = feature_cache["detrended_rms"][TARGET_DATASET][train_mask]
                    y_target_raw = target_labels[train_mask]
                    train_classes = set(y_target_raw.tolist())
                    test_classes = set(y_test_raw.tolist())
                    missing = sorted(test_classes - train_classes)

                    row = {
                        "Variant": "detrended_rms", "Source Datasets": "|".join(SOURCE_DATASETS),
                        "Target Dataset": TARGET_DATASET, "Test Condition": str(condition),
                        "Mode": mode, "Encoder Profile": encoder_profile, "Head Shape": head_shape,
                        "Balance Strategy": args.balance_strategy, "Seed": seed,
                        "N Features": n_features, "N Train Target": len(X_target), "N Test": len(X_test),
                        "ACC": np.nan, "Bal Acc": np.nan, "Macro F1": np.nan,
                        "ROC-AUC OvR Macro": np.nan, "Status": "ERROR", "Error": "",
                    }
                    if missing:
                        row.update(Status="SKIPPED", Error=f"Test classes absent from target train: {missing}")
                        report(f"[SKIPPED] condition={condition}, missing classes={missing}")
                        results.append(row)
                        pd.DataFrame(results).to_csv(results_path, index=False)
                        continue

                    target_encoder = LabelEncoder().fit(y_target_raw)
                    y_target = target_encoder.transform(y_target_raw)
                    y_test = target_encoder.transform(y_test_raw)
                    train_data = {TARGET_DATASET: (X_target, y_target)}
                    for source in SOURCE_DATASETS:
                        # Independently encoded vocabularies: equal integer IDs
                        # from different domains never enter one SupCon batch.
                        source_encoder = LabelEncoder().fit(labels_cache[source])
                        train_data[source] = (
                            feature_cache["detrended_rms"][source],
                            source_encoder.transform(labels_cache[source]),
                        )

                    report(
                        f"[RUN] mode={mode} encoder={encoder_profile} head={head_shape} "
                        f"balance={args.balance_strategy} seed={seed} condition={condition}"
                    )
                    try:
                        with redirect_stdout(_Tee(sys.stdout, log)):
                            metrics = train_and_evaluate(
                                train_data, TARGET_DATASET, X_test, y_test,
                                mode=mode, encoder_profile=encoder_profile, head_shape=head_shape,
                                balance_strategy=args.balance_strategy, epochs=args.epochs,
                                finetune_epochs=args.finetune_epochs, batch_size=args.batch_size,
                                learning_rate=args.learning_rate, contrastive_weight=args.contrastive_weight,
                                temperature=args.temperature, seed=seed,
                            )
                        row.update({
                            "ACC": metrics["accuracy"], "Bal Acc": metrics["balanced_accuracy"],
                            "Macro F1": metrics["macro_f1"], "ROC-AUC OvR Macro": metrics["roc_auc_ovr_macro"],
                            "Balance Counts Before": json.dumps(metrics["counts_before"]),
                            "Balance Counts After": json.dumps(metrics["counts_after"]),
                            "Status": "OK",
                        })
                        report(
                            f"  Macro-F1={row['Macro F1']:.4f} ACC={row['ACC']:.4f} "
                            f"BalAcc={row['Bal Acc']:.4f} AUC={row['ROC-AUC OvR Macro']}"
                        )
                    except Exception as exc:
                        row.update(Status="ERROR", Error=f"{type(exc).__name__}: {exc}")
                        report(f"[ERROR] {row['Error']}")
                    results.append(row)
                    pd.DataFrame(results).to_csv(results_path, index=False)

    df = pd.DataFrame(results)
    summary, summary_across_seeds = build_experiment_summaries(df)
    if summary is not None:
        summary.to_csv(os.path.join(output_dir, "summary.csv"), index=False)
    if summary_across_seeds is not None:
        summary_across_seeds.to_csv(os.path.join(output_dir, "summary_across_seeds.csv"), index=False)
    print(f"Results: {results_path}")
    print(f"Config: {config_path}")
    return output_dir


def parse_args():
    parser = argparse.ArgumentParser(description="CWRU_12k + MFPT -> CWRU_48k SupCon MLP LOCO")
    parser.add_argument("--modes", nargs="+", choices=["baseline", "joint", "pretrain"], default=["baseline"])
    parser.add_argument("--encoder-profiles", nargs="+", choices=["baseline", "expanded"], default=["baseline"])
    parser.add_argument("--head-shapes", nargs="+", choices=["asymmetric", "symmetric", "linear"], default=["asymmetric"])
    parser.add_argument("--balance-strategy", choices=["none", "undersample", "smote", "hybrid"], default="none")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--contrastive-weight", type=float, default=0.1)
    parser.add_argument("--temperature", type=float, default=0.1)
    return parser.parse_args()


if __name__ == "__main__":
    run_experiment(parse_args())
