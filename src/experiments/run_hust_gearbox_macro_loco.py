"""Exploratory Macro-LOCO study for HUST_Gearbox (RF and SVM only).

Inputs are the already prepared `detrended_rms` `.npy` windows. Recording IDs
are recovered from `sNNNNN_wNN.npy`; malformed names or inconsistent metadata
are rejected to prevent window-level leakage.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import logging
from pathlib import Path
import platform
import re
import subprocess
import sys
from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC

from src.config import PROJECT_ROOT, RESULTS_DIR
from src.experiments.run_baseline_loco_v2 import default_141_feature_extractor


PREPROCESSING = "detrended_rms"
DATASET = "HUST_Gearbox"
EXPECTED_CLASSES = {"Class_Broken", "Class_Missing", "Class_Normal"}
SPEED_GROUPS = ["20Hz", "25Hz", "30Hz", "35Hz", "40Hz", "VS_0_40Hz"]
LOAD_GROUPS = [f"Load_{i}" for i in range(5)]
EXPECTED_CONDITIONS = {
    *(f"Cond_{speed}_{load}" for speed in (20, 25, 30, 35, 40) for load in range(5)),
    *(f"Cond_L{load}_VS_0_40_0" for load in range(5)),
}
FILENAME_RE = re.compile(r"^(s\d+)_w\d+\.npy$", re.IGNORECASE)
DEFAULT_DATA_ROOT = PROJECT_ROOT / "data" / "processed_tl" / PREPROCESSING
DEFAULT_OUTPUT_ROOT = RESULTS_DIR / "experiment_planning" / "hust_gearbox"


def derive_recording_id(filename: str) -> str:
    """Return the original recording ID encoded in the prepared window name."""
    match = FILENAME_RE.fullmatch(filename)
    if not match:
        raise ValueError(
            f"Nome de janela sem ID de gravação reconhecível: {filename!r}; "
            "esperado sNNNNN_wNN.npy."
        )
    return match.group(1).lower()


def condition_groups(condition: str) -> tuple[str, str]:
    """Map an audited condition name to its speed and load macro-groups."""
    constant = re.fullmatch(r"Cond_(20|25|30|35|40)_([0-4])", condition)
    if constant:
        speed, load = constant.groups()
        return f"{speed}Hz", f"Load_{load}"
    variable = re.fullmatch(r"Cond_L([0-4])_VS_0_40_0", condition)
    if variable:
        return "VS_0_40Hz", f"Load_{variable.group(1)}"
    raise ValueError(f"Condição HUST_Gearbox não prevista na auditoria: {condition!r}")


def build_fold_masks(
    conditions: np.ndarray, speed_groups: np.ndarray, load_groups: np.ndarray,
    protocol: str, held_out: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a macro-LOCO split and assert all requested groups are test-only."""
    groups = speed_groups if protocol == "speed" else load_groups
    test_mask = groups == held_out
    if not test_mask.any() or test_mask.all():
        raise ValueError(f"Fold inválido ou sem dados de treino: {protocol}={held_out}")
    train_mask = ~test_mask
    if np.intersect1d(conditions[train_mask], conditions[test_mask]).size:
        raise AssertionError("Condição operacional vazou entre treino e teste.")
    return train_mask, test_mask


def load_windows(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load windows and validate conditions, classes, and recording provenance."""
    if not data_dir.is_dir():
        raise FileNotFoundError(
            f"Dataset não encontrado: {data_dir}. Necessário: "
            "data/processed_tl/detrended_rms/HUST_Gearbox/<condição>/<classe>/sNNNNN_wNN.npy. "
            "Os arquivos devem vir da preparação existente, sem baixar dados automaticamente."
        )

    paths = sorted(data_dir.glob("*/*/*.npy"))
    if not paths:
        raise FileNotFoundError(f"Nenhuma janela .npy encontrada sob {data_dir}.")

    windows: list[np.ndarray] = []
    conditions: list[str] = []
    labels: list[str] = []
    recording_ids: list[str] = []
    source_recording_ids: list[str] = []
    sample_ids: list[str] = []
    speed: list[str] = []
    load: list[str] = []
    recording_metadata: dict[str, tuple[str, str]] = {}
    for path in paths:
        condition = path.parent.parent.name
        label = path.parent.name
        source_recording_id = derive_recording_id(path.name)
        # The current preprocessing output can reuse the same sNNNNN prefix in
        # different condition folders, so use its full provenance path as key.
        recording_id = f"{condition}/{label}/{source_recording_id}"
        speed_group, load_group = condition_groups(condition)
        previous = recording_metadata.setdefault(recording_id, (condition, label))
        if previous != (condition, label):
            raise ValueError(
                f"Gravação {recording_id} aparece em mais de uma condição/classe: "
                f"{previous} e {(condition, label)}."
            )
        arr = np.asarray(np.load(path, allow_pickle=False), dtype=np.float32).reshape(-1)
        if arr.size == 0 or not np.isfinite(arr).all():
            raise ValueError(f"Janela vazia ou não finita: {path}")
        windows.append(arr)
        conditions.append(condition)
        labels.append(label)
        recording_ids.append(recording_id)
        source_recording_ids.append(source_recording_id)
        sample_ids.append(path.relative_to(data_dir.parent).as_posix())
        speed.append(speed_group)
        load.append(load_group)

    observed_conditions = set(conditions)
    if observed_conditions != EXPECTED_CONDITIONS:
        raise ValueError(
            "Conjunto de condições não corresponde à auditoria (30 condições). "
            f"Ausentes={sorted(EXPECTED_CONDITIONS-observed_conditions)}; "
            f"extras={sorted(observed_conditions-EXPECTED_CONDITIONS)}"
        )
    observed_classes = set(labels)
    if observed_classes != EXPECTED_CLASSES:
        raise ValueError(
            "Vocabulário de classes não corresponde à auditoria específica "
            f"(3 classes: {sorted(EXPECTED_CLASSES)}); observado={sorted(observed_classes)}. "
            "Não será aplicada remarcação de rótulos implícita."
        )
    return (
        np.stack(windows), np.asarray(labels, dtype=str), np.asarray(conditions, dtype=str),
        np.asarray(recording_ids, dtype=str), np.asarray(source_recording_ids, dtype=str),
        np.asarray(sample_ids, dtype=str), np.asarray(speed, dtype=str),
    )


def unique_run_dir(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    candidate = root / stamp
    suffix = 1
    while candidate.exists():
        candidate = root / f"{stamp}_{suffix:02d}"
        suffix += 1
    candidate.mkdir()
    return candidate


def git_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", choices=["speed", "load"], default="speed")
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--fold", help="Um fold para execução reduzida (ex.: 20Hz ou Load_4).")
    parser.add_argument("--seed", type=int, action="append", help="Pode repetir; default: 42,100,2024,7,999.")
    args = parser.parse_args()

    fold_names = SPEED_GROUPS if args.protocol == "speed" else LOAD_GROUPS
    if args.fold and args.fold not in fold_names:
        parser.error(f"Fold incompatível com --protocol {args.protocol}: {args.fold}")
    seeds = args.seed if args.seed is not None else [42, 100, 2024, 7, 999]
    selected_folds = [args.fold] if args.fold else fold_names

    data_dir = args.data_root / DATASET
    X_windows, y_text, conditions, recording_ids, source_recording_ids, sample_ids, speed_groups = load_windows(data_dir)
    load_groups = np.asarray([condition_groups(c)[1] for c in conditions], dtype=str)
    if len(set(map(len, X_windows))) != 1:
        raise ValueError("Comprimentos de janela divergentes no conjunto HUST_Gearbox.")
    logger = logging.getLogger("hust_gearbox_macro_loco")
    logger.setLevel(logging.INFO)
    run_dir = unique_run_dir(args.output_root)
    handler = logging.FileHandler(run_dir / "run.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)
    logger.addHandler(logging.StreamHandler(sys.stdout))
    logger.info("Run directory: %s", run_dir)
    logger.info("Loading %d windows; extracting canonical 141 features.", len(X_windows))
    X = default_141_feature_extractor(X_windows, 25600)

    label_encoder = LabelEncoder().fit(y_text)
    y = label_encoder.transform(y_text)
    model_configs = {
        "RandomForest": {"n_estimators": 100, "max_features": "sqrt", "class_weight": "balanced", "n_jobs": 1},
        "SVM_RBF": {"C": 1.0, "kernel": "rbf", "probability": True, "class_weight": "balanced"},
    }
    try:
        import sklearn
        sklearn_version = sklearn.__version__
    except ImportError:  # pragma: no cover
        sklearn_version = "unavailable"
    script_path = Path(__file__).resolve()
    config: dict[str, Any] = {
        "experiment": "hust_gearbox_macro_loco_exploratory",
        "protocol": args.protocol,
        "folds": selected_folds,
        "seeds": seeds,
        "data_root": str(data_dir.resolve()),
        "preprocessing": PREPROCESSING,
        "sampling_rate_hz": 25600,
        "feature_extractor": "default_141_feature_extractor",
        "feature_count": int(X.shape[1]),
        "models": model_configs,
        "hyperparameter_search": False,
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn_version},
        "git_commit": git_version(),
        "script_sha256": hashlib.sha256(script_path.read_bytes()).hexdigest(),
        "created_at": datetime.now().astimezone().isoformat(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

    result_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    group_values = speed_groups if args.protocol == "speed" else load_groups
    for fold_name in selected_folds:
        train_mask, test_mask = build_fold_masks(
            conditions, speed_groups, load_groups, args.protocol, fold_name
        )
        train_recordings = sorted(set(recording_ids[train_mask]))
        test_recordings = sorted(set(recording_ids[test_mask]))
        train_source_recordings = sorted(set(source_recording_ids[train_mask]))
        test_source_recordings = sorted(set(source_recording_ids[test_mask]))
        if set(train_recordings) & set(test_recordings):
            raise AssertionError("Vazamento de gravação entre treino e teste.")
        train_ids = sample_ids[train_mask].tolist()
        test_ids = sample_ids[test_mask].tolist()
        if len(set(y[train_mask])) != len(label_encoder.classes_):
            raise ValueError(f"Treino de {fold_name} não contém todas as classes.")
        counts = {label: int(np.sum(y_text[test_mask] == label)) for label in label_encoder.classes_}
        train_counts = {label: int(np.sum(y_text[train_mask] == label)) for label in label_encoder.classes_}
        condition_groups_in_test = sorted(set(conditions[test_mask]))
        for seed in seeds:
            for model_name, params in model_configs.items():
                if model_name == "RandomForest":
                    model = RandomForestClassifier(random_state=seed, **params)
                else:
                    model = make_pipeline(
                        StandardScaler(),
                        SVC(random_state=seed, **params),
                    )
                model.fit(X[train_mask], y[train_mask])
                predicted = model.predict(X[test_mask])
                probabilities = model.predict_proba(X[test_mask])
                auc = float(roc_auc_score(
                    y[test_mask], probabilities, labels=np.arange(len(label_encoder.classes_)),
                    multi_class="ovr", average="macro",
                ))
                result_rows.append({
                    "protocol": args.protocol, "fold": fold_name, "seed": seed,
                    "model": model_name, "accuracy": float(accuracy_score(y[test_mask], predicted)),
                    "balanced_accuracy": float(balanced_accuracy_score(y[test_mask], predicted)),
                    "macro_f1": float(f1_score(y[test_mask], predicted, average="macro")),
                    "macro_roc_auc_ovr": auc, "n_train": int(train_mask.sum()),
                    "n_test": int(test_mask.sum()), "test_class_support": json.dumps(counts, sort_keys=True),
                    "train_class_support": json.dumps(train_counts, sort_keys=True),
                    "model_config": json.dumps({**params, "random_state": seed}, sort_keys=True),
                })
                logger.info("%s seed=%s %s: F1=%.4f AUC=%.4f", fold_name, seed, model_name,
                            result_rows[-1]["macro_f1"], auc)
        manifest_rows.append({
            "protocol": args.protocol, "fold": fold_name,
            "held_out_conditions": json.dumps(condition_groups_in_test),
            "train_recording_ids": json.dumps(train_recordings),
            "test_recording_ids": json.dumps(test_recordings),
            "train_source_recording_prefixes": json.dumps(train_source_recordings),
            "test_source_recording_prefixes": json.dumps(test_source_recordings),
            "train_sample_ids": json.dumps(train_ids), "test_sample_ids": json.dumps(test_ids),
            "train_condition_groups": json.dumps(sorted(set(group_values[train_mask]))),
            "test_condition_groups": json.dumps(sorted(set(group_values[test_mask]))),
            "train_n_windows": int(train_mask.sum()), "test_n_windows": int(test_mask.sum()),
            "train_class_support": json.dumps(train_counts, sort_keys=True),
            "test_class_support": json.dumps(counts, sort_keys=True),
            "recording_overlap": json.dumps(sorted(set(train_recordings) & set(test_recordings))),
        })

    results_df = pd.DataFrame(result_rows)
    manifest_df = pd.DataFrame(manifest_rows)
    results_df.to_csv(run_dir / "results.csv", index=False)
    manifest_df.to_csv(run_dir / "fold_manifest.csv", index=False)
    summary = results_df.groupby(["fold", "model"])[
        ["accuracy", "balanced_accuracy", "macro_f1", "macro_roc_auc_ovr"]
    ].agg(["mean", "std"]).fillna(0.0)
    summary_lines = [
        "| Fold | Modelo | Accuracy | Balanced Accuracy | Macro-F1 | Macro ROC-AUC OvR |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for (fold, model_name), row in summary.iterrows():
        summary_lines.append(
            f"| {fold} | {model_name} | {row[('accuracy', 'mean')]:.4f} ± {row[('accuracy', 'std')]:.4f} "
            f"| {row[('balanced_accuracy', 'mean')]:.4f} ± {row[('balanced_accuracy', 'std')]:.4f} "
            f"| {row[('macro_f1', 'mean')]:.4f} ± {row[('macro_f1', 'std')]:.4f} "
            f"| {row[('macro_roc_auc_ovr', 'mean')]:.4f} ± {row[('macro_roc_auc_ovr', 'std')]:.4f} |"
        )
    report_lines = [
        "# Estudo exploratório HUST_Gearbox — Macro-LOCO", "",
        f"- Protocolo: `{args.protocol}`; folds: {', '.join(selected_folds)}.",
        f"- Seeds: {', '.join(map(str, seeds))}; modelos: Random Forest e SVM RBF.",
        f"- Janelas: {len(X_windows)}; atributos: {X.shape[1]}; classes: {', '.join(label_encoder.classes_)}.",
        "- Métrica primária: Macro-F1. AUC reportada como OvR macro.",
        "- Busca de hiperparâmetros: não realizada; configurações fixas registradas em `config.json`.",
        "- IDs de gravação disjuntos entre treino e teste verificados para todos os folds.", "",
        "## Resultados por fold e modelo", "",
        *summary_lines, "",
        "## Limitações", "",
        "Estudo exploratório em uma máquina e representação pré-processada disponível; não constitui experimento de TL, DA ou DG.",
    ]
    (run_dir / "report.md").write_text("\n".join(report_lines), encoding="utf-8")
    logger.info("Completed; outputs: %s", run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
