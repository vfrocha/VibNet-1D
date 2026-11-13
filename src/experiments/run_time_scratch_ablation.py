"""Experimento de Ablação From Scratch no Domínio do Tempo (VibNet1DTime - Fase 1).

Avalia a arquitetura `ResNet18_1D` sob protocolo Leave-One-Condition-Out (LOCO)
nos datasets `CWRU_12k`, `UOEMD`, `HUST_Gearbox` e `CWRU_48k`, variando:
- Fatores de resolução temporal (`--k_factors`, onde `target_len = k * 1024`);
- Estratégias de reamostragem (`--resample_modes`: `uniform`, `fft`, `poly`, `linear`);
- Estratégia de normalização por janela (`--norm_mode`: `raw`, `window_zscore`, `window_rms`).

100% isolado dos pipelines tabulares existentes.
"""

from __future__ import annotations

import argparse
from datetime import datetime
import os
import sys
import time
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

from src.config import DATA_ROOT, RESULTS_DIR
from src.data.time_resampler import downsample_window, normalize_window
from src.models.resnet1d_time import train_and_eval_resnet1d_scratch

TARGET_DATASETS: List[str] = ["CWRU_12k", "UOEMD", "HUST_Gearbox", "CWRU_48k"]
DEFAULT_TASKS: List[str] = ["diagnosis"]
DEFAULT_K_FACTORS: List[int] = [2, 4, 6, 8]
DEFAULT_RESAMPLE_MODES: List[str] = ["uniform", "fft", "poly", "linear"]
DEFAULT_NORM_MODE: str = "window_zscore"


class Logger:
    """Duplica a saída padrão (stdout) para o terminal e para um arquivo de log."""

    def __init__(self, filename: str) -> None:
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message: str) -> None:
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self) -> None:
        self.terminal.flush()
        self.log.flush()


def load_dataset_raw_windows(
    dataset_name: str,
    data_root: str | os.PathLike = DATA_ROOT,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Carrega todas as janelas `.npy` de `data/processed/<dataset>/<condition>/<class>/*.npy`.

    Retorna `(X_raw_list, y_raw_str_array, cond_raw_array)` mantendo `X_raw` como
    lista de arrays 1D em memória.
    """
    dataset_path = os.path.join(str(data_root), dataset_name)
    if not os.path.exists(dataset_path):
        print(f"[Aviso] Dataset {dataset_name} não encontrado em: {dataset_path}")
        return [], np.array([]), np.array([])

    X_raw: List[np.ndarray] = []
    y_raw_str: List[str] = []
    cond_raw: List[str] = []

    for root, _, files in os.walk(dataset_path):
        for file in sorted(files):
            if not file.endswith(".npy"):
                continue
            class_name = os.path.basename(root)
            cond_name = os.path.basename(os.path.dirname(root))
            file_path = os.path.join(root, file)

            arr = np.load(file_path)
            if arr.ndim == 1:
                X_raw.append(np.asarray(arr, dtype=np.float32))
                y_raw_str.append(class_name)
                cond_raw.append(cond_name)
            elif arr.ndim == 2:
                for row in arr:
                    X_raw.append(np.asarray(row, dtype=np.float32))
                    y_raw_str.append(class_name)
                    cond_raw.append(cond_name)

    return X_raw, np.asarray(y_raw_str), np.asarray(cond_raw)


def map_labels_for_task(
    dataset_name: str,
    X_raw: Sequence[np.ndarray],
    y_raw_str: np.ndarray,
    cond_raw: np.ndarray,
    task: str,
) -> Tuple[List[np.ndarray], np.ndarray, np.ndarray]:
    """Filtra e mapeia rótulos conforme a tarefa (`diagnosis` ou `detection`).

    - Ignora classes normais/saudáveis na `CWRU_48k` em diagnóstico;
    - Mapeia classes normais para `'Universal_Normal'` (diagnóstico) ou `0`/`1` (detecção).
    """
    task_clean = task.strip().lower()
    X_filtered: List[np.ndarray] = []
    y_mapped: List[str] = []
    cond_filtered: List[str] = []

    for sig, lbl, cond in zip(X_raw, y_raw_str.tolist(), cond_raw.tolist()):
        is_normal = "normal" in lbl.lower() or "healthy" in lbl.lower()

        if dataset_name == "CWRU_48k" and is_normal:
            continue

        X_filtered.append(sig)
        cond_filtered.append(cond)
        if task_clean == "detection":
            y_mapped.append("Normal" if is_normal else "Fault")
        else:
            y_mapped.append("Universal_Normal" if is_normal else f"{dataset_name}_{lbl}")

    return X_filtered, np.asarray(y_mapped), np.asarray(cond_filtered)


def preprocess_windows_batch(
    X_windows: Sequence[np.ndarray],
    target_len: int,
    resample_mode: str,
    norm_mode: str,
) -> np.ndarray:
    """Aplica `downsample_window` e `normalize_window` em cada janela 1D da lista."""
    n_samples = len(X_windows)
    out = np.empty((n_samples, target_len), dtype=np.float32)
    for idx, sig in enumerate(X_windows):
        sig_ds = downsample_window(sig, target_len=target_len, mode=resample_mode)
        out[idx] = normalize_window(sig_ds, norm_mode=norm_mode)
    return out


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Ablação From Scratch no Domínio do Tempo (VibNet1DTime - ResNet18_1D)."
    )
    parser.add_argument(
        "--k_factors",
        nargs="+",
        type=int,
        default=DEFAULT_K_FACTORS,
        help="Fatores inteiros k onde target_len = k * 1024 (default: 2 4 6 8).",
    )
    parser.add_argument(
        "--resample_modes",
        nargs="+",
        type=str,
        default=DEFAULT_RESAMPLE_MODES,
        choices=["uniform", "fft", "poly", "linear"],
        help="Modos de reamostragem (default: uniform fft poly linear).",
    )
    parser.add_argument(
        "--norm_mode",
        type=str,
        default=DEFAULT_NORM_MODE,
        choices=["raw", "window_zscore", "window_rms"],
        help="Modo de normalização por janela (default: window_zscore).",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        default=TARGET_DATASETS,
        help="Lista de datasets alvo (default: CWRU_12k UOEMD HUST_Gearbox CWRU_48k).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        type=str,
        default=DEFAULT_TASKS,
        choices=["diagnosis", "detection"],
        help="Tarefas avaliadas (default: diagnosis).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
        help="Número de épocas de treinamento da ResNet18_1D (default: 30).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Tamanho de lote (batch_size) no treinamento (default: 64).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Taxa de aprendizado inicial do Adam (default: 1e-3).",
    )
    return parser.parse_args(argv)


def run_time_scratch_ablation(argv: Sequence[str] | None = None) -> str:
    args = parse_args(argv)
    os.makedirs(RESULTS_DIR, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(str(RESULTS_DIR), f"time_scratch_ablation_{timestamp}.csv")
    log_file = os.path.join(str(RESULTS_DIR), f"log_time_scratch_ablation_{timestamp}.txt")
    sys.stdout = Logger(log_file)

    print(
        f"{'=' * 80}\n"
        f" VIBNET-1D TIME: ABLAÇÃO FROM SCRATCH (RESNET18_1D)\n"
        f"{'=' * 80}"
    )
    print(f"Datasets       : {args.datasets}")
    print(f"Tasks          : {args.tasks}")
    print(f"K_Factors      : {args.k_factors} (target_len = k * 1024)")
    print(f"Resample_Modes : {args.resample_modes}")
    print(f"Norm_Mode      : {args.norm_mode}")
    print(f"Epochs / BS    : {args.epochs} / {args.batch_size} (lr={args.lr})")
    print(f"Arquivo CSV    : {csv_file}\n")

    # 1. Cache em memória das listas X_raw por dataset
    print("--- FASE 1: CARREGAMENTO E CACHE DE JANELAS BRUTAS NA RAM ---")
    raw_cache: Dict[str, Tuple[List[np.ndarray], np.ndarray, np.ndarray]] = {}
    for ds_name in args.datasets:
        X_raw, y_raw_str, cond_raw = load_dataset_raw_windows(ds_name, DATA_ROOT)
        if len(X_raw) > 0:
            raw_cache[ds_name] = (X_raw, y_raw_str, cond_raw)
            print(f"  -> {ds_name}: {len(X_raw)} janelas carregadas na RAM.")

    master_results: List[Dict[str, object]] = []

    # 2. Loop principal de ablação (LOCO)
    print("\n--- FASE 2: AVALIAÇÃO LOCO NO DOMÍNIO DO TEMPO ---")
    for task in args.tasks:
        task_lower = task.strip().lower()
        print(f"\n{'*' * 60}\n TAREFA: {task_lower.upper()}\n{'*' * 60}")

        for ds_name in args.datasets:
            if task_lower == "detection" and ds_name == "CWRU_48k":
                print(f"[{ds_name}] Pulando em detection (ausência de classe Normal).")
                continue
            if ds_name not in raw_cache:
                continue

            X_raw_ds, y_raw_ds, cond_raw_ds = raw_cache[ds_name]
            X_task_list, y_task_str, cond_task = map_labels_for_task(
                ds_name, X_raw_ds, y_raw_ds, cond_raw_ds, task_lower
            )
            if len(X_task_list) == 0:
                print(f"[{ds_name}] Nenhuma janela válida após filtro de tarefa.")
                continue

            unique_conds = sorted(np.unique(cond_task).tolist())
            print(f"\n{'#' * 50}\n DATASET: {ds_name} ({len(unique_conds)} Condições LOCO)\n{'#' * 50}")

            for k_factor in args.k_factors:
                target_len = int(k_factor) * 1024
                for resample_mode in args.resample_modes:
                    print(
                        f"\n  [Config] Dataset={ds_name} | k={k_factor} (L={target_len}) | "
                        f"Resample={resample_mode} | Norm={args.norm_mode}"
                    )
                    X_proc_all = preprocess_windows_batch(
                        X_task_list,
                        target_len=target_len,
                        resample_mode=resample_mode,
                        norm_mode=args.norm_mode,
                    )

                    for test_cond in unique_conds:
                        test_mask = cond_task == test_cond
                        train_mask = ~test_mask

                        X_train_fold = X_proc_all[train_mask]
                        y_train_fold_str = y_task_str[train_mask]
                        X_test_fold_raw = X_proc_all[test_mask]
                        y_test_fold_str = y_task_str[test_mask]

                        if len(X_train_fold) == 0 or len(X_test_fold_raw) == 0:
                            print(f"    -> [{test_cond}] Fold vazio. Pulando.")
                            continue

                        # LabelEncoder local estritamente no fold de treino
                        le_local = LabelEncoder()
                        y_train_enc = le_local.fit_transform(y_train_fold_str)

                        if len(le_local.classes_) < 2:
                            print(
                                f"    -> [{test_cond}] Menos de 2 classes no treino "
                                f"({le_local.classes_}). Pulando."
                            )
                            continue

                        # Filtra amostras de teste cujas classes não existam no treino
                        valid_test_idx = [
                            i
                            for i, lbl in enumerate(y_test_fold_str)
                            if lbl in le_local.classes_
                        ]
                        if len(valid_test_idx) == 0:
                            print(
                                f"    -> [{test_cond}] Nenhuma amostra de teste pertence "
                                "às classes vistas no treino. Pulando."
                            )
                            continue

                        X_test_fold = X_test_fold_raw[valid_test_idx]
                        y_test_enc = le_local.transform(y_test_fold_str[valid_test_idx])

                        if task_lower == "detection" and len(np.unique(y_test_enc)) < 2:
                            print(
                                f"    -> [{test_cond}] Teste sem ambas as classes 0/1 "
                                "para detection. Pulando."
                            )
                            continue

                        t0 = time.perf_counter()
                        try:
                            bal_acc, macro_f1, roc_auc = train_and_eval_resnet1d_scratch(
                                X_train=X_train_fold,
                                y_train=y_train_enc,
                                X_test=X_test_fold,
                                y_test=y_test_enc,
                                task=task_lower,
                                epochs=args.epochs,
                                batch_size=args.batch_size,
                                lr=args.lr,
                            )
                            train_time_sec = time.perf_counter() - t0
                            print(
                                f"    -> Fold={test_cond} | BalAcc={bal_acc:.4f} | "
                                f"MacroF1={macro_f1:.4f} | AUC={roc_auc:.4f} "
                                f"({train_time_sec:.2f}s)"
                            )
                        except Exception as exc:
                            train_time_sec = time.perf_counter() - t0
                            print(f"    -> [ERRO] Falha no fold {test_cond}: {exc}")
                            bal_acc, macro_f1, roc_auc = 0.0, 0.0, 0.0

                        master_results.append(
                            {
                                "Dataset": ds_name,
                                "Task": task_lower.capitalize(),
                                "Test_Condition": test_cond,
                                "K_Factor": int(k_factor),
                                "Target_Len": int(target_len),
                                "Resample_Mode": resample_mode,
                                "Norm_Mode": args.norm_mode,
                                "Bal_Acc": float(bal_acc),
                                "Macro_F1": float(macro_f1),
                                "ROC_AUC": float(roc_auc),
                                "Train_Time_Sec": float(round(train_time_sec, 4)),
                            }
                        )

                        # Salvamento incremental a cada fold
                        pd.DataFrame(master_results).to_csv(csv_file, index=False)

    print(f"\n[SUCESSO] Experimento concluído. Resultados salvos em: {csv_file}")
    return csv_file


if __name__ == "__main__":
    run_time_scratch_ablation()

