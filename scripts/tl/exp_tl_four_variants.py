#!/usr/bin/env python3
"""
exp_tl_four_variants.py

Experimento controlado de Transfer Learning para diagnóstico:

    Sources:
        CWRU_12k + MFPT

    Target:
        CWRU_48k

Variantes de pré-processamento já geradas em data/processed_tl/:
    1. raw
    2. detrended
    3. raw_rms
    4. detrended_rms

Modelos:
    - MLP
    - TabNet

Protocolo:
    - As quatro variantes são utilizadas diretamente; este script NÃO
      aplica detrend ou normalização adicional.
    - Todos os dados das sources participam do pré-treinamento.
    - No target, é aplicado Leave-One-Condition-Out (LOCO):
        * uma condição é teste;
        * as demais condições do target são usadas no fine-tuning.
    - A avaliação é feita somente se todas as classes presentes no teste
      também estiverem presentes no treino do target.
    - Macro F1 é a métrica primária.
    - Resultados são salvos incrementalmente.

IMPORTANTE:
    train_and_evaluate_multihead() do projeto atual executa uma padronização
    StandardScaler por dataset na etapa de modelagem. Portanto, a análise
    deste experimento mede o efeito das variantes na extração de features
    + treinamento, e não uma passagem dos valores absolutos das features
    diretamente à rede sem padronização posterior.
"""

import argparse
import json
import os
import random
import sys
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------------------------------
# Projeto
# -----------------------------------------------------------------------------
sys.path.append(
    os.path.abspath(
        os.path.join(
            os.path.dirname(__file__),
            "../..",
        )
    )
)

from src.features.extractors_v2 import extract_advanced_features
from src.features.signalai_wrapper import extract_fusion_features
from src.models.build_tabnet_resnet import (
    train_and_evaluate_multihead,
)


# =============================================================================
# CONFIGURAÇÃO
# =============================================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../..",
    )
)

DATA_ROOT = os.path.join(
    PROJECT_ROOT,
    "data",
    "processed_tl",
)

RESULTS_ROOT = os.path.join(
    PROJECT_ROOT,
    "results",
    "tl_four_variants",
)

SOURCE_DATASETS = [
    "CWRU_12k",
    "MFPT",
]

TARGET_DATASET = "CWRU_48k"

DATASETS_CONFIG = {
    "CWRU_12k": 12000,
    "MFPT": 48828,
    "CWRU_48k": 48000,
}

VARIANTS = {
    "raw": os.path.join(DATA_ROOT, "raw"),
    "detrended": os.path.join(DATA_ROOT, "detrended"),
    "raw_rms": os.path.join(DATA_ROOT, "raw_rms"),
    "detrended_rms": os.path.join(
        DATA_ROOT,
        "detrended_rms",
    ),
}

MODELS = [
    "mlp",
    "tabnet",
]

EPOCHS = 15
BATCH_SIZE = 512

METRIC_PRIMARY = "Macro F1"


# =============================================================================
# LOG
# =============================================================================

class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(
            filename,
            "w",
            encoding="utf-8",
        )

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


# =============================================================================
# REPRODUTIBILIDADE
# =============================================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    except Exception:
        pass


# =============================================================================
# DADOS
# =============================================================================

def find_npy_files(root):
    """Lista todos os .npy recursivamente."""
    files = []

    if not os.path.isdir(root):
        return files

    for current_root, _, filenames in os.walk(root):
        for filename in sorted(filenames):
            if filename.endswith(".npy"):
                files.append(
                    os.path.join(
                        current_root,
                        filename,
                    )
                )

    return sorted(files)


def load_dataset_variant(
    dataset_name,
    variant_name,
):
    """
    Carrega diretamente uma das quatro variantes já processadas.

    Retorna:
        X_raw       : lista de janelas
        labels_raw  : labels locais (nome da pasta)
        conditions  : condições LOCO (diretório acima da classe)
    """
    if dataset_name not in DATASETS_CONFIG:
        raise KeyError(
            f"Dataset não configurado: {dataset_name}"
        )

    if variant_name not in VARIANTS:
        raise KeyError(
            f"Variante não configurada: {variant_name}"
        )

    dataset_root = os.path.join(
        VARIANTS[variant_name],
        dataset_name,
    )

    if not os.path.isdir(dataset_root):
        raise FileNotFoundError(
            f"Dataset/variante não encontrado:\n"
            f"{dataset_root}"
        )

    files = find_npy_files(
        dataset_root
    )

    if not files:
        raise FileNotFoundError(
            f"Nenhum .npy encontrado em:\n"
            f"{dataset_root}"
        )

    X = []
    labels = []
    conditions = []
    relative_paths = []

    for file_path in files:
        # Estrutura esperada:
        # dataset / condition / class / sample.npy
        relative = os.path.relpath(
            file_path,
            dataset_root,
        )

        parts = relative.split(
            os.sep
        )

        if len(parts) < 3:
            raise ValueError(
                "Estrutura de diretórios inesperada: "
                f"{file_path}"
            )

        condition = parts[-3]
        class_name = parts[-2]

        signal = np.load(
            file_path,
            allow_pickle=False,
        )

        signal = np.asarray(
            signal,
            dtype=np.float64,
        ).reshape(-1)

        X.append(signal)
        labels.append(class_name)
        conditions.append(condition)
        relative_paths.append(
            relative
        )

    return (
        X,
        np.asarray(labels),
        np.asarray(conditions),
        relative_paths,
    )


# =============================================================================
# FEATURES
# =============================================================================

def extract_features(
    X_windows,
    sampling_rate,
):
    """
    Extrai o conjunto completo atual:

        extract_advanced_features
        +
        extract_fusion_features

    Não aplica nenhuma transformação temporal adicional.
    """
    if len(X_windows) == 0:
        raise ValueError(
            "Nenhuma janela para extração de features."
        )

    X_array = np.asarray(
        X_windows,
        dtype=np.float64,
    )

    if X_array.ndim != 2:
        raise ValueError(
            "As janelas devem formar uma matriz 2D. "
            f"Shape recebido: {X_array.shape}"
        )

    X_fusion = extract_fusion_features(
        X_array,
        sampling_rate,
        extract_advanced_features,
    )

    X_clean = np.nan_to_num(
        np.asarray(
            X_fusion,
            dtype=np.float32,
        ),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    if X_clean.ndim == 1:
        X_clean = X_clean.reshape(
            len(X_windows),
            -1,
        )

    if X_clean.shape[0] != len(X_windows):
        raise ValueError(
            "Número de amostras de features diferente do "
            f"número de janelas: "
            f"{X_clean.shape[0]} != {len(X_windows)}"
        )

    return X_clean


def build_feature_cache(
    variants,
    datasets,
):
    """
    Carrega e extrai features para todas as combinações.

    Estrutura:
        feature_cache[variant][dataset] -> X
        labels_cache[dataset]          -> y
        conditions_cache[dataset]      -> conditions
    """
    feature_cache = {
        variant: {}
        for variant in variants
    }

    labels_cache = {}
    conditions_cache = {}
    paths_cache = {}

    dimensions = {}

    print()
    print("=" * 90)
    print("FASE 1: CARGA E EXTRAÇÃO DE FEATURES")
    print("=" * 90)

    for dataset_name in datasets:
        labels_ref = None
        conditions_ref = None

        for variant_name in variants:
            print()
            print(
                f"[{dataset_name}] "
                f"Variante={variant_name}"
            )

            (
                X_windows,
                labels,
                conditions,
                relative_paths,
            ) = load_dataset_variant(
                dataset_name,
                variant_name,
            )

            fs = DATASETS_CONFIG[
                dataset_name
            ]

            X_features = extract_features(
                X_windows,
                fs,
            )

            feature_cache[
                variant_name
            ][dataset_name] = X_features

            dimensions[
                (variant_name, dataset_name)
            ] = X_features.shape[1]

            print(
                f"  Janelas : {len(X_windows)}"
            )
            print(
                f"  Features: {X_features.shape[1]}"
            )
            print(
                f"  fs      : {fs} Hz"
            )

            # Os dados e as classes/condições não devem mudar entre
            # variantes. Esta verificação garante alinhamento.
            if labels_ref is None:
                labels_ref = labels
                conditions_ref = conditions
                paths_cache[
                    dataset_name
                ] = relative_paths

            else:
                if not np.array_equal(
                    labels_ref,
                    labels,
                ):
                    raise ValueError(
                        f"Labels diferentes entre variantes "
                        f"para {dataset_name}."
                    )

                if not np.array_equal(
                    conditions_ref,
                    conditions,
                ):
                    raise ValueError(
                        f"Condições diferentes entre variantes "
                        f"para {dataset_name}."
                    )

                if paths_cache[
                    dataset_name
                ] != relative_paths:
                    raise ValueError(
                        f"Estrutura de arquivos diferente entre "
                        f"variantes para {dataset_name}."
                    )

        labels_cache[
            dataset_name
        ] = labels_ref

        conditions_cache[
            dataset_name
        ] = conditions_ref

    all_dims = set(
        dimensions.values()
    )

    if len(all_dims) != 1:
        raise ValueError(
            "Dimensões de features inconsistentes entre "
            f"datasets/variantes: {dimensions}"
        )

    n_features = next(
        iter(all_dims)
    )

    print()
    print(
        f"Dimensão final das features: "
        f"{n_features}"
    )

    return (
        feature_cache,
        labels_cache,
        conditions_cache,
        n_features,
    )


# =============================================================================
# RÓTULOS
# =============================================================================

def encode_dataset_labels(
    labels,
):
    """
    Codifica classes separadamente por dataset.

    Isso é compatível com o modelo multi-head atual:
    cada domínio possui sua própria cabeça e seu próprio LabelEncoder.
    """
    encoder = LabelEncoder()
    encoded = encoder.fit_transform(
        labels
    )

    return encoded, encoder


# =============================================================================
# VALIDAÇÃO DO LOCO
# =============================================================================

def valid_loco_split(
    y_train_str,
    y_test_str,
):
    """
    Verifica se todas as classes do teste estão representadas
    no treino do target.

    Não fazemos a prática de remover amostras de classes ausentes,
    pois isso mudaria o problema de diagnóstico.
    """
    train_classes = set(
        y_train_str
    )

    test_classes = set(
        y_test_str
    )

    missing = sorted(
        test_classes -
        train_classes
    )

    return (
        len(missing) == 0,
        missing,
    )


# =============================================================================
# RESUMO
# =============================================================================

def build_summaries(
    results_df,
    output_dir,
):
    """
    Gera:
        summary_by_variant_model.csv
        summary.csv
    """
    valid = results_df[
        results_df["Status"] == "OK"
    ].copy()

    if valid.empty:
        return None, None

    by_variant_model = (
        valid
        .groupby(
            [
                "Variant",
                "Model",
            ],
            dropna=False,
        )
        .agg(
            N_Folds=(
                "Test Condition",
                "count",
            ),
            N_Test_Samples=(
                "N Test",
                "sum",
            ),
            ACC_Mean=(
                "ACC",
                "mean",
            ),
            ACC_Std=(
                "ACC",
                "std",
            ),
            Bal_Acc_Mean=(
                "Bal Acc",
                "mean",
            ),
            Bal_Acc_Std=(
                "Bal Acc",
                "std",
            ),
            Macro_F1_Mean=(
                "Macro F1",
                "mean",
            ),
            Macro_F1_Std=(
                "Macro F1",
                "std",
            ),
            Macro_F1_Median=(
                "Macro F1",
                "median",
            ),
            Macro_F1_Min=(
                "Macro F1",
                "min",
            ),
            Macro_F1_Max=(
                "Macro F1",
                "max",
            ),
            ROC_AUC_Mean=(
                "ROC-AUC",
                "mean",
            ),
            ROC_AUC_Std=(
                "ROC-AUC",
                "std",
            ),
        )
        .reset_index()
    )

    by_variant = (
        by_variant_model
        .groupby(
            "Variant",
            dropna=False,
        )
        .agg(
            N_Models=(
                "Model",
                "nunique",
            ),
            N_Folds_Total=(
                "N_Folds",
                "sum",
            ),
            Macro_F1_Selection_Score=(
                "Macro_F1_Mean",
                "mean",
            ),
            Macro_F1_Model_Std=(
                "Macro_F1_Mean",
                "std",
            ),
            Macro_F1_Worst_Model=(
                "Macro_F1_Mean",
                "min",
            ),
            Macro_F1_Best_Model=(
                "Macro_F1_Mean",
                "max",
            ),
            ACC_Selection_Mean=(
                "ACC_Mean",
                "mean",
            ),
            Bal_Acc_Selection_Mean=(
                "Bal_Acc_Mean",
                "mean",
            ),
            ROC_AUC_Selection_Mean=(
                "ROC_AUC_Mean",
                "mean",
            ),
        )
        .reset_index()
        .sort_values(
            "Macro_F1_Selection_Score",
            ascending=False,
        )
    )

    by_variant_model_file = os.path.join(
        output_dir,
        "summary_by_variant_model.csv",
    )

    summary_file = os.path.join(
        output_dir,
        "summary.csv",
    )

    by_variant_model.to_csv(
        by_variant_model_file,
        index=False,
    )

    by_variant.to_csv(
        summary_file,
        index=False,
    )

    return (
        by_variant_model,
        by_variant,
    )


# =============================================================================
# EXPERIMENTO
# =============================================================================

def run_experiment(
    variants,
    models,
    seed,
    epochs,
    batch_size,
):
    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    output_dir = os.path.join(
        RESULTS_ROOT,
        f"run_{timestamp}",
    )

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    log_file = os.path.join(
        output_dir,
        "experiment.log",
    )

    results_file = os.path.join(
        output_dir,
        "results.csv",
    )

    config_file = os.path.join(
        output_dir,
        "config.json",
    )

    original_stdout = sys.stdout
    logger = Logger(
        log_file
    )
    sys.stdout = logger

    try:
        set_seed(
            seed
        )

        config = {
            "source_datasets": SOURCE_DATASETS,
            "target_dataset": TARGET_DATASET,
            "variants": variants,
            "models": models,
            "seed": seed,
            "epochs": epochs,
            "batch_size": batch_size,
            "primary_metric": METRIC_PRIMARY,
            "data_root": DATA_ROOT,
            "dataset_sampling_rates": DATASETS_CONFIG,
        }

        with open(
            config_file,
            "w",
            encoding="utf-8",
        ) as f:
            json.dump(
                config,
                f,
                indent=2,
                ensure_ascii=False,
            )

        print("=" * 90)
        print(
            "EXPERIMENTO: TL + DIAGNOSIS + "
            "QUATRO VARIANTES"
        )
        print("=" * 90)
        print()
        print(
            "Sources: "
            + ", ".join(SOURCE_DATASETS)
        )
        print(
            f"Target: {TARGET_DATASET}"
        )
        print(
            "Modelos: "
            + ", ".join(models)
        )
        print(
            "Variantes: "
            + ", ".join(variants)
        )
        print(
            f"Epochs={epochs} | "
            f"Batch size={batch_size} | "
            f"Seed={seed}"
        )
        print()
        print(
            "Métrica primária: Macro F1"
        )
        print(
            "Avaliação target: LOCO por condição"
        )
        print(
            "Pré-processamento temporal adicional neste "
            "script: nenhum"
        )

        # ---------------------------------------------------------
        # Carga das quatro variantes.
        # ---------------------------------------------------------
        (
            feature_cache,
            labels_cache,
            conditions_cache,
            n_features,
        ) = build_feature_cache(
            variants=variants,
            datasets=SOURCE_DATASETS + [
                TARGET_DATASET
            ],
        )

        print()
        print("=" * 90)
        print(
            "FASE 2: TRANSFER LEARNING "
            "SOURCE → TARGET"
        )
        print("=" * 90)

        target_labels_raw = labels_cache[
            TARGET_DATASET
        ]

        target_conditions = np.unique(
            conditions_cache[
                TARGET_DATASET
            ]
        )

        print(
            f"Condições LOCO no target: "
            f"{len(target_conditions)}"
        )
        print(
            "Condições: "
            + ", ".join(
                map(
                    str,
                    target_conditions,
                )
            )
        )

        results = []

        for variant in variants:

            print()
            print("=" * 90)
            print(
                f"VARIANTE: {variant}"
            )
            print("=" * 90)

            # Features da variante atual.
            target_X = feature_cache[
                variant
            ][
                TARGET_DATASET
            ]

            for test_condition in target_conditions:

                print()
                print(
                    "-" * 90
                )
                print(
                    f"LOCO | Target={TARGET_DATASET} | "
                    f"Test condition={test_condition}"
                )
                print(
                    "-" * 90
                )

                target_test_mask = (
                    conditions_cache[
                        TARGET_DATASET
                    ] == test_condition
                )

                target_train_mask = (
                    ~target_test_mask
                )

                X_target_train = (
                    target_X[
                        target_train_mask
                    ]
                )

                y_target_train_raw = (
                    target_labels_raw[
                        target_train_mask
                    ]
                )

                X_test = (
                    target_X[
                        target_test_mask
                    ]
                )

                y_test_raw = (
                    target_labels_raw[
                        target_test_mask
                    ]
                )

                valid_split, missing = (
                    valid_loco_split(
                        y_target_train_raw,
                        y_test_raw,
                    )
                )

                if not valid_split:
                    reason = (
                        "Classes presentes no teste e ausentes "
                        f"no treino target: {missing}"
                    )

                    print(
                        f"[PULADO] {reason}"
                    )

                    for model in models:
                        results.append(
                            {
                                "Variant": variant,
                                "Model": model.upper(),
                                "Source Datasets": "|".join(
                                    SOURCE_DATASETS
                                ),
                                "Target Dataset": TARGET_DATASET,
                                "Test Condition": str(
                                    test_condition
                                ),
                                "N Features": n_features,
                                "N Train Target": len(
                                    X_target_train
                                ),
                                "N Test": len(
                                    X_test
                                ),
                                "ACC": np.nan,
                                "Bal Acc": np.nan,
                                "Macro F1": np.nan,
                                "ROC-AUC": np.nan,
                                "Status": "SKIPPED",
                                "Error": reason,
                                "Seed": seed,
                            }
                        )

                    continue

                # -----------------------------------------------------
                # LabelEncoder do target.
                # -----------------------------------------------------
                (
                    y_target_train,
                    target_encoder,
                ) = encode_dataset_labels(
                    y_target_train_raw
                )

                y_test = target_encoder.transform(
                    y_test_raw
                )

                # -----------------------------------------------------
                # Monta source + target para o modelo multi-head.
                # -----------------------------------------------------
                train_data_dict = {}

                for source_dataset in SOURCE_DATASETS:
                    X_source = feature_cache[
                        variant
                    ][
                        source_dataset
                    ]

                    y_source_raw = labels_cache[
                        source_dataset
                    ]

                    (
                        y_source,
                        source_encoder,
                    ) = encode_dataset_labels(
                        y_source_raw
                    )

                    train_data_dict[
                        source_dataset
                    ] = (
                        X_source,
                        y_source,
                    )

                train_data_dict[
                    TARGET_DATASET
                ] = (
                    X_target_train,
                    y_target_train,
                )

                for model in models:

                    print(
                        f"\n[TRAIN] "
                        f"Variant={variant} | "
                        f"Model={model.upper()}"
                    )

                    set_seed(
                        seed
                    )

                    try:
                        (
                            bal_acc,
                            macro_f1,
                            roc_auc,
                            aux_output,
                        ) = train_and_evaluate_multihead(
                            train_data_dict=train_data_dict,
                            target_dataset_name=TARGET_DATASET,
                            X_test=X_test,
                            y_test=y_test,
                            task="diagnosis",
                            epochs=epochs,
                            batch_size=batch_size,
                            encoder_type=model,
                        )

                        y_pred = None

                        if isinstance(
                            aux_output,
                            dict,
                        ):
                            y_pred = aux_output.get(
                                "y_pred"
                            )

                        if y_pred is None:
                            raise RuntimeError(
                                "train_and_evaluate_multihead não "
                                "retornou y_pred no aux_output."
                            )

                        y_pred = np.asarray(
                            y_pred
                        ).reshape(-1)

                        if len(y_pred) != len(y_test):
                            raise RuntimeError(
                                "Quantidade de predições diferente "
                                f"do teste: "
                                f"{len(y_pred)} != {len(y_test)}"
                            )

                        acc = float(
                            np.mean(
                                y_pred ==
                                y_test
                            )
                        )

                        print(
                            f"  ACC      = {acc:.4f}"
                        )
                        print(
                            f"  Bal Acc  = {float(bal_acc):.4f}"
                        )
                        print(
                            f"  Macro F1 = {float(macro_f1):.4f}"
                        )
                        print(
                            f"  ROC-AUC  = {float(roc_auc):.4f}"
                        )

                        results.append(
                            {
                                "Variant": variant,
                                "Model": model.upper(),
                                "Source Datasets": "|".join(
                                    SOURCE_DATASETS
                                ),
                                "Target Dataset": TARGET_DATASET,
                                "Test Condition": str(
                                    test_condition
                                ),
                                "N Features": n_features,
                                "N Train Target": len(
                                    X_target_train
                                ),
                                "N Test": len(
                                    X_test
                                ),
                                "ACC": acc,
                                "Bal Acc": float(
                                    bal_acc
                                ),
                                "Macro F1": float(
                                    macro_f1
                                ),
                                "ROC-AUC": float(
                                    roc_auc
                                ),
                                "Status": "OK",
                                "Error": "",
                                "Seed": seed,
                            }
                        )

                    except Exception as exc:
                        print(
                            f"  [ERRO] {exc}"
                        )

                        results.append(
                            {
                                "Variant": variant,
                                "Model": model.upper(),
                                "Source Datasets": "|".join(
                                    SOURCE_DATASETS
                                ),
                                "Target Dataset": TARGET_DATASET,
                                "Test Condition": str(
                                    test_condition
                                ),
                                "N Features": n_features,
                                "N Train Target": len(
                                    X_target_train
                                ),
                                "N Test": len(
                                    X_test
                                ),
                                "ACC": np.nan,
                                "Bal Acc": np.nan,
                                "Macro F1": np.nan,
                                "ROC-AUC": np.nan,
                                "Status": "ERROR",
                                "Error": str(
                                    exc
                                ),
                                "Seed": seed,
                            }
                        )

                    finally:
                        try:
                            import torch

                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        except Exception:
                            pass

                    # Salva progresso imediatamente.
                    pd.DataFrame(
                        results
                    ).to_csv(
                        results_file,
                        index=False,
                    )

        # ---------------------------------------------------------
        # Resumos.
        # ---------------------------------------------------------
        results_df = pd.DataFrame(
            results
        )

        results_df.to_csv(
            results_file,
            index=False,
        )

        print()
        print("=" * 90)
        print(
            "FASE 3: RESUMOS"
        )
        print("=" * 90)

        (
            summary_variant_model,
            summary_variant,
        ) = build_summaries(
            results_df=results_df,
            output_dir=output_dir,
        )

        if (
            summary_variant_model is not None
            and summary_variant is not None
        ):
            print()
            print(
                "Resumo por variante/modelo:"
            )
            print(
                summary_variant_model.to_string(
                    index=False
                )
            )

            print()
            print(
                "Resumo final por variante:"
            )
            print(
                summary_variant.to_string(
                    index=False
                )
            )

            selected = (
                summary_variant.iloc[0]
            )

            print()
            print("=" * 90)
            print(
                "SELEÇÃO DA VARIANTE"
            )
            print("=" * 90)
            print(
                "Critério: maior Macro F1 médio "
                "entre os modelos, após média dos "
                "folds LOCO."
            )
            print(
                f"Melhor variante: "
                f"{selected['Variant']}"
            )
            print(
                f"Macro F1 Selection Score: "
                f"{selected['Macro_F1_Selection_Score']:.4f}"
            )

        else:
            print(
                "[AVISO] Não houve resultados válidos "
                "para gerar resumo."
            )

        print()
        print("=" * 90)
        print(
            "ARQUIVOS"
        )
        print("=" * 90)
        print(
            f"Diretório: {output_dir}"
        )
        print(
            f"Detalhado: {results_file}"
        )
        print(
            f"Log     : {log_file}"
        )

    finally:
        sys.stdout = original_stdout
        logger.close()

    return output_dir


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "TL diagnóstico: CWRU_12k + MFPT -> CWRU_48k "
            "com quatro variantes."
        )
    )

    parser.add_argument(
        "--variants",
        nargs="+",
        default=list(
            VARIANTS.keys()
        ),
        choices=list(
            VARIANTS.keys()
        ),
        help=(
            "Variantes a executar. "
            "Por padrão: todas as quatro."
        ),
    )

    parser.add_argument(
        "--models",
        nargs="+",
        default=MODELS,
        choices=MODELS,
        help=(
            "Modelos a executar. "
            "Por padrão: mlp tabnet."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed de reprodutibilidade.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=EPOCHS,
        help="Número de epochs.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=BATCH_SIZE,
        help="Batch size.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_experiment(
        variants=args.variants,
        models=args.models,
        seed=args.seed,
        epochs=args.epochs,
        batch_size=args.batch_size,
    )
