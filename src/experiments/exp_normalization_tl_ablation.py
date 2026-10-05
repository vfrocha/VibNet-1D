import os
import sys
import random
from datetime import datetime

import numpy as np
import pandas as pd
from scipy.signal import detrend
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder

# -----------------------------------------------------------------------------
# Projeto
# -----------------------------------------------------------------------------
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.features.extractors_v2 import extract_advanced_features
from src.features.signalai_wrapper import extract_fusion_features
from src.models.build_tabnet_resnet import train_and_evaluate_multihead

# =============================================================================
# CONFIGURAÇÃO EXPERIMENTAL
# =============================================================================
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))

TASK = 'diagnosis'

TARGET_DATASETS = ['CWRU_12k', 'PU', 'HUST_Gearbox', 'CWRU_48k']

DATASETS_CONFIG = {
    'CWRU_12k': 12000,
    'CWRU_48k': 48000,
    'UOEMD': 42000,
    'HUST_Gearbox': 25600,
    'HUST': 51200,
    'PU': 64000,
    'UORED': 42000,
    'Mechanical_Gear': 5000,
    'Electric_Motor': 50000,
}

# -----------------------------------------------------------------------------
# Matriz de ablação
#
# preprocessing values:
#   raw             = sinal original, sem detrend e sem RMS
#   rms_raw         = sinal original / RMS(sinal original)
#   detrend         = sinal - tendência linear
#   rms_detrend     = sinal detrended / RMS(sinal detrended)
#
# Extras:
#   RMS = RMS do sinal original da janela
#   b1  = inclinação da tendência linear
#   b2  = intercepto da tendência linear
# -----------------------------------------------------------------------------
EXPERIMENT_MATRIX = [
    {
        'id': 'E0_RMS_Detrend',
        'description': 'RMS + detrend em pretrain, fine-tune e teste',
        'pretrain': 'rms_detrend',
        'finetune': 'rms_detrend',
        'test': 'rms_detrend',
        'add_parameters': False,
    },
    {
        'id': 'E1_RMS_Sem_Detrend',
        'description': 'RMS sem detrend em pretrain, fine-tune e teste',
        'pretrain': 'rms_raw',
        'finetune': 'rms_raw',
        'test': 'rms_raw',
        'add_parameters': False,
    },
    {
        'id': 'E2_RMS_Detrend_Treino_Raw_Teste',
        'description': 'RMS + detrend no treino; raw sem detrend no teste',
        'pretrain': 'rms_detrend',
        'finetune': 'rms_detrend',
        'test': 'raw',
        'add_parameters': False,
    },
    {
        'id': 'E3_RMS_Detrend_Com_Parametros',
        'description': 'RMS + detrend com RMS, b1 e b2 como features',
        'pretrain': 'rms_detrend',
        'finetune': 'rms_detrend',
        'test': 'rms_detrend',
        'add_parameters': True,
    },
    {
        'id': 'E4_RMS_Detrend_Treino_Raw_Teste_Com_Parametros',
        'description': 'RMS + detrend no treino; raw sem detrend no teste; com RMS, b1 e b2',
        'pretrain': 'rms_detrend',
        'finetune': 'rms_detrend',
        'test': 'raw',
        'add_parameters': True,
    },
]

MODELS = ['mlp', 'tabnet']

EPOCHS = 15
BATCH_SIZE = 512

# =============================================================================
# LOG
# =============================================================================
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, 'w', encoding='utf-8')

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
# PARÂMETROS DA JANELA
# =============================================================================
def calculate_window_parameters(signal):
    """
    Calcula os parâmetros de transformação da janela original.

    A tendência linear é:

        x(t) ~= b1 * t + b2

    onde t é o índice da amostra (0, 1, ..., N-1).

    Retorna:
        rms_raw: RMS do sinal original
        b1:      coeficiente angular da tendência
        b2:      intercepto da tendência
        rms_detrended: RMS após remoção da tendência
    """
    x = np.asarray(signal, dtype=np.float64).reshape(-1)
    if x.size == 0:
        return 0.0, 0.0, 0.0, 0.0

    rms_raw = float(np.sqrt(np.mean(x ** 2)))

    if x.size >= 2:
        t = np.arange(x.size, dtype=np.float64)
        b1, b2 = np.polyfit(t, x, 1)
        trend = b1 * t + b2
        residual = x - trend
    else:
        b1 = 0.0
        b2 = float(x[0])
        residual = x.copy()

    rms_detrended = float(np.sqrt(np.mean(residual ** 2)))

    return rms_raw, float(b1), float(b2), rms_detrended


# =============================================================================
# TRANSFORMAÇÃO TEMPORAL
# =============================================================================
def transform_window(signal, strategy):
    """Aplica exatamente uma das quatro transformações temporais definidas."""
    x = np.asarray(signal, dtype=np.float64).reshape(-1)

    if strategy == 'raw':
        return x

    if strategy == 'rms_raw':
        rms = np.sqrt(np.mean(x ** 2))
        return x / rms if rms > 0 else x

    if strategy == 'detrend':
        return detrend(x)

    if strategy == 'rms_detrend':
        x_dt = detrend(x)
        rms = np.sqrt(np.mean(x_dt ** 2))
        return x_dt / rms if rms > 0 else x_dt

    raise ValueError(f'Estratégia temporal desconhecida: {strategy}')


# =============================================================================
# CARGA DOS DATASETS
# =============================================================================
def load_entire_dataset(dataset_name):
    if dataset_name not in DATASETS_CONFIG:
        raise KeyError(f'Dataset sem frequência configurada: {dataset_name}')

    dataset_path = os.path.join(DATA_ROOT, dataset_name)
    if not os.path.exists(dataset_path):
        return [], [], []

    X_raw = []
    y_raw = []
    cond_raw = []

    for root, _, files in os.walk(dataset_path):
        for filename in sorted(files):
            if not filename.endswith('.npy'):
                continue

            class_name = os.path.basename(root)
            cond_name = os.path.basename(os.path.dirname(root))
            X_raw.append(np.load(os.path.join(root, filename)))
            y_raw.append(class_name)
            cond_raw.append(cond_name)

    return X_raw, np.asarray(y_raw), np.asarray(cond_raw)


# =============================================================================
# EXTRAÇÃO DE FEATURES + PARÂMETROS
# =============================================================================
def build_feature_cache():
    """
    Extrai as features temporais para cada transformação necessária.

    Além das features do VibNet/SignalAI, mantém uma matriz de três parâmetros
    por janela: [RMS_original, b1, b2]. Esses parâmetros são derivados apenas
    da janela original e são adicionados somente nas estratégias E3/E4.
    """
    strategies = sorted({
        item['pretrain'] for item in EXPERIMENT_MATRIX
    } | {
        item['finetune'] for item in EXPERIMENT_MATRIX
    } | {
        item['test'] for item in EXPERIMENT_MATRIX
    })

    feature_cache = {strategy: {} for strategy in strategies}
    parameter_cache = {}
    labels_raw = {}
    conditions = {}
    feature_dimensions = {}

    print('\n' + '=' * 100)
    print('FASE 1: CARGA, PARÂMETROS E EXTRAÇÃO DAS FEATURES')
    print('=' * 100)
    print('Parâmetros adicionais: [RMS_original, b1, b2]')

    for dataset_name in DATASETS_CONFIG:
        fs = DATASETS_CONFIG[dataset_name]
        X_raw, y_raw, cond_raw = load_entire_dataset(dataset_name)

        if len(X_raw) == 0:
            print(f'  [AVISO] {dataset_name}: nenhum arquivo .npy encontrado.')
            continue

        labels_raw[dataset_name] = y_raw
        conditions[dataset_name] = cond_raw

        # Calcula RMS, b1 e b2 uma única vez por janela.
        params = np.asarray([
            calculate_window_parameters(window)[:3]
            for window in X_raw
        ], dtype=np.float32)
        params = np.nan_to_num(params, nan=0.0, posinf=0.0, neginf=0.0)
        parameter_cache[dataset_name] = params

        print(f'  -> {dataset_name}: {len(X_raw)} janelas | fs={fs} Hz')

        for strategy in strategies:
            print(f'     extraindo: {strategy} ...')
            X_transformed = [
                transform_window(window, strategy)
                for window in X_raw
            ]

            X_fusion = extract_fusion_features(
                np.asarray(X_transformed),
                fs,
                extract_advanced_features,
            )

            X_clean = np.nan_to_num(
                np.asarray(X_fusion, dtype=np.float32),
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )

            if X_clean.ndim == 1:
                X_clean = X_clean.reshape(len(y_raw), -1)

            if X_clean.shape[0] != len(y_raw):
                raise ValueError(
                    f'{dataset_name}/{strategy}: número de linhas de features '
                    f'({X_clean.shape[0]}) diferente do número de amostras '
                    f'({len(y_raw)}).'
                )

            feature_dimensions.setdefault(dataset_name, X_clean.shape[1])
            if feature_dimensions[dataset_name] != X_clean.shape[1]:
                raise ValueError(
                    f'{dataset_name}: dimensão de features inconsistente entre estratégias.'
                )

            feature_cache[strategy][dataset_name] = X_clean

        print(f'     Features temporais: {feature_dimensions[dataset_name]}')
        print('     Features extras: 3 [RMS, b1, b2]')

    available = set(labels_raw.keys())
    missing_datasets = [ds for ds in DATASETS_CONFIG if ds not in available]
    if missing_datasets:
        raise FileNotFoundError(
            'Datasets configurados ausentes: ' + ', '.join(missing_datasets)
        )

    all_dims = set(feature_dimensions.values())
    if len(all_dims) != 1:
        raise ValueError(
            f'Os datasets não possuem a mesma dimensão de features: {feature_dimensions}'
        )

    n_features = next(iter(all_dims))
    return feature_cache, parameter_cache, labels_raw, conditions, n_features


# =============================================================================
# RÓTULOS DE DIAGNOSIS
# =============================================================================
def build_diagnosis_labels(labels_raw, available_datasets):
    mapped = {}

    for dataset_name in available_datasets:
        values = []
        for label in labels_raw[dataset_name]:
            label_lower = label.lower()
            is_normal = 'normal' in label_lower or 'healthy' in label_lower

            if is_normal:
                values.append('Universal_Normal')
            else:
                values.append(f'{dataset_name}_{label}')

        mapped[dataset_name] = np.asarray(values)

    return mapped


# =============================================================================
# AUXILIARES
# =============================================================================
def compose_features(base_features, parameters, add_parameters):
    if not add_parameters:
        return base_features

    if len(base_features) != len(parameters):
        raise ValueError(
            f'Quantidade de amostras incompatível: features={len(base_features)} '
            f'vs parâmetros={len(parameters)}.'
        )

    return np.hstack([
        base_features,
        parameters.astype(np.float32),
    ]).astype(np.float32)


def extract_predictions(aux_output, n_expected):
    candidates = []

    if isinstance(aux_output, dict):
        for key in ('y_pred', 'predictions', 'preds', 'yhat'):
            if key in aux_output:
                candidates.append(aux_output[key])
    elif isinstance(aux_output, tuple) and len(aux_output) == 2:
        candidates.extend(aux_output)
    else:
        candidates.append(aux_output)

    for candidate in candidates:
        try:
            arr = np.asarray(candidate)
        except Exception:
            continue

        if arr.ndim == 1 and arr.size == n_expected:
            return arr.reshape(-1)

        if arr.ndim == 2 and arr.shape[0] == n_expected and arr.shape[1] > 1:
            return np.argmax(arr, axis=1)

        if arr.ndim == 2 and arr.shape == (n_expected, 1):
            return arr[:, 0]

    return None


def feature_set(feature_cache, parameter_cache, strategy, dataset_name, add_parameters):
    return compose_features(
        feature_cache[strategy][dataset_name],
        parameter_cache[dataset_name],
        add_parameters,
    )


# =============================================================================
# EXPERIMENTO
# =============================================================================
def run_experiment(seed=42):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    prefix = f'norm_tl_diagnosis_ablation_{timestamp}'

    log_file = os.path.join(RESULTS_DIR, f'{prefix}.log')
    csv_file = os.path.join(RESULTS_DIR, f'{prefix}_results.csv')
    summary_target_file = os.path.join(RESULTS_DIR, f'{prefix}_summary_by_target_model.csv')
    summary_strategy_file = os.path.join(RESULTS_DIR, f'{prefix}_summary_by_strategy_model.csv')
    summary_overall_file = os.path.join(RESULTS_DIR, f'{prefix}_summary.csv')

    original_stdout = sys.stdout
    logger = Logger(log_file)
    sys.stdout = logger

    try:
        print('=' * 100)
        print('EXPERIMENTO: ABLATION TL + DIAGNOSIS + DETREND/RMS')
        print('=' * 100)
        print(f'Targets: {", ".join(TARGET_DATASETS)}')
        print(f'Modelos: {", ".join(MODELS)}')
        print(f'Tarefa: {TASK}')
        print(f'Epochs: {EPOCHS} | Batch size: {BATCH_SIZE}')
        print('\nMATRIZ EXPERIMENTAL:')
        for exp in EXPERIMENT_MATRIX:
            print(f"  {exp['id']}: {exp['description']}")
            print(
                f"       pretrain={exp['pretrain']} | "
                f"finetune={exp['finetune']} | "
                f"test={exp['test']} | "
                f"extras={exp['add_parameters']}"
            )

        feature_cache, parameter_cache, labels_raw, conditions, n_temporal_features = build_feature_cache()
        available_datasets = sorted(labels_raw.keys())
        labels_mapped = build_diagnosis_labels(labels_raw, available_datasets)

        print(f'\nDatasets disponíveis: {len(available_datasets)}')
        print(f'Features temporais: {n_temporal_features}')
        print(f'Features com extras: {n_temporal_features + 3}')
        print('Datasets:', ', '.join(available_datasets))

        missing_targets = [ds for ds in TARGET_DATASETS if ds not in available_datasets]
        if missing_targets:
            raise FileNotFoundError(
                'TARGET_DATASETS ausentes: ' + ', '.join(missing_targets)
            )

        results = []

        # ---------------------------------------------------------------------
        # FASE 2: MULTI-TARGET LOCO
        # ---------------------------------------------------------------------
        print('\n' + '=' * 100)
        print('FASE 2: MULTI-TARGET LOCO')
        print('=' * 100)

        for target_dataset in TARGET_DATASETS:
            source_datasets = [
                ds for ds in available_datasets
                if ds != target_dataset
            ]

            print('\n' + '=' * 100)
            print(f'TARGET DATASET: {target_dataset}')
            print('=' * 100)
            print(f'Sources ({len(source_datasets)}): {", ".join(source_datasets)}')

            target_conditions = np.unique(conditions[target_dataset])
            print(f'Condições LOCO: {len(target_conditions)}')

            for exp in EXPERIMENT_MATRIX:
                print('\n' + '-' * 100)
                print(f"{exp['id']}: {exp['description']}")
                print('-' * 100)

                # -------------------------------------------------------------
                # TARGET: a representação de treino/fine-tune é definida por
                # finetune; o conjunto de teste usa explicitamente test.
                # -------------------------------------------------------------
                X_target_full = feature_set(
                    feature_cache,
                    parameter_cache,
                    exp['finetune'],
                    target_dataset,
                    exp['add_parameters'],
                )
                X_target_test_full = feature_set(
                    feature_cache,
                    parameter_cache,
                    exp['test'],
                    target_dataset,
                    exp['add_parameters'],
                )

                y_target_full = labels_mapped[target_dataset]
                cond_target = conditions[target_dataset]

                for test_cond in target_conditions:
                    print(f'\n[LOCO] Target={target_dataset} | Test condition={test_cond}')

                    test_mask = cond_target == test_cond
                    train_target_mask = ~test_mask

                    X_target_train = X_target_full[train_target_mask]
                    y_target_train = y_target_full[train_target_mask]
                    X_test = X_target_test_full[test_mask]
                    y_test_str = y_target_full[test_mask]

                    if len(X_target_train) == 0 or len(X_test) == 0:
                        print('  [AVISO] Dobra vazia. Pulando.')
                        continue

                    train_data_dict_dl = {}
                    target_encoder = LabelEncoder()

                    # ---------------------------------------------------------
                    # SOURCES: pretrain
                    # ---------------------------------------------------------
                    for source_ds in source_datasets:
                        X_source = feature_set(
                            feature_cache,
                            parameter_cache,
                            exp['pretrain'],
                            source_ds,
                            exp['add_parameters'],
                        )
                        y_source = labels_mapped[source_ds]

                        if len(X_source) == 0:
                            continue

                        source_encoder = LabelEncoder()
                        y_source_enc = source_encoder.fit_transform(y_source)
                        train_data_dict_dl[source_ds] = (X_source, y_source_enc)

                    # ---------------------------------------------------------
                    # TARGET: fine-tune
                    # ---------------------------------------------------------
                    y_target_train_enc = target_encoder.fit_transform(y_target_train)
                    train_data_dict_dl[target_dataset] = (
                        X_target_train,
                        y_target_train_enc,
                    )

                    valid_test_idx = [
                        idx for idx, label in enumerate(y_test_str)
                        if label in target_encoder.classes_
                    ]

                    if not valid_test_idx:
                        print(
                            '  [AVISO] Nenhuma classe do teste está presente no '
                            'treino target. Pulando.'
                        )
                        continue

                    X_test_valid = X_test[valid_test_idx]
                    y_test_valid_str = y_test_str[valid_test_idx]
                    y_test_enc = target_encoder.transform(y_test_valid_str)

                    n_features_current = X_target_train.shape[1]
                    print(f'  Features usadas: {n_features_current}')
                    print(f'  Target train: {len(X_target_train)}')
                    print(f'  Teste: {len(X_test_valid)}')

                    for model_type in MODELS:
                        model_name = f'{model_type.upper()} (TL)'
                        print(f'  -> Treinando {model_name}...')

                        try:
                            set_seed(seed)

                            output = train_and_evaluate_multihead(
                                train_data_dict=train_data_dict_dl,
                                target_dataset_name=target_dataset,
                                X_test=X_test_valid,
                                y_test=y_test_enc,
                                task=TASK,
                                epochs=EPOCHS,
                                batch_size=BATCH_SIZE,
                                encoder_type=model_type,
                            )

                            if not isinstance(output, tuple) or len(output) != 4:
                                raise RuntimeError(
                                    'train_and_evaluate_multihead deve retornar '
                                    '(bal_acc, macro_f1, roc_auc, aux_output).'
                                )

                            bal_acc, macro_f1, roc_auc, aux_output = output
                            y_pred = extract_predictions(
                                aux_output,
                                n_expected=len(y_test_enc),
                            )

                            if y_pred is None:
                                raise RuntimeError(
                                    'Não foi possível identificar y_pred no 4º retorno.'
                                )

                            acc = accuracy_score(y_test_enc, y_pred)

                            print(
                                f'     [{model_name}] '
                                f'ACC={acc:.4f} | '
                                f'Bal Acc={bal_acc:.4f} | '
                                f'Macro F1={macro_f1:.4f} | '
                                f'ROC-AUC={roc_auc:.4f}'
                            )

                            results.append({
                                'Experiment ID': exp['id'],
                                'Description': exp['description'],
                                'Pretrain Strategy': exp['pretrain'],
                                'Finetune Strategy': exp['finetune'],
                                'Test Strategy': exp['test'],
                                'Additional Parameters': exp['add_parameters'],
                                'Additional Features': 'RMS,b1,b2' if exp['add_parameters'] else '',
                                'Target Dataset': target_dataset,
                                'Source Datasets': '|'.join(source_datasets),
                                'N Sources': len(source_datasets),
                                'Task': TASK.capitalize(),
                                'Test Condition': str(test_cond),
                                'Model': model_name,
                                'N Temporal Features': n_temporal_features,
                                'N Features': n_features_current,
                                'N Train Target': len(X_target_train),
                                'N Test': len(X_test_valid),
                                'ACC': float(acc),
                                'Bal Acc': float(bal_acc),
                                'Macro F1': float(macro_f1),
                                'ROC-AUC': float(roc_auc),
                                'Seed': seed,
                            })

                        except Exception as exc:
                            print(f'     [ERRO] {model_name}: {exc}')
                            results.append({
                                'Experiment ID': exp['id'],
                                'Description': exp['description'],
                                'Pretrain Strategy': exp['pretrain'],
                                'Finetune Strategy': exp['finetune'],
                                'Test Strategy': exp['test'],
                                'Additional Parameters': exp['add_parameters'],
                                'Additional Features': 'RMS,b1,b2' if exp['add_parameters'] else '',
                                'Target Dataset': target_dataset,
                                'Source Datasets': '|'.join(source_datasets),
                                'N Sources': len(source_datasets),
                                'Task': TASK.capitalize(),
                                'Test Condition': str(test_cond),
                                'Model': model_name,
                                'N Temporal Features': n_temporal_features,
                                'N Features': n_features_current,
                                'N Train Target': len(X_target_train),
                                'N Test': len(X_test_valid),
                                'ACC': np.nan,
                                'Bal Acc': np.nan,
                                'Macro F1': np.nan,
                                'ROC-AUC': np.nan,
                                'Seed': seed,
                                'Error': str(exc),
                            })

                    # Salva progressivamente para não perder resultados em uma
                    # execução longa.
                    pd.DataFrame(results).to_csv(csv_file, index=False)

        # ---------------------------------------------------------------------
        # FASE 3: RESUMOS
        # ---------------------------------------------------------------------
        results_df = pd.DataFrame(results)
        if results_df.empty:
            print('\n[ERRO] Nenhum resultado foi produzido.')
            return

        target_model_summary = (
            results_df
            .groupby(
                [
                    'Target Dataset',
                    'Experiment ID',
                    'Pretrain Strategy',
                    'Finetune Strategy',
                    'Test Strategy',
                    'Additional Parameters',
                    'Model',
                ],
                dropna=False,
            )
            .agg(
                N_Folds=('Test Condition', 'count'),
                ACC_Mean=('ACC', 'mean'),
                ACC_Std=('ACC', 'std'),
                Bal_Acc_Mean=('Bal Acc', 'mean'),
                Bal_Acc_Std=('Bal Acc', 'std'),
                Macro_F1_Mean=('Macro F1', 'mean'),
                Macro_F1_Std=('Macro F1', 'std'),
                Macro_F1_Median=('Macro F1', 'median'),
                Macro_F1_Min=('Macro F1', 'min'),
                Macro_F1_Max=('Macro F1', 'max'),
                ROC_AUC_Mean=('ROC-AUC', 'mean'),
                ROC_AUC_Std=('ROC-AUC', 'std'),
            )
            .reset_index()
        )

        strategy_summary = (
            target_model_summary
            .groupby(
                [
                    'Experiment ID',
                    'Pretrain Strategy',
                    'Finetune Strategy',
                    'Test Strategy',
                    'Additional Parameters',
                    'Model',
                ],
                dropna=False,
            )
            .agg(
                N_Targets=('Target Dataset', 'nunique'),
                Macro_F1_Mean_Targets=('Macro_F1_Mean', 'mean'),
                Macro_F1_Std_Targets=('Macro_F1_Mean', 'std'),
                Macro_F1_Min_Target=('Macro_F1_Mean', 'min'),
                Macro_F1_Max_Target=('Macro_F1_Mean', 'max'),
                ACC_Mean_Targets=('ACC_Mean', 'mean'),
                Bal_Acc_Mean_Targets=('Bal_Acc_Mean', 'mean'),
                ROC_AUC_Mean_Targets=('ROC_AUC_Mean', 'mean'),
            )
            .reset_index()
        )

        overall = (
            strategy_summary
            .groupby(
                [
                    'Experiment ID',
                    'Pretrain Strategy',
                    'Finetune Strategy',
                    'Test Strategy',
                    'Additional Parameters',
                ],
                dropna=False,
            )
            .agg(
                N_Models=('Model', 'nunique'),
                Macro_F1_Selection_Score=('Macro_F1_Mean_Targets', 'mean'),
                Macro_F1_Model_Std=('Macro_F1_Mean_Targets', 'std'),
                Macro_F1_Worst_Target=('Macro_F1_Min_Target', 'min'),
                Macro_F1_Best_Target=('Macro_F1_Max_Target', 'max'),
                ACC_Selection_Mean=('ACC_Mean_Targets', 'mean'),
                Bal_Acc_Selection_Mean=('Bal_Acc_Mean_Targets', 'mean'),
                ROC_AUC_Selection_Mean=('ROC_AUC_Mean_Targets', 'mean'),
            )
            .reset_index()
            .sort_values('Macro_F1_Selection_Score', ascending=False)
        )

        target_model_summary.to_csv(summary_target_file, index=False)
        strategy_summary.to_csv(summary_strategy_file, index=False)
        overall.to_csv(summary_overall_file, index=False)

        print('\n' + '=' * 100)
        print('RESUMO POR TARGET / MODELO / ESTRATÉGIA')
        print('=' * 100)
        print(target_model_summary.to_string(index=False))

        print('\n' + '=' * 100)
        print('RESUMO POR ESTRATÉGIA E MODELO')
        print('=' * 100)
        print(strategy_summary.to_string(index=False))

        print('\n' + '=' * 100)
        print('COMPARAÇÃO FINAL E0-E4')
        print('=' * 100)
        print(overall.to_string(index=False))

        selected = overall.iloc[0]
        print('\n' + '=' * 100)
        print('CRITÉRIO PRIMÁRIO DE SELEÇÃO')
        print('=' * 100)
        print(
            'Macro F1 médio entre targets, dando peso igual aos targets e aos modelos.'
        )
        print(
            f"Melhor experimento: {selected['Experiment ID']} | "
            f"Macro F1={selected['Macro_F1_Selection_Score']:.4f}"
        )
        print(
            '\nIMPORTANTE: a escolha final deve considerar também a consistência '
            'entre targets/modelos e a variabilidade das dobras LOCO.'
        )

        print('\n' + '=' * 100)
        print('ARQUIVOS GERADOS')
        print('=' * 100)
        print(f'Detalhado : {csv_file}')
        print(f'Por target: {summary_target_file}')
        print(f'Por modelo: {summary_strategy_file}')
        print(f'Geral     : {summary_overall_file}')
        print(f'Log       : {log_file}')

    finally:
        sys.stdout = original_stdout
        logger.close()


if __name__ == '__main__':
    run_experiment(seed=42)
