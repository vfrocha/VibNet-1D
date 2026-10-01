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

# Datasets que serão avaliados como domínios-alvo.
# Em cada rodada, o target atual é retirado do conjunto de sources,
# enquanto todos os demais datasets disponíveis participam do pré-treinamento.
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

# Matriz de normalização mantida exatamente nas cinco estratégias do estudo.
EXPERIMENT_MATRIX = [
    {'id': '0_Baseline',        'pretrain': 'raw',           'finetune': 'raw'},
    {'id': '1_ZScore_Puro',     'pretrain': 'window_zscore', 'finetune': 'window_zscore'},
    {'id': '2_RMS_Puro',        'pretrain': 'window_rms',    'finetune': 'window_rms'},
    {'id': '3_ZScore_Raw',      'pretrain': 'window_zscore', 'finetune': 'raw'},
    {'id': '4_RMS_Raw',         'pretrain': 'window_rms',    'finetune': 'raw'},
]

# Os dois modelos solicitados.
MODELS = ['mlp', 'tabnet']

# Hiperparâmetros mantidos iguais ao experimento TL/DL original.
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
    """Define sementes dos geradores disponíveis no script."""
    random.seed(seed)
    np.random.seed(seed)

    # PyTorch é opcional aqui porque o módulo do projeto já pode gerenciá-lo.
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


# =============================================================================
# NORMALIZAÇÃO NO DOMÍNIO TEMPORAL
# =============================================================================
def normalize_time_window(signal, strategy):
    """
    Normalização por janela aplicada à série temporal antes da extração das
    features. O detrend é mantido como etapa base das três estratégias.
    """
    signal_detrend = detrend(signal)

    if strategy == 'raw':
        return signal_detrend

    if strategy == 'window_zscore':
        std = np.std(signal_detrend)
        return signal_detrend / std if std > 0 else signal_detrend

    if strategy == 'window_rms':
        rms = np.sqrt(np.mean(signal_detrend ** 2))
        return signal_detrend / rms if rms > 0 else signal_detrend

    raise ValueError(f'Estratégia de normalização desconhecida: {strategy}')


# =============================================================================
# CARGA DOS DATASETS
# =============================================================================
def load_entire_dataset(dataset_name):
    """Carrega todas as janelas, rótulos e condições de um dataset."""
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
# EXTRAÇÃO COMPLETA DAS FEATURES
# =============================================================================
def build_feature_cache():
    """
    Extrai, para cada dataset, o vetor completo de features disponibilizado por
    extract_fusion_features (VibNet + SignAI na implementação atual).

    A extração é feita uma vez por estratégia e armazenada em cache.
    """
    strategies = sorted({
        item['pretrain'] for item in EXPERIMENT_MATRIX
    } | {
        item['finetune'] for item in EXPERIMENT_MATRIX
    })

    feature_cache = {strategy: {} for strategy in strategies}
    labels_raw = {}
    conditions = {}
    feature_dimensions = {}

    # Carrega todos os datasets uma única vez; o target é definido na FASE 2.
    datasets_to_load = list(DATASETS_CONFIG.keys())

    print('\n' + '=' * 90)
    print('FASE 1: CARGA E EXTRAÇÃO COMPLETA DAS FEATURES')
    print('=' * 90)

    for dataset_name in datasets_to_load:
        fs = DATASETS_CONFIG[dataset_name]
        X_raw, y_raw, cond_raw = load_entire_dataset(dataset_name)

        if len(X_raw) == 0:
            print(f'  [AVISO] {dataset_name}: nenhum arquivo .npy encontrado.')
            continue

        labels_raw[dataset_name] = y_raw
        conditions[dataset_name] = cond_raw

        print(f'  -> {dataset_name}: {len(X_raw)} janelas | fs={fs} Hz')

        for strategy in strategies:
            X_transformed = [
                normalize_time_window(window, strategy)
                for window in X_raw
            ]

            # Todas as features disponibilizadas pela combinação atual de
            # extract_fusion_features + extract_advanced_features.
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

        print(
            f'     Features utilizadas: {feature_dimensions[dataset_name]}'
        )

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
    return feature_cache, labels_raw, conditions, n_features


# =============================================================================
# RÓTULOS DE DIAGNOSIS
# =============================================================================
def build_diagnosis_labels(labels_raw, available_datasets):
    """
    Mantém o protocolo de diagnosis do experimento anterior:
      - normal/healthy -> Universal_Normal
      - falhas -> identificador específico do dataset

    Isso mantém heads locais por dataset compatíveis com o train_and_evaluate_multihead.
    """
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
# PREDIÇÕES -> ACCURACY
# =============================================================================
def extract_predictions(aux_output, n_expected):
    """
    Tenta localizar y_pred no quarto retorno da função
    train_and_evaluate_multihead.

    O experimento TL/DL existente descarta esse quarto retorno com '_'. Como o
    código-fonte dessa função não está neste script, esta função aceita os
    formatos mais comuns sem assumir que o objeto seja obrigatoriamente um array.
    """
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

        # Predições de classe diretamente.
        if arr.ndim == 1 and arr.size == n_expected:
            return arr.reshape(-1)

        # Probabilidades/logits: uma linha por amostra.
        if arr.ndim == 2 and arr.shape[0] == n_expected and arr.shape[1] > 1:
            return np.argmax(arr, axis=1)

        # Algumas implementações podem devolver uma coluna (n, 1).
        if arr.ndim == 2 and arr.shape == (n_expected, 1):
            return arr[:, 0]

    return None


# =============================================================================
# EXPERIMENTO
# =============================================================================
def run_experiment(seed=42):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    set_seed(seed)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(
        RESULTS_DIR,
        f'log_norm_tl_diagnosis_multitarget_{timestamp}.txt'
    )
    csv_file = os.path.join(
        RESULTS_DIR,
        f'norm_tl_diagnosis_multitarget_results_{timestamp}.csv'
    )
    summary_file = os.path.join(
        RESULTS_DIR,
        f'norm_tl_diagnosis_multitarget_summary_{timestamp}.csv'
    )

    original_stdout = sys.stdout
    logger = Logger(log_file)
    sys.stdout = logger

    try:
        print('=' * 90)
        print('EXPERIMENTO: TL + DIAGNOSIS + NORMALIZAÇÃO')
        print('=' * 90)
        print(f'Targets/Teste: {", ".join(TARGET_DATASETS)}')
        print('UOEMD será source sempre que não for o target (e, neste protocolo, nunca é target).')
        print(f'Modelos: {", ".join(MODELS)}')
        print(f'Tarefa: {TASK}')
        print(f'Epochs: {EPOCHS} | Batch size: {BATCH_SIZE}')
        print('Critério principal para seleção da estratégia: Macro F1 médio entre as dobras LOCO.')

        feature_cache, labels_raw, conditions, n_features = build_feature_cache()
        available_datasets = sorted(labels_raw.keys())
        labels_mapped = build_diagnosis_labels(labels_raw, available_datasets)

        print(f'Número de datasets disponíveis: {len(available_datasets)}')
        print(f'Número de features utilizadas: {n_features}')
        print('Datasets:', ', '.join(available_datasets))

        results = []

        # ---------------------------------------------------------------------
        # FASE 2: MATRIZ EXPERIMENTAL + LEAVE-ONE-DATASET-OUT + LOCO
        # ---------------------------------------------------------------------
        print('\n' + '=' * 90)
        print('FASE 2: MATRIZ EXPERIMENTAL / MULTI-TARGET LOCO')
        print('=' * 90)

        available_datasets = sorted(labels_raw.keys())
        missing_targets = [ds for ds in TARGET_DATASETS if ds not in available_datasets]
        if missing_targets:
            raise FileNotFoundError(
                'TARGET_DATASETS ausentes: ' + ', '.join(missing_targets)
            )

        for target_dataset in TARGET_DATASETS:
            source_datasets = [
                ds for ds in available_datasets
                if ds != target_dataset
            ]

            print('\n' + '=' * 90)
            print(f'TARGET DATASET: {target_dataset}')
            print('=' * 90)
            print(f'Sources ({len(source_datasets)}): {", ".join(source_datasets)}')

            target_conditions = np.unique(conditions[target_dataset])
            print(f'Condições LOCO: {len(target_conditions)}')

            for exp in EXPERIMENT_MATRIX:
                print('\n' + '-' * 90)
                print(f"ESTRATÉGIA {exp['id']}")
                print(f"  Pretrain: {exp['pretrain']}")
                print(f"  Fine-tune: {exp['finetune']}")
                print('-' * 90)

                X_target_full = feature_cache[exp['finetune']][target_dataset]
                y_target_full = labels_mapped[target_dataset]
                cond_target = conditions[target_dataset]

                for test_cond in target_conditions:
                    print(f'\n[LOCO] Target={target_dataset} | Test condition={test_cond}')

                    test_mask = cond_target == test_cond
                    train_target_mask = ~test_mask

                    X_target_train = X_target_full[train_target_mask]
                    y_target_train = y_target_full[train_target_mask]
                    X_test = X_target_full[test_mask]
                    y_test_str = y_target_full[test_mask]

                    if len(X_target_train) == 0 or len(X_test) == 0:
                        print('  [AVISO] Dobra vazia. Pulando.')
                        continue

                    train_data_dict_dl = {}
                    target_encoder = LabelEncoder()

                    # ---------------------------------------------------------
                    # SOURCES: todos os datasets exceto o target atual.
                    # Todos usam a estratégia de PRETRAIN.
                    # ---------------------------------------------------------
                    for source_ds in source_datasets:
                        X_source = feature_cache[exp['pretrain']][source_ds]
                        y_source = labels_mapped[source_ds]

                        if len(X_source) == 0:
                            continue

                        source_encoder = LabelEncoder()
                        y_source_enc = source_encoder.fit_transform(y_source)

                        train_data_dict_dl[source_ds] = (
                            X_source,
                            y_source_enc,
                        )

                    # ---------------------------------------------------------
                    # TARGET: somente as condições diferentes da condição teste.
                    # Usa a estratégia de FINE-TUNE.
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

                    for model_type in MODELS:
                        model_name = f'{model_type.upper()} (TL)'
                        print(f'  -> Treinando {model_name}...')

                        try:
                            # Garante a mesma seed inicial para cada combinação
                            # estratégia x target x condição x modelo.
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
                                    'train_and_evaluate_multihead deve retornar 4 valores: '
                                    '(bal_acc, macro_f1, roc_auc, aux_output).'
                                )

                            bal_acc, macro_f1, roc_auc, aux_output = output

                            y_pred = extract_predictions(
                                aux_output,
                                n_expected=len(y_test_enc),
                            )

                            if y_pred is None:
                                raise RuntimeError(
                                    'Não foi possível identificar y_pred no 4º retorno de '
                                    'train_and_evaluate_multihead. Para calcular accuracy_score, '
                                    'essa função precisa retornar as predições (ou um objeto que as contenha).'
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
                                'Pretrain Strategy': exp['pretrain'],
                                'Finetune Strategy': exp['finetune'],
                                'Target Dataset': target_dataset,
                                'Source Datasets': '|'.join(source_datasets),
                                'N Sources': len(source_datasets),
                                'Task': TASK.capitalize(),
                                'Test Condition': str(test_cond),
                                'Model': model_name,
                                'N Features': n_features,
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
                                'Pretrain Strategy': exp['pretrain'],
                                'Finetune Strategy': exp['finetune'],
                                'Target Dataset': target_dataset,
                                'Source Datasets': '|'.join(source_datasets),
                                'N Sources': len(source_datasets),
                                'Task': TASK.capitalize(),
                                'Test Condition': str(test_cond),
                                'Model': model_name,
                                'N Features': n_features,
                                'N Train Target': len(X_target_train),
                                'N Test': len(X_test_valid),
                                'ACC': np.nan,
                                'Bal Acc': np.nan,
                                'Macro F1': np.nan,
                                'ROC-AUC': np.nan,
                                'Seed': seed,
                                'Error': str(exc),
                            })

                    pd.DataFrame(results).to_csv(csv_file, index=False)

        # ---------------------------------------------------------------------
        # FASE 3: RESUMOS PARA SELEÇÃO DA ESTRATÉGIA
        # ---------------------------------------------------------------------
        results_df = pd.DataFrame(results)

        if results_df.empty:
            print('\n[ERRO] Nenhum resultado foi produzido.')
            return

        # Resumo por target + modelo + estratégia.
        target_model_summary = (
            results_df
            .groupby(['Target Dataset', 'Experiment ID', 'Pretrain Strategy',
                      'Finetune Strategy', 'Model'], dropna=False)
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
                ROC_AUC_Mean=('ROC-AUC', 'mean'),
                ROC_AUC_Std=('ROC-AUC', 'std'),
            )
            .reset_index()
        )

        # Primeiro dá peso igual a cada target: média das dobras LOCO dentro
        # de cada target; depois média entre os quatro targets.
        strategy_summary = (
            target_model_summary
            .groupby(['Experiment ID', 'Pretrain Strategy',
                      'Finetune Strategy', 'Model'], dropna=False)
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

        # Métrica principal consolidada: Macro F1, dando peso igual a
        # MLP e TabNet depois de equilibrar os targets.
        overall = (
            strategy_summary
            .groupby(['Experiment ID', 'Pretrain Strategy', 'Finetune Strategy'],
                     dropna=False)
            .agg(
                N_Models=('Model', 'nunique'),
                Macro_F1_Selection_Score=('Macro_F1_Mean_Targets', 'mean'),
                Macro_F1_Model_Std=('Macro_F1_Mean_Targets', 'std'),
                Macro_F1_Worst_Target=('Macro_F1_Min_Target', 'min'),
                ACC_Selection_Mean=('ACC_Mean_Targets', 'mean'),
                Bal_Acc_Selection_Mean=('Bal_Acc_Mean_Targets', 'mean'),
                ROC_AUC_Selection_Mean=('ROC_AUC_Mean_Targets', 'mean'),
            )
            .reset_index()
            .sort_values('Macro_F1_Selection_Score', ascending=False)
        )

        target_model_summary.to_csv(
            summary_file.replace('.csv', '_by_target_model.csv'),
            index=False,
        )
        strategy_summary.to_csv(
            summary_file.replace('.csv', '_by_strategy_model.csv'),
            index=False,
        )
        overall.to_csv(summary_file, index=False)

        print('\n' + '=' * 90)
        print('RESUMO POR TARGET / MODELO / ESTRATÉGIA')
        print('=' * 90)
        print(target_model_summary.to_string(index=False))

        print('\n' + '=' * 90)
        print('RESUMO POR ESTRATÉGIA E MODELO')
        print('=' * 90)
        print(strategy_summary.to_string(index=False))

        print('\n' + '=' * 90)
        print('COMPARAÇÃO FINAL DAS ESTRATÉGIAS')
        print('=' * 90)
        print(overall.to_string(index=False))

        selected = overall.iloc[0]
        print('\n' + '=' * 90)
        print('CRITÉRIO PRIMÁRIO DE SELEÇÃO')
        print('=' * 90)
        print(
            'Macro F1 médio entre targets (peso igual por target) e entre '
            'MLP/TabNet.'
        )
        print(
            f"Estratégia com maior Macro F1 Selection Score: "
            f"{selected['Experiment ID']} | "
            f"Score={selected['Macro_F1_Selection_Score']:.4f}"
        )
        print(
            'Recomendação metodológica para a análise: verificar também a '
            'variabilidade entre condições e a consistência entre MLP e TabNet '
            'antes de fixar a estratégia para a etapa seguinte.'
        )

        print(f'\n[SUCESSO] Resultados detalhados: {csv_file}')
        print(f'[SUCESSO] Resumo: {summary_file}')
        print(f'[SUCESSO] Log: {log_file}')

    finally:
        sys.stdout = original_stdout
        logger.close()


if __name__ == '__main__':
    run_experiment(seed=42)
