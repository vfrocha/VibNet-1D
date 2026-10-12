import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Adiciona a raiz do projeto ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.dataloader import load_vibration_data
from src.features.extractors_v2 import extract_advanced_features
from src.features.signalai_wrapper import extract_fusion_features

# Importando os modelos modulares
from src.models.build_sklearn import get_random_forest, get_svm, get_xgboost, train_and_evaluate
from src.models.build_tabular import get_tabnet_classifier, train_and_evaluate_tabnet

# --- CONFIGURAÇÃO DO PROTOCOLO DO MERT (Tabela 3-3 e Tabela 4-3) ---
TASKS = ["diagnosis"]
FS = 48000
MERT_WINDOW_SIZE = 1024  # 21.3 ms a 48 kHz

# Domínios 1 a 8 (Treino): Todas as cargas de 0.007" e 0.014"
TRAIN_DOMAINS_1_TO_8 = [
    "Load_0HP_Sev_0.007", "Load_1HP_Sev_0.007", "Load_2HP_Sev_0.007", "Load_3HP_Sev_0.007", # Domínios 1-4
    "Load_0HP_Sev_0.014", "Load_1HP_Sev_0.014", "Load_2HP_Sev_0.014", "Load_3HP_Sev_0.014"  # Domínios 5-8
]

# Domínios 9 a 12 (Severidade 0.021")
TEST_DOMAIN_9 = ["Load_0HP_Sev_0.021"] # Domínio 9 exato da Tabela 4-3 (1797 RPM / 0 HP)
ALL_021_DOMAINS = [
    "Load_0HP_Sev_0.021", "Load_1HP_Sev_0.021", "Load_2HP_Sev_0.021", "Load_3HP_Sev_0.021"  # Domínios 9-12
]

ALL_CONDITIONS = TRAIN_DOMAINS_1_TO_8 + ALL_021_DOMAINS

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))


def slice_into_subwindows(X_raw, y_raw, subwindow_size=1024):
    """
    Fatia janelas de 1 segundo (48.000 pontos) em sub-janelas não sobrepostas
    de 1024 pontos em memória, sem precisar alterar as pastas no disco.
    """
    X_sub, y_sub = [], []
    for signal, label in zip(X_raw, y_raw):
        n_Possible = len(signal) // subwindow_size
        for i in range(n_Possible):
            start = i * subwindow_size
            end = start + subwindow_size
            X_sub.append(signal[start:end])
            y_sub.append(label)
    return np.array(X_sub, dtype=np.float32), np.array(y_sub)


def apply_window_preprocessing(X_windows, mode="none"):
    """
    Aplica normalização individual por janela (como na Seção 3.2 da tese do Mert)
    para tornar as features invariantes à escala de energia da severidade.
    Opções: 'none', 'rms', 'zscore', 'minmax'
    """
    if mode == "none":
        return X_windows
    
    X_norm = np.empty_like(X_windows, dtype=np.float32)
    for i, w in enumerate(X_windows):
        if mode == "rms":
            rms = np.sqrt(np.mean(w ** 2))
            X_norm[i] = w / rms if rms > 1e-8 else w
        elif mode == "zscore":
            std = np.std(w)
            X_norm[i] = (w - np.mean(w)) / std if std > 1e-8 else (w - np.mean(w))
        elif mode == "minmax":
            w_min, w_max = np.min(w), np.max(w)
            X_norm[i] = (w - w_min) / (w_max - w_min) if (w_max - w_min) > 1e-8 else w
    return X_norm


def evaluate_all_models(X_train, y_train, X_test, y_test, dataset_name, task, test_cond, prep_mode):
    fold_results = []
    base_info = {
        "Dataset": dataset_name,
        "Task": task.capitalize(),
        "Test Condition": test_cond,
        "Window Preprocessing": prep_mode
    }

    # A) Random Forest
    print(f"     -> Treinando Random Forest...")
    try:
        rf_pipeline, rf_grid = get_random_forest()
        rf_acc, rf_f1, rf_auc, _ = train_and_evaluate(rf_pipeline, rf_grid, X_train, y_train, X_test, y_test, task=task)
    except Exception as e:
        print(f"        [AVISO Random Forest] O modelo falhou: {e}")
        rf_acc, rf_f1, rf_auc = 0.0, 0.0, 0.0
    fold_results.append({**base_info, "Model": "Random Forest", "Bal Acc": rf_acc, "Macro F1": rf_f1, "ROC-AUC": rf_auc})

    # B) SVM
    print(f"     -> Treinando SVM...")
    try:
        svm_pipeline, svm_grid = get_svm()
        svm_acc, svm_f1, svm_auc, _ = train_and_evaluate(svm_pipeline, svm_grid, X_train, y_train, X_test, y_test, task=task)
    except Exception as e:
        print(f"        [AVISO SVM] O modelo falhou: {e}")
        svm_acc, svm_f1, svm_auc = 0.0, 0.0, 0.0
    fold_results.append({**base_info, "Model": "SVM", "Bal Acc": svm_acc, "Macro F1": svm_f1, "ROC-AUC": svm_auc})

    # C) XGBoost
    print(f"     -> Treinando XGBoost...")
    try:
        xgb_pipeline, xgb_grid = get_xgboost()
        xgb_acc, xgb_f1, xgb_auc, _ = train_and_evaluate(xgb_pipeline, xgb_grid, X_train, y_train, X_test, y_test, task=task)
    except Exception as e:
        print(f"        [AVISO XGBoost] O modelo falhou: {e}")
        xgb_acc, xgb_f1, xgb_auc = 0.0, 0.0, 0.0
    fold_results.append({**base_info, "Model": "XGBoost", "Bal Acc": xgb_acc, "Macro F1": xgb_f1, "ROC-AUC": xgb_auc})

    # D) TabNet
    print(f"     -> Treinando TabNet...")
    try:
        tabnet_model = get_tabnet_classifier()
        tabnet_acc, tabnet_f1, tabnet_auc, *_ = train_and_evaluate_tabnet(
            model=tabnet_model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, task=task
        )
    except Exception as e:
        print(f"        [AVISO TABNET] O modelo falhou: {e}")
        tabnet_acc, tabnet_f1, tabnet_auc = 0.0, 0.0, 0.0
    fold_results.append({**base_info, "Model": "TabNet", "Bal Acc": tabnet_acc, "Macro F1": tabnet_f1, "ROC-AUC": tabnet_auc})

    return fold_results


def run_mert_comparison():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = os.path.join(RESULTS_DIR, f'cwru48k_mert_comparison_{timestamp}.csv')
    master_results = []

    print(f"{'='*75}\n COMPARAÇÃO COM TESE DO MERT: CWRU 48k (DOMÍNIOS 1-8 -> DOMÍNIO 9 | 1024 pts)\n{'='*75}")

    # 1. Carrega as 12 condições em cache (janelas originais de 48.000 pontos)
    print(">>> Carregando condições originais em cache...")
    cached_data = {}
    for cond in ALL_CONDITIONS:
        _, _, X_cond, y_cond, _ = load_vibration_data(
            data_root=DATA_ROOT, dataset_name="CWRU_48k", test_condition=cond, task="diagnosis"
        )
        cached_data[cond] = (X_cond, y_cond)

    # 2. Monta o conjunto de Treino (Domínios 1 a 8) e Teste (Domínio 9: Load_0HP_Sev_0.021)
    X_train_48k = np.concatenate([cached_data[c][0] for c in TRAIN_DOMAINS_1_TO_8 if len(cached_data[c][0]) > 0], axis=0)
    y_train_48k = np.concatenate([cached_data[c][1] for c in TRAIN_DOMAINS_1_TO_8 if len(cached_data[c][0]) > 0], axis=0)

    X_test_48k = np.concatenate([cached_data[c][0] for c in TEST_DOMAIN_9 if len(cached_data[c][0]) > 0], axis=0)
    y_test_48k = np.concatenate([cached_data[c][1] for c in TEST_DOMAIN_9 if len(cached_data[c][0]) > 0], axis=0)

    # 3. Fatia as janelas de 48.000 pontos para 1.024 pontos (como na tese do Mert)
    X_train_1024, y_train_1024 = slice_into_subwindows(X_train_48k, y_train_48k, MERT_WINDOW_SIZE)
    X_test_1024, y_test_1024   = slice_into_subwindows(X_test_48k, y_test_48k, MERT_WINDOW_SIZE)

    print(f"\n[Resumo do Fatiamento para 1024 pontos]")
    print(f" -> Treino (Domínios 1-8): {len(X_train_48k)} janelas de 1s viraram {len(X_train_1024)} janelas de 1024 pts")
    print(f" -> Teste  (Domínio 9)   : {len(X_test_48k)} janelas de 1s viraram {len(X_test_1024)} janelas de 1024 pts")

    # 4. Avalia tanto sem pré-normalização ('none') quanto com normalização 'rms' por janela (Mert Seção 3.2.3)
    for prep_mode in ["none", "rms"]:
        print(f"\n{'#'*65}\n TESTANDO COM PRÉ-PROCESSAMENTO POR JANELA: {prep_mode.upper()}\n{'#'*65}")

        X_tr_prep = apply_window_preprocessing(X_train_1024, mode=prep_mode)
        X_te_prep = apply_window_preprocessing(X_test_1024, mode=prep_mode)

        # Extração de Features (VibNet + SignAI)
        X_train_fusion = extract_fusion_features(X_tr_prep, FS, extract_advanced_features)
        X_test_fusion  = extract_fusion_features(X_te_prep, FS, extract_advanced_features)

        X_train_clean = np.nan_to_num(np.array(X_train_fusion, dtype=np.float32))
        X_test_clean  = np.nan_to_num(np.array(X_test_fusion, dtype=np.float32))

        if X_train_clean.ndim == 1: X_train_clean = X_train_clean.reshape(len(y_train_1024), -1)
        if X_test_clean.ndim == 1:  X_test_clean = X_test_clean.reshape(len(y_test_1024), -1)

        results = evaluate_all_models(
            X_train_clean, y_train_1024, X_test_clean, y_test_1024,
            dataset_name="CWRU_48k_1024pts", task="diagnosis",
            test_cond="Domain_9 (Load_0HP_Sev_0.021)", prep_mode=prep_mode
        )
        master_results.extend(results)
        pd.DataFrame(master_results).to_csv(csv_filename, index=False)

    print(f"\n{'='*75}\n CONCLUÍDO! Resultados salvos em: {csv_filename}\n{'='*75}")

if __name__ == "__main__":
    run_mert_comparison()
