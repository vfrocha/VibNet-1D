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

# --- CONFIGURAÇÃO GLOBAL FOCADA ---
# Mantivemos apenas "diagnosis" pois a CWRU 48k não possui classe normal nesta config para "detection"
TASKS = ["diagnosis"] 

FS = 48000
ALL_CONDITIONS = [
    "Load_0HP_Sev_0.007", "Load_0HP_Sev_0.021", "Load_1HP_Sev_0.014", "Load_2HP_Sev_0.007",
    "Load_2HP_Sev_0.021", "Load_3HP_Sev_0.014", "Load_0HP_Sev_0.014", "Load_1HP_Sev_0.007",
    "Load_1HP_Sev_0.021", "Load_2HP_Sev_0.014", "Load_3HP_Sev_0.007", "Load_3HP_Sev_0.021"
]

# As severidades que usaremos como dobras (folds) no Leave-One-Severity-Out
TARGET_SEVERITIES = ["0.007", "0.014", "0.021"]

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))

# --- MÓDULO DE AVALIAÇÃO (Intacto) ---
def evaluate_all_models(X_train, y_train, X_test, y_test, dataset_name, task, test_cond):
    fold_results = []
    base_info = {"Dataset": dataset_name, "Task": task.capitalize(), "Test Condition": test_cond}

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

# --- ORQUESTRADOR PRINCIPAL ---
def run_severity_baselines():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = os.path.join(RESULTS_DIR, f'cwru48k_severity_loco_results_{timestamp}.csv')
    
    master_results = []

    print(f"{'='*70}\n INICIANDO EXPERIMENTOS CWRU 48k (LEAVE-ONE-SEVERITY-OUT)\n{'='*70}")

    for task in TASKS:
        print(f"\n\n{'#'*60}\n TAREFA ATUAL: {task.upper()}\n{'#'*60}")
        
        # 1. Carregamento em Cache (Buscamos todas as 12 condições uma única vez para ganhar tempo)
        print(">>> Carregando e mapeando todas as condições (Cache)...")
        cached_data = {}
        for cond in ALL_CONDITIONS:
            # Aproveitamos o X_test_raw do seu dataloader, que sempre retorna isoladamente a condição solicitada
            _, _, X_cond, y_cond, _ = load_vibration_data(
                data_root=DATA_ROOT, dataset_name="CWRU_48k", test_condition=cond, task=task
            )
            cached_data[cond] = (X_cond, y_cond)

        # 2. Loop pelas Severidades (A nova dobra LOCO)
        for test_sev in TARGET_SEVERITIES:
            print(f"\n >>> Dobra Alvo (Teste): Severidade {test_sev} | Treino: Demais Severidades")
            
            X_train_list, y_train_list = [], []
            X_test_list, y_test_list = [], []
            
            # Agrupamento dinâmico baseado na string da severidade
            for cond in ALL_CONDITIONS:
                X_c, y_c = cached_data[cond]
                
                if len(X_c) == 0:
                    continue
                    
                # Se a severidade alvo (ex: '0.007') estiver no nome da condição, vai pro Teste.
                if test_sev in cond:
                    X_test_list.append(X_c)
                    y_test_list.append(y_c)
                # Caso contrário, vai pro Treino.
                else:
                    X_train_list.append(X_c)
                    y_train_list.append(y_c)
                    
            if not X_train_list or not X_test_list:
                print(f"      [Aviso] Dados insuficientes para severidade {test_sev}. Pulando.")
                continue
                
            # Concatenamos as listas para formar os arrays NumPy finais
            X_train_raw = np.concatenate(X_train_list, axis=0)
            y_train = np.concatenate(y_train_list, axis=0)
            X_test_raw = np.concatenate(X_test_list, axis=0)
            y_test = np.concatenate(y_test_list, axis=0)

            print(f"      [Info] Janelas de Treino: {len(X_train_raw)} | Janelas de Teste: {len(X_test_raw)}")

            # 3. Feature Fusion Modular (VibNet + SignAI)
            X_train_fusion = extract_fusion_features(X_train_raw, FS, extract_advanced_features)
            X_test_fusion  = extract_fusion_features(X_test_raw, FS, extract_advanced_features)

            # 4. Limpeza de Dados (Anti-Pandas / Anti-NaN)
            X_train_clean = np.nan_to_num(np.array(X_train_fusion, dtype=np.float32))
            X_test_clean  = np.nan_to_num(np.array(X_test_fusion, dtype=np.float32))
            
            if X_train_clean.ndim == 1: X_train_clean = X_train_clean.reshape(len(y_train), -1)
            if X_test_clean.ndim == 1:  X_test_clean = X_test_clean.reshape(len(y_test), -1)

            # 5. Avaliação Limpa e Modular
            current_results = evaluate_all_models(
                X_train_clean, y_train, X_test_clean, y_test, 
                dataset_name="CWRU_48k", task=task, test_cond=f"Severity_{test_sev}"
            )
            
            master_results.extend(current_results)

            # 6. Salvamento Incremental
            df = pd.DataFrame(master_results)
            df.to_csv(csv_filename, index=False)

    print(f"\n{'='*70}\n EXPERIMENTOS CONCLUÍDOS! Relatório salvo em: {csv_filename}\n{'='*70}")

if __name__ == "__main__":
    run_severity_baselines()
