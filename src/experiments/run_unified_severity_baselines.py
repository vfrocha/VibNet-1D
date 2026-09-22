import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Adiciona a raiz do projeto ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.dataloader import load_vibration_data
from src.features.extractors_v2 import extract_advanced_features
from src.features.signalai_wrapper import extract_fusion_features

# Importando os modelos modulares
from src.models.build_sklearn import get_random_forest, get_svm, get_xgboost, train_and_evaluate
from src.models.build_tabular import get_tabnet_classifier, train_and_evaluate_tabnet

# --- CONFIGURAÇÃO GLOBAL UNIFICADA ---
TASKS = ["diagnosis"] 

# Configuração dinâmica para abranger as duas bases de dados com suas respectivas frequências e pastas
DATASETS_CONFIG = {
    "CWRU_48k": {
        "fs": 48000,
        "conditions": [
            "Load_0HP_Sev_0.007", "Load_0HP_Sev_0.014", "Load_0HP_Sev_0.021",
            "Load_1HP_Sev_0.007", "Load_1HP_Sev_0.014", "Load_1HP_Sev_0.021",
            "Load_2HP_Sev_0.007", "Load_2HP_Sev_0.014", "Load_2HP_Sev_0.021",
            "Load_3HP_Sev_0.007", "Load_3HP_Sev_0.014", "Load_3HP_Sev_0.021"
        ]
    },
    "CWRU_12k_Severity": {
        "fs": 12000,
        "conditions": [
            "Load_0HP_Sev_0.007", "Load_0HP_Sev_0.014", "Load_0HP_Sev_0.021", "Load_0HP_Sev_0.028",
            "Load_1HP_Sev_0.007", "Load_1HP_Sev_0.014", "Load_1HP_Sev_0.021", "Load_1HP_Sev_0.028",
            "Load_2HP_Sev_0.007", "Load_2HP_Sev_0.014", "Load_2HP_Sev_0.021", "Load_2HP_Sev_0.028",
            "Load_3HP_Sev_0.007", "Load_3HP_Sev_0.014", "Load_3HP_Sev_0.021", "Load_3HP_Sev_0.028"
        ]
    }
}

# Severidades de teste cruzado (0.028 ignorado como alvo pois a 48k não possui essa severidade)
TARGET_SEVERITIES = ["0.007", "0.014", "0.021"]

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))

def _safe_accuracy(y_true, X_test, extra_out):
    """Extrai a acurácia simples com segurança a partir dos retornos da função genérica de treino."""
    try:
        if hasattr(extra_out, 'predict'):
            y_pred = extra_out.predict(X_test)
        else:
            y_pred = extra_out
        return accuracy_score(y_true, y_pred)
    except Exception:
        return 0.0

# --- MÓDULO DE AVALIAÇÃO COM NOVA MÉTRICA ---
def evaluate_all_models(X_train, y_train, X_test, y_test, dataset_name, task, test_cond):
    fold_results = []
    base_info = {"Dataset": dataset_name, "Task": task.capitalize(), "Test Condition": test_cond}

    # A) Random Forest
    print(f"     -> Treinando Random Forest...")
    try:
        rf_pipeline, rf_grid = get_random_forest()
        rf_bacc, rf_f1, rf_auc, rf_extra = train_and_evaluate(rf_pipeline, rf_grid, X_train, y_train, X_test, y_test, task=task)
        rf_acc = _safe_accuracy(y_test, X_test, rf_extra)
    except Exception as e:
        print(f"        [AVISO Random Forest] O modelo falhou: {e}")
        rf_acc, rf_bacc, rf_f1, rf_auc = 0.0, 0.0, 0.0, 0.0
    fold_results.append({**base_info, "Model": "Random Forest", "Accuracy": rf_acc, "Bal Acc": rf_bacc, "Macro F1": rf_f1, "ROC-AUC": rf_auc})

    # B) SVM
    print(f"     -> Treinando SVM...")
    try:
        svm_pipeline, svm_grid = get_svm()
        svm_bacc, svm_f1, svm_auc, svm_extra = train_and_evaluate(svm_pipeline, svm_grid, X_train, y_train, X_test, y_test, task=task)
        svm_acc = _safe_accuracy(y_test, X_test, svm_extra)
    except Exception as e:
        print(f"        [AVISO SVM] O modelo falhou: {e}")
        svm_acc, svm_bacc, svm_f1, svm_auc = 0.0, 0.0, 0.0, 0.0
    fold_results.append({**base_info, "Model": "SVM", "Accuracy": svm_acc, "Bal Acc": svm_bacc, "Macro F1": svm_f1, "ROC-AUC": svm_auc})

    # C) XGBoost
    print(f"     -> Treinando XGBoost...")
    try:
        xgb_pipeline, xgb_grid = get_xgboost()
        xgb_bacc, xgb_f1, xgb_auc, xgb_extra = train_and_evaluate(xgb_pipeline, xgb_grid, X_train, y_train, X_test, y_test, task=task)
        xgb_acc = _safe_accuracy(y_test, X_test, xgb_extra)
    except Exception as e:
        print(f"        [AVISO XGBoost] O modelo falhou: {e}")
        xgb_acc, xgb_bacc, xgb_f1, xgb_auc = 0.0, 0.0, 0.0, 0.0
    fold_results.append({**base_info, "Model": "XGBoost", "Accuracy": xgb_acc, "Bal Acc": xgb_bacc, "Macro F1": xgb_f1, "ROC-AUC": xgb_auc})

    # D) TabNet
    print(f"     -> Treinando TabNet...")
    try:
        tabnet_model = get_tabnet_classifier()
        tabnet_bacc, tabnet_f1, tabnet_auc, tabnet_extra = train_and_evaluate_tabnet(
            model=tabnet_model, X_train=X_train, y_train=y_train, X_test=X_test, y_test=y_test, task=task
        )
        tabnet_acc = _safe_accuracy(y_test, X_test, tabnet_extra)
    except Exception as e:
        print(f"        [AVISO TABNET] O modelo falhou: {e}")
        tabnet_acc, tabnet_bacc, tabnet_f1, tabnet_auc = 0.0, 0.0, 0.0, 0.0
    
    fold_results.append({**base_info, "Model": "TabNet", "Accuracy": tabnet_acc, "Bal Acc": tabnet_bacc, "Macro F1": tabnet_f1, "ROC-AUC": tabnet_auc})

    return fold_results

# --- ORQUESTRADOR PRINCIPAL ---
def run_unified_severity_baselines():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = os.path.join(RESULTS_DIR, f'cwru_unified_severity_loco_{timestamp}.csv')
    
    master_results = []

    print(f"{'='*80}\n EXPERIMENTOS UNIFICADOS CWRU (12k e 48k) - SEVERITY LOCO\n{'='*80}")

    for task in TASKS:
        print(f"\n\n{'#'*60}\n TAREFA ATUAL: {task.upper()}\n{'#'*60}")
        
        for dataset_name, config in DATASETS_CONFIG.items():
            fs = config["fs"]
            conditions = config["conditions"]
            
            print(f"\n{'='*50}\n PROCESSANDO DATASET: {dataset_name} (fs={fs}Hz)\n{'='*50}")
            
            # 1. Carregamento em Cache Local por Dataset
            print(">>> Carregando e mapeando condições (Cache)...")
            cached_data = {}
            for cond in conditions:
                _, _, X_cond, y_cond, _ = load_vibration_data(
                    data_root=DATA_ROOT, dataset_name=dataset_name, test_condition=cond, task=task
                )
                cached_data[cond] = (X_cond, y_cond)

            # 2. Loop pelas Severidades (Dobra LOCO)
            for test_sev in TARGET_SEVERITIES:
                print(f"\n >>> Dobra Alvo (Teste): Severidade {test_sev} | Treino: Demais Severidades")
                
                X_train_list, y_train_list = [], []
                X_test_list, y_test_list = [], []
                
                for cond in conditions:
                    X_c, y_c = cached_data[cond]
                    if len(X_c) == 0:
                        continue
                        
                    if test_sev in cond:
                        X_test_list.append(X_c)
                        y_test_list.append(y_c)
                    else:
                        X_train_list.append(X_c)
                        y_train_list.append(y_c)
                        
                if not X_train_list or not X_test_list:
                    print(f"      [Aviso] Dados insuficientes para severidade {test_sev}. Pulando.")
                    continue
                    
                X_train_raw = np.concatenate(X_train_list, axis=0)
                y_train_raw = np.concatenate(y_train_list, axis=0)
                X_test_raw = np.concatenate(X_test_list, axis=0)
                y_test_raw = np.concatenate(y_test_list, axis=0)

                print(f"      [Info] Janelas de Treino: {len(X_train_raw)} | Janelas de Teste: {len(X_test_raw)}")

                # 3. Feature Fusion Modular (VibNet + SignAI)
                X_train_fusion = extract_fusion_features(X_train_raw, fs, extract_advanced_features)
                X_test_fusion  = extract_fusion_features(X_test_raw, fs, extract_advanced_features)

                # 4. Limpeza de Dados
                X_train_clean = np.nan_to_num(np.array(X_train_fusion, dtype=np.float32))
                X_test_clean  = np.nan_to_num(np.array(X_test_fusion, dtype=np.float32))
                
                if X_train_clean.ndim == 1: X_train_clean = X_train_clean.reshape(len(y_train_raw), -1)
                if X_test_clean.ndim == 1:  X_test_clean = X_test_clean.reshape(len(y_test_raw), -1)

                # 5. Mapeamento Estrito de Rótulos [0, 1, 2] (Blindagem para XGBoost e TabNet)
                le = LabelEncoder()
                y_train_enc = le.fit_transform(y_train_raw)
                
                # Retém apenas amostras no teste cujas classes foram vistas no treino
                valid_test_idx = [i for i, lbl in enumerate(y_test_raw) if lbl in le.classes_]
                if len(valid_test_idx) == 0:
                    continue
                X_test_clean = X_test_clean[valid_test_idx]
                y_test_enc = le.transform(y_test_raw[valid_test_idx])

                # 6. Avaliação
                current_results = evaluate_all_models(
                    X_train_clean, y_train_enc, X_test_clean, y_test_enc, 
                    dataset_name=dataset_name, task=task, test_cond=f"Severity_{test_sev}"
                )
                
                master_results.extend(current_results)

                # 7. Salvamento Incremental Unificado
                df = pd.DataFrame(master_results)
                df.to_csv(csv_filename, index=False)

    print(f"\n{'='*80}\n [SUCESSO] EXPERIMENTOS CONCLUÍDOS! Relatório final unificado salvo em: {csv_filename}\n{'='*80}")

if __name__ == "__main__":
    run_unified_severity_baselines()
