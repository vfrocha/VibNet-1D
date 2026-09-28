import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Adiciona a raiz do projeto ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.dataloader import load_vibration_data
from src.features.extractors_v2 import extract_advanced_features
from src.features.signalai_wrapper import extract_fusion_features

# Importando os modelos modulares
from src.models.build_sklearn import get_random_forest, get_svm, get_xgboost, train_and_evaluate
from src.models.build_tabular import get_tabnet_classifier, train_and_evaluate_tabnet
from src.models.build_pytorch_baselines import train_and_evaluate_pure_mlp

# --- CONFIGURAÇÃO GLOBAL ---
TASKS = ["diagnosis"] #"detection"

BASELINE_CONFIGS = {
    "CWRU_12k": {
        "fs": 12000,
        "conditions": ["Load_0HP", "Load_1HP", "Load_2HP", "Load_3HP"]
    },
    "CWRU_48k": {
        "fs": 48000,
        "conditions": ["Load_0HP_Sev_0.007", "Load_0HP_Sev_0.021", "Load_1HP_Sev_0.014", "Load_2HP_Sev_0.007",
                       "Load_2HP_Sev_0.021", "Load_3HP_Sev_0.014", "Load_0HP_Sev_0.014", "Load_1HP_Sev_0.007",
                       "Load_1HP_Sev_0.021",  "Load_2HP_Sev_0.014",  "Load_3HP_Sev_0.007", "Load_3HP_Sev_0.021"] 
    },
    "PU": {
        "fs": 64000,
        "conditions": ["C1_1500rpm_0.7Nm_1000N", "C2_900rpm_0.7Nm_1000N", 
                       "C3_1500rpm_0.1Nm_1000N", "C4_1500rpm_0.7Nm_400N"]
    },
    "HUST_Gearbox": {
        "fs": 25600,
        "conditions": [
            "Cond_20_0", "Cond_20_1", "Cond_20_2", "Cond_20_3", "Cond_20_4",
            "Cond_25_0", "Cond_25_1", "Cond_25_2", "Cond_25_3", "Cond_25_4",
            "Cond_30_0", "Cond_30_1", "Cond_30_2", "Cond_30_3", "Cond_30_4",
            "Cond_35_0", "Cond_35_1", "Cond_35_2", "Cond_35_3", "Cond_35_4",
            "Cond_40_0", "Cond_40_1", "Cond_40_2", "Cond_40_3", "Cond_40_4",
            "Cond_L0_VS_0_40_0", "Cond_L1_VS_0_40_0", "Cond_L2_VS_0_40_0", 
            "Cond_L3_VS_0_40_0", "Cond_L4_VS_0_40_0"
        ]
    }
}

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))

# --- MÓDULO DE AVALIAÇÃO ---
def evaluate_all_models(X_train, y_train, X_test, y_test, dataset_name, task, test_cond):
    """
    Treina e avalia todos os modelos configurados, retornando uma lista de dicionários de resultados.
    """
    fold_results = []
    base_info = {"Dataset": dataset_name, "Task": task.capitalize(), "Test Condition": test_cond}

    # A) Random Forest
    print(f"     -> Treinando Random Forest...")
    try:
        rf_pipeline, rf_grid = get_random_forest()
        rf_acc, rf_bacc, rf_f1, rf_auc, *_ = (rf_pipeline, rf_grid, X_train, y_train, X_test, y_test, task=task)
    except Exception as e:
        print(f"        [AVISO Random Forest] O modelo falhou: {e}")
        rf_acc, rf_bacc, rf_f1, rf_auc = 0.0, 0.0, 0.0, 0.0
    fold_results.append({**base_info, "Model": "Random Forest", "Accuracy": rf_acc, "Bal Acc": rf_bacc, "Macro F1": rf_f1, "ROC-AUC": rf_auc})

    # B) SVM
    print(f"     -> Treinando SVM...")
    try:
        svm_pipeline, svm_grid = get_svm()
        svm_acc, svm_bacc, svm_f1, svm_auc, *_ = (svm_pipeline, svm_grid, X_train, y_train, X_test, y_test, task=task)
    except Exception as e:
        print(f"        [AVISO SVM] O modelo falhou: {e}")
        svm_acc, svm_bacc, svm_f1, svm_auc = 0.0, 0.0, 0.0, 0.0
    fold_results.append({**base_info, "Model": "SVM", "Accuracy": svm_acc, "Bal Acc": svm_bacc, "Macro F1": svm_f1, "ROC-AUC": svm_auc})

    # C) XGBoost
    print(f"     -> Treinando XGBoost...")
    try:
        xgb_pipeline, xgb_grid = get_xgboost()
        xgb_acc, xgb_bacc, xgb_f1, xgb_auc, *_ = (xgb_pipeline, xgb_grid, X_train, y_train, X_test, y_test, task=task)
    except Exception as e:
        print(f"        [AVISO XGBoost] O modelo falhou: {e}")
        xgb_acc, xgb_bacc, xgb_f1, xgb_auc = 0.0, 0.0, 0.0, 0.0
    fold_results.append({**base_info, "Model": "XGBoost", "Accuracy": xgb_acc, "Bal Acc": xgb_bacc, "Macro F1": xgb_f1, "ROC-AUC": xgb_auc})

    # =========================================================================
    # BLINDAGEM DE ESCALA PARA DEEP LEARNING (MLP e TABNET)
    # =========================================================================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # D) TabNet Puro
    print(f"     -> Treinando TabNet Puro...")
    try:
        tabnet_model = get_tabnet_classifier()
        tabnet_acc, tabnet_bacc, tabnet_f1, tabnet_auc, *_ = _tabnet(
            model=tabnet_model, X_train=X_train_scaled, y_train=y_train, X_test=X_test_scaled, y_test=y_test, task=task
        )
    except Exception as e:
        print(f"        [AVISO TABNET] O modelo falhou: {e}")
        tabnet_acc, tabnet_bacc, tabnet_f1, tabnet_auc = 0.0, 0.0, 0.0, 0.0
    fold_results.append({**base_info, "Model": "TabNet", "Accuracy": tabnet_acc, "Bal Acc": tabnet_bacc, "Macro F1": tabnet_f1, "ROC-AUC": tabnet_auc})

    # E) Pure MLP (From Scratch - PyTorch)
    print(f"     -> Treinando PyTorch MLP (From Scratch)...")
    try:
        mlp_acc, mlp_bacc, mlp_f1, mlp_auc, *_ = _pure_mlp(
            X_train_scaled, y_train, X_test_scaled, y_test, task=task
        )
    except Exception as e:
        print(f"        [AVISO MLP] O modelo falhou: {e}")
        mlp_acc, mlp_bacc, mlp_f1, mlp_auc = 0.0, 0.0, 0.0, 0.0
        
    fold_results.append({
        **base_info, 
        "Model": "MLP (Scratch)", 
        "Accuracy": mlp_acc,
        "Bal Acc": mlp_bacc, 
        "Macro F1": mlp_f1, 
        "ROC-AUC": mlp_auc
    })
    
    return fold_results

# --- ORQUESTRADOR PRINCIPAL ---
def run_baselines():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = os.path.join(RESULTS_DIR, f'baseline_loco_results_{timestamp}.csv')
    
    master_results = []

    print(f"{'='*70}\n INICIANDO EXPERIMENTOS BASELINE (LOCO)\n{'='*70}")

    for task in TASKS:
        print(f"\n\n{'#'*60}\n TAREFA ATUAL: {task.upper()}\n{'#'*60}")

        for dataset_name, config in BASELINE_CONFIGS.items():
            
            if task == "detection" and dataset_name == "CWRU_48k":
                print(f"\n[{dataset_name}] Ignorado para Detection (Não possui classe Normal).")
                continue
            
            fs = config["fs"]
            conditions = config["conditions"]
            
            print(f"\n[{dataset_name}] Processando {len(conditions)} dobras LOCO...")
            
            for test_cond in conditions:
                print(f"\n  >>> Dobra Alvo (Teste): {test_cond} | Treino: Demais Condições")
                
                X_train_raw, y_train, X_test_raw, y_test, le = load_vibration_data(
                    data_root=DATA_ROOT, dataset_name=dataset_name, test_condition=test_cond, task=task
                )
                
                if len(X_train_raw) == 0:
                    print(f"      [Aviso] Dados insuficientes para {test_cond} na tarefa {task}. Pulando.")
                    continue

                # 2. Feature Fusion Modular (VibNet + SignAI)
                X_train_fusion = extract_fusion_features(X_train_raw, fs, extract_advanced_features)
                X_test_fusion  = extract_fusion_features(X_test_raw, fs, extract_advanced_features)

                # 3. Limpeza de Dados (Anti-Pandas / Anti-NaN)
                X_train_clean = np.nan_to_num(np.array(X_train_fusion, dtype=np.float32))
                X_test_clean  = np.nan_to_num(np.array(X_test_fusion, dtype=np.float32))
                
                if X_train_clean.ndim == 1: X_train_clean = X_train_clean.reshape(len(y_train), -1)
                if X_test_clean.ndim == 1:  X_test_clean = X_test_clean.reshape(len(y_test), -1)

                # 4. Avaliação Limpa e Modular
                current_results = evaluate_all_models(
                    X_train_clean, y_train, X_test_clean, y_test, dataset_name, task, test_cond
                )
                
                master_results.extend(current_results)

                # 5. Salvamento Incremental
                df = pd.DataFrame(master_results)
                df.to_csv(csv_filename, index=False)

    print(f"\n{'='*70}\n EXPERIMENTOS CONCLUÍDOS! Relatório salvo em: {csv_filename}\n{'='*70}")

if __name__ == "__main__":
    run_baselines()
