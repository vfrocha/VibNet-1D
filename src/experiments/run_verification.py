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

# --- CONFIGURAÇÃO GLOBAL ---
TASKS = ["diagnosis"]

BASELINE_CONFIGS = {
    # CWRU_12k readicionado para comparação direta
    "CWRU_12k": {
        "fs": 12000,
        "conditions": ["Load_0HP", "Load_1HP", "Load_2HP", "Load_3HP"]
    },
    # CWRU_48k com divisão focada apenas no Load (ignorando severidade)
    "CWRU_48k": {
        "fs": 48000,
        "conditions": ["Load_0HP", "Load_1HP", "Load_2HP", "Load_3HP"]
    }
}

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))

# --- MÓDULO DE AVALIAÇÃO ---
def evaluate_all_models(X_train, y_train, X_test, y_test, dataset_name, task, test_cond, pipeline_name):
    """
    Treina e avalia todos os modelos configurados, retornando uma lista de dicionários de resultados.
    """
    fold_results = []
    base_info = {"Dataset": dataset_name, "Task": task.capitalize(), "Test Condition": test_cond, "Pipeline": pipeline_name,}

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

    return fold_results

# --- ORQUESTRADOR PRINCIPAL ---
def run_baselines():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = os.path.join(RESULTS_DIR, f'comparativo_signai_cwru12k_48k_results_{timestamp}.csv')
    
    master_results = []

    print(f"{'='*70}\n INICIANDO EXPERIMENTOS COMPARATIVOS (SIGNAI ONLY - LOCO)\n{'='*70}")

    for task in TASKS:
        print(f"\n\n{'#'*60}\n TAREFA ATUAL: {task.upper()}\n{'#'*60}")

        for dataset_name, config in BASELINE_CONFIGS.items():
            
            # --- REGRA DE EXCEÇÃO ---
            if task == "detection" and dataset_name == "CWRU_48k":
                print(f"\n[{dataset_name}] Ignorado para Detection (Não possui classe Normal).")
                continue
            # ------------------------------------
            
            fs = config["fs"]
            conditions = config["conditions"]
            
            print(f"\n[{dataset_name}] Processando {len(conditions)} dobras LOCO...")
            
            for test_cond in conditions:
                print(f"\n  >>> Dobra Alvo (Teste): {test_cond} | Treino: Demais Condições")
                
                # 1. Carregamento Inteligente (LOCO)
                X_train_raw, y_train, X_test_raw, y_test, le = load_vibration_data(
                    data_root=DATA_ROOT, dataset_name=dataset_name, test_condition=test_cond, task=task
                )
                
                if len(X_train_raw) == 0:
                    print(f"      [Aviso] Dados insuficientes para {test_cond} na tarefa {task}. Pulando.")
                    continue

                pipeline_name = "time_and_frequency"
                
                # 2. Feature Fusion Modular (Extrai tudo mas filtraremos depois)
                X_train_fusion = extract_fusion_features(X_train_raw, fs, extract_advanced_features, pipeline_name)
                X_test_fusion  = extract_fusion_features(X_test_raw, fs, extract_advanced_features, pipeline_name)

                # 3. Limpeza de Dados (Anti-Pandas / Anti-NaN)
                X_train_clean = np.nan_to_num(np.array(X_train_fusion, dtype=np.float32))
                X_test_clean  = np.nan_to_num(np.array(X_test_fusion, dtype=np.float32))
                
                if X_train_clean.ndim == 1: X_train_clean = X_train_clean.reshape(len(y_train), -1)
                if X_test_clean.ndim == 1:  X_test_clean = X_test_clean.reshape(len(y_test), -1)

                # ====================================================================
                # ADAPTAÇÃO: MANTER APENAS AS FEATURES DO SIGNAI
                # O array fusionado tem 141 features (16 do VibNet + 125 do SignAI).
                # Fatiamos para excluir as 16 primeiras e reter apenas as 125 do SignAI.
                # ====================================================================
                X_train_clean = X_train_clean[:, 16:]
                X_test_clean  = X_test_clean[:, 16:]
                print(f"      [SignAI Wrapper] Features isoladas: Shape atual de treino {X_train_clean.shape}")

                # 4. Avaliação Limpa e Modular
                current_results = evaluate_all_models(
                    X_train_clean, y_train, X_test_clean, y_test, dataset_name, task, test_cond, pipeline_name
                )
                
                master_results.extend(current_results)

                # 5. Salvamento Incremental
                df = pd.DataFrame(master_results)
                df.to_csv(csv_filename, index=False)

    print(f"\n{'='*70}\n EXPERIMENTOS CONCLUÍDOS! Relatório salvo em: {csv_filename}\n{'='*70}")

if __name__ == "__main__":
    run_baselines()
