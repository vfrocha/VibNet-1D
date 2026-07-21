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

# Importando os modelos modulares (incluindo o XGBoost)
from src.models.build_sklearn import get_random_forest, get_svm, get_xgboost, train_and_evaluate
from src.models.build_tabular import get_tabnet_classifier, train_and_evaluate_tabnet

# --- CONFIGURAÇÃO DOS DATASETS DE TESTE (BASELINE LOCO) ---
BASELINE_CONFIGS = {
    "CWRU_12k": {
        "fs": 12000,
        "task": "diagnosis",
        "conditions": ["Load_0HP", "Load_1HP", "Load_2HP", "Load_3HP"]
    },
    "UOEMD": {
        "fs": 10000, # Atualize com o Sampling Rate real da UOEMD
        "task": "diagnosis",
        "conditions": ["Cond_1", "Cond_2", "Cond_3", "Cond_4", "Cond_5", "Cond_6", "Cond_7", "Cond_8"]
    },
    "HUST_gearbox": {
        "fs": 25600, # Atualize com o Sampling Rate real da HUST
        "task": "diagnosis",
        "conditions": ["Speed1_Load1", "Speed1_Load2", "Speed2_Load1"] # Atualize com as pastas reais
    }
}

DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))

def run_baselines():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    csv_filename = os.path.join(RESULTS_DIR, f'baseline_loco_results_{timestamp}.csv')
    
    master_results = []

    print(f"{'='*70}\n INICIANDO EXPERIMENTOS BASELINE (LOCO) - SEM TRANSFER LEARNING\n{'='*70}")

    for dataset_name, config in BASELINE_CONFIGS.items():
        fs = config["fs"]
        task = config["task"]
        conditions = config["conditions"]
        
        print(f"\n[{dataset_name}] Processando {len(conditions)} dobras LOCO (Leave-One-Condition-Out)...")
        
        for test_cond in conditions:
            print(f"\n  >>> Dobra Alvo (Teste): {test_cond} | Treino: Demais Condições")
            
            # 1. Carregamento Inteligente (LOCO)
            X_train_raw, y_train, X_test_raw, y_test, le = load_vibration_data(
                data_root=DATA_ROOT, dataset_name=dataset_name, test_condition=test_cond, task=task
            )
            
            if len(X_train_raw) == 0:
                print(f"      [Aviso] Dados para {test_cond} não encontrados. Pulando.")
                continue

            # 2. Feature Fusion Modular (VibNet + SignAI)
            X_train_fusion = extract_fusion_features(X_train_raw, fs, extract_advanced_features)
            X_test_fusion  = extract_fusion_features(X_test_raw, fs, extract_advanced_features)

            # --- BLINDAGEM DE DADOS: Limpeza Rigorosa ---
            # Remove qualquer resquício do Pandas e converte para float puro
            # Transforma NaNs e Infinitos em 0.0 para não quebrar as redes neurais
            X_train_clean = np.nan_to_num(np.array(X_train_fusion, dtype=np.float32))
            X_test_clean  = np.nan_to_num(np.array(X_test_fusion, dtype=np.float32))
            
            # Garante dimensionalidade 2D [n_amostras, n_features]
            if X_train_clean.ndim == 1:
                X_train_clean = X_train_clean.reshape(len(y_train), -1)
            if X_test_clean.ndim == 1:
                X_test_clean = X_test_clean.reshape(len(y_test), -1)

            # 3. Treinamento e Avaliação Clássica
            
            # A) Random Forest
            print(f"     -> Treinando Random Forest...")
            rf_pipeline, rf_grid = get_random_forest()
            rf_acc, rf_f1, _ = train_and_evaluate(rf_pipeline, rf_grid, X_train_clean, y_train, X_test_clean, y_test)
            master_results.append({"Dataset": dataset_name, "Task": task.capitalize(), "Test Condition": test_cond, "Model": "Random Forest", "Bal Acc": rf_acc, "Macro F1": rf_f1})

            # B) SVM
            print(f"     -> Treinando SVM...")
            svm_pipeline, svm_grid = get_svm()
            svm_acc, svm_f1, _ = train_and_evaluate(svm_pipeline, svm_grid, X_train_clean, y_train, X_test_clean, y_test)
            master_results.append({"Dataset": dataset_name, "Task": task.capitalize(), "Test Condition": test_cond, "Model": "SVM", "Bal Acc": svm_acc, "Macro F1": svm_f1})

            # C) XGBoost
            print(f"     -> Treinando XGBoost...")
            xgb_pipeline, xgb_grid = get_xgboost()
            xgb_acc, xgb_f1, _ = train_and_evaluate(xgb_pipeline, xgb_grid, X_train_clean, y_train, X_test_clean, y_test)
            master_results.append({"Dataset": dataset_name, "Task": task.capitalize(), "Test Condition": test_cond, "Model": "XGBoost", "Bal Acc": xgb_acc, "Macro F1": xgb_f1})

            # 4. Modelos de Deep Learning Tabular (TabNet)
            print(f"     -> Treinando TabNet...")
            try:
                # O USO DE ARGUMENTOS NOMEADOS (kwargs) AQUI É VITAL PARA EVITAR O ERRO ANTERIOR
                tabnet_acc, tabnet_f1, _ = train_and_evaluate_tabnet(
                    X_train=X_train_clean, 
                    y_train=y_train, 
                    X_test=X_test_clean, 
                    y_test=y_test, 
                    task=task
                )
            except Exception as e:
                print(f"        [ERRO FATAL TABNET] O modelo quebrou devido a: {e}")
                import traceback
                traceback.print_exc()
                tabnet_acc, tabnet_f1 = 0.0, 0.0
            
            master_results.append({"Dataset": dataset_name, "Task": task.capitalize(), "Test Condition": test_cond, "Model": "TabNet", "Bal Acc": tabnet_acc, "Macro F1": tabnet_f1})

            # Salvando no CSV incrementalmente
            df = pd.DataFrame(master_results)
            df.to_csv(csv_filename, index=False)

    print(f"\n{'='*70}\n EXPERIMENTOS CONCLUÍDOS! Relatório salvo em: {csv_filename}\n{'='*70}")

if __name__ == "__main__":
    run_baselines()
