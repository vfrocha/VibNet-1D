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

# Importando os modelos modulares (incluindo o XGBoost que adicionamos)
from src.models.build_sklearn import get_random_forest, get_svm, get_xgboost, train_and_evaluate
from src.models.build_tabular import get_tabnet_classifier, train_and_evaluate_tabnet
from src.models.build_fttransformer import train_and_evaluate_ft_transformer
from src.models.build_tabnet_resnet import train_and_evaluate_hybrid

# --- CONFIGURAÇÃO DOS DATASETS DE TESTE (BASELINE LOCO) ---
# Aqui nós definimos apenas os datasets alvos da pesquisa.
# ATENÇÃO: Preencha as listas de condições reais de acordo com as pastas do seu sistema.
BASELINE_CONFIGS = {
    "CWRU_12k": {
        "fs": 12000,
        "task": "diagnosis",
        "conditions": ["Load_0HP", "Load_1HP", "Load_2HP", "Load_3HP"]
    },
    "UOEMD": {
        "fs": 10000, # Atualize com o Sampling Rate real da UOEMD
        "task": "diagnosis",
        "conditions": ["Load_Loaded_Speed_15Hz","Load_Loaded_Speed_Dec_45_to_15Hz","Load_No_Load_Speed_15Hz","Load_No_Load_Speed_Dec_45_to_15Hz",
            "Load_Loaded_Speed_30Hz","Load_Loaded_Speed_Dec_60_to_30Hz","Load_No_Load_Speed_30Hz","Load_No_Load_Speed_Dec_60_to_30Hz",
            "Load_Loaded_Speed_45Hz","Load_Loaded_Speed_Inc_15_to_45Hz","Load_No_Load_Speed_45Hz","Load_No_Load_Speed_Inc_15_to_45Hz",
            "Load_Loaded_Speed_60Hz","Load_Loaded_Speed_Inc_30_to_60Hz","Load_No_Load_Speed_60Hz","Load_No_Load_Speed_Inc_30_to_60Hz"]
    },
    "HUST_Gearbox": {
        "fs": 25600, # Atualize com o Sampling Rate real da HUST
        "task": "diagnosis",
        # Como são 30 condições, liste os nomes das pastas aqui:
        "conditions": ["Cond_0"] 
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
        
        # O Loop LOCO: Cada condição se torna o conjunto de teste uma vez
        for test_cond in conditions:
            print(f"\n  >>> Dobra Alvo (Teste): {test_cond} | Treino: Demais Condições")
            
            # 1. Carregamento Inteligente (O Dataloader já separa Treino/Teste pelo LOCO)
            X_train_raw, y_train, X_test_raw, y_test, le = load_vibration_data(
                data_root=DATA_ROOT, dataset_name=dataset_name, test_condition=test_cond, task=task
            )
            
            if len(X_train_raw) == 0:
                print(f"      [Aviso] Dados para {test_cond} não encontrados. Pulando.")
                continue

            # 2. Feature Fusion Modular (VibNet + SignAI)
            X_train_fusion = extract_fusion_features(X_train_raw, fs, extract_advanced_features)
            X_test_fusion  = extract_fusion_features(X_test_raw, fs, extract_advanced_features)

            # 3. Treinamento e Avaliação (Modelos Clássicos State-of-the-Art)
            
            # A) Random Forest
            print(f"     -> Treinando Random Forest...")
            rf_pipeline, rf_grid = get_random_forest()
            rf_acc, rf_f1, _ = train_and_evaluate(rf_pipeline, rf_grid, X_train_fusion, y_train, X_test_fusion, y_test)
            master_results.append({"Dataset": dataset_name, "Task": task.capitalize(), "Test Condition": test_cond, "Model": "Random Forest", "Bal Acc": rf_acc, "Macro F1": rf_f1})

            # B) SVM
            print(f"     -> Treinando SVM...")
            svm_pipeline, svm_grid = get_svm()
            svm_acc, svm_f1, _ = train_and_evaluate(svm_pipeline, svm_grid, X_train_fusion, y_train, X_test_fusion, y_test)
            master_results.append({"Dataset": dataset_name, "Task": task.capitalize(), "Test Condition": test_cond, "Model": "SVM", "Bal Acc": svm_acc, "Macro F1": svm_f1})

            # C) XGBoost
            print(f"     -> Treinando XGBoost...")
            xgb_pipeline, xgb_grid = get_xgboost()
            xgb_acc, xgb_f1, _ = train_and_evaluate(xgb_pipeline, xgb_grid, X_train_fusion, y_train, X_test_fusion, y_test)
            master_results.append({"Dataset": dataset_name, "Task": task.capitalize(), "Test Condition": test_cond, "Model": "XGBoost", "Bal Acc": xgb_acc, "Macro F1": xgb_f1})

            # 4. Modelos de Deep Learning Tabular (Podem usar as mesmas features!)
            # ATENÇÃO: Remova o bloco do FT-Transformer/TabNet se não quiser testá-los nesta etapa
            print(f"     -> Treinando TabNet...")
            try:
                tabnet_acc, tabnet_f1, _ = train_and_evaluate_tabnet(X_train_fusion, y_train, X_test_fusion, y_test, task)
            except Exception:
                tabnet_acc, tabnet_f1 = 0.0, 0.0
            master_results.append({"Dataset": dataset_name, "Task": task.capitalize(), "Test Condition": test_cond, "Model": "TabNet", "Bal Acc": tabnet_acc, "Macro F1": tabnet_f1})

            # Vai salvando no CSV incrementalmente para não perder dados se o PC desligar
            df = pd.DataFrame(master_results)
            df.to_csv(csv_filename, index=False)

    print(f"\n{'='*70}\n EXPERIMENTOS CONCLUÍDOS! Relatório salvo em: {csv_filename}\n{'='*70}")

if __name__ == "__main__":
    run_baselines()
