import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Adiciona a raiz do projeto ao path para o Python achar a pasta 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.dataloader import load_vibration_data
# Importamos AMBOS os extratores
from src.features.extractors import extract_time_features, extract_freq_features
from src.models.build_sklearn import get_random_forest, get_svm, train_and_evaluate

# --- LOGGER PARA SALVAR NO TERMINAL E NO ARQUIVO ---
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding='utf-8')
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

# --- CONFIGURAÇÕES ---
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))
DATASET = "CWRU_12k"
CONDITIONS = ["Load_0HP", "Load_1HP", "Load_2HP", "Load_3HP"]

def run_statistical_fusion_experiment():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(RESULTS_DIR, f"log_exp04_fusion_{DATASET}_{timestamp}.txt")
    csv_file = os.path.join(RESULTS_DIR, f"resultados_exp04_fusion_{DATASET}_{timestamp}.csv")
    
    sys.stdout = Logger(log_file)
    
    print(f"\n{'='*50}")
    print(f"INICIANDO EXPERIMENTO 4: STATISTICAL FUSION (Time + Frequency)")
    print(f"Dataset: {DATASET}")
    print(f"Arquivo de log: {log_file}")
    print(f"{'='*50}")

    results = []

    for test_cond in CONDITIONS:
        print(f"\n--- Fold: Testando na condição inédita {test_cond} ---")

        # 1. Carrega os sinais 1D brutos
        X_train_raw, y_train, X_test_raw, y_test, label_encoder = load_vibration_data(
            data_root=DATA_ROOT, dataset_name=DATASET, test_condition=test_cond
        )

        if len(X_train_raw) == 0 or len(X_test_raw) == 0:
            print(f"  [Aviso] Dados insuficientes para {test_cond}. Verifique as pastas.")
            continue

        # 2. Extração Dupla (Tempo e Frequência)
        print("  -> Extraindo características de Tempo...")
        X_train_time = extract_time_features(X_train_raw)
        X_test_time = extract_time_features(X_test_raw)

        print("  -> Extraindo características de Frequência...")
        X_train_freq = extract_freq_features(X_train_raw, fs=12000)
        X_test_freq = extract_freq_features(X_test_raw, fs=12000)

        # 3. A Fusão (Concatenando as colunas)
        # O np.hstack junta matrizes lado a lado. 
        # Ex: (N, 19) + (N, 6) = (N, 25 features)
        X_train_fusion = np.hstack((X_train_time, X_train_freq))
        X_test_fusion = np.hstack((X_test_time, X_test_freq))
        
        print(f"  -> Shape final do vetor de fusão: Treino {X_train_fusion.shape}, Teste {X_test_fusion.shape}")

        # 4. Treina e avalia os modelos com GridSearch
        
        # --- Random Forest ---
        print("\n  [Iniciando Random Forest - Fusion]")
        rf_pipeline, rf_grid = get_random_forest()
        rf_acc, rf_f1, rf_best_params = train_and_evaluate(
            rf_pipeline, rf_grid, X_train_fusion, y_train, X_test_fusion, y_test
        )
        
        results.append({
            "Test Condition": test_cond,
            "Model": "Random Forest (Fusion)",
            "Bal Accuracy": rf_acc,
            "Macro F1": rf_f1,
            "Best Params": str(rf_best_params)
        })

        # --- SVM ---
        print("\n  [Iniciando SVM - Fusion]")
        svm_pipeline, svm_grid = get_svm()
        svm_acc, svm_f1, svm_best_params = train_and_evaluate(
            svm_pipeline, svm_grid, X_train_fusion, y_train, X_test_fusion, y_test
        )
        
        results.append({
            "Test Condition": test_cond,
            "Model": "SVM (Fusion)",
            "Bal Accuracy": svm_acc,
            "Macro F1": svm_f1,
            "Best Params": str(svm_best_params)
        })

    # --- Relatório Final ---
    if results:
        df_results = pd.DataFrame(results)
        print("\n\n" + "="*50)
        print("RELATÓRIO FINAL - FUSÃO DE CARACTERÍSTICAS")
        print("="*50)
        print(df_results.to_string(index=False))
        
        print("\n--- RESUMO (Média e Desvio Padrão) ---")
        summary = df_results.groupby("Model")[["Bal Accuracy", "Macro F1"]].agg(['mean', 'std'])
        print(summary.to_string())
        
        df_results.to_csv(csv_file, index=False)
        print(f"\n[Sucesso] Tabela de resultados de Fusão exportada para: {csv_file}")

if __name__ == "__main__":
    run_statistical_fusion_experiment()
