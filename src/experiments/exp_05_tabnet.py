import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Adiciona a raiz do projeto ao path para o Python achar a pasta 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.dataloader import load_vibration_data
from src.features.extractors import extract_time_features, extract_freq_features
from src.models.build_tabular import get_tabnet_classifier, train_and_evaluate_tabnet

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

# Aqui definimos as duas tarefas exigidas pelo esqueleto do artigo!
TASKS = ["detection", "diagnosis"] 

def run_tabnet_experiment():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(RESULTS_DIR, f"log_exp05_tabnet_{DATASET}_{timestamp}.txt")
    csv_file = os.path.join(RESULTS_DIR, f"resultados_exp05_tabnet_{DATASET}_{timestamp}.csv")
    
    sys.stdout = Logger(log_file)
    
    print(f"\n{'='*60}")
    print(f"EXPERIMENTO 5: DEEP LEARNING TABULAR (TabNet + Fusion)")
    print(f"Dataset: {DATASET}")
    print(f"Arquivo de log: {log_file}")
    print(f"{'='*60}")

    results = []

    # Loop 1: Alterna entre Tarefa de Detecção e Diagnóstico
    for task in TASKS:
        print(f"\n\n{'='*40}")
        print(f" INICIANDO TAREFA: {task.upper()}")
        print(f"{'='*40}")
        
        # Loop 2: Leave-One-Load-Out
        for test_cond in CONDITIONS:
            print(f"\n--- Fold: Condição inédita {test_cond} | Tarefa: {task} ---")

            # 1. Carrega os sinais baseados na Tarefa atual
            X_train_raw, y_train, X_test_raw, y_test, label_encoder = load_vibration_data(
                data_root=DATA_ROOT, dataset_name=DATASET, test_condition=test_cond, task=task
            )

            if len(X_train_raw) == 0 or len(X_test_raw) == 0:
                print(f"  [Aviso] Dados insuficientes para {test_cond}. Pulando...")
                continue

            # 2. Extração de Características (O TabNet precisa da matriz pronta)
            print("  -> Extraindo características de Tempo e Frequência...")
            X_train_time = extract_time_features(X_train_raw)
            X_test_time = extract_time_features(X_test_raw)

            X_train_freq = extract_freq_features(X_train_raw, fs=12000)
            X_test_freq = extract_freq_features(X_test_raw, fs=12000)

            # Fusão de todos os descritores estatísticos
            X_train_fusion = np.hstack((X_train_time, X_train_freq))
            X_test_fusion = np.hstack((X_test_time, X_test_freq))

            # 3. Treina e avalia o TabNet
            model = get_tabnet_classifier()
            
            bal_acc, macro_f1, roc_auc, trained_model = train_and_evaluate_tabnet(
                model=model, 
                X_train=X_train_fusion, 
                y_train=y_train, 
                X_test=X_test_fusion, 
                y_test=y_test, 
                task=task
            )
            
            results.append({
                "Task": task.capitalize(),
                "Test Condition": test_cond,
                "Model": "TabNet (Fusion)",
                "Bal Accuracy": bal_acc,
                "Macro F1": macro_f1,
                "ROC-AUC": roc_auc
            })

    # --- Relatório Final ---
    if results:
        df_results = pd.DataFrame(results)
        print("\n\n" + "="*60)
        print("RELATÓRIO FINAL - TABNET (AUTOENCODER TABULAR)")
        print("="*60)
        print(df_results.to_string(index=False))
        
        print("\n--- RESUMO POR TAREFA (Média e Desvio Padrão) ---")
        summary = df_results.groupby("Task")[["Bal Accuracy", "Macro F1", "ROC-AUC"]].agg(['mean', 'std'])
        print(summary.to_string())
        
        df_results.to_csv(csv_file, index=False)
        print(f"\n[Sucesso] Tabela de resultados TabNet exportada para: {csv_file}")

if __name__ == "__main__":
    run_tabnet_experiment()
