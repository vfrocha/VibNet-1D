import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns

# Garante que os imports da pasta src funcionem
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.dataloader import load_vibration_data
from src.features.extractors import extract_time_features, extract_freq_features
from src.models.build_tabular import get_tabnet_classifier, train_and_evaluate_tabnet

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

FEATURE_NAMES = [
    "T_Mean", "T_Std", "T_Skewness", "T_Kurtosis", "T_Peak2Peak", 
    "T_RMS", "T_CrestFactor", "T_ShapeFactor", "T_ImpulseFactor", "T_MarginFactor", 
    "T_Energy", "T_Var", "T_Min", "T_Max", "T_AbsMean", 
    "T_RootAmp", "T_Clearance", "T_Complexity", "T_ZeroCross", 
    "F_Centroid", "F_Bandwidth", "F_Flatness", "F_DomFreq", "F_Rolloff", "F_Entropy" 
]

def run_universal_tabnet(dataset_name, conditions, data_root, results_dir):
    """
    Motor universal que executa a validação cruzada (Leave-One-Out) com TabNet
    para qualquer Dataset fornecido.
    """
    os.makedirs(results_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(results_dir, f"log_tabnet_{dataset_name}_{timestamp}.txt")
    csv_file = os.path.join(results_dir, f"resultados_tabnet_{dataset_name}_{timestamp}.csv")
    
    sys.stdout = Logger(log_file)
    
    print(f"\n{'='*60}")
    print(f"MOTOR VIBNET TABULAR (Autoencoder)")
    print(f"Dataset Alvo: {dataset_name}")
    print(f"Condições (Folds): {conditions}")
    print(f"{'='*60}")

    tasks = ["detection", "diagnosis"]
    results = []
    xai_importances = []

    for task in tasks:
        print(f"\n\n{'='*40}")
        print(f" TAREFA: {task.upper()}")
        print(f"{'='*40}")
        
        for test_cond in conditions:
            print(f"\n--- Fold Inédito: {test_cond} | Tarefa: {task} ---")

            # 1. Carregamento Blindado
            X_train_raw, y_train, X_test_raw, y_test, label_encoder = load_vibration_data(
                data_root=data_root, dataset_name=dataset_name, test_condition=test_cond, task=task
            )

            # Trava de segurança (se a Detecção não achou o Normal)
            if len(X_train_raw) == 0:
                print(f"  [Skip] Abortando fold {test_cond} para a tarefa {task}.")
                continue

            # 2. Extração Estatística
            print("  -> Extraindo características (Tempo + Freq)...")
            X_train_fusion = np.hstack((extract_time_features(X_train_raw), extract_freq_features(X_train_raw)))
            X_test_fusion = np.hstack((extract_time_features(X_test_raw), extract_freq_features(X_test_raw)))

            # 3. Treinamento do TabNet
            model = get_tabnet_classifier()
            bal_acc, macro_f1, roc_auc, trained_model = train_and_evaluate_tabnet(
                model=model, X_train=X_train_fusion, y_train=y_train, 
                X_test=X_test_fusion, y_test=y_test, task=task
            )
            
            if task == 'diagnosis':
                xai_importances.append(trained_model.feature_importances_)
            
            results.append({
                "Dataset": dataset_name,
                "Task": task.capitalize(),
                "Test Condition": test_cond,
                "Bal Accuracy": bal_acc,
                "Macro F1": macro_f1,
                "ROC-AUC": roc_auc
            })

    # Relatório Final
    if results:
        df_results = pd.DataFrame(results)
        print("\n\n" + "="*60)
        print(f"RELATÓRIO FINAL: {dataset_name}")
        print("="*60)
        print(df_results.to_string(index=False))
        
        summary = df_results.groupby("Task")[["Bal Accuracy", "Macro F1", "ROC-AUC"]].agg(['mean', 'std'])
        print(f"\n--- RESUMO ({dataset_name}) ---")
        print(summary.to_string())
        df_results.to_csv(csv_file, index=False)
        print(f"\n[Sucesso] Tabela exportada para: {csv_file}")

        # Gráfico XAI
        if xai_importances:
            mean_importances = np.mean(xai_importances, axis=0)
            df_xai = pd.DataFrame({'Feature': FEATURE_NAMES, 'Importance': mean_importances})
            df_xai = df_xai.sort_values(by='Importance', ascending=False)
            
            plt.figure(figsize=(12, 8))
            sns.set_theme(style="whitegrid")
            sns.barplot(x='Importance', y='Feature', data=df_xai, palette='viridis')
            plt.title(f'TabNet Feature Importance - {dataset_name}', fontsize=16, pad=15)
            plt.xlabel('Mean Attention Weight', fontsize=12)
            plt.ylabel('Features', fontsize=12)
            plt.tight_layout()
            
            xai_plot_file = os.path.join(results_dir, f"xai_{dataset_name}_{timestamp}.png")
            plt.savefig(xai_plot_file, dpi=300)
            plt.close()
            print(f"  [Sucesso] XAI salvo em: {xai_plot_file}")
