import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

# Adiciona a raiz do projeto ao path
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

# --- CONFIGURAÇÃO GLOBAL DOS DATASETS ---
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))

# Mapeamento de todas as bases e suas dobras (Ajuste os nomes se necessário)
ALL_DATASETS = {
    "CWRU_12k": ["Load_0HP", "Load_1HP", "Load_2HP", "Load_3HP"],
    "HUST_Dataset": ["Load_0W", "Load_200W", "Load_400W"],
    "PU_Dataset": ["C1_1500rpm_0.7Nm_1000N", "C2_900rpm_0.7Nm_1000N", "C3_1500rpm_0.1Nm_1000N", "C4_1500rpm_0.7Nm_400N"],
    "UORED": { # Grupos Virtuais
        "Group_A": ["Bearing_1", "Bearing_6", "Bearing_11", "Bearing_16"],
        "Group_B": ["Bearing_2", "Bearing_7", "Bearing_12", "Bearing_17"],
        "Group_C": ["Bearing_3", "Bearing_8", "Bearing_13", "Bearing_18"],
        "Group_D": ["Bearing_4", "Bearing_9", "Bearing_14", "Bearing_19"],
        "Group_E": ["Bearing_5", "Bearing_10", "Bearing_15", "Bearing_20"]
    }
}

TASKS = ["detection", "diagnosis"]

def evaluate_sklearn_model(model, X_train, y_train, X_test, y_test, task, model_name):
    """Função unificada para avaliar Modelos Clássicos (RF, SVM) com as mesmas métricas do TabNet"""
    print(f"     -> Treinando {model_name}...")
    
    # Normalização Padrão
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    model.fit(X_train_s, y_train)
    
    y_pred = model.predict(X_test_s)
    y_pred_probs = model.predict_proba(X_test_s)
    
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    
    if task == 'detection':
        roc_auc = roc_auc_score(y_test, y_pred_probs[:, 1])
        macro_f1 = f1_score(y_test, y_pred, average='binary')
    else:
        try:
            roc_auc = roc_auc_score(y_test, y_pred_probs, multi_class='ovr')
        except ValueError:
            roc_auc = 0.0
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        
    print(f"        [{model_name}] Bal Acc: {bal_acc:.4f} | F1: {macro_f1:.4f} | ROC-AUC: {roc_auc:.4f}")
    return bal_acc, macro_f1, roc_auc


def run_master_orchestrator():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(RESULTS_DIR, f"log_master_strategy1_{timestamp}.txt")
    csv_file = os.path.join(RESULTS_DIR, f"resultados_master_strategy1_{timestamp}.csv")
    sys.stdout = Logger(log_file)
    
    print(f"{'='*70}\nORQUESTRADOR MASTER: ESTRATÉGIA 1 (Baseline Comparativo)\n{'='*70}")

    master_results = []

    # 1. Loop de Datasets
    for dataset_name, conditions in ALL_DATASETS.items():
        print(f"\n\n{'#'*60}\n INICIANDO DATASET: {dataset_name}\n{'#'*60}")
        
        # Prepara a iteração (suporta Listas simples ou Dicionários de Grupos Virtuais)
        iterable_conditions = conditions.items() if isinstance(conditions, dict) else [(c, c) for c in conditions]
        
        # 2. Loop de Tarefas
        for task in TASKS:
            print(f"\n{'='*40}\n TAREFA: {task.upper()}\n{'='*40}")
            
            # 3. Loop de Dobras (Folds)
            for fold_name, test_cond_val in iterable_conditions:
                print(f"\n--- Fold: {fold_name} ---")

                # Carrega os dados da dobra atual
                X_train_raw, y_train, X_test_raw, y_test, le = load_vibration_data(
                    data_root=DATA_ROOT, dataset_name=dataset_name, test_condition=test_cond_val, task=task
                )

                if len(X_train_raw) == 0:
                    print(f"  [Skip] Dados insuficientes para {fold_name}.")
                    continue

                # Extração Única de Features para TODOS os modelos
                print("  -> Extraindo características estatísticas (Tempo + Freq)...")
                X_train_fusion = np.hstack((extract_time_features(X_train_raw), extract_freq_features(X_train_raw)))
                X_test_fusion = np.hstack((extract_time_features(X_test_raw), extract_freq_features(X_test_raw)))

                # --- 4. A BATALHA DOS MODELOS ---
                
                # A) Random Forest Clássico
                rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
                rf_acc, rf_f1, rf_auc = evaluate_sklearn_model(rf, X_train_fusion, y_train, X_test_fusion, y_test, task, "Random Forest")
                master_results.append({"Dataset": dataset_name, "Task": task.capitalize(), "Fold": fold_name, "Model": "Random Forest", "Bal Acc": rf_acc, "Macro F1": rf_f1, "ROC-AUC": rf_auc})
                
                # B) SVM Clássico (probability=True exigido para o ROC-AUC)
                svm = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
                svm_acc, svm_f1, svm_auc = evaluate_sklearn_model(svm, X_train_fusion, y_train, X_test_fusion, y_test, task, "SVM (RBF)")
                master_results.append({"Dataset": dataset_name, "Task": task.capitalize(), "Fold": fold_name, "Model": "SVM (RBF)", "Bal Acc": svm_acc, "Macro F1": svm_f1, "ROC-AUC": svm_auc})
                
                # C) TabNet (Deep Learning Tabular Proposto)
                print(f"     -> Treinando TabNet...")
                tabnet_model = get_tabnet_classifier()
                tab_acc, tab_f1, tab_auc, _ = train_and_evaluate_tabnet(
                    model=tabnet_model, X_train=X_train_fusion, y_train=y_train, 
                    X_test=X_test_fusion, y_test=y_test, task=task
                )
                master_results.append({"Dataset": dataset_name, "Task": task.capitalize(), "Fold": fold_name, "Model": "TabNet", "Bal Acc": tab_acc, "Macro F1": tab_f1, "ROC-AUC": tab_auc})

    # --- RELATÓRIO FINAL ---
    if master_results:
        df = pd.DataFrame(master_results)
        df.to_csv(csv_file, index=False)
        print(f"\n\n[SUCESSO ABSOLUTO] Tabela Master exportada para: {csv_file}")
        
        # Exibe um resumo incrível formatado por Dataset e Modelo
        print("\n--- RESUMO GERAL (MACRO F1 MÉDIO POR MODELO) ---")
        summary = df[df['Task'] == 'Diagnosis'].groupby(['Dataset', 'Model'])['Macro F1'].mean().unstack()
        print(summary.to_string())

if __name__ == "__main__":
    run_master_orchestrator()
