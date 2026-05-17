import os
import sys
import pandas as pd

# Adiciona a raiz do projeto ao path para o Python achar a pasta 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.dataloader import load_vibration_data
from src.features.extractors import extract_time_features
from src.models.build_sklearn import get_random_forest, get_svm, train_and_evaluate

# --- CONFIGURAÇÕES ---
# Garante que o caminho aponte para VibNet-1D/data/processed
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
DATASET = "CWRU_12k" # Mude para PU, UORED, ou HUST quando quiser testar os outros

# Condições baseadas na função get_names que fizemos no dataloader
CONDITIONS = ["Load_0HP", "Load_1HP", "Load_2HP", "Load_3HP"]

def run_statistical_experiment():
    print(f"\n{'='*50}")
    print(f"INICIANDO EXPERIMENTO 1: STATISTICAL (Time Domain)")
    print(f"Dataset: {DATASET}")
    print(f"{'='*50}")

    results = []

    # Loop Leave-One-Load-Out (Cross-Validation sem viés de similaridade)
    for test_cond in CONDITIONS:
        print(f"\n--- Fold: Testando na condição inédita {test_cond} ---")

        # 1. Carrega os sinais 1D brutos
        X_train_raw, y_train, X_test_raw, y_test, label_encoder = load_vibration_data(
            data_root=DATA_ROOT,
            dataset_name=DATASET,
            test_condition=test_cond
        )

        if len(X_train_raw) == 0 or len(X_test_raw) == 0:
            print(f"  [Aviso] Dados insuficientes para {test_cond}. Verifique as pastas.")
            continue

        # 2. Extrai as métricas estatísticas (Transforma vetor longo em vetor curto de features)
        X_train_feat = extract_time_features(X_train_raw)
        X_test_feat = extract_time_features(X_test_raw)

        # 3. Treina e avalia os modelos "Caixa Preta"
        
        # --- Random Forest ---
        rf_model = get_random_forest()
        rf_acc, rf_f1, _ = train_and_evaluate(rf_model, X_train_feat, y_train, X_test_feat, y_test)
        
        results.append({
            "Test Condition": test_cond,
            "Model": "Random Forest",
            "Bal Accuracy": rf_acc,
            "Macro F1": rf_f1
        })

        # --- SVM ---
        svm_model = get_svm()
        svm_acc, svm_f1, _ = train_and_evaluate(svm_model, X_train_feat, y_train, X_test_feat, y_test)
        
        results.append({
            "Test Condition": test_cond,
            "Model": "SVM",
            "Bal Accuracy": svm_acc,
            "Macro F1": svm_f1
        })

    # --- Relatório Final ---
    if results:
        df_results = pd.DataFrame(results)
        print("\n\n" + "="*50)
        print("RELATÓRIO FINAL")
        print("="*50)
        print(df_results.to_string(index=False))
        
        print("\n--- RESUMO (Média e Desvio Padrão) ---")
        summary = df_results.groupby("Model")[["Bal Accuracy", "Macro F1"]].agg(['mean', 'std'])
        print(summary.to_string())

if __name__ == "__main__":
    run_statistical_experiment()
