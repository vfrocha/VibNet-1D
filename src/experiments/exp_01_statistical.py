import os
import sys
import pandas as pd
from datetime import datetime

# Adiciona a raiz do projeto ao path para o Python achar a pasta 'src'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.dataloader import load_vibration_data
from src.features.extractors import extract_time_features
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

def run_statistical_experiment():
    # Cria a pasta results se não existir
    os.makedirs(RESULTS_DIR, exist_ok=True)
    
    # Configura os arquivos de saída com a data e hora exatas da execução
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(RESULTS_DIR, f"log_exp01_{DATASET}_{timestamp}.txt")
    csv_file = os.path.join(RESULTS_DIR, f"resultados_exp01_{DATASET}_{timestamp}.csv")
    
    # Redireciona o print para o terminal E para o arquivo de log
    sys.stdout = Logger(log_file)
    
    print(f"\n{'='*50}")
    print(f"INICIANDO EXPERIMENTO 1: STATISTICAL (Time Domain)")
    print(f"Dataset: {DATASET}")
    print(f"Arquivo de log: {log_file}")
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

        # 2. Extrai as métricas estatísticas
        X_train_feat = extract_time_features(X_train_raw)
        X_test_feat = extract_time_features(X_test_raw)

        # 3. Treina e avalia os modelos com GridSearch
        
        # --- Random Forest ---
        print("\n  [Iniciando Random Forest]")
        rf_pipeline, rf_grid = get_random_forest()
        rf_acc, rf_f1, rf_best_params = train_and_evaluate(
            rf_pipeline, rf_grid, X_train_feat, y_train, X_test_feat, y_test
        )
        
        results.append({
            "Test Condition": test_cond,
            "Model": "Random Forest",
            "Bal Accuracy": rf_acc,
            "Macro F1": rf_f1,
            "Best Params": str(rf_best_params)
        })

        # --- SVM ---
        print("\n  [Iniciando SVM]")
        svm_pipeline, svm_grid = get_svm()
        svm_acc, svm_f1, svm_best_params = train_and_evaluate(
            svm_pipeline, svm_grid, X_train_feat, y_train, X_test_feat, y_test
        )
        
        results.append({
            "Test Condition": test_cond,
            "Model": "SVM",
            "Bal Accuracy": svm_acc,
            "Macro F1": svm_f1,
            "Best Params": str(svm_best_params)
        })

    # --- Relatório Final e Exportação ---
    if results:
        df_results = pd.DataFrame(results)
        print("\n\n" + "="*50)
        print("RELATÓRIO FINAL")
        print("="*50)
        print(df_results.to_string(index=False))
        
        print("\n--- RESUMO (Média e Desvio Padrão) ---")
        summary = df_results.groupby("Model")[["Bal Accuracy", "Macro F1"]].agg(['mean', 'std'])
        print(summary.to_string())
        
        # Exporta a tabela limpa para CSV
        df_results.to_csv(csv_file, index=False)
        print(f"\n[Sucesso] Tabela de resultados brutos exportada para: {csv_file}")

if __name__ == "__main__":
    run_statistical_experiment()
