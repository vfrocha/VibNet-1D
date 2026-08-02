import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Importações do Scikit-Learn diretas para não depender do build_sklearn.py
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

# Adiciona a raiz do projeto ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.features.extractors_v2 import extract_advanced_features
from src.features.signalai_wrapper import extract_fusion_features

# --- CONFIGURAÇÃO GLOBAL ---
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))

# Expandido para usar todas as 9 bases que possuem a classe "Normal" mapeada!
DATASETS_CONFIG = {
    "CWRU_12k": 12000,
    "CWRU_48k": 48000,
    "UOEMD": 42000,
    "HUST_Gearbox": 25600,
    "HUST": 51200,
    "PU": 64000,
    "UORED": 200000,          # ou 42000, dependendo do FS que você definiu no dataloader
    "Mechanical_Gear": 5000,
    "Electric_Motor": 50000
}

TASK = "detection" 

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

def load_entire_dataset_for_tl(dataset_name, fs):
    """
    Carrega TODOS os arquivos de uma base de dados específica.
    Converte os rótulos para Binário (0 = Normal, 1 = Fault).
    """
    dataset_path = os.path.join(DATA_ROOT, dataset_name)
    if not os.path.exists(dataset_path):
        print(f"[Aviso] Dataset {dataset_name} não encontrado no disco.")
        return [], []
        
    X_raw = []
    y_raw = []
    
    # Percorre todas as pastas (Condições) e subpastas (Classes)
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.npy'):
                class_name = os.path.basename(root)
                file_path = os.path.join(root, file)
                
                # Mapeamento Universal Binário
                if 'normal' in class_name.lower() or 'healthy' in class_name.lower():
                    label = 0 # Saudável
                else:
                    label = 1 # Falha
                    
                # CWRU_48k não tem dados normais, pegamos só as falhas para usar no treino
                if dataset_name == "CWRU_48k" and label == 0:
                    continue 
                    
                X_raw.append(np.load(file_path))
                y_raw.append(label)
                
    return X_raw, y_raw

def evaluate_transfer_models(X_train, y_train, X_test, y_test, source_name, target_name):
    """
    Avalia a generalização treinando nas origens (source) e testando no alvo (target).
    """
    results = []
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "SVM (RBF)": SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    for model_name, model in models.items():
        print(f"     -> Treinando {model_name}...")
        try:
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            y_probs = model.predict_proba(X_test_s)
            
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average='binary')
            
            try:
                roc_auc = roc_auc_score(y_test, y_probs[:, 1])
            except ValueError:
                roc_auc = 0.0 # Ocorre se o Target tiver apenas 1 classe presente no teste
                
            print(f"        [{model_name}] Bal Acc: {bal_acc:.4f} | F1: {macro_f1:.4f} | ROC-AUC: {roc_auc:.4f}")
            
            results.append({
                "Source Domains": source_name,
                "Target Domain": target_name,
                "Task": "Detection",
                "Model": model_name,
                "Bal Acc": bal_acc,
                "Macro F1": macro_f1,
                "ROC-AUC": roc_auc
            })
        except Exception as e:
            print(f"        [ERRO] Falha ao treinar {model_name}: {e}")
            
    return results

def run_transfer_learning():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(RESULTS_DIR, f"log_transfer_learning_{timestamp}.txt")
    csv_file = os.path.join(RESULTS_DIR, f"tl_results_{timestamp}.csv")
    sys.stdout = Logger(log_file)
    
    print(f"{'='*80}\nEXPERIMENTOS DE TRANSFER LEARNING (MULTI-SOURCE -> SINGLE TARGET)\n{'='*80}")
    print("Tarefa: DETECTION (0 = Normal, 1 = Fault)")
    print(f"Datsets disponíveis: {list(DATASETS_CONFIG.keys())}\n")

    master_results = []
    db_features = {}
    db_labels = {}
    
    print("--- FASE 1: EXTRAÇÃO DE FEATURES DE TODAS AS MÁQUINAS ---")
    for ds_name, fs in DATASETS_CONFIG.items():
        X_raw, y_raw = load_entire_dataset_for_tl(ds_name, fs)
        if len(X_raw) > 0:
            # FIX AQUI: Converter lista para NumPy Array para compatibilidade com o SignAI Wrapper
            X_raw = np.array(X_raw)
            
            print(f"  -> {ds_name} carregado: {X_raw.shape[0]} janelas. Extraindo 141 features (fs={fs}Hz)...")
            X_fusion = extract_fusion_features(X_raw, fs, extract_advanced_features)
            
            X_clean = np.nan_to_num(np.array(X_fusion, dtype=np.float32))
            if X_clean.ndim == 1: X_clean = X_clean.reshape(len(y_raw), -1)
            
            db_features[ds_name] = X_clean
            db_labels[ds_name] = np.array(y_raw)
        else:
            print(f"  -> {ds_name}: Nenhuma amostra encontrada. Pulando.")

    # --- FASE 2: VALIDAÇÃO CRUZADA LEAVE-ONE-DOMAIN-OUT ---
    print("\n--- FASE 2: TREINAMENTO MULTI-SOURCE ---")
    available_datasets = list(db_features.keys())
    
    for target_ds in available_datasets:
        # Se a base alvo não possui dados normais E falhas simultaneamente, o teste quebra.
        if len(np.unique(db_labels[target_ds])) < 2:
            print(f"\n[!] Pulando {target_ds} como Alvo de Teste (Não possui dados saudáveis para medir F1-Score).")
            continue
            
        print(f"\n{'#'*60}")
        print(f" ALVO DE TESTE (TARGET DOMAIN): {target_ds}")
        
        source_datasets = [ds for ds in available_datasets if ds != target_ds]
        print(f" TREINADO EM (SOURCE DOMAINS): {', '.join(source_datasets)}")
        print(f"{'#'*60}")
        
        X_train_list = [db_features[ds] for ds in source_datasets]
        y_train_list = [db_labels[ds] for ds in source_datasets]
        
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        
        X_test = db_features[target_ds]
        y_test = db_labels[target_ds]
        
        print(f"  -> Volume de Treino: {X_train.shape[0]} amostras de {len(source_datasets)} máquinas")
        print(f"  -> Volume de Teste: {X_test.shape[0]} amostras (Máquina Desconhecida)")
        
        source_name_str = f"All_Except_{target_ds}"
        
        current_results = evaluate_transfer_models(X_train, y_train, X_test, y_test, source_name_str, target_ds)
        master_results.extend(current_results)
        
        df = pd.DataFrame(master_results)
        df.to_csv(csv_file, index=False)

    print(f"\n[SUCESSO] Tabela de Transfer Learning exportada para: {csv_file}")
    
    if master_results:
        df = pd.DataFrame(master_results)
        print("\n--- RESUMO GERAL DO TRANSFER LEARNING (MACRO F1) ---")
        summary = df.groupby(['Target Domain', 'Model'])['Macro F1'].mean().unstack()
        print(summary.to_string())

if __name__ == "__main__":
    run_transfer_learning()
