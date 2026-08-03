import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

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

# Todas as bases
DATASETS_CONFIG = {
    "CWRU_12k": 12000,
    "CWRU_48k": 48000,
    "UOEMD": 42000,
    "HUST_Gearbox": 25600,
    "HUST": 51200,
    "PU": 64000,
    "UORED": 42000,          
    "Mechanical_Gear": 5000,
    "Electric_Motor": 50000
}

# Bases que queremos ativamente analisar o F1-Score do Teste (os alvos do Baseline)
TARGET_DATASETS = ["CWRU_12k", "UOEMD", "HUST_Gearbox"] 

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
    Carrega TODOS os arquivos de uma base, capturando a 'Condição' (Folder Pai) 
    para podermos isolar os testes exatamente como no Baseline LOCO.
    """
    dataset_path = os.path.join(DATA_ROOT, dataset_name)
    if not os.path.exists(dataset_path):
        print(f"[Aviso] Dataset {dataset_name} não encontrado no disco.")
        return [], [], []
        
    X_raw, y_raw, cond_raw = [], [], []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.npy'):
                # Exemplo: root = ".../CWRU_12k/Load_0HP/Class_Normal"
                class_name = os.path.basename(root)
                cond_name = os.path.basename(os.path.dirname(root))
                file_path = os.path.join(root, file)
                
                # Mapeamento Universal Binário
                if 'normal' in class_name.lower() or 'healthy' in class_name.lower():
                    label = 0 # Saudável
                else:
                    label = 1 # Falha
                    
                if dataset_name == "CWRU_48k" and label == 0:
                    continue 
                    
                X_raw.append(np.load(file_path))
                y_raw.append(label)
                cond_raw.append(cond_name)
                
    return X_raw, y_raw, cond_raw

def evaluate_transfer_models(X_train, y_train, X_test, y_test, target_name, test_cond):
    """
    Avalia a generalização. As métricas são as mesmas do baseline para viabilizar 
    o cruzamento do compare_results.py
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
        print(f"       -> Treinando {model_name}...")
        try:
            model.fit(X_train_s, y_train)
            y_pred = model.predict(X_test_s)
            y_probs = model.predict_proba(X_test_s)
            
            bal_acc = balanced_accuracy_score(y_test, y_pred)
            macro_f1 = f1_score(y_test, y_pred, average='binary')
            
            try:
                roc_auc = roc_auc_score(y_test, y_probs[:, 1])
            except ValueError:
                roc_auc = 0.0 
                
            print(f"          [{model_name}] Bal Acc: {bal_acc:.4f} | F1: {macro_f1:.4f} | ROC-AUC: {roc_auc:.4f}")
            
            # A nomenclatura exata para o compare_results.py achar
            results.append({
                "Dataset": target_name,
                "Task": "Detection",
                "Test Condition": test_cond,
                "Model": model_name,
                "Bal Acc": bal_acc,
                "Macro F1": macro_f1,
                "ROC-AUC": roc_auc
            })
        except Exception as e:
            print(f"          [ERRO] Falha ao treinar {model_name}: {e}")
            
    return results

def run_transfer_learning():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(RESULTS_DIR, f"log_tl_loco_{timestamp}.txt")
    csv_file = os.path.join(RESULTS_DIR, f"tl_loco_results_{timestamp}.csv")
    sys.stdout = Logger(log_file)
    
    print(f"{'='*80}\nEXPERIMENTOS DE TL + LOCO (EXTERNAL SOURCE + TARGET TRAIN -> TARGET TEST)\n{'='*80}")
    print("Tarefa: DETECTION (0 = Normal, 1 = Fault)")
    print(f"Datsets disponíveis para extração de Conhecimento: {list(DATASETS_CONFIG.keys())}\n")

    master_results = []
    
    db_features = {}
    db_labels = {}
    db_conds = {}
    
    print("--- FASE 1: EXTRAÇÃO DE FEATURES DE TODAS AS MÁQUINAS ---")
    for ds_name, fs in DATASETS_CONFIG.items():
        X_raw, y_raw, cond_raw = load_entire_dataset_for_tl(ds_name, fs)
        if len(X_raw) > 0:
            X_raw = np.array(X_raw)
            print(f"  -> {ds_name} carregado: {X_raw.shape[0]} janelas. Extraindo 141 features (fs={fs}Hz)...")
            
            X_fusion = extract_fusion_features(X_raw, fs, extract_advanced_features)
            X_clean = np.nan_to_num(np.array(X_fusion, dtype=np.float32))
            if X_clean.ndim == 1: X_clean = X_clean.reshape(len(y_raw), -1)
            
            db_features[ds_name] = X_clean
            db_labels[ds_name] = np.array(y_raw)
            db_conds[ds_name] = np.array(cond_raw)
        else:
            print(f"  -> {ds_name}: Nenhuma amostra encontrada. Pulando.")

    # --- FASE 2: VALIDAÇÃO TL-LOCO (IDÊNTICA AO BASELINE) ---
    print("\n--- FASE 2: TESTE CONDIÇÃO POR CONDIÇÃO (Idêntico ao Baseline) ---")
    available_datasets = list(db_features.keys())
    
    for target_ds in TARGET_DATASETS:
        if target_ds not in db_features: continue
        
        print(f"\n{'#'*60}\n ALVO GERAL: {target_ds}\n{'#'*60}")
        unique_conds = np.unique(db_conds[target_ds])
        
        for test_cond in unique_conds:
            print(f"\n   --- Dobra de Teste: {test_cond} ---")
            
            # 1. Isola o conjunto de Teste Exato (Igual ao Baseline)
            test_mask = (db_conds[target_ds] == test_cond)
            X_test = db_features[target_ds][test_mask]
            y_test = db_labels[target_ds][test_mask]
            
            if len(np.unique(y_test)) < 2:
                print(f"      [Aviso] Dados insuficientes/Falta da classe Normal para {test_cond}. Pulando.")
                continue
                
            # 2. Isola o conjunto de Treino do Próprio Target (Igual ao Baseline)
            train_target_mask = (db_conds[target_ds] != test_cond)
            X_train_local = db_features[target_ds][train_target_mask]
            y_train_local = db_labels[target_ds][train_target_mask]
            
            # 3. Puxa as 8 Máquinas Externas (O Diferencial do Transfer Learning)
            X_external_list = [db_features[ds] for ds in available_datasets if ds != target_ds]
            y_external_list = [db_labels[ds] for ds in available_datasets if ds != target_ds]
            
            # 4. Une tudo no grande conjunto de Treinamento
            X_train_global = np.vstack(X_external_list + [X_train_local])
            y_train_global = np.concatenate(y_external_list + [y_train_local])
            
            print(f"      -> Treinando com: {X_train_global.shape[0]} janelas (Treino Baseline + 8 Máquinas Externas)")
            print(f"      -> Testando em: {X_test.shape[0]} janelas ({test_cond} Puro)")
            
            current_results = evaluate_transfer_models(
                X_train_global, y_train_global, X_test, y_test, target_ds, test_cond
            )
            master_results.extend(current_results)
            
            df = pd.DataFrame(master_results)
            df.to_csv(csv_file, index=False)

    print(f"\n[SUCESSO] Tabela TL LOCO exportada para: {csv_file}")

if __name__ == "__main__":
    run_transfer_learning()
