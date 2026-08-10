import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

# Adiciona a raiz do projeto ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.features.extractors_v2 import extract_advanced_features
from src.features.signalai_wrapper import extract_fusion_features
from src.models.build_tabnet_resnet import train_and_evaluate_hybrid

# --- CONFIGURAÇÃO GLOBAL ---
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))

TASKS = ["detection", "diagnosis"]

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

TARGET_DATASETS = ["CWRU_12k", "UOEMD", "HUST_Gearbox", "CWRU_48k"] 

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
    Retorna os dados brutos e os Rótulos em STRING, preservando 
    a nomenclatura exata para posterior mapeamento das tasks.
    """
    dataset_path = os.path.join(DATA_ROOT, dataset_name)
    if not os.path.exists(dataset_path):
        print(f"[Aviso] Dataset {dataset_name} não encontrado no disco.")
        return [], [], []
        
    X_raw, y_raw_str, cond_raw = [], [], []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.npy'):
                class_name = os.path.basename(root)
                cond_name = os.path.basename(os.path.dirname(root))
                file_path = os.path.join(root, file)
                
                # Ignoramos a normalização Binária/Multiclasse aqui!
                # Fazemos apenas uma filtragem de segurança
                if dataset_name == "CWRU_48k" and ('normal' in class_name.lower() or 'healthy' in class_name.lower()):
                    continue 
                    
                X_raw.append(np.load(file_path))
                y_raw_str.append(class_name)
                cond_raw.append(cond_name)
                
    return X_raw, y_raw_str, cond_raw

def evaluate_transfer_models(X_train, y_train, X_test, y_test, target_name, test_cond, task):
    """
    Avalia a generalização de Modelos Rasos Clássicos + Redes Neurais Híbridas (MLP/TabNet)
    """
    results = []
    
    # 1. MODELOS CLÁSSICOS (Scikit-Learn)
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
            if task == 'detection':
                macro_f1 = f1_score(y_test, y_pred, average='binary')
                try: roc_auc = roc_auc_score(y_test, y_probs[:, 1])
                except ValueError: roc_auc = 0.0 
            else:
                macro_f1 = f1_score(y_test, y_pred, average='macro')
                try: roc_auc = roc_auc_score(y_test, y_probs, multi_class='ovr')
                except ValueError: roc_auc = 0.0 
                
            print(f"          [{model_name}] Bal Acc: {bal_acc:.4f} | F1: {macro_f1:.4f} | ROC-AUC: {roc_auc:.4f}")
            results.append({"Dataset": target_name, "Task": task.capitalize(), "Test Condition": test_cond, "Model": model_name, "Bal Acc": bal_acc, "Macro F1": macro_f1, "ROC-AUC": roc_auc})
        except Exception as e:
            print(f"          [ERRO] Falha ao treinar {model_name}: {e}")
            
    # 2. MODELOS PROFUNDOS: ESTUDO DE ABLAÇÃO (MLP vs TabNet)
    dl_encoders = ['mlp', 'tabnet']
    
    for enc in dl_encoders:
        model_name = f"Hybrid DL ({enc.upper()})"
        print(f"       -> Treinando {model_name}...")
        try:
            bal_acc, macro_f1, roc_auc, _ = train_and_evaluate_hybrid(
                X_train, y_train, X_test, y_test, 
                task=task, 
                epochs=15,          
                batch_size=512,      
                encoder_type=enc
            )
            print(f"          [{model_name}] Bal Acc: {bal_acc:.4f} | F1: {macro_f1:.4f} | ROC-AUC: {roc_auc:.4f}")
            results.append({"Dataset": target_name, "Task": task.capitalize(), "Test Condition": test_cond, "Model": model_name, "Bal Acc": bal_acc, "Macro F1": macro_f1, "ROC-AUC": roc_auc})
        except Exception as e:
            print(f"          [ERRO] Falha ao treinar {model_name}: {e}")

    return results

def run_transfer_learning():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(RESULTS_DIR, f"log_tl_completo_{timestamp}.txt")
    csv_file = os.path.join(RESULTS_DIR, f"tl_completo_results_{timestamp}.csv")
    sys.stdout = Logger(log_file)
    
    print(f"{'='*80}\n EXPERIMENTO FINAL DE TRANSFER LEARNING (DETECTION & DIAGNOSIS)\n{'='*80}")
    print(f"Alvos (Targets): {TARGET_DATASETS}")
    print(f"Bases de Conhecimento: {list(DATASETS_CONFIG.keys())}\n")

    master_results = []
    
    db_features = {}
    db_labels_raw = {}
    db_conds = {}
    
    print("--- FASE 1: EXTRAÇÃO GLOBAL NA MEMÓRIA RAM ---")
    for ds_name, fs in DATASETS_CONFIG.items():
        X_raw, y_raw_str, cond_raw = load_entire_dataset_for_tl(ds_name, fs)
        if len(X_raw) > 0:
            X_raw = np.array(X_raw)
            print(f"  -> {ds_name} carregado: {X_raw.shape[0]} janelas. Extraindo 141 features (fs={fs}Hz)...")
            
            X_fusion = extract_fusion_features(X_raw, fs, extract_advanced_features)
            X_clean = np.nan_to_num(np.array(X_fusion, dtype=np.float32))
            if X_clean.ndim == 1: X_clean = X_clean.reshape(len(y_raw_str), -1)
            
            db_features[ds_name] = X_clean
            db_labels_raw[ds_name] = np.array(y_raw_str)
            db_conds[ds_name] = np.array(cond_raw)
        else:
            print(f"  -> {ds_name}: Nenhuma amostra encontrada. Pulando.")

    available_datasets = list(db_features.keys())

    print("\n--- FASE 2: TREINAMENTO MULTI-TASK & MULTI-DOMAIN ---")
    
    for task in TASKS:
        print(f"\n\n{'*'*60}\n INICIANDO TAREFA: {task.upper()}\n{'*'*60}")
        
        # 1. Estratégia Inteligente de Mapeamento de Rótulos (Label Harmonization)
        db_labels_mapped = {}
        for ds in available_datasets:
            mapped = []
            for lbl in db_labels_raw[ds]:
                is_normal = ('normal' in lbl.lower() or 'healthy' in lbl.lower())
                
                if task == 'detection':
                    mapped.append(0 if is_normal else 1)
                elif task == 'diagnosis':
                    if is_normal:
                        mapped.append("Universal_Normal") # Unifica os dados saudáveis do mundo inteiro
                    else:
                        mapped.append(f"{ds}_{lbl}")      # Isola o defeito para não cruzar classes diferentes
            
            db_labels_mapped[ds] = np.array(mapped)

        # 2. Execução da Validação Cruzada (LOCO + External)
        for target_ds in TARGET_DATASETS:
            if target_ds not in db_features: continue
            
            if task == 'detection' and target_ds == "CWRU_48k":
                print(f"\n   [!] Pulando CWRU_48k para Detecção (não possui dados Normal)")
                continue
                
            print(f"\n{'#'*40}\n ALVO: {target_ds}\n{'#'*40}")
            unique_conds = np.unique(db_conds[target_ds])
            
            for test_cond in unique_conds:
                print(f"\n   --- Dobra de Teste: {test_cond} ---")
                
                test_mask = (db_conds[target_ds] == test_cond)
                X_test_raw = db_features[target_ds][test_mask]
                y_test_raw = db_labels_mapped[target_ds][test_mask]
                
                if len(np.unique(y_test_raw)) < 2 and task == 'detection':
                    print(f"      [Aviso] Faltam classes 0/1 para F1-Score em {test_cond}. Pulando.")
                    continue
                    
                train_target_mask = (db_conds[target_ds] != test_cond)
                X_train_local = db_features[target_ds][train_target_mask]
                y_train_local = db_labels_mapped[target_ds][train_target_mask]
                
                X_external_list = [db_features[ds] for ds in available_datasets if ds != target_ds]
                y_external_list = [db_labels_mapped[ds] for ds in available_datasets if ds != target_ds]
                
                X_train_global = np.vstack(X_external_list + [X_train_local])
                y_train_global = np.concatenate(y_external_list + [y_train_local])
                
                # 3. Label Encoding para transformar strings em Inteiros para a Rede Neural e Sklearn
                le = LabelEncoder()
                y_train_encoded = le.fit_transform(y_train_global)
                
                # Proteção: Garantir que o Teste só contenha classes que o modelo viu no Treino
                valid_test_idx = [i for i, label in enumerate(y_test_raw) if label in le.classes_]
                if len(valid_test_idx) < len(y_test_raw):
                    print(f"      [Aviso] {len(y_test_raw)-len(valid_test_idx)} janelas descartadas (Classe inédita no Treino)")
                
                X_test = X_test_raw[valid_test_idx]
                y_test_encoded = le.transform(y_test_raw[valid_test_idx])
                
                if len(y_test_encoded) == 0: continue
                
                print(f"      -> Treinando com: {X_train_global.shape[0]} janelas ({len(le.classes_)} classes simultâneas)")
                print(f"      -> Testando em: {X_test.shape[0]} janelas ({test_cond} Puro)")
                
                current_results = evaluate_transfer_models(
                    X_train_global, y_train_encoded, X_test, y_test_encoded, target_ds, test_cond, task
                )
                master_results.extend(current_results)
                
                df = pd.DataFrame(master_results)
                df.to_csv(csv_file, index=False)

    print(f"\n[SUCESSO] Tabela Completa de Transfer Learning exportada para: {csv_file}")

if __name__ == "__main__":
    run_transfer_learning()
