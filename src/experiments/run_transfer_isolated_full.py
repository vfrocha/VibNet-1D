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
from src.models.build_tabnet_resnet import train_and_evaluate_multihead 

# --- CONFIGURAÇÃO GLOBAL ---
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))

TASKS = ["diagnosis"] #["detection", "diagnosis"]

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
    Carrega os dados brutos e Rótulos. Ignora CWRU_48k saudável para evitar falhas de label.
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
                
                if dataset_name == "CWRU_48k" and ('normal' in class_name.lower() or 'healthy' in class_name.lower()):
                    continue 
                    
                X_raw.append(np.load(file_path))
                y_raw_str.append(class_name)
                cond_raw.append(cond_name)
                
    return X_raw, y_raw_str, cond_raw

def evaluate_transfer_models(X_train_c, y_train_c, X_test_c, y_test_c, train_data_dict_dl, X_test_dl, y_test_dl, target_name, test_cond, task):
    """Treina os Baselines Clássicos e a Rede Multi-Head na mesma dobra."""
    results = []
    
    # 1. MODELOS CLÁSSICOS (RF e SVM)
    models = {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "SVM (RBF)": SVC(kernel='rbf', probability=True, random_state=42)
    }
    
    for model_name, model in models.items():
        print(f"       -> Treinando {model_name}...")
        try:
            model.fit(X_train_c, y_train_c)
            y_pred = model.predict(X_test_c)
            y_probs = model.predict_proba(X_test_c)
            
            bal_acc = balanced_accuracy_score(y_test_c, y_pred)
            if task == 'detection':
                macro_f1 = f1_score(y_test_c, y_pred, average='binary')
                try: roc_auc = roc_auc_score(y_test_c, y_probs[:, 1])
                except ValueError: roc_auc = 0.0 
            else:
                macro_f1 = f1_score(y_test_c, y_pred, average='macro')
                try: roc_auc = roc_auc_score(y_test_c, y_probs, multi_class='ovr')
                except ValueError: roc_auc = 0.0 
                
            print(f"          [{model_name}] Bal Acc: {bal_acc:.4f} | F1: {macro_f1:.4f} | ROC-AUC: {roc_auc:.4f}")
            results.append({"Dataset": target_name, "Task": task.capitalize(), "Test Condition": test_cond, "Model": model_name, "Bal Acc": bal_acc, "Macro F1": macro_f1, "ROC-AUC": roc_auc})
        except Exception as e:
            print(f"          [ERRO] Falha ao treinar {model_name}: {e}")
            
    # 2. MODELOS PROFUNDOS: MULTI-HEAD (Ajuste de Magnitude Interno)
    if len(y_test_dl) > 0:
        dl_encoders = ['mlp', 'tabnet']
        for enc in dl_encoders:
            model_name = f"Multi-Head DL ({enc.upper()})"
            print(f"       -> Treinando {model_name}...")
            try:
                bal_acc, macro_f1, roc_auc, _ = train_and_evaluate_multihead(
                    train_data_dict=train_data_dict_dl,
                    target_dataset_name=target_name,
                    X_test=X_test_dl,
                    y_test=y_test_dl,
                    task=task,
                    epochs=15,          
                    batch_size=512,      
                    encoder_type=enc
                )
                print(f"          [{model_name}] Bal Acc: {bal_acc:.4f} | F1: {macro_f1:.4f} | ROC-AUC: {roc_auc:.4f}")
                results.append({"Dataset": target_name, "Task": task.capitalize(), "Test Condition": test_cond, "Model": model_name, "Bal Acc": bal_acc, "Macro F1": macro_f1, "ROC-AUC": roc_auc})
            except Exception as e:
                print(f"          [ERRO] Falha ao treinar {model_name}: {e}")
    else:
        print("       -> [Aviso] Multi-Head DL ignorado (Classes insuficientes no teste).")

    return results

def run_transfer_isolated_full():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(RESULTS_DIR, f"log_tl_isolated_full_{timestamp}.txt")
    csv_file = os.path.join(RESULTS_DIR, f"tl_isolated_full_results_{timestamp}.csv")
    sys.stdout = Logger(log_file)
    
    print(f"{'='*80}\n EXPERIMENTO COMPLETO DE TRANSFER LEARNING (Z-SCORE ISOLADO)\n{'='*80}")

    master_results = []
    db_features, db_labels_raw, db_conds = {}, {}, {}
    
    print("--- FASE 1: EXTRAÇÃO DE FEATURES (1 VEZ NA RAM) ---")
    for ds_name, fs in DATASETS_CONFIG.items():
        X_raw, y_raw_str, cond_raw = load_entire_dataset_for_tl(ds_name, fs)
        if len(X_raw) > 0:
            X_raw = np.array(X_raw)
            print(f"  -> {ds_name} carregado: {X_raw.shape[0]} amostras. Extraindo features...")
            
            X_fusion = extract_fusion_features(X_raw, fs, extract_advanced_features)
            X_clean = np.nan_to_num(np.array(X_fusion, dtype=np.float32))
            
            db_features[ds_name] = X_clean
            db_labels_raw[ds_name] = np.array(y_raw_str)
            db_conds[ds_name] = np.array(cond_raw)

    available_datasets = list(db_features.keys())

    print("\n--- FASE 2: AVALIAÇÃO LOCO CROSS-DOMAIN ---")
    for task in TASKS:
        print(f"\n\n{'*'*60}\n INICIANDO TAREFA: {task.upper()}\n{'*'*60}")
        
        # Mapeamento Unificado de Labels
        db_labels_mapped = {}
        for ds in available_datasets:
            mapped = []
            for lbl in db_labels_raw[ds]:
                is_normal = ('normal' in lbl.lower() or 'healthy' in lbl.lower())
                if task == 'detection':
                    mapped.append(0 if is_normal else 1)
                elif task == 'diagnosis':
                    if is_normal: mapped.append("Universal_Normal") 
                    else: mapped.append(f"{ds}_{lbl}")      
            db_labels_mapped[ds] = np.array(mapped)

        for target_ds in TARGET_DATASETS:
            if target_ds not in db_features: continue
            if task == 'detection' and target_ds == "CWRU_48k": continue
                
            print(f"\n{'#'*40}\n ALVO: {target_ds}\n{'#'*40}")
            unique_conds = np.unique(db_conds[target_ds])
            
            for test_cond in unique_conds:
                print(f"\n   --- Dobra de Teste: {test_cond} ---")
                
                test_mask = (db_conds[target_ds] == test_cond)
                X_test_raw = db_features[target_ds][test_mask]
                y_test_raw = db_labels_mapped[target_ds][test_mask]
                
                if len(np.unique(y_test_raw)) < 2 and task == 'detection':
                    print(f"      [Aviso] Faltam classes 0/1 para {test_cond}. Pulando.")
                    continue
                    
                train_target_mask = ~test_mask
                X_train_local = db_features[target_ds][train_target_mask]
                y_train_local = db_labels_mapped[target_ds][train_target_mask]
                
                # =========================================================
                # ESTRUTURA CLÁSSICA (RF/SVM) - NORMALIZAÇÃO ISOLADA
                # =========================================================
                X_train_classical_list = []
                y_train_classical_list = []
                target_scaler = None

                for ds in available_datasets:
                    scaler = StandardScaler()
                    if ds == target_ds:
                        if len(X_train_local) == 0: continue
                        X_tr_s = scaler.fit_transform(X_train_local)
                        target_scaler = scaler
                        y_tr_c = y_train_local
                    else:
                        X_tr_s = scaler.fit_transform(db_features[ds])
                        y_tr_c = db_labels_mapped[ds]
                        
                    X_train_classical_list.append(X_tr_s)
                    y_train_classical_list.append(y_tr_c)

                X_train_classical = np.vstack(X_train_classical_list)
                y_train_classical_raw = np.concatenate(y_train_classical_list)
                
                le_global = LabelEncoder()
                y_train_classical_enc = le_global.fit_transform(y_train_classical_raw)
                
                # Filtro de teste
                valid_test_idx_c = [i for i, lbl in enumerate(y_test_raw) if lbl in le_global.classes_]
                if len(valid_test_idx_c) == 0: continue
                X_test_valid_c = X_test_raw[valid_test_idx_c]
                y_test_c_enc = le_global.transform(y_test_raw[valid_test_idx_c])
                X_test_classical_s = target_scaler.transform(X_test_valid_c)

                # =========================================================
                # ESTRUTURA DL MULTI-HEAD - DICIONÁRIO DE DADOS
                # =========================================================
                train_data_dict_dl = {}
                le_target_local = LabelEncoder()

                for ds in available_datasets:
                    le_local = LabelEncoder()
                    if ds == target_ds:
                        if len(X_train_local) > 0:
                            y_tr_local = le_local.fit_transform(y_train_local)
                            train_data_dict_dl[ds] = (X_train_local, y_tr_local)
                            le_target_local = le_local
                    else:
                        if len(db_features[ds]) > 0:
                            y_tr_local = le_local.fit_transform(db_labels_mapped[ds])
                            train_data_dict_dl[ds] = (db_features[ds], y_tr_local)

                valid_test_idx_dl = [i for i, lbl in enumerate(y_test_raw) if lbl in le_target_local.classes_]
                X_test_dl = X_test_raw[valid_test_idx_dl]
                y_test_dl_enc = le_target_local.transform(y_test_raw[valid_test_idx_dl]) if len(valid_test_idx_dl) > 0 else []

                # Executa a Avaliação
                current_results = evaluate_transfer_models(
                    X_train_classical, y_train_classical_enc, X_test_classical_s, y_test_c_enc,
                    train_data_dict_dl, X_test_dl, y_test_dl_enc,
                    target_ds, test_cond, task
                )
                master_results.extend(current_results)
                
                pd.DataFrame(master_results).to_csv(csv_file, index=False)

    print(f"\n[SUCESSO] Relatório Final Exportado: {csv_file}")

if __name__ == "__main__":
    run_transfer_isolated_full()
