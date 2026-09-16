import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

# Adiciona a raiz do projeto ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.features.extractors_v2 import extract_advanced_features
from src.features.signalai_wrapper import extract_fusion_features
from src.models.build_tabnet_resnet import train_and_evaluate_multihead 

# --- CONFIGURAÇÃO GLOBAL ---
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))

TASKS = ["diagnosis"] # Foco na tarefa de diagnóstico

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

# Você pode incluir outros alvos aqui se desejar rodar o baseline para todos
TARGET_DATASETS = ["UOEMD"]

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

def evaluate_tabnet_scratch(train_data_dict_dl, X_test_dl, y_test_dl, target_name, test_cond, task):
    """Treina exclusivamente o TabNet na máquina alvo (sem transfer learning)."""
    results = []
    
    if len(y_test_dl) == 0:
        return results

    model_name = "Multi-Head DL (TABNET)"
    combo_name = "Baseline (Scratch)" # Identificador claro para o merge com o Grid Search

    try:
        bal_acc, macro_f1, roc_auc, _ = train_and_evaluate_multihead(
            train_data_dict=train_data_dict_dl,
            target_dataset_name=target_name,
            X_test=X_test_dl,
            y_test=y_test_dl,
            task=task,
            epochs=15,          
            batch_size=512,      
            encoder_type='tabnet'
        )
        print(f"          [{model_name}] Bal Acc: {bal_acc:.4f} | F1: {macro_f1:.4f} | ROC-AUC: {roc_auc:.4f}")
        
        # O Dicionário de resultados tem as MESMAS CHAVES do Grid Search
        results.append({
            "Target Dataset": target_name, 
            "Task": task.capitalize(), 
            "Test Condition": test_cond, 
            "Source Combo": combo_name, 
            "Model": model_name, 
            "Bal Acc": bal_acc, 
            "Macro F1": macro_f1, 
            "ROC-AUC": roc_auc
        })
    except Exception as e:
        print(f"          [ERRO] Falha ao treinar {model_name}: {e}")

    return results

def run_baseline_scratch():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(RESULTS_DIR, f"log_baseline_scratch_{timestamp}.txt")
    csv_file = os.path.join(RESULTS_DIR, f"baseline_scratch_results_{timestamp}.csv")
    sys.stdout = Logger(log_file)
    
    print(f"{'='*80}\n EXPERIMENTO BASELINE (TREINAMENTO FROM SCRATCH)\n{'='*80}")

    master_results = []
    db_features, db_labels_raw, db_conds = {}, {}, {}
    
    # Diferente do grid search, carregamos apenas os TARGET_DATASETS para economizar RAM
    print("--- FASE 1: EXTRAÇÃO DE FEATURES DOS ALVOS ---")
    for ds_name in TARGET_DATASETS:
        if ds_name not in DATASETS_CONFIG: continue
        fs = DATASETS_CONFIG[ds_name]
        
        X_raw, y_raw_str, cond_raw = load_entire_dataset_for_tl(ds_name, fs)
        if len(X_raw) > 0:
            X_raw = np.array(X_raw)
            print(f"  -> {ds_name} carregado: {X_raw.shape[0]} amostras. Extraindo features...")
            
            X_fusion = extract_fusion_features(X_raw, fs, extract_advanced_features)
            X_clean = np.nan_to_num(np.array(X_fusion, dtype=np.float32))
            
            db_features[ds_name] = X_clean
            db_labels_raw[ds_name] = np.array(y_raw_str)
            db_conds[ds_name] = np.array(cond_raw)

    print("\n--- FASE 2: AVALIAÇÃO LOCO (ISOLADA POR MÁQUINA) ---")
    for task in TASKS:
        # Mapeamento Unificado de Labels
        db_labels_mapped = {}
        for ds in db_features.keys():
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
                
            print(f"\n{'#'*40}\n ALVO FIXO: {target_ds}\n{'#'*40}")
            unique_conds = np.unique(db_conds[target_ds])
            
            for test_cond in unique_conds:
                print(f"\n   --- Dobra de Teste: {test_cond} ---")
                
                test_mask = (db_conds[target_ds] == test_cond)
                X_test_raw = db_features[target_ds][test_mask]
                y_test_raw = db_labels_mapped[target_ds][test_mask]
                
                train_target_mask = ~test_mask
                X_train_local = db_features[target_ds][train_target_mask]
                y_train_local = db_labels_mapped[target_ds][train_target_mask]
                
                print(f"      -> Treinando DO ZERO na própria máquina: {target_ds}")
                
                # =========================================================
                # SETUP DL MULTI-HEAD - APENAS COM A MÁQUINA LOCAL
                # =========================================================
                train_data_dict_dl = {}
                le_target_local = LabelEncoder()

                # Adiciona APENAS o Treino Local (Target)
                if len(X_train_local) > 0:
                    y_tr_local = le_target_local.fit_transform(y_train_local)
                    train_data_dict_dl[target_ds] = (X_train_local, y_tr_local)

                # Prepara o teste
                valid_test_idx_dl = [i for i, lbl in enumerate(y_test_raw) if lbl in le_target_local.classes_]
                if len(valid_test_idx_dl) == 0: continue
                
                X_test_dl = X_test_raw[valid_test_idx_dl]
                y_test_dl_enc = le_target_local.transform(y_test_raw[valid_test_idx_dl])
                
                # Executa a Avaliação
                current_results = evaluate_tabnet_scratch(
                    train_data_dict_dl, X_test_dl, y_test_dl_enc,
                    target_ds, test_cond, task
                )
                master_results.extend(current_results)
                
                # Salva o CSV a cada iteração
                pd.DataFrame(master_results).to_csv(csv_file, index=False)

    print(f"\n[SUCESSO] Relatório de Baseline Exportado: {csv_file}")

if __name__ == "__main__":
    run_baseline_scratch()
