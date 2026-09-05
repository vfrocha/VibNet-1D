import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.signal import detrend
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

# Todos os datasets mapeados para Transfer Learning[cite: 9]
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
    """Carrega as janelas brutas e rótulos do disco.[cite: 9]"""
    dataset_path = os.path.join(DATA_ROOT, dataset_name)
    if not os.path.exists(dataset_path):
        return [], [], []
        
    X_raw, y_raw_str, cond_raw = [], [], []
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.npy'):
                class_name = os.path.basename(root)
                cond_name = os.path.basename(os.path.dirname(root))
                if dataset_name == "CWRU_48k" and ('normal' in class_name.lower() or 'healthy' in class_name.lower()):
                    continue 
                X_raw.append(np.load(os.path.join(root, file)))
                y_raw_str.append(class_name)
                cond_raw.append(cond_name)
                
    return X_raw, y_raw_str, cond_raw

def normalize_time_window(sinal, strategy):
    """Aplica a normalização no nível do sinal (Onda Física)[cite: 4, 7]"""
    sinal_detrend = detrend(sinal)
    if strategy == 'raw':
        return sinal_detrend
    elif strategy == 'window_zscore':
        std = np.std(sinal_detrend)
        return sinal_detrend / std if std > 0 else sinal_detrend
    elif strategy == 'window_rms':
        rms = np.sqrt(np.mean(sinal_detrend**2))
        return sinal_detrend / rms if rms > 0 else sinal_detrend
    return sinal_detrend

def run_normalization_tl_experiment():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(RESULTS_DIR, f"log_norm_tl_dl_{timestamp}.txt")
    csv_file = os.path.join(RESULTS_DIR, f"norm_tl_dl_results_{timestamp}.csv")
    sys.stdout = Logger(log_file)
    
    print(f"{'='*80}\n ABLATION STUDY: TRANSFER LEARNING DL COM NORMALIZAÇÃO DE JANELA\n{'='*80}")
    
    experiment_matrix = [
        {'id': '0_Baseline', 'pretrain': 'raw', 'finetune': 'raw'},
        {'id': '1_Estrategia_A', 'pretrain': 'window_zscore', 'finetune': 'window_zscore'},
        {'id': '2_Estrategia_B', 'pretrain': 'window_rms', 'finetune': 'window_rms'},
        {'id': '3_Estrategia_C_com_A', 'pretrain': 'window_zscore', 'finetune': 'raw'},
        {'id': '4_Estrategia_C_com_B', 'pretrain': 'window_rms', 'finetune': 'raw'}
    ]

    # --- FASE 1: PRÉ-PROCESSAMENTO COM CACHE (Evita re-extrair 5 vezes) ---
    print("--- FASE 1: EXTRAÇÃO MULTI-ESTRATÉGIA EM CACHE ---")
    
    feature_cache = {'raw': {}, 'window_zscore': {}, 'window_rms': {}}
    db_labels_raw = {}
    db_conds = {}
    
    for ds_name, fs in DATASETS_CONFIG.items():
        X_raw_list, y_raw_str, cond_raw = load_entire_dataset_for_tl(ds_name, fs)
        if not X_raw_list: continue
        
        db_labels_raw[ds_name] = np.array(y_raw_str)
        db_conds[ds_name] = np.array(cond_raw)
        
        print(f"  -> Processando {ds_name} ({len(X_raw_list)} amostras)...")
        for transform_type in feature_cache.keys():
            X_transformed = [normalize_time_window(w, transform_type) for w in X_raw_list]
            X_fusion = extract_fusion_features(np.array(X_transformed), fs, extract_advanced_features)
            X_clean = np.nan_to_num(np.array(X_fusion, dtype=np.float32))
            if X_clean.ndim == 1: X_clean = X_clean.reshape(len(y_raw_str), -1)
            feature_cache[transform_type][ds_name] = X_clean

    available_datasets = list(db_labels_raw.keys())
    master_results = []

    # --- FASE 2: MATRIZ EXPERIMENTAL & TRANSFER LEARNING ---
    for exp in experiment_matrix:
        print(f"\n\n{'='*80}\n EXECUTANDO: {exp['id']} (Source: {exp['pretrain']} | Target: {exp['finetune']})\n{'='*80}")
        
        for task in TASKS:
            print(f"\n{'*'*40} TAREFA: {task.upper()} {'*'*40}")
            
            # Mapeamento Global de Rótulos[cite: 9]
            db_labels_mapped = {}
            for ds in available_datasets:
                mapped = []
                for lbl in db_labels_raw[ds]:
                    is_normal = ('normal' in lbl.lower() or 'healthy' in lbl.lower())
                    if task == 'detection':
                        mapped.append(0 if is_normal else 1)
                    elif task == 'diagnosis':
                        mapped.append("Universal_Normal" if is_normal else f"{ds}_{lbl}")      
                db_labels_mapped[ds] = np.array(mapped)

            for target_ds in TARGET_DATASETS:
                if target_ds not in available_datasets: continue
                if task == 'detection' and target_ds == "CWRU_48k": continue
                
                unique_conds = np.unique(db_conds[target_ds])
                
                for test_cond in unique_conds:
                    print(f"\n   [Alvo: {target_ds} | Dobra: {test_cond}]")
                    
                    test_mask = (db_conds[target_ds] == test_cond)
                    train_target_mask = ~test_mask
                    
                    # Target usa o conjunto de features definido na estratégia de FINETUNE
                    X_target_full = feature_cache[exp['finetune']][target_ds]
                    y_target_full = db_labels_mapped[target_ds]
                    
                    if len(np.unique(y_target_full[test_mask])) < 2 and task == 'detection':
                        continue # Pula se faltar a classe 0/1 no teste
                        
                    # =========================================================================
                    # SETUP DL MULTI-HEAD - DICIONÁRIO DE DADOS (Label Encoding Local)[cite: 9]
                    # =========================================================================
                    train_data_dict_dl = {}
                    le_target_local = LabelEncoder()

                    for ds in available_datasets:
                        le_local = LabelEncoder()
                        
                        if ds == target_ds:
                            # Target (Finetune Strategy)
                            X_tr_local = X_target_full[train_target_mask]
                            if len(X_tr_local) > 0:
                                y_tr_local = le_local.fit_transform(y_target_full[train_target_mask])
                                train_data_dict_dl[ds] = (X_tr_local, y_tr_local)
                                le_target_local = le_local
                        else:
                            # Sources (Pretrain Strategy)
                            X_source = feature_cache[exp['pretrain']][ds]
                            if len(X_source) > 0:
                                y_tr_source = le_local.fit_transform(db_labels_mapped[ds])
                                train_data_dict_dl[ds] = (X_source, y_tr_source)

                    # Preparando dados de teste do Alvo
                    valid_test_idx = [i for i, lbl in enumerate(y_target_full[test_mask]) if lbl in le_target_local.classes_]
                    if len(valid_test_idx) == 0: continue
                    
                    X_test_dl = X_target_full[test_mask][valid_test_idx]
                    y_test_dl_enc = le_target_local.transform(y_target_full[test_mask][valid_test_idx])

                    # =========================================================================
                    # TREINAMENTO PROFUNDO (MLP & TABNET)[cite: 9]
                    # =========================================================================
                    dl_encoders = ['mlp', 'tabnet']
                    for enc in dl_encoders:
                        model_name = f"{enc.upper()} (TL)"
                        print(f"      -> Treinando {model_name}...")
                        try:
                            bal_acc, macro_f1, roc_auc, _ = train_and_evaluate_multihead(
                                train_data_dict=train_data_dict_dl,
                                target_dataset_name=target_ds,
                                X_test=X_test_dl,
                                y_test=y_test_dl_enc,
                                task=task,
                                epochs=15,          
                                batch_size=512,      
                                encoder_type=enc
                            )
                            print(f"         [{model_name}] Bal Acc: {bal_acc:.4f} | F1: {macro_f1:.4f}")
                            master_results.append({
                                "Exp Strategy": exp['id'],
                                "Dataset": target_ds,
                                "Task": task.capitalize(),
                                "Test Condition": test_cond,
                                "Model": model_name,
                                "Bal Acc": bal_acc,
                                "Macro F1": macro_f1,
                                "ROC-AUC": roc_auc
                            })
                        except Exception as e:
                            print(f"         [ERRO] Falha ao treinar {model_name}: {e}")
                            
                    # Exporta dinamicamente a cada loop para evitar perda de dados
                    pd.DataFrame(master_results).to_csv(csv_file, index=False)

    print(f"\n[SUCESSO] Relatório Final da Matriz DL Exportado: {csv_file}")

if __name__ == "__main__":
    run_normalization_tl_experiment()
