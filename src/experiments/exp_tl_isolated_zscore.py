import os
import sys
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import balanced_accuracy_score, f1_score

# Adiciona a raiz do projeto ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.features.extractors_v2 import extract_advanced_features
from src.features.signalai_wrapper import extract_fusion_features

# --- CONFIGURAÇÃO GLOBAL (TODAS AS BASES RESTAURADAS) ---
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))

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

TARGET_DATASET = "UOEMD"
# Defina aqui a condição que será separada estritamente para o Teste
TARGET_TEST_COND = "Load_No_Load_Speed_15Hz" # Exemplo para UOEMD

def load_entire_dataset_for_tl(dataset_name):
    """
    Carrega os dados brutos e os Rótulos.
    Mapeia automaticamente para a tarefa de DETECTION (Normal vs Fault).
    """
    dataset_path = os.path.join(DATA_ROOT, dataset_name)
    if not os.path.exists(dataset_path):
        print(f"[Aviso] Dataset {dataset_name} não encontrado no disco.")
        return [], [], []
        
    X_raw, y_raw_str, cond_raw = [], [], []
    healthy_keywords = ['normal', 'baseline', 'healthy', 'k001', 'k002', 'k003', 'k004', 'k005', 'k006']
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.npy'):
                class_name = os.path.basename(root)
                cond_name = os.path.basename(os.path.dirname(root))
                file_path = os.path.join(root, file)
                
                # Ignora classe Normal do CWRU_48k se houver inconsistência
                if dataset_name == "CWRU_48k" and any(k in class_name.lower() for k in healthy_keywords):
                    continue
                    
                # Padronização Universal para Tarefa de Detecção
                if any(keyword in class_name.lower() for keyword in healthy_keywords) or class_name.upper() == 'CLASS_H':
                    mapped_label = 'Normal'
                else:
                    mapped_label = 'Fault'
                    
                X_raw.append(np.load(file_path))
                y_raw_str.append(mapped_label)
                cond_raw.append(cond_name)
                
    return np.array(X_raw), np.array(y_raw_str), np.array(cond_raw)

def run_isolated_zscore_tl():
    print(f"{'='*80}\n EXPERIMENTO DE TRANSFER LEARNING COM Z-SCORE ISOLADO\n{'='*80}")
    
    db_features_raw = {}
    db_labels = {}
    db_conds = {}

    # 1. EXTRAÇÃO DE FEATURES
    print("--- FASE 1: EXTRAÇÃO BRUTA ---")
    for ds_name, fs in DATASETS_CONFIG.items():
        X_raw, y_raw, cond_raw = load_entire_dataset_for_tl(ds_name)
        
        if len(X_raw) == 0:
            continue
            
        print(f"  -> {ds_name}: Extraindo features de {X_raw.shape[0]} amostras...")
        X_features = extract_fusion_features(X_raw, fs, extract_advanced_features)
        X_features = np.nan_to_num(np.array(X_features, dtype=np.float32))
        
        db_features_raw[ds_name] = X_features
        db_labels[ds_name] = y_raw
        db_conds[ds_name] = cond_raw

    # 2. SEPARAÇÃO LOCO NO ALVO
    print(f"\n--- FASE 2: SEPARAÇÃO LOCO NO ALVO ({TARGET_DATASET}) ---")
    mask_test = np.array([TARGET_TEST_COND in cond for cond in db_conds[TARGET_DATASET]])
    mask_train = ~mask_test
    
    print(f"  -> Amostras de Treino no Alvo: {np.sum(mask_train)}")
    print(f"  -> Amostras de Teste no Alvo: {np.sum(mask_test)}")

    # 3. NORMALIZAÇÃO Z-SCORE ISOLADA POR MÁQUINA
    print("\n--- FASE 3: NORMALIZAÇÃO Z-SCORE ISOLADA ---")
    X_train_list_scaled = []
    y_train_list = []
    target_scaler = None 
    
    for ds_name in db_features_raw.keys():
        scaler = StandardScaler()
        
        if ds_name == TARGET_DATASET:
            X_tr_raw = db_features_raw[ds_name][mask_train]
            y_tr = db_labels[ds_name][mask_train]
            if len(X_tr_raw) > 0:
                X_tr_scaled = scaler.fit_transform(X_tr_raw)
                target_scaler = scaler 
                X_train_list_scaled.append(X_tr_scaled)
                y_train_list.append(y_tr)
                print(f"  -> {ds_name} (Alvo - Treino): Padronizado isoladamente.")
        else:
            X_tr_raw = db_features_raw[ds_name]
            y_tr = db_labels[ds_name]
            X_tr_scaled = scaler.fit_transform(X_tr_raw)
            X_train_list_scaled.append(X_tr_scaled)
            y_train_list.append(y_tr)
            print(f"  -> {ds_name} (Source Inteira): Padronizado isoladamente.")

    # 4. CONCATENAÇÃO GLOBAL
    X_train_global = np.vstack(X_train_list_scaled)
    y_train_global = np.concatenate(y_train_list)
    
    le_global = LabelEncoder()
    y_train_enc = le_global.fit_transform(y_train_global)
    
    # 5. PREPARAÇÃO DO TESTE
    X_test_raw = db_features_raw[TARGET_DATASET][mask_test]
    y_test_raw = db_labels[TARGET_DATASET][mask_test]
    
    X_test_scaled = target_scaler.transform(X_test_raw)
    y_test_enc = le_global.transform(y_test_raw)
    
    print(f"\nMatriz de Treino Global Final: {X_train_global.shape}")
    print(f"Matriz de Teste Final: {X_test_scaled.shape}")
    print(f"Classes identificadas: {le_global.classes_}")

    # 6. TREINAMENTO
    print("\n--- FASE 4: TREINAMENTO E AVALIAÇÃO ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train_global, y_train_enc)
    
    y_pred = rf.predict(X_test_scaled)
    acc = balanced_accuracy_score(y_test_enc, y_pred)
    f1 = f1_score(y_test_enc, y_pred, average='macro')
    
    print(f"Random Forest (Multi-Source TL Isolado) -> Bal Acc: {acc:.4f} | Macro F1: {f1:.4f}")

if __name__ == "__main__":
    run_isolated_zscore_tl()
