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

# --- CONFIGURAÇÃO GLOBAL ---
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))

DATASETS_CONFIG = {
    "CWRU_12k": 12000,
    "HUST_Gearbox": 25600,
    "UOEMD": 42000
}
TARGET_DATASET = "UOEMD"

def mock_load_data(dataset_name):
    """Função adaptadora para puxar seus dados. Substitua pelo seu dataloader real."""
    # Simulação de carregamento para estrutura do script
    X_raw = np.random.rand(100, 12000) # Exemplo: 100 janelas
    y_raw = np.array([f"{dataset_name}_Fault"] * 50 + [f"{dataset_name}_Normal"] * 50)
    conds = np.array(["Cond_A"] * 50 + ["Cond_B"] * 50)
    return X_raw, y_raw, conds

def run_isolated_zscore_tl():
    print(f"{'='*80}\n EXPERIMENTO DE TRANSFER LEARNING COM Z-SCORE ISOLADO\n{'='*80}")
    
    db_features_raw = {}
    db_labels = {}
    db_conds = {}

    # 1. EXTRAÇÃO DE FEATURES (Sem Z-Score AINDA)
    print("--- FASE 1: EXTRAÇÃO BRUTA ---")
    for ds_name, fs in DATASETS_CONFIG.items():
        X_raw, y_raw, cond_raw = mock_load_data(ds_name)
        
        # Extrai as 141 features tabulares
        X_features = extract_fusion_features(X_raw, fs, extract_advanced_features)
        X_features = np.nan_to_num(np.array(X_features, dtype=np.float32))
        
        db_features_raw[ds_name] = X_features
        db_labels[ds_name] = y_raw
        db_conds[ds_name] = cond_raw
        print(f"  -> {ds_name}: {X_features.shape[0]} amostras extraídas.")

    # 2. SEPARAÇÃO TREINO / TESTE (LOCO no Alvo)
    print("\n--- FASE 2: SEPARAÇÃO LOCO NO ALVO ---")
    target_test_cond = "Cond_B" # Exemplo de condição que vai ficar de fora para teste
    
    mask_test = (db_conds[TARGET_DATASET] == target_test_cond)
    mask_train = ~mask_test
    
    # 3. O SEGREDO DA NORMALIZAÇÃO ISOLADA POR MÁQUINA
    print("\n--- FASE 3: NORMALIZAÇÃO Z-SCORE ISOLADA ---")
    X_train_list_scaled = []
    y_train_list = []
    
    target_scaler = None # O scaler do alvo precisa ser salvo para normalizar o teste depois
    
    for ds_name in db_features_raw.keys():
        scaler = StandardScaler()
        
        if ds_name == TARGET_DATASET:
            # Pega APENAS o treino da máquina alvo para fitar o Scaler (Evita Data Leakage)
            X_tr_raw = db_features_raw[ds_name][mask_train]
            y_tr = db_labels[ds_name][mask_train]
            
            X_tr_scaled = scaler.fit_transform(X_tr_raw)
            target_scaler = scaler # Salva a média e desvio apenas do alvo
            
            print(f"  -> {ds_name} (Alvo - Treino) Padronizado isoladamente.")
        else:
            # Máquinas externas: padroniza a máquina inteira
            X_tr_raw = db_features_raw[ds_name]
            y_tr = db_labels[ds_name]
            
            X_tr_scaled = scaler.fit_transform(X_tr_raw)
            print(f"  -> {ds_name} (Source Inteira) Padronizado isoladamente.")
            
        X_train_list_scaled.append(X_tr_scaled)
        y_train_list.append(y_tr)

    # 4. CONCATENAÇÃO GLOBAL (Agora é seguro juntar!)
    X_train_global = np.vstack(X_train_list_scaled)
    y_train_global = np.concatenate(y_train_list)
    
    # Label Encoding Universal
    le_global = LabelEncoder()
    y_train_enc = le_global.fit_transform(y_train_global)
    
    # 5. PREPARAÇÃO DO TESTE
    X_test_raw = db_features_raw[TARGET_DATASET][mask_test]
    y_test_raw = db_labels[TARGET_DATASET][mask_test]
    
    # Padroniza o teste usando o scaler do ALVO (Salvo no passo 3)
    X_test_scaled = target_scaler.transform(X_test_raw)
    y_test_enc = le_global.transform(y_test_raw)
    
    print(f"\nMatriz de Treino Global Final: {X_train_global.shape}")
    print(f"Matriz de Teste Final ({TARGET_DATASET}): {X_test_scaled.shape}")

    # 6. TREINAMENTO DOS MODELOS
    print("\n--- FASE 4: TREINAMENTO E AVALIAÇÃO ---")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_global, y_train_enc)
    
    y_pred = rf.predict(X_test_scaled)
    acc = balanced_accuracy_score(y_test_enc, y_pred)
    f1 = f1_score(y_test_enc, y_pred, average='macro')
    
    print(f"Random Forest -> Bal Acc: {acc:.4f} | Macro F1: {f1:.4f}")
    
if __name__ == "__main__":
    run_isolated_zscore_tl()
