import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from scipy.signal import detrend

from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier

# Adiciona a raiz do projeto ao path para importar seus módulos originais
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.features.extractors_v2 import extract_advanced_features
from src.features.signalai_wrapper import extract_fusion_features
from src.models.build_tabnet_resnet import train_and_evaluate_multihead

# --- CONFIGURAÇÃO GLOBAL ---
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))

TARGET_DATASET = "UOEMD"
# Frequência de amostragem hipotética baseada nas conversas (ex: HUST 25.6kHz, PU 64kHz)
SOURCE_DATASETS = {"CWRU_12k": 12000, "HUST_Gearbox": 25600, "PU": 64000} 
TARGET_FS = 42000

def normalize_time_window(sinal, strategy):
    """
    Aplica a normalização diretamente na série temporal (janela de 1s).
    O detrend é aplicado como base para zerar a média (remover offset DC)
    sem distorcer a amplitude física da aceleração.
    """
    # 1. Base obrigatória: Remove a tendência linear[cite: 7]
    sinal_detrend = detrend(sinal)
    
    # 2. Estratégias do Flávio
    if strategy == 'raw':
        return sinal_detrend
        
    elif strategy == 'window_zscore':
        std = np.std(sinal_detrend)
        if std > 0:
            return sinal_detrend / std
        return sinal_detrend
        
    elif strategy == 'window_rms':
        # Calcula o RMS original da janela (energia média quadrática)
        rms = np.sqrt(np.mean(sinal_detrend**2))
        if rms > 0:
            return sinal_detrend / rms
        return sinal_detrend
        
    return sinal_detrend

def load_and_extract(dataset_name, fs, window_strategy):
    """
    Carrega os dados .npy do disco, aplica a normalização na janela temporal,
    e extrai a matriz de features tabulares fundidas (SignAI + VibNet).
    """
    dataset_path = os.path.join(DATA_ROOT, dataset_name)
    if not os.path.exists(dataset_path):
        return None, [], []
        
    X_windows = []
    y_str = []
    conds = []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.npy'):
                class_name = os.path.basename(root)
                cond_name = os.path.basename(os.path.dirname(root))
                
                sinal = np.load(os.path.join(root, file))
                
                # APLICA A NORMALIZAÇÃO NO SINAL DE TEMPO BRUTO
                sinal_norm = normalize_time_window(sinal, window_strategy)
                
                X_windows.append(sinal_norm)
                y_str.append(class_name)
                conds.append(cond_name)
                
    if len(X_windows) == 0:
        return None, [], []

    print(f"      -> Extraindo features para {dataset_name} ({window_strategy})...")
    # Extrai as 141 features combinadas (VibNet + SignAI)
    X_features = extract_fusion_features(np.array(X_windows), fs, extract_advanced_features)
    X_features_clean = np.nan_to_num(np.array(X_features, dtype=np.float32))
    
    return X_features_clean, np.array(y_str), np.array(conds)

def run_abc_matrix_experiment():
    print(f"{'='*80}\n EXPERIMENTO: MATRIZ DE NORMALIZAÇÃO DE JANELAS (LODO-CV)\n{'='*80}")
    
    # A grade desenhada na ótica do Flávio
    experiment_matrix = [
        {'id': '0_Baseline', 'pretrain': 'raw', 'finetune': 'raw'},
        {'id': '1_Estrategia_A_Pura', 'pretrain': 'window_zscore', 'finetune': 'window_zscore'},
        {'id': '2_Estrategia_B_Pura', 'pretrain': 'window_rms', 'finetune': 'window_rms'},
        {'id': '3_Estrategia_C_com_A', 'pretrain': 'window_zscore', 'finetune': 'raw'},
        {'id': '4_Estrategia_C_com_B', 'pretrain': 'window_rms', 'finetune': 'raw'}
    ]
    
    master_results = []
    
    for exp in experiment_matrix:
        print(f"\n\n{'*'*60}\n EXECUTANDO: {exp['id']}\n{'*'*60}")
        print(f"  -> Sinal na Fase Pré-Treino (Source): {exp['pretrain']}")
        print(f"  -> Sinal na Fase Teste/Target: {exp['finetune']}")
        
        # --- ETAPA 1: Processamento dos Sources (Pré-Treino) ---
        X_source_list = []
        y_source_list = []
        
        for ds_name, fs in SOURCE_DATASETS.items():
            X_feat, y_str, _ = load_and_extract(ds_name, fs, exp['pretrain'])
            if X_feat is not None:
                # Transforma rótulos em formato global ex: "CWRU_12k_InnerRace"
                y_global = np.array([f"{ds_name}_{lbl}" for lbl in y_str])
                X_source_list.append(X_feat)
                y_source_list.append(y_global)
                
        X_source_full = np.vstack(X_source_list) if X_source_list else np.array([])
        y_source_full = np.concatenate(y_source_list) if y_source_list else np.array([])
        
        # --- ETAPA 2: Processamento do Target (Fine-Tuning/Teste) ---
        X_target_feat, y_target_str, conds_target = load_and_extract(TARGET_DATASET, TARGET_FS, exp['finetune'])
        y_target_global = np.array([f"{TARGET_DATASET}_{lbl}" for lbl in y_target_str])
        
        unique_conds = np.unique(conds_target)
        
        # --- ETAPA 3: Validação Leave-One-Condition-Out (LOCO) ---
        for test_cond in unique_conds:
            print(f"\n   --- Avaliando Dobra de Teste: {test_cond} ---")
            
            # Mascaramento do Target
            test_mask = (conds_target == test_cond)
            train_mask = ~test_mask
            
            X_target_train = X_target_feat[train_mask]
            y_target_train = y_target_global[train_mask]
            
            X_target_test = X_target_feat[test_mask]
            y_target_test = y_target_global[test_mask]
            
            # Concatena Source + Treino do Target
            if len(X_source_full) > 0:
                X_train_final = np.vstack([X_source_full, X_target_train])
                y_train_final = np.concatenate([y_source_full, y_target_train])
            else:
                X_train_final, y_train_final = X_target_train, y_target_train
                
            # OBRIGATÓRIO: Padronização (Z-Score) apenas no espaço tabular de features[cite: 4, 7]
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_final)
            X_test_scaled = scaler.transform(X_target_test)
            
            le = LabelEncoder()
            y_train_enc = le.fit_transform(y_train_final)
            
            # Proteção de índices no teste
            valid_idx = [i for i, lbl in enumerate(y_target_test) if lbl in le.classes_]
            if not valid_idx: continue
            
            X_test_scaled = X_test_scaled[valid_idx]
            y_test_enc = le.transform(y_target_test[valid_idx])
            
            # --- ETAPA 4: Classificação (Baseline Multi-Domínio Clássico) ---
            # (Aqui você pode plugar a sua Multi-Head DL passando os dataloaders)
            rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            rf.fit(X_train_scaled, y_train_enc)
            
            from sklearn.metrics import balanced_accuracy_score, f1_score
            y_pred = rf.predict(X_test_scaled)
            bal_acc = balanced_accuracy_score(y_test_enc, y_pred)
            f1 = f1_score(y_test_enc, y_pred, average='macro')
            
            print(f"      [RESULTADO] Random Forest -> Bal Acc: {bal_acc:.4f} | Macro F1: {f1:.4f}")
            
            master_results.append({
                "Experiment ID": exp['id'],
                "Target": TARGET_DATASET,
                "Test Condition": test_cond,
                "Model": "Random Forest",
                "Bal Acc": bal_acc,
                "Macro F1": f1
            })

    # Exportação dos resultados
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(RESULTS_DIR, f"normalization_abc_results_{timestamp}.csv")
    pd.DataFrame(master_results).to_csv(csv_file, index=False)
    print(f"\n[SUCESSO] Relatório exportado para: {csv_file}")

if __name__ == "__main__":
    run_abc_matrix_experiment()
