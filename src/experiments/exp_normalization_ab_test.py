import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime

# Adiciona a raiz do projeto ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.features.extractors_v2 import extract_advanced_features
from src.features.signalai_wrapper import extract_fusion_features
from src.models.build_tabnet_resnet import train_and_evaluate_multihead
from sklearn.preprocessing import LabelEncoder

# --- CONFIGURAÇÃO GLOBAL ---
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))

# Selecionamos um Target e alguns Sources para rodar rápido o teste de conceito
TARGET_DATASET = "UOEMD"
SOURCE_DATASETS = {"CWRU_12k": 12000, "HUST_Gearbox": 25600, "PU": 64000}

def load_data_with_normalization_strategy(dataset_name, fs, strategy):
    """
    Carrega o dataset e aplica diferentes estratégias de normalização nas janelas de tempo.
    Estratégias suportadas: 'global_zscore', 'window_zscore', 'raw'
    """
    dataset_path = os.path.join(DATA_ROOT, dataset_name)
    if not os.path.exists(dataset_path):
        return [], [], []
        
    X_raw, y_raw_str, cond_raw = [], [], []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.npy'):
                class_name = os.path.basename(root)
                cond_name = os.path.basename(os.path.dirname(root))
                
                sinal = np.load(os.path.join(root, file))
                
                # Aplica as ideias do orientador
                if strategy == 'window_zscore':
                    std = np.std(sinal)
                    if std > 0:
                        sinal = (sinal - np.mean(sinal)) / std
                        
                X_raw.append(sinal)
                y_raw_str.append(class_name)
                cond_raw.append(cond_name)
                
    X_raw_np = np.array(X_raw, dtype=np.float32)
    
    if strategy == 'global_zscore' and len(X_raw_np) > 0:
        dataset_mean = np.mean(X_raw_np)
        dataset_std = np.std(X_raw_np)
        if dataset_std > 0:
            X_raw_np = (X_raw_np - dataset_mean) / dataset_std
            
    return list(X_raw_np), y_raw_str, cond_raw

def run_normalization_ab_test():
    print(f"{'='*80}\n ABLATION TEST: ESTRATÉGIAS DE NORMALIZAÇÃO DE SINAIS HETEROGÊNEOS\n{'='*80}")
    
    strategies = [
        {'name': 'Baseline_Global_ZScore', 'pretrain': 'global_zscore', 'finetune': 'global_zscore'},
        {'name': 'Window_ZScore_Only', 'pretrain': 'window_zscore', 'finetune': 'window_zscore'},
        {'name': 'Asymmetric_Training', 'pretrain': 'window_zscore', 'finetune': 'raw'}
    ]
    
    master_results = []
    
    for strat in strategies:
        print(f"\n\n{'*'*60}\n TESTANDO ESTRATÉGIA: {strat['name']}\n{'*'*60}")
        print(f"  -> Pre-training mode: {strat['pretrain']}")
        print(f"  -> Fine-tuning mode: {strat['finetune']}")
        
        train_data_dict_dl = {}
        
        # 1. Carrega os Source Datasets (Pré-treino) com a estratégia definida
        for ds_name, fs in SOURCE_DATASETS.items():
            X_raw, y_str, _ = load_data_with_normalization_strategy(ds_name, fs, strat['pretrain'])
            if len(X_raw) > 0:
                # Se for testar a injeção do RMS Original, você pode calcular aqui e concatenar no X_features
                X_features = extract_fusion_features(np.array(X_raw), fs, extract_advanced_features)
                X_clean = np.nan_to_num(np.array(X_features, dtype=np.float32))
                
                le_local = LabelEncoder()
                y_enc = le_local.fit_transform(y_str)
                train_data_dict_dl[ds_name] = (X_clean, y_enc)
                print(f"    - {ds_name} (Source) carregado. {X_clean.shape[0]} amostras.")

        # 2. Carrega o Target Dataset com a estratégia de Fine-Tuning
        X_target_raw, y_target_str, cond_target = load_data_with_normalization_strategy(TARGET_DATASET, 42000, strat['finetune'])
        X_target_features = extract_fusion_features(np.array(X_target_raw), 42000, extract_advanced_features)
        X_target_clean = np.nan_to_num(np.array(X_target_features, dtype=np.float32))
        
        cond_target = np.array(cond_target)
        y_target_str = np.array(y_target_str)
        
        le_target = LabelEncoder()
        
        # 3. LOCO Validation no Target
        unique_conds = np.unique(cond_target)
        for test_cond in unique_conds:
            print(f"\n   --- Testando no domínio alvo ({TARGET_DATASET}) - Dobra: {test_cond} ---")
            
            test_mask = (cond_target == test_cond)
            train_mask = ~test_mask
            
            X_train_target = X_target_clean[train_mask]
            y_train_target = le_target.fit_transform(y_target_str[train_mask])
            
            X_test_target = X_target_clean[test_mask]
            
            valid_test_idx = [i for i, lbl in enumerate(y_target_str[test_mask]) if lbl in le_target.classes_]
            if len(valid_test_idx) == 0: continue
            
            X_test_target_valid = X_test_target[valid_test_idx]
            y_test_target_valid = le_target.transform(y_target_str[test_mask][valid_test_idx])
            
            # Adiciona o alvo no dicionário de treino para a rede Multi-Head
            train_data_dict_dl[TARGET_DATASET] = (X_train_target, y_train_target)
            
            # Executa o treino
            try:
                bal_acc, macro_f1, roc_auc, _ = train_and_evaluate_multihead(
                    train_data_dict=train_data_dict_dl,
                    target_dataset_name=TARGET_DATASET,
                    X_test=X_test_target_valid,
                    y_test=y_test_target_valid,
                    task='diagnosis',
                    epochs=15,
                    batch_size=512,
                    encoder_type='mlp' # Pode testar 'tabnet' também
                )
                print(f"      [RESULTADO] Bal Acc: {bal_acc:.4f} | F1: {macro_f1:.4f}")
                master_results.append({
                    "Strategy": strat['name'], 
                    "Test Condition": test_cond, 
                    "Bal Acc": bal_acc, 
                    "Macro F1": macro_f1
                })
            except Exception as e:
                print(f"      [ERRO] na execução da rede: {e}")

    # Salva os resultados para análise
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_file = os.path.join(RESULTS_DIR, f"normalization_ab_test_{timestamp}.csv")
    pd.DataFrame(master_results).to_csv(csv_file, index=False)
    print(f"\n[SUCESSO] Relatório exportado para: {csv_file}")

if __name__ == "__main__":
    run_normalization_ab_test()
