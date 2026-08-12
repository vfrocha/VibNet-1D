import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import StandardScaler

# Adiciona a raiz ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.features.extractors_v2 import extract_advanced_features
from src.features.signalai_wrapper import extract_fusion_features
from src.data.dataloader import load_vibration_data

# Importa os novos modelos de anomalia!
from src.models.build_anomaly import evaluate_classical_anomaly, evaluate_autoencoder_anomaly

# --- CONFIGURAÇÃO GLOBAL ---
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))

BASELINE_CONFIGS = {
    "CWRU_12k": {
        "fs": 12000,
        "conditions": ["Load_0HP", "Load_1HP", "Load_2HP", "Load_3HP"]
    },
    "HUST_Gearbox": {
        "fs": 25600,
        "conditions": ["Cond_20_0", "Cond_20_1", "Cond_20_2", "Cond_20_3", "Cond_20_4",
                       "Cond_25_0", "Cond_25_1", "Cond_25_2", "Cond_25_3", "Cond_25_4",
                       "Cond_30_0", "Cond_30_1", "Cond_30_2", "Cond_30_3", "Cond_30_4",
                       "Cond_35_0", "Cond_35_1", "Cond_35_2", "Cond_35_3", "Cond_35_4",
                       "Cond_40_0", "Cond_40_1", "Cond_40_2", "Cond_40_3", "Cond_40_4",
                       "Cond_L0_VS_0_40_0", "Cond_L1_VS_0_40_0", "Cond_L2_VS_0_40_0", "Cond_L3_VS_0_40_0", "Cond_L4_VS_0_40_0"]
    },
    "UOEMD": {
        "fs": 42000,
        "conditions": ["Load_Loaded_Speed_15Hz", "Load_Loaded_Speed_30Hz", "Load_Loaded_Speed_45Hz", "Load_Loaded_Speed_60Hz",
                       "Load_Loaded_Speed_Dec_45_to_15Hz", "Load_Loaded_Speed_Dec_60_to_30Hz", "Load_Loaded_Speed_Inc_15_to_45Hz",
                       "Load_Loaded_Speed_Inc_30_to_60Hz", "Load_No_Load_Speed_15Hz", "Load_No_Load_Speed_30Hz",
                       "Load_No_Load_Speed_45Hz", "Load_No_Load_Speed_60Hz", "Load_No_Load_Speed_Dec_45_to_15Hz",
                       "Load_No_Load_Speed_Dec_60_to_30Hz", "Load_No_Load_Speed_Inc_15_to_45Hz", "Load_No_Load_Speed_Inc_30_to_60Hz"]
    }
}

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

def run_anomaly_detection():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(RESULTS_DIR, f"log_anomaly_{timestamp}.txt")
    csv_file = os.path.join(RESULTS_DIR, f"anomaly_results_{timestamp}.csv")
    sys.stdout = Logger(log_file)
    
    print(f"{'='*80}\nEXPERIMENTOS DE DETECÇÃO DE ANOMALIAS (ONE-CLASS CLASSIFICATION)\n{'='*80}")
    print("Regra: O modelo será treinado EXCLUSIVAMENTE com a classe Normal (0).")
    
    master_results = []
    
    for dataset_name, config in BASELINE_CONFIGS.items():
        fs = config["fs"]
        conditions = config["conditions"]
        print(f"\n{'#'*60}\n DATASET: {dataset_name} | Fs: {fs}Hz\n{'#'*60}")
        
        for test_cond in conditions:
            print(f"\n   --- Dobra de Teste: {test_cond} ---")
            
            # 1. Carrega dados no esquema LOCO passando a task='detection'
            # O dataloader retorna 5 variáveis: X_train, y_train_enc, X_test, y_test_enc, label_encoder
            X_train_raw, y_train_encoded, X_test_raw, y_test_encoded, le = load_vibration_data(
                DATA_ROOT, dataset_name, test_cond, task='detection'
            )
            
            # Identifica qual número o LabelEncoder atribuiu à classe 'Normal'
            try:
                normal_idx = le.transform(['Normal'])[0]
            except ValueError:
                print(f"      [Aviso] Classe 'Normal' não encontrada no treino para {test_cond}. Pulando.")
                continue
                
            # Mapeamento Estrito para Anomalia: 0 = Normal, 1 = Falha
            y_train = (y_train_encoded != normal_idx).astype(int)
            y_test  = (y_test_encoded != normal_idx).astype(int)
            
            # ---------------------------------------------------------
            # A MÁGICA DA ANOMALIA: FILTRAR O TREINO APENAS PARA NORMAL
            # ---------------------------------------------------------
            mask_normal_train = (y_train == 0)
            X_train_raw_normal = np.array(X_train_raw)[mask_normal_train]
            
            if len(X_train_raw_normal) == 0:
                print(f"      [Aviso] Nenhum dado Normal no treino para {test_cond}. Pulando.")
                continue
                
            print(f"      -> Treinamento (Apenas Saudável): {len(X_train_raw_normal)} amostras")
            print(f"      -> Teste (Saudável + Falhas): {len(X_test_raw)} amostras")
            
            # 2. Extração de Features (Aplica na nova matriz filtrada e no teste)
            X_train_fusion = extract_fusion_features(X_train_raw_normal, fs, extract_advanced_features)
            X_test_fusion  = extract_fusion_features(X_test_raw, fs, extract_advanced_features)
            
            X_train_clean = np.nan_to_num(np.array(X_train_fusion, dtype=np.float32))
            X_test_clean  = np.nan_to_num(np.array(X_test_fusion, dtype=np.float32))
            
            # 3. Padronização Robusta (Fit apenas no Saudável!)
            scaler = StandardScaler()
            X_train_s = scaler.fit_transform(X_train_clean)
            X_test_s = scaler.transform(X_test_clean)
            
            # 4. Avaliação: Modelos Clássicos One-Class
            print("      -> Avaliando Isolation Forest e One-Class SVM...")
            results_classical = evaluate_classical_anomaly(X_train_s, X_test_s, y_test)
            for r in results_classical:
                r.update({"Dataset": dataset_name, "Test Condition": test_cond})
                print(f"         [{r['Model']}] F1 (Anomaly): {r['F1 (Anomaly)']:.4f}")
            
            # 5. Avaliação: Autoencoder Deep Learning com Limiares
            print("      -> Avaliando MLP Autoencoder com limiares estatísticos...")
            results_ae = evaluate_autoencoder_anomaly(X_train_s, X_test_s, y_test)
            for r in results_ae:
                r.update({"Dataset": dataset_name, "Test Condition": test_cond})
                print(f"         [{r['Model']}] F1 (Anomaly): {r['F1 (Anomaly)']:.4f}")
            
            # Adiciona na tabela geral
            master_results.extend(results_classical)
            master_results.extend(results_ae)
            
            pd.DataFrame(master_results).to_csv(csv_file, index=False)

    print(f"\n[SUCESSO] Resultados de Anomalia exportados para: {csv_file}")

if __name__ == "__main__":
    run_anomaly_detection()
