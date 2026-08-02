import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime
from tqdm import tqdm

# Adiciona a raiz do projeto ao path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.features.extractors_v2 import extract_advanced_features
from src.features.signalai_wrapper import extract_fusion_features
from src.models.build_sklearn import evaluate_all_models

# --- CONFIGURAÇÃO GLOBAL ---
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))
RESULTS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../results'))

# Lista das bases que participarão do Transfer Learning
DATASETS_CONFIG = {
    "CWRU_12k": 12000,
    "CWRU_48k": 48000,
    "UOEMD": 42000,
    "HUST_Gearbox": 25600
}

# Focaremos na detecção para garantir harmonia entre rótulos de máquinas diferentes
TASK = "detection" 

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
    Carrega TODOS os arquivos de uma base de dados específica.
    Como é Transfer Learning, não separamos por condição (Fold), carregamos a máquina inteira.
    Converte os rótulos para Binário (0 = Normal, 1 = Fault).
    """
    dataset_path = os.path.join(DATA_ROOT, dataset_name)
    if not os.path.exists(dataset_path):
        print(f"[Aviso] Dataset {dataset_name} não encontrado no disco.")
        return [], []
        
    X_raw = []
    y_raw = []
    
    print(f"  -> Lendo arquivos de {dataset_name}...")
    # Percorre todas as pastas (Condições) e subpastas (Classes)
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.npy'):
                class_name = os.path.basename(root)
                file_path = os.path.join(root, file)
                
                # Mapeamento Universal Binário (Harmonização)
                if 'normal' in class_name.lower():
                    label = 0 # Saudável
                else:
                    label = 1 # Falha (Qualquer tipo)
                    
                # Ignora a CWRU_48k se não tiver dados normais, pois enviesaria a detecção
                if dataset_name == "CWRU_48k" and label == 0:
                    continue # Segurança extra, embora já saibamos que 48k não tem normal
                    
                X_raw.append(np.load(file_path))
                y_raw.append(label)
                
    return X_raw, y_raw

def run_transfer_learning():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(RESULTS_DIR, f"log_transfer_learning_{timestamp}.txt")
    csv_file = os.path.join(RESULTS_DIR, f"tl_results_{timestamp}.csv")
    sys.stdout = Logger(log_file)
    
    print(f"{'='*80}\nEXPERIMENTOS DE TRANSFER LEARNING (MULTI-SOURCE -> SINGLE TARGET)\n{'='*80}")
    print("Tarefa: DETECTION (0 = Normal, 1 = Fault)")
    print(f"Datsets disponíveis: {list(DATASETS_CONFIG.keys())}\n")

    master_results = []
    
    # Pré-carrega e extrai as features de todas as bases para economizar memória e tempo
    # A MÁGICA: O fs é diferente para cada base, mas o extrator gera sempre 141 colunas padronizadas!
    db_features = {}
    db_labels = {}
    
    print("--- FASE 1: EXTRAÇÃO DE FEATURES DE TODAS AS MÁQUINAS ---")
    for ds_name, fs in DATASETS_CONFIG.items():
        X_raw, y_raw = load_entire_dataset_for_tl(ds_name, fs)
        if len(X_raw) > 0:
            print(f"  -> Extraindo Fusion Features (141 cols) para {ds_name} (fs={fs}Hz)...")
            X_fusion = extract_fusion_features(X_raw, fs, extract_advanced_features)
            
            # Limpeza de NaNs
            X_clean = np.nan_to_num(np.array(X_fusion, dtype=np.float32))
            if X_clean.ndim == 1: X_clean = X_clean.reshape(len(y_raw), -1)
            
            db_features[ds_name] = X_clean
            db_labels[ds_name] = np.array(y_raw)
            print(f"  -> {ds_name} pronto: {X_clean.shape[0]} amostras.\n")

    # --- FASE 2: VALIDAÇÃO CRUZADA LEAVE-ONE-DOMAIN-OUT ---
    print("\n--- FASE 2: TREINAMENTO MULTI-SOURCE ---")
    available_datasets = list(db_features.keys())
    
    for target_ds in available_datasets:
        # Se CWRU_48k não tem dados normais, não podemos usá-lo como alvo de teste para Detecção
        if target_ds == "CWRU_48k" or len(np.unique(db_labels[target_ds])) < 2:
            print(f"\n[!] Pulando {target_ds} como Alvo (Não possui as duas classes para Detecção).")
            continue
            
        print(f"\n{'#'*60}")
        print(f" ALVO DE TESTE (TARGET DOMAIN): {target_ds}")
        
        # Define os domínios de origem (Source) - Tudo que não for o Target
        source_datasets = [ds for ds in available_datasets if ds != target_ds]
        print(f" TREINADO EM (SOURCE DOMAINS): {', '.join(source_datasets)}")
        print(f"{'#'*60}")
        
        # Concatena todas as amostras das bases Source
        X_train_list = [db_features[ds] for ds in source_datasets]
        y_train_list = [db_labels[ds] for ds in source_datasets]
        
        X_train = np.vstack(X_train_list)
        y_train = np.concatenate(y_train_list)
        
        # Pega a base Target para Teste
        X_test = db_features[target_ds]
        y_test = db_labels[target_ds]
        
        print(f"  -> Tamanho do Treino (Múltiplas Máquinas): {X_train.shape[0]} amostras")
        print(f"  -> Tamanho do Teste (Máquina Desconhecida): {X_test.shape[0]} amostras")
        
        # O modelo vai prever as métricas e salvar
        # Usamos 'dataset_name' na tabela como os Sources, e 'test_cond' como o Target
        source_name_str = "+".join(source_datasets)
        
        current_results = evaluate_all_models(
            X_train, y_train, X_test, y_test, 
            dataset_name=source_name_str, 
            task=TASK, 
            test_cond=target_ds # O Fold agora é o Dataset inteiro!
        )
        
        master_results.extend(current_results)
        
        # Salva incrementalmente
        df = pd.DataFrame(master_results)
        df.to_csv(csv_file, index=False)

    print(f"\n[SUCESSO] Tabela de Transfer Learning exportada para: {csv_file}")
    
    if master_results:
        df = pd.DataFrame(master_results)
        print("\n--- RESUMO GERAL DO TRANSFER LEARNING (MACRO F1) ---")
        summary = df.groupby(['Test Condition', 'Model'])['Macro F1'].mean().unstack()
        print(summary.to_string())

if __name__ == "__main__":
    run_transfer_learning()
