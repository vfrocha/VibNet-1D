import os
import sys
import numpy as np

# Adiciona a raiz do projeto ao path para localizar a pasta data corretamente
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# --- CONFIGURAÇÃO GLOBAL ---
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

def run_amplitude_audit():
    print(f"{'='*70}\n AUDITORIA DE AMPLITUDES FÍSICAS NOS DADOS PROCESSADOS (.npy)\n{'='*70}")

    for dataset_name in DATASETS_CONFIG.keys():
        dataset_path = os.path.join(DATA_ROOT, dataset_name)
        
        if not os.path.exists(dataset_path):
            print(f"[{dataset_name}] [AVISO] Pasta não encontrada em: {dataset_path}")
            print("-" * 70)
            continue

        global_max = -np.inf
        global_min = np.inf
        stds = []
        file_count = 0

        # Percorre as pastas lendo os arquivos .npy usando a mesma lógica do seu dataloader
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if file.endswith('.npy'):
                    file_path = os.path.join(root, file)
                    
                    # Carrega a janela de vibração individualmente
                    sinal = np.load(file_path)
                    
                    # Atualiza os limites globais do dataset
                    current_max = np.max(sinal)
                    current_min = np.min(sinal)
                    
                    if current_max > global_max: global_max = current_max
                    if current_min < global_min: global_min = current_min
                    
                    # Salva a energia (desvio padrão) para calcular a média do dataset
                    stds.append(np.std(sinal))
                    file_count += 1

        if file_count > 0:
            mean_std = np.mean(stds)
            print(f"Dataset: {dataset_name.upper()} ({file_count} janelas analisadas)")
            print(f"  -> Pico Máximo Global: {global_max:.4f}")
            print(f"  -> Pico Mínimo Global: {global_min:.4f}")
            print(f"  -> Desvio Padrão Médio (Energia): {mean_std:.4f}")
            
            # Diagnóstico da Unidade de Medida
            # Valores acima de ~15 ou 20 geralmente indicam sinais elétricos brutos (mV ou V)
            # Valores entre -5 e 5 geralmente indicam sinais já convertidos para gravidade (g)
            if global_max > 15.0 or global_min < -15.0:
                print(f"  -> [Diagnóstico]: Amplitudes elevadas. Sinal provavelmente em Volts (mV) brutos do DAQ.")
            else:
                print(f"  -> [Diagnóstico]: Amplitudes compactas. Sinal provavelmente calibrado em aceleração (g) ou m/s².")
        else:
            print(f"[{dataset_name}] [AVISO] Nenhum arquivo .npy encontrado nas subpastas.")
            
        print("-" * 70)

if __name__ == "__main__":
    run_amplitude_audit()
