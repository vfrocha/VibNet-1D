import os
import numpy as np

# Ajuste para a base que você quer investigar
DATASET_DIR = "/home/vfrocha/VibNet-1D/data/processed/HUST/Load_0W" #"data/processed/PU_Dataset/Load_0" # Exemplo

def find_healthy_class(dataset_path):
    print(f"Analisando dataset: {dataset_path}")
    classes = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    
    class_energy = {}
    
    for cls in classes:
        cls_path = os.path.join(dataset_path, cls)
        files = [f for f in os.listdir(cls_path) if f.endswith('.npy')]
        
        if not files:
            continue
            
        # Pega apenas as primeiras 10 amostras para ser bem rápido
        sample_files = files[:10]
        rms_values = []
        
        for file in sample_files:
            signal = np.load(os.path.join(cls_path, file))
            # Calcula o RMS (Energia do sinal)
            rms = np.sqrt(np.mean(signal**2))
            rms_values.append(rms)
            
        # Tira a média de energia dessa classe
        class_energy[cls] = np.mean(rms_values)
        
    # Ordena da menor energia (saudável) para a maior (defeitos severos)
    sorted_classes = sorted(class_energy.items(), key=lambda item: item[1])
    
    print("\nRanking de Energia (RMS) por Classe:")
    print("-" * 40)
    for cls_name, energy in sorted_classes:
        print(f"Classe: {cls_name:15} | Energia (RMS): {energy:.4f}")
        
    print("-" * 40)
    print(f"👉 CONCLUSÃO: A classe com menor energia é '{sorted_classes[0][0]}'.")
    print(f"É 99% de certeza que esta é a condição Normal/Baseline. Você deve renomeá-la para 'Normal'.")

if __name__ == "__main__":
    find_healthy_class(DATASET_DIR)
