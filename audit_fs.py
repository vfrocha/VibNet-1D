import os
import scipy.io as sio
import numpy as np

RAW_DIR = "data/raw" # Ajuste para a sua pasta de dados brutos (mat ou csv)

def audit_datasets():
    print("="*60)
    print(" AUDITORIA DE DADOS BRUTOS (Verificação de $fs$)")
    print("="*60)
    
    if not os.path.exists(RAW_DIR):
        print(f"Pasta {RAW_DIR} não encontrada.")
        return

    for dataset in os.listdir(RAW_DIR):
        dataset_path = os.path.join(RAW_DIR, dataset)
        if not os.path.isdir(dataset_path): continue
        
        print(f"\n[{dataset.upper()}]")
        
        # Pega até 3 arquivos de exemplo dentro das subpastas
        files_checked = 0
        for root, dirs, files in os.walk(dataset_path):
            for file in files:
                if files_checked >= 3: break
                
                file_path = os.path.join(root, file)
                try:
                    if file.endswith('.mat'):
                        mat_data = sio.loadmat(file_path)
                        # Procura o array principal de vibração dentro do .mat
                        for key in mat_data.keys():
                            if not key.startswith('__'):
                                shape = mat_data[key].shape
                                # Pega o maior eixo
                                length = max(shape) if len(shape) > 0 else 0
                                if length > 1000: # Ignora metadados curtos
                                    print(f"  -> Arquivo: {file} | Subpasta: {os.path.basename(root)} | Pontos totais: {length}")
                                    break
                    elif file.endswith('.npy'):
                        data = np.load(file_path)
                        print(f"  -> Arquivo: {file} | Subpasta: {os.path.basename(root)} | Pontos totais: {max(data.shape)}")
                except Exception as e:
                    pass
                
                files_checked += 1

if __name__ == "__main__":
    audit_datasets()
