import os
import scipy.io as sio
import numpy as np
import pandas as pd

RAW_DIR = "data/raw"

def audit_datasets():
    print("="*60)
    print(f" AUDITORIA DE DADOS BRUTOS (Buscando em {os.path.abspath(RAW_DIR)})")
    print("="*60)
    
    if not os.path.exists(RAW_DIR):
        print(f"Pasta {RAW_DIR} não encontrada.")
        return

    itens_na_raiz = os.listdir(RAW_DIR)
    print(f"[*] Itens encontrados na pasta raiz: {len(itens_na_raiz)}")

    arquivos_auditados = 0
    
    for root, dirs, files in os.walk(RAW_DIR):
        for file in files:
            file_path = os.path.join(root, file)
            tamanho = 0
            
            try:
                if file.endswith('.mat'):
                    mat_data = sio.loadmat(file_path)
                    for key in mat_data.keys():
                        if not key.startswith('__'):
                            shape = mat_data[key].shape
                            length = max(shape) if len(shape) > 0 else 0
                            if length > 1000:
                                tamanho = length
                                break
                elif file.endswith('.npy'):
                    data = np.load(file_path)
                    tamanho = max(data.shape)
                elif file.endswith('.csv'):
                    # Lê só a primeira coluna para ser rápido
                    df = pd.read_csv(file_path, usecols=[0])
                    tamanho = len(df)
                    
                if tamanho > 0:
                    pasta_pai = os.path.basename(root)
                    print(f" -> [{pasta_pai}] Arquivo: {file} | Pontos: {tamanho}")
                    arquivos_auditados += 1
                    
            except Exception as e:
                pass
                
            # Para não poluir a tela, audita no máximo 15 arquivos no total
            if arquivos_auditados >= 15:
                print("\n[*] Auditoria concluída (limite de 15 arquivos atingido para amostra).")
                return
                
    if arquivos_auditados == 0:
        print("\n[!] Nenhum arquivo .mat, .npy ou .csv válido foi encontrado nas subpastas.")

if __name__ == "__main__":
    audit_datasets()
