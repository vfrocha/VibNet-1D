import os
import pandas as pd
import vibdata.raw as raw_datasets

RAW_DATA_DIR = "/home/vfrocha/VibNet_Project/raw_data"

def audit_datasets_with_vibdata():
    print("="*70)
    print(" AUDITORIA DE DADOS BRUTOS VIA VIBDATA (Verificação de $fs$)")
    print("="*70)
    
    datasets = ["UORED", "CWRU", "PU", "HUST"]

    for ds_name in datasets:
        print(f"\n[{ds_name.upper()}]")
        
        try:
            # Invoca a classe nativa da biblioteca
            raw_cls = getattr(raw_datasets, f"{ds_name}_raw")
            ds = raw_cls(RAW_DATA_DIR, download=False)
        except Exception as e: 
            print(f" -> Erro ao carregar dataset: {e}")
            continue
            
        print(f" -> Total de arquivos/gravações contínuas: {len(ds)}")
        
        # Pega 3 exemplos de metadados de cada base para compararmos
        for i in range(min(3, len(ds))):
            try:
                item = ds[i]
                meta = item['metainfo']
                if isinstance(meta, pd.DataFrame): 
                    meta = meta.iloc[0]
                
                # Extrai o sinal e pega o tamanho (total de pontos da gravação)
                sig = item.get('signal')
                if isinstance(sig, list): sig = sig[0]
                tamanho_pontos = max(sig.shape) if hasattr(sig, 'shape') else len(sig)
                
                # Tenta puxar o Sampling Rate nativo que veio da máquina
                # Algumas bases chamam de sample_rate, outras de fs, etc.
                fs_declarado = meta.get('sample_rate', meta.get('fs', meta.get('sampling_rate', 'NÃO INFORMADO')))
                
                print(f"    - Amostra {i}: Condição = {meta.get('label', 'N/A')} | Fs Declarado = {fs_declarado} Hz | Pontos Totais = {tamanho_pontos}")
                
            except Exception as e:
                print(f"    - Falha ao ler amostra {i}: {e}")

if __name__ == "__main__":
    audit_datasets_with_vibdata()
