import os
import pandas as pd

# Define o caminho para a pasta onde as janelas (.npy) estão processadas
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../data/processed'))

def generate_dataset_statistics():
    print("Mapeando a distribuição de classes nos datasets processados...\n")
    
    stats = []
    
    if not os.path.exists(DATA_ROOT):
        print(f"[ERRO] Diretório não encontrado: {DATA_ROOT}")
        return
        
    # Lista todas as pastas de datasets
    datasets = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    
    for ds_name in datasets:
        ds_path = os.path.join(DATA_ROOT, ds_name)
        
        class_counts = {}
        total_amostras = 0
        
        # Varre todas as subpastas (Condições e Classes)
        for root, dirs, files in os.walk(ds_path):
            npy_files = [f for f in files if f.endswith('.npy')]
            
            if len(npy_files) > 0:
                # O nome da classe geralmente é a última pasta no caminho (root)
                class_name = os.path.basename(root)
                
                if class_name not in class_counts:
                    class_counts[class_name] = 0
                    
                class_counts[class_name] += len(npy_files)
                total_amostras += len(npy_files)
        
        # Formata os dados obtidos para a construção da tabela
        for class_name, count in class_counts.items():
            porcentagem = (count / total_amostras) * 100 if total_amostras > 0 else 0
            stats.append({
                "Dataset": ds_name,
                "Classe": class_name,
                "Nº de Exemplos (Janelas)": count,
                "Representação no Dataset": f"{porcentagem:.1f}%"
            })
            
    if len(stats) > 0:
        df = pd.DataFrame(stats)
        # Ordena alfabeticamente pelo dataset e depois pela classe
        df = df.sort_values(by=["Dataset", "Classe"])
        
        print("="*80)
        print(" DISTRIBUIÇÃO DE CLASSES POR DATASET (Janelas de 1 Segundo)")
        print("="*80)
        print(df.to_markdown(index=False))
        
        # Salva a tabela em CSV para planejamento e relatórios
        os.makedirs("results", exist_ok=True)
        csv_path = "results/distribuicao_classes_datasets.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n[SUCESSO] Tabela exportada para: {csv_path}")
        
    else:
        print("Nenhuma amostra .npy encontrada nas subpastas. Verifique se a extração foi concluída.")

if __name__ == "__main__":
    generate_dataset_statistics()
