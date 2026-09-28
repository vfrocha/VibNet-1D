import os
import pandas as pd

# =============================================================================
# CONFIGURAÇÕES DE DIRETÓRIO
# =============================================================================
# DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), 'data/processed'))
# LABELS_CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), 'labels.csv')) 

# Se estiver rodando no Google Colab, descomente e use os caminhos abaixo:
DATA_ROOT = "../data/processed"
LABELS_CSV_PATH = "/home/vfrocha/vibdata/vibdata/resources/labels.csv"

def load_labels_mapping(csv_path):
    """
    Lê o arquivo labels.csv oficial da vibdata e constrói um 
    dicionário dinâmico de mapeamento: dict[dataset][class_id] = nome_fisico
    """
    mapping = {}
    if not os.path.exists(csv_path):
        print(f"[AVISO] Arquivo '{csv_path}' não encontrado. As classes manterão os nomes das pastas.")
        return mapping

    try:
        df = pd.read_csv(csv_path)
        print(f"[INFO] labels.csv carregado com sucesso. Estruturando mapeamento dinâmico...")
        
        for _, row in df.iterrows():
            # Extrai usando as colunas padrão encontradas na documentação da vibdata
            ds = str(row.get('dataset', '')).strip()
            id_lbl = str(row.get('id_label', '')).strip()
            name_lbl = str(row.get('labels_name', '')).strip()
            
            if ds and id_lbl:
                if ds not in mapping:
                    mapping[ds] = {}
                
                # O script é esperto: mapeia tanto o ID numérico quanto a pasta "Class_ID"
                mapping[ds][id_lbl] = name_lbl
                mapping[ds][f"Class_{id_lbl}"] = name_lbl
                
    except Exception as e:
        print(f"[ERRO] Falha ao processar o arquivo CSV: {e}")
        
    return mapping

def get_physical_name(ds_name, class_folder, mapping):
    """Tenta traduzir o nome numérico da pasta para o defeito físico."""
    
    # 1. Tenta achar a pasta diretamente no dicionário do dataset
    if ds_name in mapping and class_folder in mapping[ds_name]:
        return mapping[ds_name][class_folder]
    
    # 2. Remove o prefixo "Class_" e tenta buscar apenas pelo número
    clean_id = class_folder.replace("Class_", "")
    if ds_name in mapping and clean_id in mapping[ds_name]:
        return mapping[ds_name][clean_id]
        
    # 3. Fallback Universal: Se o nome for Normal/Healthy, a gente garante a legibilidade
    folder_lower = class_folder.lower()
    if 'normal' in folder_lower or 'healthy' in folder_lower or clean_id == '0':
        return "Healthy (Normal)"
        
    # 4. Se não achar tradução, retorna o nome original (ex: para evitar quebrar a tabela)
    return class_folder

def generate_dataset_statistics():
    print("="*80)
    print(" MAPEANDO DISTRIBUIÇÃO E TRADUZINDO RÓTULOS AUTOMATICAMENTE")
    print("="*80)
    
    # Injeta a inteligência do CSV no dicionário
    mapping_dict = load_labels_mapping(LABELS_CSV_PATH)
    
    stats = []
    
    if not os.path.exists(DATA_ROOT):
        print(f"[ERRO] Diretório não encontrado: {DATA_ROOT}")
        return
        
    datasets = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    
    for ds_name in datasets:
        ds_path = os.path.join(DATA_ROOT, ds_name)
        
        class_counts = {}
        total_amostras = 0
        
        # Varre as subpastas procurando as janelas recém-geradas
        for root, dirs, files in os.walk(ds_path):
            npy_files = [f for f in files if f.endswith('.npy')]
            
            if len(npy_files) > 0:
                class_folder = os.path.basename(root)
                
                # A mágica acontece aqui: A tradução física sob demanda!
                physical_name = get_physical_name(ds_name, class_folder, mapping_dict)
                
                if physical_name not in class_counts:
                    class_counts[physical_name] = 0
                    
                class_counts[physical_name] += len(npy_files)
                total_amostras += len(npy_files)
        
        # Formata para a exportação
        for class_name, count in class_counts.items():
            porcentagem = (count / total_amostras) * 100 if total_amostras > 0 else 0
            stats.append({
                "Dataset": ds_name,
                "Defeito Físico (Rótulo)": class_name,
                "Nº de Exemplos (Janelas)": count,
                "Representação": f"{porcentagem:.1f}%"
            })
            
    if len(stats) > 0:
        df = pd.DataFrame(stats)
        df = df.sort_values(by=["Dataset", "Defeito Físico (Rótulo)"])
        
        print("\n" + "="*80)
        print(" DISTRIBUIÇÃO DE CLASSES POR DATASET (TRADUZIDA)")
        print("="*80)
        print(df.to_markdown(index=False))
        
        os.makedirs("results", exist_ok=True)
        csv_path = "results/distribuicao_classes_oficial.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n[SUCESSO] Tabela final exportada para: {csv_path}")
        
    else:
        print("Nenhuma amostra .npy encontrada. Verifique se a extração foi concluída.")

if __name__ == "__main__":
    generate_dataset_statistics()
