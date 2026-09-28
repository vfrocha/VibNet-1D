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
    """Lê o labels.csv e mapeia dinamicamente, ignorando problemas de case e espaços."""
    mapping = {}
    if not os.path.exists(csv_path):
        print(f"[AVISO] Arquivo '{csv_path}' não encontrado.")
        return mapping

    try:
        df = pd.read_csv(csv_path)
        # Padroniza os nomes das colunas (remove espaços e coloca em minúsculo)
        df.columns = [str(c).strip().lower() for c in df.columns]
        
        # Tenta encontrar as colunas dinamicamente caso os nomes divirjam
        col_ds = next((c for c in df.columns if 'dataset' in c or 'data' in c), 'dataset')
        col_id = next((c for c in df.columns if 'id' in c or 'class' in c), 'id_label')
        col_name = next((c for c in df.columns if 'name' in c or 'label' in c and c != col_id), 'labels_name')
        
        if col_ds in df.columns and col_id in df.columns and col_name in df.columns:
            for _, row in df.iterrows():
                ds = str(row[col_ds]).strip()
                id_lbl = str(row[col_id]).strip()
                name_lbl = str(row[col_name]).strip()
                
                if ds and id_lbl:
                    if ds not in mapping:
                        mapping[ds] = {}
                    mapping[ds][id_lbl] = name_lbl
                    mapping[ds][f"Class_{id_lbl}"] = name_lbl
        else:
            print(f"[AVISO] As colunas esperadas não foram encontradas no CSV. Colunas detectadas: {df.columns.tolist()}")
                
    except Exception as e:
        print(f"[ERRO] Falha ao processar o arquivo CSV: {e}")
        
    return mapping

def get_physical_name(ds_name, class_folder, mapping):
    """Faz o cruzamento inteligente entre o nome da pasta e as regras do CSV."""
    clean_id = class_folder.replace("Class_", "")
    
    # 1. Busca Flexível pelo nome do Dataset
    # Ex: Mapeia regras do "CWRU" (do CSV) para "CWRU_12k_Severity" (da Pasta)
    matched_ds = None
    if ds_name in mapping:
        matched_ds = ds_name
    else:
        for map_ds in mapping.keys():
            if map_ds.lower() in ds_name.lower():
                matched_ds = map_ds
                break

    # 2. Aplica a tradução exata se o dataset foi encontrado
    if matched_ds:
        if class_folder in mapping[matched_ds]:
            return mapping[matched_ds][class_folder]
        if clean_id in mapping[matched_ds]:
            return mapping[matched_ds][clean_id]

    # 3. Limpeza Automática para classes Textuais
    # Ex: Class_Combined_Fault -> Combined Fault
    if class_folder.startswith("Class_") and not clean_id.isdigit():
        return clean_id.replace("_", " ")

    # 4. Fallback Universal de Segurança
    folder_lower = class_folder.lower()
    if 'normal' in folder_lower or 'healthy' in folder_lower or clean_id == '0':
        return "Healthy (Normal)"

    return class_folder

def generate_dataset_statistics():
    print("="*80)
    print(" MAPEANDO DISTRIBUIÇÃO E TRADUZINDO RÓTULOS (CORREÇÃO FLEXÍVEL)")
    print("="*80)
    
    mapping_dict = load_labels_mapping(LABELS_CSV_PATH)
    if mapping_dict:
        print(f"[INFO] Datasets detectados no labels.csv: {list(mapping_dict.keys())}")
    
    stats = []
    
    if not os.path.exists(DATA_ROOT):
        print(f"[ERRO] Diretório não encontrado: {DATA_ROOT}")
        return
        
    datasets = [d for d in os.listdir(DATA_ROOT) if os.path.isdir(os.path.join(DATA_ROOT, d))]
    
    for ds_name in datasets:
        ds_path = os.path.join(DATA_ROOT, ds_name)
        class_counts = {}
        total_amostras = 0
        
        for root, dirs, files in os.walk(ds_path):
            npy_files = [f for f in files if f.endswith('.npy')]
            
            if len(npy_files) > 0:
                class_folder = os.path.basename(root)
                physical_name = get_physical_name(ds_name, class_folder, mapping_dict)
                
                if physical_name not in class_counts:
                    class_counts[physical_name] = 0
                    
                class_counts[physical_name] += len(npy_files)
                total_amostras += len(npy_files)
        
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
        print(" DISTRIBUIÇÃO DE CLASSES POR DATASET")
        print("="*80)
        print(df.to_markdown(index=False))
        
        os.makedirs("results", exist_ok=True)
        csv_path = "results/distribuicao_classes_oficial.csv"
        df.to_csv(csv_path, index=False)
        print(f"\n[SUCESSO] Tabela formatada exportada para: {csv_path}")

if __name__ == "__main__":
    generate_dataset_statistics()
