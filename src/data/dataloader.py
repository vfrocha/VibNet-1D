import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def load_vibration_data(data_root, dataset_name, test_condition):
    """
    Carrega arrays .npy de um dataset específico e aplica o split Leave-One-Load-Out.
    
    Parâmetros:
    - data_root: Caminho para a pasta 'data/processed'
    - dataset_name: Nome do dataset (ex: 'CWRU_12k', 'UORED')
    - test_condition: Nome da pasta da condição que será usada apenas para teste
    
    Retorna:
    - X_train, y_train, X_test, y_test, label_encoder
    """
    dataset_dir = os.path.join(data_root, dataset_name)
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset não encontrado: {dataset_dir}")

    train_files, train_labels = [], []
    test_files, test_labels = [], []

    # Mapear todas as condições disponíveis
    conditions = [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]
    
    print(f"Carregando {dataset_name} | Condição de Teste: {test_condition}")
    
    for cond in conditions:
        cond_path = os.path.join(dataset_dir, cond)
        classes = [c for c in os.listdir(cond_path) if os.path.isdir(os.path.join(cond_path, c))]
        
        is_test = (cond == test_condition)
        
        for cls_name in classes:
            cls_path = os.path.join(cond_path, cls_name)
            files = [os.path.join(cls_path, f) for f in os.listdir(cls_path) if f.endswith('.npy')]
            
            if is_test:
                test_files.extend(files)
                test_labels.extend([cls_name] * len(files))
            else:
                train_files.extend(files)
                train_labels.extend([cls_name] * len(files))

    # Carregar os arrays NumPy na memória
    print("-> Lendo arquivos .npy...")
    X_train = np.array([np.load(f) for f in tqdm(train_files, desc="Treino")])
    X_test = np.array([np.load(f) for f in tqdm(test_files, desc="Teste")])
    
    # Converter rótulos de texto ('Class_Normal') para inteiros (0, 1, 2...)
    le = LabelEncoder()
    y_train = le.fit_transform(train_labels)
    y_test = le.transform(test_labels)

    return X_train, y_train, X_test, y_test, le
