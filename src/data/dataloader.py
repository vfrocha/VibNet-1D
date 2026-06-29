import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

def load_vibration_data(data_root, dataset_name, test_condition, task):
    """
    Carrega e separa os dados de vibração em conjuntos de treino e teste,
    garantindo que as janelas correspondam estritamente a 1 segundo físico.
    Previne erros de CUDA (Index Out of Bounds) ao ajustar o LabelEncoder
    APENAS nas classes que sobreviveram ao filtro temporal.
    """
    print(f"Carregando {dataset_name} | Condição: {test_condition} | Tarefa: {task.upper()}")
    
    # 1. Mapeamento de caminhos e descoberta de arquivos
    dataset_path = os.path.join(data_root, dataset_name)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Caminho do dataset não encontrado: {dataset_path}")
        
    all_files = []
    all_labels = []
    
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                label = os.path.basename(root)
                all_files.append(file_path)
                all_labels.append(label)
                
    # 2. Filtro de Tarefa Inteligente (Mapeamento de Normal/Fault)
    processed_labels = []
    healthy_keywords = ['normal', 'baseline', 'healthy', 'k001', 'k002', 'k003', 'k004', 'k005', 'k006']
    
    if task.lower() == 'detection':
        for label in all_labels:
            if any(keyword in label.lower() for keyword in healthy_keywords) or label.upper() == 'H':
                processed_labels.append('Normal')
            else:
                processed_labels.append('Fault')
    else:
        for label in all_labels:
            if any(keyword in label.lower() for keyword in healthy_keywords) or label.upper() == 'H':
                processed_labels.append('Class_Normal')
            else:
                processed_labels.append(label)
    
    # 3. Divisão Inicial de Treino e Teste
    train_files_raw = []
    y_train_raw = []
    test_files_raw = []
    y_test_raw = []
    
    for file, label in zip(all_files, processed_labels):
        is_test = False
        if isinstance(test_condition, list):
            is_test = any(cond in file for cond in test_condition)
        else:
            is_test = (test_condition in file)
            
        if is_test:
            test_files_raw.append(file)
            y_test_raw.append(label)
        else:
            train_files_raw.append(file)
            y_train_raw.append(label)

    # 4. Descoberta Automática de Janelas (Com base na auditoria)
    EXPECTED_SIZES = {
        "cwru_12k": 12000,
        "cwru_48k": 48000,
        "hust": 51200,
        "uored": 42000,
        "pu": 64000
    }

    expected_len = None
    for key, val in EXPECTED_SIZES.items():
        if key in dataset_name.lower():
            expected_len = val
            break
            
    if expected_len is None and len(train_files_raw) > 0:
        from collections import Counter
        sample_lens = [len(np.load(f)) for f in train_files_raw[:100]]
        expected_len = Counter(sample_lens).most_common(1)[0][0]
    elif expected_len is None:
        expected_len = 12000

    print(f"  -> [DEBUG Dataloader] Tamanho de janela de 1s esperado: {expected_len} pontos")

    # 5. Carregamento e Filtro Físico (Descarte de Caudas)
    print("-> Lendo arquivos .npy...")
    
    temp_X_train = [np.load(f) for f in tqdm(train_files_raw, desc="Treino")]
    if temp_X_train:
        valid_idx_train = [i for i, x in enumerate(temp_X_train) if len(x) == expected_len]
        X_train = np.array([temp_X_train[i] for i in valid_idx_train])
        y_train_filtered = [y_train_raw[i] for i in valid_idx_train]
    else:
        X_train = np.array([])
        y_train_filtered = []

    temp_X_test = [np.load(f) for f in tqdm(test_files_raw, desc="Teste")]
    if temp_X_test:
        valid_idx_test = [i for i, x in enumerate(temp_X_test) if len(x) == expected_len]
        X_test_temp = [temp_X_test[i] for i in valid_idx_test]
        y_test_filtered = [y_test_raw[i] for i in valid_idx_test]
    else:
        X_test_temp = []
        y_test_filtered = []

    # 6. SEGURANÇA MÁXIMA DE CLASSES (Proteção contra CUDA Device-Side Assert)
    le = LabelEncoder()
    if len(y_train_filtered) > 0:
        # Ajusta o encoder APENAS nas classes que realmente sobreviveram ao janelamento
        le.fit(y_train_filtered)
        y_train_encoded = le.transform(y_train_filtered)
        
        # Filtra o conjunto de teste: remove classes que o modelo nunca viu no treino
        valid_test_final_idx = [i for i, label in enumerate(y_test_filtered) if label in le.classes_]
        dropped = len(y_test_filtered) - len(valid_test_final_idx)
        
        if dropped > 0:
            print(f"  -> [AVISO] {dropped} janelas de teste ignoradas (pertencem a classes que sumiram do Treino).")
            
        y_test_final = [y_test_filtered[i] for i in valid_test_final_idx]
        X_test_final = [X_test_temp[i] for i in valid_test_final_idx]
        
        X_test_encoded = np.array(X_test_final) if len(X_test_final) > 0 else np.array([])
        y_test_encoded = le.transform(y_test_final) if len(y_test_final) > 0 else np.array([])
    else:
        y_train_encoded = np.array([])
        X_test_encoded = np.array([])
        y_test_encoded = np.array([])

    unique_train = np.unique(y_train_filtered) if len(y_train_filtered) > 0 else []
    print(f"  -> [DEBUG Dataloader] Classes FINAIS no Treino: {unique_train}")

    return X_train, y_train_encoded, X_test_encoded, y_test_encoded, le
