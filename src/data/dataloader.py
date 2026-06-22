import os
import numpy as np
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder

def load_vibration_data(data_root, dataset_name, test_condition, task):
    """
    Carrega e separa os dados de vibração em conjuntos de treino e teste,
    garantindo que as janelas correspondam estritamente a 1 segundo físico.
    Descarta caudas e frações de janelas incompletas para evitar erros de shape.
    Também detecta e alerta sobre misturas de arquivos com tamanhos de janelas antigos.
    """
    print(f"Carregando {dataset_name} | Condição: {test_condition} | Tarefa: {task.upper()}")
    
    # 1. Mapeamento de caminhos e descoberta de arquivos
    dataset_path = os.path.join(data_root, dataset_name)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Caminho do dataset não encontrado: {dataset_path}")
        
    all_files = []
    all_labels = []
    
    # Varre a árvore de diretórios do dataset para coletar arquivos e classes
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.npy'):
                file_path = os.path.join(root, file)
                # Determina a classe baseada na subpasta imediata
                label = os.path.basename(root)
                all_files.append(file_path)
                all_labels.append(label)
                
    # 2. Filtro de Tarefa (Detecção Binária vs Diagnóstico Multiclasse)
    processed_labels = []
    if task.lower() == 'detection':
        # Binário: Normal vs Falha (tudo que não é Normal vira Fault)
        for label in all_labels:
            if 'normal' in label.lower():
                processed_labels.append('Normal')
            else:
                processed_labels.append('Fault')
    else:
        # Multiclasse: Diagnóstico completo das classes de defeito
        processed_labels = all_labels

    print(f"  -> [DEBUG Dataloader] Classes mapeadas para {task.upper()}: {np.unique(processed_labels)}")
    
    # 3. Divisão de Treino/Teste baseada nas Condições (LOLO / Grupos Virtuais)
    train_files = []
    y_train = []
    test_files = []
    y_test = []
    
    for file, label in zip(all_files, processed_labels):
        # Verifica se o arquivo pertence a alguma das condições de teste
        is_test = False
        if isinstance(test_condition, list):
            # Se for um grupo virtual de rolamentos (UORED)
            is_test = any(cond in file for cond in test_condition)
        else:
            # Se for uma dobra simples de carga/RPM (CWRU, HUST, PU)
            is_test = (test_condition in file)
            
        if is_test:
            test_files.append(file)
            y_test.append(label)
        else:
            train_files.append(file)
            y_train.append(label)
            
    # Encoder de Rótulos para manter consistência numérica
    le = LabelEncoder()
    le.fit(processed_labels)
    y_train = le.transform(y_train) if len(y_train) > 0 else np.array([])
    y_test = le.transform(y_test) if len(y_test) > 0 else np.array([])

    # Mapeamento do tamanho de 1 segundo físico para cada dataset
    EXPECTED_SIZES = {
        "cwru_12k": 12000,
        "cwru_48k": 48000,
        "hust": 25600,
        "uored": 200000,
        "pu": 64000
    }

    # Determinar tamanho esperado de 1 segundo para este dataset
    expected_len = None
    for key, val in EXPECTED_SIZES.items():
        if key in dataset_name.lower():
            expected_len = val
            break
            
    # Fallback se não encontrar o nome mapeado (usa a moda/valor mais frequente)
    if expected_len is None and len(train_files) > 0:
        from collections import Counter
        # Carrega uma pequena amostra para descobrir a moda do tamanho do sinal
        sample_lens = [len(np.load(f)) for f in train_files[:100]]
        expected_len = Counter(sample_lens).most_common(1)[0][0]
    elif expected_len is None:
        expected_len = 12000 # Fallback final absoluto

    print(f"  -> [DEBUG Dataloader] Tamanho de janela de 1s esperado para {dataset_name}: {expected_len} pontos")

    # 4. Carregamento Físico e Descarte de Janelas Incompletas (Caudas)
    print("-> Lendo arquivos .npy...")
    
    # A) Processamento do Conjunto de Treino
    temp_X_train = [np.load(f) for f in tqdm(train_files, desc="Treino")]
    if temp_X_train:
        # Mostra estatísticas de tamanho antes de filtrar para o usuário auditar o cache
        lengths = [len(x) for x in temp_X_train]
        unique_lens, counts = np.unique(lengths, return_counts=True)
        print(f"  -> [DEBUG Dataloader] Distribuição de tamanhos no Treino: {dict(zip(unique_lens, counts))}")
        
        valid_idx_train = [i for i, x in enumerate(temp_X_train) if len(x) == expected_len]
        
        X_train = np.array([temp_X_train[i] for i in valid_idx_train])
        train_files = [train_files[i] for i in valid_idx_train]
        
        # Filtra os rótulos de forma segura usando variável temporária
        y_train_filtered = [y_train[i] for i in valid_idx_train]
        y_train = np.array(y_train_filtered)
        
        # Alerta se o descarte de janelas inconsistentes esvaziou as classes
        unique_classes_train = np.unique(y_train)
        if len(unique_classes_train) < 2 and task.lower() == 'detection':
            print("\n" + "!"*80)
            print(f"⚠️  AVISO CRÍTICO DE CONFIGURAÇÃO DE DADOS (Treino - {dataset_name}) ⚠️")
            print(f"O conjunto de treino acabou com apenas {len(unique_classes_train)} classe: {unique_classes_train}")
            print("Isso ocorre quando arquivos antigos processados (janelas menores) estão misturados")
            print("na mesma pasta com os novos arquivos de 1 segundo!")
            print(f"Por favor, APAGUE COMPLETAMENTE a pasta em: 'data/processed/{dataset_name}'")
            print("e execute o script de geração de datasets novamente para limpar o cache!")
            print("!"*80 + "\n")
    else:
        X_train = np.array([])
        y_train = np.array([])

    # B) Processamento do Conjunto de Teste
    temp_X_test = [np.load(f) for f in tqdm(test_files, desc="Teste")]
    if temp_X_test:
        lengths = [len(x) for x in temp_X_test]
        unique_lens, counts = np.unique(lengths, return_counts=True)
        print(f"  -> [DEBUG Dataloader] Distribuição de tamanhos no Teste: {dict(zip(unique_lens, counts))}")
        
        valid_idx_test = [i for i, x in enumerate(temp_X_test) if len(x) == expected_len]
        
        X_test = np.array([temp_X_test[i] for i in valid_idx_test])
        test_files = [test_files[i] for i in valid_idx_test]
        
        # Filtra os rótulos de forma segura usando variável temporária
        y_test_filtered = [y_test[i] for i in valid_idx_test]
        y_test = np.array(y_test_filtered)
        
        # Alerta se o descarte de janelas inconsistentes esvaziou as classes no teste
        unique_classes_test = np.unique(y_test)
        if len(unique_classes_test) < 2 and task.lower() == 'detection':
            print("\n" + "!"*80)
            print(f"⚠️  AVISO CRÍTICO DE CONFIGURAÇÃO DE DADOS (Teste - {dataset_name}) ⚠️")
            print(f"O conjunto de teste acabou com apenas {len(unique_classes_test)} classe: {unique_classes_test}")
            print("Isso ocorre quando arquivos antigos processados estão misturados no diretório.")
            print(f"Por favor, APAGUE COMPLETAMENTE a pasta em: 'data/processed/{dataset_name}' e regenere.")
            print("!"*80 + "\n")
    else:
        X_test = np.array([])
        y_test = np.array([])

    return X_train, y_train, X_test, y_test, le
