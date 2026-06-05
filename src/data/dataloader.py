import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm

def load_vibration_data(data_root, dataset_name, test_condition, task='diagnosis'):
    """
    Carrega arrays .npy de um dataset específico e aplica o split Leave-One-Load-Out.
    
    Parâmetros:
    - data_root: Caminho para a pasta 'data/processed'
    - dataset_name: Nome do dataset (ex: 'CWRU_12k', 'UORED')
    - test_condition: Nome da pasta da condição que será usada apenas para teste
    - task: 'diagnosis' (Multiclasse: tipo exato da falha) ou 'detection' (Binário: Normal vs Falha)
    
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
    
    print(f"Carregando {dataset_name} | Condição: {test_condition} | Tarefa: {task.upper()}")
    
    for cond in conditions:
        cond_path = os.path.join(dataset_dir, cond)
        classes = [c for c in os.listdir(cond_path) if os.path.isdir(os.path.join(cond_path, c))]
        
        # Permite que a condição de teste seja um Grupo Virtual (Lista de pastas)
        if isinstance(test_condition, list):
            is_test = (cond in test_condition)
        else:
            is_test = (cond == test_condition)
        
        for cls_name in classes:
            cls_path = os.path.join(cond_path, cls_name)
            files = [os.path.join(cls_path, f) for f in os.listdir(cls_path) if f.endswith('.npy')]
            
            # --- LÓGICA DE DEFINIÇÃO DA TAREFA ---
            if task == 'detection':
                # Agrupa todas as classes em apenas duas: 'Normal' e 'Fault'
                # Pressupõe que a classe normal tenha a palavra 'normal' no nome do diretório
                if 'normal' in cls_name.lower():
                    final_label = 'Normal'
                else:
                    final_label = 'Fault'
            else:
                # 'diagnosis': Mantém as classes originais separadas
                final_label = cls_name
            
            if is_test:
                test_files.extend(files)
                test_labels.extend([final_label] * len(files))
            else:
                train_files.extend(files)
                train_labels.extend([final_label] * len(files))

    # Carregar os arrays NumPy na memória
    print("-> Lendo arquivos .npy...")
    X_train = np.array([np.load(f) for f in tqdm(train_files, desc="Treino")])
    X_test = np.array([np.load(f) for f in tqdm(test_files, desc="Teste")])
    
    # Converter rótulos de texto para inteiros
    le = LabelEncoder()
    y_train = le.fit_transform(train_labels)
    
    # DEBUG: Mostra exatamente o que o Dataloader enxergou
    print(f"  -> [DEBUG Dataloader] Classes mapeadas para {task.upper()}: {le.classes_}")
    
    # --- NOVA REGRA DE DESIGN: Trava de Segurança para Detecção ---
    # Se a tarefa for 'detection' e o LabelEncoder só achou 1 classe (tudo 'Fault'),
    # significa que a pasta 'Normal' não existe. Abortamos para não quebrar o ROC-AUC.
    if task == 'detection' and len(le.classes_) < 2:
        print(f"\n  [Aviso Defensivo] Dataset {dataset_name} não possui a classe 'Normal'. "
              f"A tarefa de Detecção Binária será ignorada para este fold.")
        # Retornamos arrays vazios para acionar o 'continue' do script principal
        return np.array([]), np.array([]), np.array([]), np.array([]), None
    
    y_test = le.transform(test_labels)

    return X_train, y_train, X_test, y_test, le
