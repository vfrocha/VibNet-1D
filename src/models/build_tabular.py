import os
import numpy as np
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

def get_tabnet_classifier(seed=42):
    """
    Instancia a arquitetura TabNet com hiperparâmetros ajustados 
    para dados contínuos de vibração (sem variáveis categóricas).
    """
    # A escolha entre CPU ou GPU (cuda) é automática no TabNet,
    # mas garantimos a reproducibilidade
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Hiperparâmetros baseados em recomendações para datasets de engenharia (médio porte)
    clf = TabNetClassifier(
        n_d=16, # Dimensão da camada de previsão
        n_a=16, # Dimensão da camada de atenção (XAI)
        n_steps=5, # Quantidade de saltos na arquitetura (profundidade)
        gamma=1.5, # Grau de relaxamento da esparsidade
        n_independent=2, # Camadas independentes por step
        n_shared=2, # Camadas compartilhadas por step
        lambda_sparse=1e-4, # Penalidade para forçar a atenção apenas no que importa
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(step_size=20, gamma=0.9),
        mask_type='entmax', # 'entmax' ou 'sparsemax' (entmax costuma ser mais suave)
        verbose=0
    )
    
    return clf

def train_and_evaluate_tabnet(model, X_train, y_train, X_test, y_test, task='diagnosis'):
    """
    Padroniza os dados estatísticos, treina o Autoencoder Tabular (TabNet)
    utilizando early stopping, e calcula as métricas do artigo.
    
    task pode ser 'diagnosis' (multiclasse) ou 'detection' (binário)
    """
    print(f"  -> Treinando arquitetura TabNet (Autoencoder Tabular)...")
    
    # 1. Normalização Z-score (Essencial para redes neurais e exigência do esqueleto)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 2. Treinamento
    model.fit(
        X_train=X_train_scaled, y_train=y_train,
        eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)],
        eval_name=['train', 'val'],
        eval_metric=['balanced_accuracy'],
        max_epochs=150, 
        patience=20, # Para se a val_accuracy não melhorar após 20 épocas
        batch_size=256, 
        virtual_batch_size=128,
        num_workers=0,
        drop_last=False
    )
    
    # 3. Predição e Métricas
    y_pred = model.predict(X_test_scaled)
    y_pred_probs = model.predict_proba(X_test_scaled)
    
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    
    # --- LOGICA CORRIGIDA E BLINDADA ---
    if task.lower() == 'detection':
        # Na detecção (Binário), focamos apenas na classe de anomalia (assumindo que seja '1')
        try:
            # Extrair índice da classe positiva (se ela existir no vetor de classes do modelo)
            # O TabNet salva as classes conhecidas em model.classes_
            pos_class_index = np.where(model.classes_ == 1)[0][0]
            roc_auc = roc_auc_score(y_test, y_pred_probs[:, pos_class_index])
        except IndexError:
            roc_auc = roc_auc_score(y_test, y_pred_probs[:, 1]) # Fallback padrão

        # F1_score binário explicitly focando no label '1'
        macro_f1 = f1_score(y_test, y_pred, average='binary', pos_label=1) 
        
        print(f"     [Detecção] Bal Acc: {bal_acc:.4f} | F1 (Bin): {macro_f1:.4f} | ROC-AUC: {roc_auc:.4f}")
        return bal_acc, macro_f1, roc_auc, model
    
    elif task.lower() == 'diagnosis':
        # No diagnóstico (Multiclasse), usamos o F1 Macro e calculamos o ROC-AUC multiclasse
        try:
            # Adicionado o 'average="macro"' também no ROC-AUC multiclasse
            roc_auc = roc_auc_score(y_test, y_pred_probs, multi_class='ovr', average='macro')
        except ValueError:
            roc_auc = 0.0 # Caso extremo onde só há uma classe real no teste
            
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        print(f"     [Diagnóstico] Bal Acc: {bal_acc:.4f} | Macro F1: {macro_f1:.4f} | ROC-AUC: {roc_auc:.4f}")
        return bal_acc, macro_f1, roc_auc, model
    
    else:
        raise ValueError(f"Tarefa desconhecida: {task}")
