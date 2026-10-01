import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import copy

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from pytorch_tabnet.pretraining import TabNetPretrainer
from pytorch_tabnet.tab_model import TabNetClassifier

# Cache em memória para não repetir o pré-treino auto-supervisionado 
# nas várias dobras LOCO do mesmo Target + Estratégia
_PRETRAINER_CACHE = {}

def train_and_evaluate_tabnet_tl(
    train_data_dict,
    target_dataset_name,
    X_test,
    y_test,
    task,
    pretrain_epochs=30,
    finetune_epochs=100,
    batch_size=512,
    seed=42
):
    """
    Transfer Learning Oficial com TabNetPretrainer (Auto-Supervisionado nas Sources)
    seguido de Fine-Tuning Supervisionado com TabNetClassifier no Target.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    source_matrices = []
    X_target_tr_s, y_target_tr = None, None
    target_scaler = None

    # 1. Padronização Z-Score individual por domínio
    for d_name, (X_tr, y_tr) in train_data_dict.items():
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)

        if d_name == target_dataset_name:
            target_scaler = scaler
            X_target_tr_s = X_tr_s
            y_target_tr = y_tr
        else:
            source_matrices.append(X_tr_s)

    if target_scaler is None or X_target_tr_s is None:
        raise ValueError(f"Target dataset '{target_dataset_name}' não encontrado em train_data_dict.")

    X_test_s = target_scaler.transform(X_test)

    # 2. Chave de Cache para as Sources (evita re-treinar o Pretrainer a cada fold LOCO)
    # A assinatura usa os nomes das sources, shapes e a soma da primeira linha para identificar a estratégia
    cache_signature = (
        target_dataset_name,
        tuple(
            (k, v[0].shape, float(np.sum(v[0][0])))
            for k, v in sorted(train_data_dict.items())
            if k != target_dataset_name
        ),
        seed
    )

    # 3. Etapa 1: Pré-treinamento Auto-Supervisionado nas Sources
    if cache_signature in _PRETRAINER_CACHE:
        unsupervised_model = copy.deepcopy(_PRETRAINER_CACHE[cache_signature])
    else:
        X_pretrain = np.vstack(source_matrices) if len(source_matrices) > 0 else X_target_tr_s

        unsupervised_model = TabNetPretrainer(
            n_d=16,
            n_a=16,
            n_steps=5,
            gamma=1.5,
            n_independent=2,
            n_shared=2,
            lambda_sparse=1e-4,
            optimizer_fn=torch.optim.Adam,
            optimizer_params=dict(lr=2e-2, weight_decay=1e-5),
            scheduler_fn=torch.optim.lr_scheduler.StepLR,
            scheduler_params=dict(step_size=10, gamma=0.9),
            mask_type='entmax',
            verbose=0,
            seed=seed
        )

        unsupervised_model.fit(
            X_train=X_pretrain,
            eval_set=[X_pretrain[:min(2048, len(X_pretrain))]],
            max_epochs=pretrain_epochs,
            patience=10,
            batch_size=batch_size,
            virtual_batch_size=128,
            num_workers=0,
            drop_last=False,
            pretraining_ratio=0.8
        )
        _PRETRAINER_CACHE[cache_signature] = copy.deepcopy(unsupervised_model)

    # 4. Etapa 2: Fine-Tuning Supervisionado no Target (mesma arquitetura do Baseline Puro)
    clf = TabNetClassifier(
        n_d=16,
        n_a=16,
        n_steps=5,
        gamma=1.5,
        n_independent=2,
        n_shared=2,
        lambda_sparse=1e-4,
        optimizer_fn=torch.optim.Adam,
        optimizer_params=dict(lr=1e-2, weight_decay=1e-5), # LR levemente menor para preservar o pré-treino
        scheduler_fn=torch.optim.lr_scheduler.StepLR,
        scheduler_params=dict(step_size=20, gamma=0.9),
        mask_type='entmax',
        verbose=0,
        seed=seed
    )

    # Nota: Usamos apenas o treino para monitorar ou sem early stopping no X_test 
    # caso haja classes desbalanceadas, ou eval_set no X_test_s se todas as classes existirem
    clf.fit(
        X_train=X_target_tr_s,
        y_train=y_target_tr,
        eval_set=[(X_target_tr_s, y_target_tr)],
        eval_name=['train'],
        eval_metric=['balanced_accuracy'],
        max_epochs=finetune_epochs,
        patience=20,
        batch_size=min(256, len(X_target_tr_s)),
        virtual_batch_size=min(128, len(X_target_tr_s)),
        num_workers=0,
        drop_last=False,
        from_unsupervised=unsupervised_model
    )

    # 5. Predição e Extração de Métricas
    preds = clf.predict(X_test_s)
    probs = clf.predict_proba(X_test_s)
    mean_attention = clf.feature_importances_

    bal_acc = balanced_accuracy_score(y_test, preds)
    if task == 'detection':
        roc_auc = roc_auc_score(y_test, probs[:, 1])
        macro_f1 = f1_score(y_test, preds, average='binary')
    else:
        try:
            roc_auc = roc_auc_score(y_test, probs, multi_class='ovr')
        except ValueError:
            roc_auc = 0.0
        macro_f1 = f1_score(y_test, preds, average='macro')

    return bal_acc, macro_f1, roc_auc, {'y_pred': preds, 'mean_attention': mean_attention}


# ---------------------------------------------------------------------------
# 1. ENCODERS
# ---------------------------------------------------------------------------
class MLPEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, output_dim)
        )
        
    def forward(self, x):
        latent = self.network(x)
        dummy_attn = torch.ones_like(x) / x.shape[1] 
        return latent, dummy_attn

class MiniTabNetEncoder(nn.Module):
    def __init__(self, input_dim, output_dim=64):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.relu = nn.ReLU()
        self.attention = nn.Linear(128, input_dim)
        self.fc_latent = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        hidden = self.relu(self.bn1(self.fc1(x)))
        attn_weights = torch.sigmoid(self.attention(hidden))
        masked_x = x * attn_weights
        latent = self.fc_latent(masked_x)
        return latent, attn_weights

# ---------------------------------------------------------------------------
# 2. BLOCO RESIDUAL 1D 
# ---------------------------------------------------------------------------
class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        identity = self.downsample(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out

# ---------------------------------------------------------------------------
# 3. A ARQUITETURA HÍBRIDA MULTI-HEAD
# ---------------------------------------------------------------------------
class HybridDLModel(nn.Module):
    def __init__(self, num_features, dataset_classes_dict, encoder_type='mlp', latent_dim=64):
        """
        Versão Otimizada: Removemos os blocos convolucionais (ResNet1D). 
        Dados tabulares não possuem localidade espacial. Agora, o vetor latente 
        altamente rico é roteado diretamente para as cabeças de classificação.
        """
        super().__init__()
        
        if encoder_type == 'mlp':
            self.encoder = MLPEncoder(input_dim=num_features, output_dim=latent_dim)
        elif encoder_type == 'tabnet':
            # Nota: Este ainda é o MiniTabNet. Se quiser o poder total, o ideal 
            # é usar o TabNetClassifier puro. Mas este roteamento direto já melhorará absurdamente.
            self.encoder = MiniTabNetEncoder(input_dim=num_features, output_dim=latent_dim)
        else:
            raise ValueError("encoder_type deve ser 'mlp' ou 'tabnet'")
            
        # Adicionamos um Bottleneck robusto antes das cabeças em vez de convoluções
        self.bottleneck = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )
        
        # Múltiplas cabeças de classificação
        self.heads = nn.ModuleDict({
            d_name: nn.Linear(64, n_classes) for d_name, n_classes in dataset_classes_dict.items()
        })

    def forward(self, x, dataset_name):
        latent, attn_weights = self.encoder(x) 
        
        # Passa pelo bottleneck linear em vez de transformar em sinal 1D
        out = self.bottleneck(latent)
        
        # Roteia a extração para a cabeça correspondente ao dataset
        logits = self.heads[dataset_name](out)
        return logits, attn_weights

# ---------------------------------------------------------------------------
# 4. FUNÇÃO DE TREINAMENTO (Otimizada para Multi-Head e Múltiplos Scalers)
# ---------------------------------------------------------------------------
def train_and_evaluate_multihead(train_data_dict, target_dataset_name, X_test, y_test, task, epochs=15, batch_size=512, encoder_type='mlp'):

    if encoder_type == 'tabnet':
        return train_and_evaluate_tabnet_tl(
            train_data_dict=train_data_dict,
            target_dataset_name=target_dataset_name,
            X_test=X_test,
            y_test=y_test,
            task=task,
            pretrain_epochs=max(25, epochs),
            finetune_epochs=100,
            batch_size=batch_size
        )
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loaders = {}
    dataset_classes_dict = {}
    num_features = None
    target_scaler = None
    
    # 1. Padronização por Dataset (sem SMOTE para não distorcer as normalizações do experimento)
    for d_name, (X_tr, y_tr) in train_data_dict.items():
        if num_features is None:
            num_features = X_tr.shape[1]
            
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        
        if d_name == target_dataset_name:
            target_scaler = scaler
            
        num_classes = 2 if task == 'detection' else len(np.unique(y_tr))
        dataset_classes_dict[d_name] = num_classes
        
        X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)
        
        # Usa lotes menores para o target para garantir mais passos de gradiente
        current_bs = min(128, batch_size) if d_name == target_dataset_name else batch_size
        train_loaders[d_name] = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=current_bs, shuffle=True)

    if target_scaler is None:
        target_scaler = StandardScaler()
        X_test_s = target_scaler.fit_transform(X_test)
        dataset_classes_dict[target_dataset_name] = 2 if task == 'detection' else len(np.unique(y_test))
    else:
        X_test_s = target_scaler.transform(X_test)

    model = HybridDLModel(num_features=num_features, dataset_classes_dict=dataset_classes_dict, encoder_type=encoder_type).to(device)
    criterion = nn.CrossEntropyLoss()
    
    # -------------------------------------------------------------------------
    # ETAPA 1: Pré-treinamento Multi-Domínio (Sources + Target)
    # -------------------------------------------------------------------------
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=1e-5)
    model.train()
    for epoch in range(epochs):
        for d_name, loader in train_loaders.items():
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device)
                optimizer.zero_grad()
                logits, _ = model(bx, dataset_name=d_name)
                loss = criterion(logits, by)
                loss.backward()
                optimizer.step()

    # -------------------------------------------------------------------------
    # ETAPA 2: Fine-Tuning Exclusivo no Target Dataset
    # -------------------------------------------------------------------------
    if target_dataset_name in train_loaders:
        ft_optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-5)
        target_loader = train_loaders[target_dataset_name]
        ft_epochs = max(20, epochs) # Garante convergência na máquina-alvo
        
        for epoch in range(ft_epochs):
            for bx, by in target_loader:
                bx, by = bx.to(device), by.to(device)
                ft_optimizer.zero_grad()
                logits, _ = model(bx, dataset_name=target_dataset_name)
                loss = criterion(logits, by)
                loss.backward()
                ft_optimizer.step()

    # -------------------------------------------------------------------------
    # AVALIAÇÃO
    # -------------------------------------------------------------------------
    model.eval()
    X_te_t = torch.tensor(X_test_s, dtype=torch.float32)
    y_te_t = torch.tensor(y_test, dtype=torch.long)
    test_loader = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=1024, shuffle=False)
    
    all_logits, all_attn = [], []
    with torch.no_grad():
        for bx, _ in test_loader:
            bx = bx.to(device)
            l, a = model(bx, dataset_name=target_dataset_name)
            all_logits.append(l.cpu())
            all_attn.append(a.cpu())
            
    final_logits = torch.cat(all_logits, dim=0)
    final_attn = torch.cat(all_attn, dim=0)
    probs = torch.softmax(final_logits, dim=1).numpy()
    preds = np.argmax(probs, axis=1)
    mean_attention = final_attn.mean(dim=0).numpy()
    
    bal_acc = balanced_accuracy_score(y_test, preds)
    if task == 'detection':
        roc_auc = roc_auc_score(y_test, probs[:, 1])
        macro_f1 = f1_score(y_test, preds, average='binary')
    else:
        try:
            roc_auc = roc_auc_score(y_test, probs, multi_class='ovr')
        except ValueError:
            roc_auc = 0.0
        macro_f1 = f1_score(y_test, preds, average='macro')
        
    # Retorno compatível com extract_predictions
    return bal_acc, macro_f1, roc_auc, {'y_pred': preds, 'mean_attention': mean_attention}
