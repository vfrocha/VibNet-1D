import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

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
    def __init__(self, num_features, dataset_classes_dict, encoder_type='mlp', latent_dim=64, expansion_size=1024):
        """
        dataset_classes_dict: Um dicionário mapeando o nome do dataset para o número de classes.
                              Ex: {'CWRU': 4, 'HUST': 3, 'UOEMD': 5}
        """
        super().__init__()
        
        if encoder_type == 'mlp':
            self.encoder = MLPEncoder(input_dim=num_features, output_dim=latent_dim)
        elif encoder_type == 'tabnet':
            self.encoder = MiniTabNetEncoder(input_dim=num_features, output_dim=latent_dim)
        else:
            raise ValueError("encoder_type deve ser 'mlp' ou 'tabnet'")
            
        self.expansion_size = expansion_size
        self.expand_layer = nn.Linear(latent_dim, expansion_size)
        self.expand_bn = nn.BatchNorm1d(expansion_size)
        self.expand_relu = nn.ReLU()
        
        self.conv_in = nn.Conv1d(1, 16, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn_in = nn.BatchNorm1d(16)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = ResidualBlock1D(16, 32, stride=2)
        self.layer2 = ResidualBlock1D(32, 64, stride=2)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # [MODIFICAÇÃO AQUI]: Substituímos a fc_out única por múltiplas cabeças
        self.heads = nn.ModuleDict({
            d_name: nn.Linear(64, n_classes) for d_name, n_classes in dataset_classes_dict.items()
        })

    def forward(self, x, dataset_name):
        latent, attn_weights = self.encoder(x) 
        expanded = self.expand_relu(self.expand_bn(self.expand_layer(latent)))
        synthetic_signal = expanded.unsqueeze(1) 
        
        out = self.relu(self.bn_in(self.conv_in(synthetic_signal)))
        out = self.pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.global_pool(out).squeeze(-1)       
        
        # [MODIFICAÇÃO AQUI]: A rede roteia a extração para a cabeça correspondente ao dataset
        logits = self.heads[dataset_name](out)
        return logits, attn_weights

# ---------------------------------------------------------------------------
# 4. FUNÇÃO DE TREINAMENTO (Otimizada para Multi-Head e Múltiplos Scalers)
# ---------------------------------------------------------------------------
def train_and_evaluate_multihead(train_data_dict, target_dataset_name, X_test, y_test, task, epochs=15, batch_size=512, encoder_type='mlp', max_samples_per_class=5000):
    """
    train_data_dict: dict contendo matrizes brutas -> {'CWRU': (X_train, y_train), ...}
    max_samples_per_class: O "teto" mágico. Se um dataset for maior que isso, ele é cortado. 
                           Se for menor, o SMOTE cria dados sintéticos até nivelar.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loaders = {}
    dataset_classes_dict = {}
    num_features = None
    target_scaler = None
    
    # 1. Padronização e Balanceamento por Dataset
    for d_name, (X_tr, y_tr) in train_data_dict.items():
        if num_features is None:
            num_features = X_tr.shape[1]
            
        # A. Padronização INDIVIDUAL (Z-Score) - Deve ocorrer ANTES do SMOTE
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        
        if d_name == target_dataset_name:
            target_scaler = scaler
            
        # =====================================================================
        # B. NOVIDADE: ESTRATÉGIA ANTI-DOMINÂNCIA (SMOTE + UnderSampling)
        # =====================================================================
        unique, counts = np.unique(y_tr, return_counts=True)
        
        # O SMOTE aumenta as classes minoritárias para igualar à maior classe daquele dataset
        if len(unique) > 1: # SMOTE precisa de pelo menos 2 classes
            try:
                smote = SMOTE(random_state=42)
                X_tr_s, y_tr = smote.fit_resample(X_tr_s, y_tr)
            except ValueError as e:
                print(f"      [Aviso] SMOTE falhou em {d_name} (poucas amostras?). Erro: {e}")
        
        # O UnderSampler 'corta' as classes que ficaram gigantes para não dominarem a rede
        unique, counts = np.unique(y_tr, return_counts=True)
        if counts.max() > max_samples_per_class:
            # Cria um dicionário limitando cada classe ao teto máximo
            sampling_strategy = {c: min(max_samples_per_class, count) for c, count in zip(unique, counts)}
            rus = RandomUnderSampler(sampling_strategy=sampling_strategy, random_state=42)
            X_tr_s, y_tr = rus.fit_resample(X_tr_s, y_tr)
        # =====================================================================
            
        num_classes = 2 if task == 'detection' else len(np.unique(y_tr))
        dataset_classes_dict[d_name] = num_classes
        
        X_tr_t = torch.tensor(X_tr_s, dtype=torch.float32)
        y_tr_t = torch.tensor(y_tr, dtype=torch.long)
        train_loaders[d_name] = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)

    # 2. Configuração do Teste (Zero-shot handling)
    if target_scaler is None:
        target_scaler = StandardScaler()
        X_test_s = target_scaler.fit_transform(X_test)
        dataset_classes_dict[target_dataset_name] = 2 if task == 'detection' else len(np.unique(y_test))
    else:
        X_test_s = target_scaler.transform(X_test)

    # 3. Inicialização do Modelo Multi-Head
    model = HybridDLModel(num_features=num_features, dataset_classes_dict=dataset_classes_dict, encoder_type=encoder_type).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 4. Laço de Treinamento
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
                
    # 5. Avaliação (Usando a cabeça do Dataset Alvo)
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
    
    # 6. Cálculo das Métricas
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
        
    return bal_acc, macro_f1, roc_auc, mean_attention
