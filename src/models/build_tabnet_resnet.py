import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# ---------------------------------------------------------------------------
# 1. ENCODERS
# ---------------------------------------------------------------------------

class MLPEncoder(nn.Module):
    """
    Encoder MLP Simples (Estilo Autoencoder Bottleneck).
    Altamente eficiente para features colineares (Extração Estatística de Sinais).
    Combina linearmente as features redundantes em um espaço latente limpo.
    """
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
        # Retorna dummy attention weights apenas para manter compatibilidade de código
        dummy_attn = torch.ones_like(x) / x.shape[1] 
        return latent, dummy_attn

class MiniTabNetEncoder(nn.Module):
    """
    Encoder focado em Atenção Esparsa (Mantido para comparação científica).
    """
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
# 3. A ARQUITETURA HÍBRIDA COMPLETA: Encoder + ResNet1D
# ---------------------------------------------------------------------------
class HybridDLModel(nn.Module):
    def __init__(self, num_features, num_classes, encoder_type='mlp', latent_dim=64, expansion_size=1024):
        super().__init__()
        
        # Seleção dinâmica do Encoder baseada na sugestão do orientador
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
        self.fc_out = nn.Linear(64, num_classes)

    def forward(self, x):
        latent, attn_weights = self.encoder(x) 
        expanded = self.expand_relu(self.expand_bn(self.expand_layer(latent)))
        synthetic_signal = expanded.unsqueeze(1) 
        
        out = self.relu(self.bn_in(self.conv_in(synthetic_signal)))
        out = self.pool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.global_pool(out).squeeze(-1)       
        logits = self.fc_out(out)
        return logits, attn_weights

# ---------------------------------------------------------------------------
# 4. FUNÇÃO DE TREINAMENTO E AVALIAÇÃO (Otimizada para Transfer Learning)
# ---------------------------------------------------------------------------
def train_and_evaluate_hybrid(X_train, y_train, X_test, y_test, task, epochs=15, batch_size=512, encoder_type='mlp'):
    """
    Recebe os dados brutos, normaliza isoladamente, treina em batch via GPU 
    e retorna as métricas idênticas às do Scikit-Learn.
    Agora permite escolher entre 'mlp' (padrão) e 'tabnet'.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Normalização Blindada
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    X_tr_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    X_te_t = torch.tensor(X_test_s, dtype=torch.float32)
    y_te_t = torch.tensor(y_test, dtype=torch.long)
    
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
    
    num_classes = 2 if task == 'detection' else len(np.unique(y_train))
    num_features = X_train.shape[1]
    
    model = HybridDLModel(num_features=num_features, num_classes=num_classes, encoder_type=encoder_type).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits, _ = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            
    model.eval()
    with torch.no_grad():
        test_loader = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=1024, shuffle=False)
        all_logits = []
        all_attn = []
        for bx, _ in test_loader:
            bx = bx.to(device)
            l, a = model(bx)
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
        
    return bal_acc, macro_f1, roc_auc, mean_attention
