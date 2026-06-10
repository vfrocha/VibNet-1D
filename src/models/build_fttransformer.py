import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# --- 1. ARQUITETURA DO FT-TRANSFORMER COM XAI ---
class MiniFTTransformer(nn.Module):
    def __init__(self, num_features, num_classes, d_token=32, n_heads=4):
        super().__init__()
        self.num_features = num_features
        
        # Tokenizador de Features Numéricas (Converte cada feature escalar em um vetor d_token)
        self.embeddings = nn.ModuleList([nn.Linear(1, d_token) for _ in range(num_features)])
        
        # Token [CLS] que vai acumular a informação de todas as features para a classificação
        self.cls_token = nn.Parameter(torch.randn(1, 1, d_token))
        
        # Bloco Transformer (Atenção Multi-Cabeça)
        self.attention = nn.MultiheadAttention(embed_dim=d_token, num_heads=n_heads, batch_first=True)
        self.layer_norm1 = nn.LayerNorm(d_token)
        
        self.ffn = nn.Sequential(
            nn.Linear(d_token, d_token * 2),
            nn.ReLU(),
            nn.Linear(d_token * 2, d_token)
        )
        self.layer_norm2 = nn.LayerNorm(d_token)
        
        # Cabeça de Classificação Final
        self.head = nn.Linear(d_token, num_classes)

    def forward(self, x):
        batch_size = x.size(0)
        
        # 1. Tokenização
        tokens = [self.embeddings[i](x[:, i:i+1]).unsqueeze(1) for i in range(self.num_features)]
        tokens = torch.cat(tokens, dim=1) # Shape: (Batch, Num_Features, d_token)
        
        # 2. Adiciona o [CLS] Token no início
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_emb = torch.cat((cls_tokens, tokens), dim=1) # Shape: (Batch, 1 + Num_Features, d_token)
        
        # 3. Mecanismo de Atenção
        # need_weights=True nos devolve a matriz de atenção para o XAI
        attn_out, attn_weights = self.attention(x_emb, x_emb, x_emb, need_weights=True)
        x_emb = self.layer_norm1(x_emb + attn_out)
        
        # 4. Feed Forward Network
        ffn_out = self.ffn(x_emb)
        x_emb = self.layer_norm2(x_emb + ffn_out)
        
        # 5. O XAI: Pegamos a atenção que o token [CLS] (índice 0) deu para as features (índices 1 em diante)
        cls_attention = attn_weights[:, 0, 1:] 
        
        # 6. Classificação usando apenas a saída do [CLS]
        logits = self.head(x_emb[:, 0, :])
        
        return logits, cls_attention


# --- 2. LOOP DE TREINAMENTO E AVALIAÇÃO PADRONIZADO ---
def train_and_evaluate_ft_transformer(X_train, y_train, X_test, y_test, task):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Normalização
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    # Tensores
    X_tr_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    X_te_t = torch.tensor(X_test_s, dtype=torch.float32)
    y_te_t = torch.tensor(y_test, dtype=torch.long)
    
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=128, shuffle=True)
    
    num_classes = 2 if task == 'detection' else len(np.unique(y_train))
    model = MiniFTTransformer(num_features=X_train.shape[1], num_classes=num_classes).to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    
    # Treinamento com Early Stopping Simples
    epochs = 50
    model.train()
    for epoch in range(epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits, _ = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            
    # Avaliação e Extração do XAI
    model.eval()
    with torch.no_grad():
        X_te_t = X_te_t.to(device)
        logits, attn_weights = model(X_te_t)
        
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        preds = np.argmax(probs, axis=1)
        
        # Média da atenção de todo o dataset de teste para o XAI
        mean_attention = attn_weights.mean(dim=0).cpu().numpy()
        
    # Métricas idênticas ao TabNet
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
