import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
import numpy as np
from src.models.build_tabnet_resnet import MLPEncoder

class PureMLPBaseline(nn.Module):
    """
    MLP Puro (From Scratch). 
    Usa a exata mesma arquitetura do bottleneck de Transfer Learning, 
    mas com uma camada final simples, sem blocos residuais.
    """
    def __init__(self, input_dim, num_classes):
        super().__init__()
        # Mesma profundidade (256 -> 128 -> 64)
        self.encoder = MLPEncoder(input_dim=input_dim, output_dim=64)
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        latent, _ = self.encoder(x)
        logits = self.classifier(latent)
        return logits

def train_and_evaluate_pure_mlp(X_train, y_train, X_test, y_test, task='diagnosis', epochs=50, batch_size=128):
    """
    Treina o MLP Puro do zero usando PyTorch, de forma modular para o baseline.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Normalização Blindada
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    
    X_tr_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.long)
    X_te_t = torch.tensor(X_test_s, dtype=torch.float32)
    y_te_t = torch.tensor(y_test, dtype=torch.long)
    
    train_loader = DataLoader(TensorDataset(X_tr_t, y_tr_t), batch_size=batch_size, shuffle=True)
    
    num_classes = 2 if task == 'detection' else len(np.unique(y_train))
    input_dim = X_train.shape[1]
    
    # 2. Inicializa o modelo com pesos aleatórios (From Scratch)
    model = PureMLPBaseline(input_dim=input_dim, num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 3. Treinamento
    model.train()
    for epoch in range(epochs):
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            logits = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            
    # 4. Avaliação
    model.eval()
    with torch.no_grad():
        test_loader = DataLoader(TensorDataset(X_te_t, y_te_t), batch_size=512, shuffle=False)
        all_logits = []
        for bx, _ in test_loader:
            bx = bx.to(device)
            logits = model(bx)
            all_logits.append(logits.cpu())
            
        final_logits = torch.cat(all_logits, dim=0)
        probs = torch.softmax(final_logits, dim=1).numpy()
        preds = np.argmax(probs, axis=1)
        
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
        
    return bal_acc, macro_f1, roc_auc, model
