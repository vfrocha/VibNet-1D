import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

# --- 1. MODELOS CLÁSSICOS DE ANOMALIA (Scikit-Learn) ---
def evaluate_classical_anomaly(X_train_normal, X_test, y_test):
    """
    Treina em dados normais. No Scikit-Learn, Inliers = 1, Outliers = -1.
    Nós mapeamos de volta para: 0 (Normal) e 1 (Fault) para manter a compatibilidade.
    """
    results = []
    
    models = {
        "Isolation Forest": IsolationForest(n_estimators=100, contamination=0.01, random_state=42),
        "One-Class SVM": OneClassSVM(kernel='rbf', gamma='scale', nu=0.01)
    }
    
    for name, model in models.items():
        # Treina APENAS com os dados normais
        model.fit(X_train_normal)
        
        # Predição: retorna 1 (normal) ou -1 (anomalia)
        preds_raw = model.predict(X_test)
        
        # Mapeamento para o nosso padrão: 1 vira 0 (Normal), -1 vira 1 (Fault)
        preds_mapped = np.where(preds_raw == 1, 0, 1)
        
        # Scores para ROC-AUC (função decision_function retorna > 0 para inliers, < 0 para outliers)
        # Invertemos o sinal para que quanto maior, maior a chance de ser anomalia
        scores = -model.decision_function(X_test)
        
        bal_acc = balanced_accuracy_score(y_test, preds_mapped)
        f1 = f1_score(y_test, preds_mapped, average='binary')
        auc = roc_auc_score(y_test, scores)
        
        results.append({
            "Model": name,
            "Bal Acc": bal_acc,
            "F1 (Anomaly)": f1,
            "ROC-AUC": auc
        })
        
    return results

# --- 2. DEEP LEARNING: AUTOENCODER ESPELHADO (PyTorch) ---
class MLPAutoencoder(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        # ENCODER: Cópia exata da profundidade usada no Transfer Learning
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.2), # Dropout mantido para evitar overfit na reconstrução
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 64) # Espaço Latente (Bottleneck idêntico)
        )
        
        # DECODER: O Espelho exato (64 -> 128 -> 256 -> input_dim)
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, input_dim) # Camada de Saída Sem Ativação para MSE Loss
        )

    def forward(self, x):
        latent = self.encoder(x)
        reconstructed = self.decoder(latent)
        return reconstructed

def evaluate_autoencoder_anomaly(X_train_normal, X_test, y_test, epochs=30, batch_size=256):
    """
    Treina um Autoencoder para reconstruir apenas os dados normais usando MSELoss.
    Aplica 3 limiares diferentes para classificar anomalias no teste.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = X_train_normal.shape[1]
    
    model = MLPAutoencoder(input_dim).to(device)
    criterion = nn.MSELoss(reduction='none') # Erro Quadrático Médio sem redução (por amostra)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    X_tr_t = torch.tensor(X_train_normal, dtype=torch.float32)
    train_loader = DataLoader(TensorDataset(X_tr_t), batch_size=batch_size, shuffle=True)
    
    # 1. TREINAMENTO (Minimizando o Erro de Reconstrução)
    model.train()
    for epoch in range(epochs):
        for batch in train_loader:
            bx = batch[0].to(device)
            optimizer.zero_grad()
            reconstructed = model(bx)
            loss = criterion(reconstructed, bx).mean()
            loss.backward()
            optimizer.step()
            
    # 2. DEFINIÇÃO DOS LIMIARES ESTATÍSTICOS
    model.eval()
    with torch.no_grad():
        X_tr_device = X_tr_t.to(device)
        recon_train = model(X_tr_device)
        # Erro MSE de reconstrução médio por amostra no treino saudável
        train_errors = criterion(recon_train, X_tr_device).mean(dim=1).cpu().numpy()
        
    mean_err = np.mean(train_errors)
    std_err = np.std(train_errors)
    median_err = np.median(train_errors)
    mad_err = np.median(np.abs(train_errors - median_err))
    
    thresholds = {
        "AE (Média + 2σ)": mean_err + 2 * std_err,
        "AE (Percentil 99)": np.percentile(train_errors, 99),
        "AE (Mediana + 3*MAD)": median_err + 3 * mad_err 
    }
    
    # 3. TESTE E AVALIAÇÃO
    X_te_t = torch.tensor(X_test, dtype=torch.float32).to(device)
    with torch.no_grad():
        recon_test = model(X_te_t)
        test_errors = criterion(recon_test, X_te_t).mean(dim=1).cpu().numpy()
        
    results = []
    
    for thr_name, threshold in thresholds.items():
        # Classificação baseada no limite de reconstrução
        preds = (test_errors > threshold).astype(int)
        
        bal_acc = balanced_accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average='binary')
        auc = roc_auc_score(y_test, test_errors)
        
        results.append({
            "Model": thr_name,
            "Bal Acc": bal_acc,
            "F1 (Anomaly)": f1,
            "ROC-AUC": auc
        })
        
    return results
