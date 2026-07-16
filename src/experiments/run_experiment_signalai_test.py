import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score

# Adiciona a raiz do projeto ao path para os imports funcionarem
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.data.dataloader import load_vibration_data
from src.features.signalai_wrapper import SignalAIWrapper

# IMPORTANTE: Importe aqui a classe do SignAI que deseja testar
from signalai.features.freq import SpectralEntropy

# --- CONFIGURAÇÃO GLOBAL ---
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))

def run_signalai_test():
    print(f"{'='*60}\n TESTE DE INTEGRAÇÃO RÁPIDO: VIBNET-1D + SIGNAI WRAPPER \n{'='*60}")

    # 1. Parâmetros simplificados para execução super rápida
    dataset_name = "CWRU_12k"
    test_condition = "Load_0HP"
    task = "diagnosis"
    fs = 12000

    print(f"\n[1] Carregando dados ({dataset_name} - Condição: {test_condition})...")
    X_train_raw, y_train, X_test_raw, y_test, le = load_vibration_data(
        data_root=DATA_ROOT, dataset_name=dataset_name, test_condition=test_condition, task=task
    )

    if len(X_train_raw) == 0:
        print("Erro: Dados não encontrados. Verifique se o DATA_ROOT está correto.")
        return

    # 2. Utilizando o Wrapper para extração de características
    print(f"\n[2] Inicializando SignalAIWrapper (fs={fs}Hz)...")
    
    # Instancia o extrator desejado da biblioteca SignAI
    signai_extractor = SpectralEntropy()
    
    # Encapsula no seu wrapper
    wrapper = SignalAIWrapper(sample_rate=fs, extractor_instance=signai_extractor)

    print(f"  -> Extraindo features de {len(X_train_raw)} janelas de treino (Isso pode levar alguns segundos)...")
    X_train_features = wrapper.fit_transform(X_train_raw)
    
    print(f"  -> Extraindo features de {len(X_test_raw)} janelas de teste...")
    X_test_features = wrapper.fit_transform(X_test_raw)

    print(f"  -> Shape final das features: Treino {X_train_features.shape}, Teste {X_test_features.shape}")

    # 3. Teste de Treinamento Rápido com Random Forest
    print("\n[3] Treinando Random Forest para validar o pipeline...")
    # Usando apenas 50 estimadores para ser ainda mais rápido no teste
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    rf.fit(X_train_features, y_train)
    
    y_pred = rf.predict(X_test_features)
    bal_acc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')

    print(f"\n{'='*40}")
    print(f" RESULTADO DO TESTE (SignAI + Random Forest)")
    print(f" Balanced Accuracy : {bal_acc:.4f}")
    print(f" Macro F1 Score    : {macro_f1:.4f}")
    print(f"{'='*40}")
    print("Sucesso! O pipeline do SignAI conversou perfeitamente com o VibNet-1D.")

if __name__ == "__main__":
    run_signalai_test()
