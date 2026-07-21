import os
import sys
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import balanced_accuracy_score, f1_score
import inspect
from vibdata.deep.signal.transforms import Transform # Classe base que os extratores usam
import signalai.features.freq as freq
import signalai.features.wavelet as wavelet
import signalai.features.custom as custom


# Adiciona a raiz do projeto ao path para os imports funcionarem
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.features.extractors_v2 import extract_advanced_features
from src.data.dataloader import load_vibration_data

# Adicione esta linha de volta no topo do seu script
from src.features.signalai_wrapper import SignalAIWrapper

# (E certifique-se de manter os imports do SignAI logo abaixo dela)
from signalai.features.freq import SpectralEntropy, SpectralCentroid, SpectralBandwidth, SpectralFlatness

# --- CONFIGURAÇÃO GLOBAL ---
DATA_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))


def get_all_signalai_extractors():
    """
    Varre os módulos da biblioteca SignAI automaticamente em busca 
    de todos os extratores de features disponíveis.
    """
    extractors = []
    
    # Lista de arquivos/módulos que vamos vasculhar
    modules_to_scan = [freq, wavelet, custom] 
    
    for module in modules_to_scan:
        # Pega todos os objetos dentro do arquivo que sejam Classes
        for name, obj in inspect.getmembers(module, inspect.isclass):
            
            # Verifica se a classe herda de 'Transform' (ou seja, é um extrator de feature válido)
            # E ignora a própria classe base 'Transform'
            if issubclass(obj, Transform) and obj is not Transform:
                try:
                    # Instancia o extrator com os parâmetros padrão (ex: SpectralEntropy())
                    instance = obj()
                    extractors.append(instance)
                except Exception as e:
                    # Se alguma feature exigir um parâmetro obrigatório sem valor padrão,
                    # ela será ignorada e avisará no console.
                    print(f"  [Aviso] Pulando '{name}': Requer parâmetros obrigatórios. Erro: {e}")
                    
    print(f"\n[INFO] Auto-Discovery concluiu: {len(extractors)} features carregadas com sucesso!\n")
    return extractors

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
    
    # Busca todas as 100+ features automaticamente
    meus_extratores = get_all_signalai_extractors()
    
    # Encapsula a LISTA GIGANTE no seu wrapper
    wrapper = SignalAIWrapper(sample_rate=fs, extractors_list=meus_extratores)

    print(f"  -> Extraindo {len(meus_extratores)} features de {len(X_train_raw)} janelas de treino (Isso VAI levar um tempo!)...")
    X_train_features = wrapper.fit_transform(X_train_raw)
    
    print(f"  -> Extraindo features de {len(X_test_raw)} janelas de teste...")
    X_test_features = wrapper.fit_transform(X_test_raw)

    print(f"  -> Shape final das features SignAI: Treino {X_train_features.shape}, Teste {X_test_features.shape}")

    print(f"  -> Shape final das features: Treino {X_train_features.shape}, Teste {X_test_features.shape}")

    print("\n[3] Extraindo features nativas do VibNet-1D...")
    
    # Passamos cada 'sinal' individualmente usando um loop rápido
    X_train_vibnet = np.array([extract_advanced_features(sinal, fs) for sinal in X_train_raw])
    X_test_vibnet  = np.array([extract_advanced_features(sinal, fs) for sinal in X_test_raw])

    print("\n[4] Fundindo (Feature Fusion) SignAI + VibNet-1D...")
    # O np.hstack junta as duas matrizes lado a lado
    X_train_combined = np.hstack((X_train_vibnet, X_train_features))
    X_test_combined  = np.hstack((X_test_vibnet, X_test_features))
    
    print(f"  -> Shape FINAL COMBINADO: Treino {X_train_combined.shape}, Teste {X_test_combined.shape}")
    # -------------------------------------------

    # 5. Teste de Treinamento Rápido com Random Forest
    print("\n[5] Treinando Random Forest com as features combinadas...")
    rf = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    
    rf.fit(X_train_combined, y_train) 
    y_pred = rf.predict(X_test_combined)

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
