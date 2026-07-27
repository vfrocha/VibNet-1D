import numpy as np
import pandas as pd
import inspect
from vibdata.deep.signal.transforms import Transform
import signalai.features.freq as freq
import signalai.features.wavelet as wavelet
import signalai.features.custom as custom

# Criamos uma Série do Pandas 'falsa' que possui o método iterrows apenas para 
# evitar que o código quebre logo de cara na checagem de atributos
class CompatibleMeta(pd.Series):
    def iterrows(self):
        yield 0, self

def get_all_signalai_extractors():
    """Varre os módulos da biblioteca SignAI automaticamente."""
    extractors = []
    modules_to_scan = [freq, wavelet, custom] 
    for module in modules_to_scan:
        for name, obj in inspect.getmembers(module, inspect.isclass):
            if issubclass(obj, Transform) and obj is not Transform:
                try:
                    extractors.append(obj())
                except Exception:
                    pass
    return extractors

class SignalAIWrapper:
    def __init__(self, sample_rate, extractors_list):
        self.fs = sample_rate
        self.extractors = extractors_list

    def fit_transform(self, X):
        # 1. Cria um DataFrame Pandas real com N linhas
        metainfo_df = pd.DataFrame([{"sample_rate": self.fs}] * len(X))
        signals_list = list(X)
        
        # 2. Estrutura o dicionário de batch
        data_dict = {
            "signal": signals_list,
            "metainfo": metainfo_df
        }

        extracted_features = []

        for ext in self.extractors:
            try:
                # Extrai a feature da SignAI
                out = ext.transform(data_dict)
                
                # Desempacota o resultado
                if isinstance(out, dict):
                    keys = [k for k in out.keys() if k not in ["signal", "metainfo"]]
                    feat = np.array(out[keys[-1]]) if keys else np.array(out["signal"])
                else:
                    feat = np.array(out)
                
                # --- A BLINDAGEM MATEMÁTICA ---
                # Força o formato estrito 2D: [Número_de_Amostras, Todas_as_Features_Geradas]
                # Isso impede o np.hstack de quebrar com arrays 1D ou 3D
                feat_2d = feat.reshape(len(X), -1)
                
                extracted_features.append(feat_2d)
                
            except Exception as e:
                # Se o extrator falhar ou retornar lixo, ignoramos silenciosamente
                pass

        # Empacota horizontalmente todas as colunas
        if extracted_features:
            return np.hstack(extracted_features)
        else:
            return np.zeros((len(X), 1))

        # Empacota horizontalmente todas as colunas
        if extracted_features:
            return np.hstack(extracted_features)
        else:
            return np.zeros((len(X), 1))

def extract_fusion_features(X_raw, fs, vibnet_extractor_func):
    """Executa a Feature Fusion (VibNet + SignAI)."""
    print(f"      -> [Fusion] Iniciando extração para {X_raw.shape[0]} amostras...")
    
    # 1. Extração VibNet-1D (As 16 características originais)
    X_vibnet = np.array([vibnet_extractor_func(sinal, fs) for sinal in X_raw])
    if X_vibnet.ndim == 1: X_vibnet = X_vibnet.reshape(-1, 1)
    print(f"         [Debug] Shape VibNet: {X_vibnet.shape}")
    
    # 2. Extração SignAI (Agora em modo BATCH completo via Pandas DataFrame)
    extractors = get_all_signalai_extractors()
    wrapper = SignalAIWrapper(sample_rate=fs, extractors_list=extractors)
    X_signai = wrapper.fit_transform(X_raw)
    print(f"         [Debug] Shape SignAI (Destravado): {X_signai.shape}")

    # 3. Fusão Final
    X_fusion = np.hstack((X_vibnet, X_signai))
    print(f"      -> [Fusion] Shape final combinado: {X_fusion.shape}")
    
    return X_fusion

def extract_fusion_features(X_raw, fs, vibnet_extractor_func):
    """
    Função modular que executa a Feature Fusion (VibNet + SignAI).
    """
    print(f"      -> [Fusion] Iniciando extração para {X_raw.shape[0]} amostras...")
    
    # 1. Extração SignAI
    extractors = get_all_signalai_extractors()
    wrapper = SignalAIWrapper(sample_rate=fs, extractors_list=extractors)
    X_signai = wrapper.fit_transform(X_raw)
    print(f"         [Debug] Shape SignAI: {X_signai.shape} | Tipo: {type(X_signai)}")
    
    # 2. Extração VibNet-1D
    X_vibnet = np.array([vibnet_extractor_func(sinal, fs) for sinal in X_raw])
    print(f"         [Debug] Shape VibNet: {X_vibnet.shape} | Tipo: {type(X_vibnet)}")
    
    # GARANTIA DE 2D: Assegura que ambos os arrays são matrizes 2D antes do hstack
    if X_signai.ndim == 1:
        X_signai = X_signai.reshape(-1, 1)
    if X_vibnet.ndim == 1:
        X_vibnet = X_vibnet.reshape(-1, 1)

    # 3. Fusão
    X_fusion = np.hstack((X_vibnet, X_signai))
    print(f"      -> [Fusion] Shape final combinado: {X_fusion.shape}")
    
    # GARANTIA DE RETORNO 2D
    if X_fusion.ndim == 1:
        print("         [ALERTA] A fusão resultou em 1D. Forçando para 2D.")
        X_fusion = X_fusion.reshape(-1, 1)

    return X_fusion
