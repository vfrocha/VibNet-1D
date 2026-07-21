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

class SignalAIWrapper:
    def __init__(self, sample_rate, extractors_list):
        self.sample_rate = sample_rate
        if not isinstance(extractors_list, list):
            self.extractors = [extractors_list]
        else:
            self.extractors = extractors_list

    def fit_transform(self, X):
        # --- 1. DRY RUN (TESTE DE SEGURANÇA) ---
        print("\n  -> [Wrapper] Validando compatibilidade dos extratores...")
        valid_extractors = []
        
        # Pega apenas a primeira amostra para o teste
        test_dict = {
            "signal": X[0, :],
            "metainfo": CompatibleMeta({"sample_rate": self.sample_rate})
        }

        for ext in self.extractors:
            try:
                # Testa se o extrator sobrevive a um sinal 1D
                _ = ext.transform(test_dict)
                valid_extractors.append(ext)
            except Exception as e:
                # Se quebrar, nós o removemos da lista sem travar o experimento
                print(f"     [Ignorado] {ext.__class__.__name__} incompatível com sinal 1D. (Erro: {type(e).__name__})")
        
        print(f"  -> [Wrapper] Restaram {len(valid_extractors)} extratores 100% estáveis para extração.\n")
        
        if len(valid_extractors) == 0:
            raise ValueError("Nenhum extrator sobreviveu ao teste. Verifique a biblioteca.")

        # --- 2. EXTRAÇÃO REAL ---
        all_samples_features = []

        for i in range(X.shape[0]):
            signal_array = X[i, :]

            signal_dict = {
                "signal": signal_array,
                "metainfo": CompatibleMeta({"sample_rate": self.sample_rate})
            }

            current_sample_features = []

            for extractor in valid_extractors:
                # Como já filtramos os ruins, isso vai rodar liso
                features_out = extractor.transform(signal_dict)

                if isinstance(features_out, dict):
                    feature_values = list(features_out.values())
                elif getattr(features_out, "shape", None) != None: 
                    feature_values = features_out.flatten().tolist()
                else:
                    feature_values = [features_out] 

                current_sample_features.extend(feature_values)

            all_samples_features.append(current_sample_features)

        return np.array(all_samples_features)

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
                    pass # Ignora silenciosamente os que precisam de parâmetros
    return extractors

def extract_fusion_features(X_raw, fs, vibnet_extractor_func):
    """
    Função modular que executa a Feature Fusion (VibNet + SignAI).
    :param X_raw: Matriz 2D de sinais puros.
    :param fs: Frequência de amostragem.
    :param vibnet_extractor_func: A sua função extract_advanced_features.
    :return: Matriz 2D com todas as features concatenadas.
    """
    print(f"      -> [Fusion] Iniciando extração dupla para {X_raw.shape[0]} amostras...")
    
    # 1. Extração SignAI
    extractors = get_all_signalai_extractors()
    wrapper = SignalAIWrapper(sample_rate=fs, extractors_list=extractors)
    X_signai = wrapper.fit_transform(X_raw)
    
    # 2. Extração VibNet-1D
    X_vibnet = np.array([vibnet_extractor_func(sinal, fs) for sinal in X_raw])
    
    # 3. Fusão
    X_fusion = np.hstack((X_vibnet, X_signai))
    print(f"      -> [Fusion] Shape final combinado: {X_fusion.shape}")
    
    return X_fusion
