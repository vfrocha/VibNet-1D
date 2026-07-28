import numpy as np
import pandas as pd
import signalai.features.pipelines as pipe_module

class SignalAIWrapper:
    def __init__(self, sample_rate, pipeline_name='all'):
        self.fs = sample_rate
        
        # Resgate Dinâmico: Procura pelo nome que o seu orientador usou no arquivo
        if hasattr(pipe_module, 'get_pipeline'):
            self.pipeline = pipe_module.get_pipeline(pipeline_name)
        elif hasattr(pipe_module, 'get_feature_pipeline'):
            self.pipeline = pipe_module.get_feature_pipeline(pipeline_name)
        elif hasattr(pipe_module, 'pipelines') and isinstance(pipe_module.pipelines, dict):
            self.pipeline = pipe_module.pipelines[pipeline_name]
        else:
            raise ImportError(f"Estrutura não reconhecida no pipelines.py. Encontrados: {dir(pipe_module)}")

    def fit_transform(self, X):
        # 1. Cria o DataFrame de metadados exigido pela SignAI
        metainfo_df = pd.DataFrame([{"sample_rate": self.fs}] * len(X))
        
        # 2. Converte a matriz de sinais do VibNet
        signals_list = list(X)
        
        # 3. Empacota tudo como a SignAI espera
        data_dict = {
            "signal": signals_list,
            "metainfo": metainfo_df
        }

        # 4. Executa a extração inteira de 71 features em 1 linha!
        try:
            out_dict = self.pipeline.transform(data_dict)
            
            # A SignAI coloca as features num dicionário. Pegamos todas as matrizes exceto os sinais puros
            feature_keys = [k for k in out_dict.keys() if k not in ["signal", "metainfo"]]
            
            extracted_features = []
            for key in feature_keys:
                feat = np.array(out_dict[key])
                if feat.ndim == 1:
                    feat = feat.reshape(-1, 1) # Blindagem matemática
                extracted_features.append(feat)
                
            # Combina tudo numa matriz 2D final
            if extracted_features:
                final_matrix = np.hstack(extracted_features)
                return final_matrix
            else:
                return np.zeros((len(X), 1))
                
        except Exception as e:
            print(f"        [ERRO SignAI] Ocorreu uma falha no Pipeline: {e}")
            import traceback
            traceback.print_exc()
            return np.zeros((len(X), 1))

def extract_fusion_features(X_raw, fs, vibnet_extractor_func):
    """Executa a Feature Fusion (VibNet + SignAI)."""
    print(f"      -> [Fusion] Iniciando extração DUPLA para {X_raw.shape[0]} amostras...")
    
    # 1. Extração VibNet-1D (As 16 características originais)
    X_vibnet = np.array([vibnet_extractor_func(sinal, fs) for sinal in X_raw])
    if X_vibnet.ndim == 1: X_vibnet = X_vibnet.reshape(-1, 1)
    
    # 2. Extração SignAI (Pipeline 'all' = 71 Features)
    wrapper = SignalAIWrapper(sample_rate=fs, pipeline_name='all')
    X_signai = wrapper.fit_transform(X_raw)

    print(f"         [Debug] Shape VibNet: {X_vibnet.shape}")
    print(f"         [Debug] Shape SignAI: {X_signai.shape}")

    # 3. Fusão Final
    X_fusion = np.hstack((X_vibnet, X_signai))
    print(f"      -> [Fusion] Shape final combinado: {X_fusion.shape}")
    
    # Limpeza rigorosa final (Anti-NaN)
    X_fusion_clean = np.nan_to_num(np.array(X_fusion, dtype=np.float32))
    
    return X_fusion_clean
