import numpy as np
import pandas as pd

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
