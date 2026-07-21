import numpy as np
import pandas as pd

class SignalAIWrapper:
    def __init__(self, sample_rate, extractors_list):
        """
        :param sample_rate: Frequência de amostragem.
        :param extractors_list: Uma lista de instâncias do SignAI.
        """
        self.sample_rate = sample_rate
        # Garante que seja uma lista
        if not isinstance(extractors_list, list):
            self.extractors = [extractors_list]
        else:
            self.extractors = extractors_list

    def fit_transform(self, X):
        all_samples_features = []

        for i in range(X.shape[0]):
            signal_array = X[i, :]

            # Nós enganamos a biblioteca dizendo que isso é uma linha de tabela do Pandas
            signal_dict = {
                "signal": signal_array,
                "metainfo": pd.Series({
                    "sample_rate": self.sample_rate
                })
            }
            # -----------------------------

            current_sample_features = []

            for extractor in self.extractors:
                # O extractor agora consegue usar o .copy(deep=False) que deu erro antes
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
