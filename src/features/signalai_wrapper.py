import numpy as np

class SignalAIWrapper:
    def __init__(self, sample_rate, extractors_list):
        """
        :param sample_rate: Frequência de amostragem.
        :param extractors_list: Uma lista de instâncias do SignAI (ex: [SpectralEntropy(), SpectralCentroid(), ...])
        """
        self.sample_rate = sample_rate
        # Se o usuário passar apenas um extrator, converte para lista para não quebrar o loop
        if not isinstance(extractors_list, list):
            self.extractors = [extractors_list]
        else:
            self.extractors = extractors_list

    def fit_transform(self, X):
        all_samples_features = []

        for i in range(X.shape[0]):
            signal_array = X[i, :]

            # Monta o dicionário exigido pela signalai
            signal_dict = {
                "signal": signal_array,
                "metainfo": {
                    "sample_rate": self.sample_rate
                }
            }

            # Armazena as features desta amostra específica
            current_sample_features = []

            # Roda todos os extratores solicitados para este sinal
            for extractor in self.extractors:
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
