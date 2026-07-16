import numpy as np
import pandas as pd
# Assume the user wants to use a specific module from signalai, like TimeDomainFeatures
from signalai.features.time import TimeFeatures # Example import, needs to be adapted based on the library structure

class SignalAIWrapper:
    def __init__(self, sample_rate, feature_extractor_class):
        """
        Inicializa o wrapper.
        :param sample_rate: Taxa de amostragem do dataset (fs).
        :param feature_extractor_class: A classe de extração do SignAI (ex: TimeFeatures).
        """
        self.sample_rate = sample_rate
        self.extractor = feature_extractor_class() # Instancia o extrator do SignAI

    def fit_transform(self, X):
        """
        Processa uma matriz de sinais e extrai as features usando a biblioteca SignAI.
        :param X: Matriz NumPy [n_amostras, tamanho_da_janela]
        :return: Matriz NumPy com as features extraídas.
        """
        all_features = []

        # Itera sobre cada amostra (sinal) individualmente
        for i in range(X.shape[0]):
            signal_array = X[i, :]

            # Monta o dicionário esperado pelo SignAI-Framework
            signal_dict = {
                "signal": signal_array,
                "sample_rate": self.sample_rate
            }

            # Chama a biblioteca
            # Dependendo de como o SignAI retorna (dicionário ou array), adaptamos:
            features_dict = self.extractor.transform(signal_dict)

            # Precisamos extrair apenas os valores numéricos do dicionário de saída
            # (Assumindo que o SignAI retorne um dicionário como {'rms': 1.2, 'kurtosis': 3.1})
            if isinstance(features_dict, dict):
                feature_values = list(features_dict.values())
            else:
                feature_values = features_dict # Se já retornar uma lista/array

            all_features.append(feature_values)

        # Converte a lista de listas de volta para uma Matriz NumPy
        return np.array(all_features)
