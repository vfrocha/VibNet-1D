import numpy as np

class SignalAIWrapper:
    def __init__(self, sample_rate, extractor_instance):
        self.sample_rate = sample_rate
        self.extractor = extractor_instance

    def fit_transform(self, X):
        all_features = []

        for i in range(X.shape[0]):
            signal_array = X[i, :]

            signal_dict = {
                "signal": signal_array,
                "metainfo": {
                    "sample_rate": self.sample_rate
                }
            }

            features_out = self.extractor.transform(signal_dict)

            if isinstance(features_out, dict):
                feature_values = list(features_out.values())
            elif getattr(features_out, "shape", None) != None: 
                feature_values = features_out.flatten().tolist()
            else:
                feature_values = [features_out] 

            all_features.append(feature_values)

        return np.array(all_features)
