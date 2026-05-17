import numpy as np
from scipy.stats import kurtosis, skew

def extract_time_features(X):
    """
    Extrai as 19 características estatísticas no domínio do tempo baseadas na literatura.
    
    Parâmetros:
    - X: Array NumPy com os sinais brutos. Formato esperado: (N_amostras, N_pontos_no_tempo)
    
    Retorna:
    - features: Array NumPy formato (N_amostras, 19_características)
    """
    print("Extraindo 19 características estatísticas avançadas...")
    
    N = X.shape[1]
    t_indices = np.arange(1, N + 1) # Vetor de tempo para métricas temporais
    
    # 1-4. Momentos Estatísticos Básicos
    mean = np.mean(X, axis=1)
    std = np.std(X, axis=1)
    var = np.var(X, axis=1)
    rms = np.sqrt(np.mean(X**2, axis=1))
    
    # 5-6. Forma da Distribuição
    kurt = kurtosis(X, axis=1, fisher=False)
    skw = skew(X, axis=1)
    
    # 7-8. Picos e Amplitude
    peak = np.max(np.abs(X), axis=1)
    peak_to_peak = np.ptp(X, axis=1) # max - min
    
    # 9-12. Fatores de Engenharia
    crest_factor = peak / (rms + 1e-8)
    clearance_factor = peak / ((np.mean(np.sqrt(np.abs(X)), axis=1))**2 + 1e-8)
    impulse_factor = peak / (np.mean(np.abs(X), axis=1) + 1e-8)
    shape_factor = rms / (np.mean(np.abs(X), axis=1) + 1e-8)
    
    # 13. Zero-Crossing Rate (ZCR)
    zcr = np.sum((X[:, :-1] * X[:, 1:]) < 0, axis=1) / (N - 1)
    
    # 14-17. Descritores Temporais
    sum_abs = np.sum(np.abs(X), axis=1) + 1e-8
    
    # Temporal Centroid
    temporal_centroid = np.sum(t_indices * np.abs(X), axis=1) / sum_abs
    
    # Effective Duration
    effective_duration = np.sum(np.abs(X)**2, axis=1) / (np.max(np.abs(X)**2, axis=1) + 1e-8)
    
    # Log Attack Time (Tempo até atingir o pico máximo absoluto)
    attack_indices = np.argmax(np.abs(X), axis=1) + 1
    log_attack_time = np.log10(attack_indices)
    
    # Temporal Decrease
    weights = 1.0 / np.arange(1, N) # 1/(i-1) para i=2...N
    diffs = X[:, 1:] - X[:, 0:1]    # (xi - x1)
    temporal_decrease = np.sum(weights * diffs, axis=1) / (np.sum(np.abs(X[:, 1:]), axis=1) + 1e-8)
    
    # 18-19. Limites do Histograma (Aproximados por Min e Max do sinal bruto)
    hist_upper = np.max(X, axis=1)
    hist_lower = np.min(X, axis=1)
    
    # Empilhar todas as 19 características
    features = np.column_stack((
        mean, std, var, rms, kurt, skw, peak, peak_to_peak, 
        crest_factor, clearance_factor, impulse_factor, shape_factor,
        zcr, temporal_centroid, effective_duration, log_attack_time, 
        temporal_decrease, hist_upper, hist_lower
    ))
    
    return features
