import numpy as np
from scipy.stats import kurtosis, skew

def extract_time_features(X):
    """
    Extrai características estatísticas no domínio do tempo para um lote de sinais.
    
    Parâmetros:
    - X: Array NumPy com os sinais brutos. Formato esperado: (N_amostras, N_pontos_no_tempo)
    
    Retorna:
    - features: Array NumPy formato (N_amostras, N_características)
    """
    print("Extraindo características estatísticas no domínio do tempo...")
    
    # Cálculos vetorizados para máxima performance
    mean = np.mean(X, axis=1)
    std = np.std(X, axis=1)
    var = np.var(X, axis=1)
    rms = np.sqrt(np.mean(X**2, axis=1))
    
    # Métricas de forma e impulsividade
    kurt = kurtosis(X, axis=1, fisher=False)
    skw = skew(X, axis=1)
    peak = np.max(np.abs(X), axis=1)
    peak_to_peak = np.ptp(X, axis=1)
    
    # Fatores de engenharia clássicos
    crest_factor = peak / (rms + 1e-8) # 1e-8 evita divisão por zero
    shape_factor = rms / (np.mean(np.abs(X), axis=1) + 1e-8)
    impulse_factor = peak / (np.mean(np.abs(X), axis=1) + 1e-8)
    
    # Empilhar todas as características como colunas
    features = np.column_stack((
        mean, std, var, rms, kurt, skw, peak, peak_to_peak, 
        crest_factor, shape_factor, impulse_factor
    ))
    
    return features
