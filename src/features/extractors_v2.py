import numpy as np
from scipy.signal import hilbert, welch, stft
from scipy.stats import entropy, kurtosis

def teager_kaiser_energy(x):
    """Operador de Energia Teager-Kaiser (TKEO)"""
    x_np = np.asarray(x)
    tkeo = np.zeros_like(x_np)
    tkeo[1:-1] = x_np[1:-1]**2 - x_np[:-2] * x_np[2:]
    return tkeo

def shannon_entropy(x, bins=100):
    """Entropia de Shannon da distribuição do sinal"""
    hist, _ = np.histogram(x, bins=bins, density=True)
    # Remove zeros para evitar log(0)
    hist = hist[hist > 0]
    return entropy(hist)

def extract_advanced_features(signal, fs):
    """
    Extrai as características avançadas definidas na nova proposta do artigo.
    :param signal: Array 1D do sinal de vibração bruto (Janela de 1 segundo).
    :param fs: Frequência de amostragem (Sampling Rate) do dataset atual.
    :return: Vetor numpy 1D com todas as novas features concatenadas.
    """
    # -------------------------------------------------------------
    # 1. DOMÍNIO DO TEMPO
    # -------------------------------------------------------------
    # Básicas
    abs_mean = np.mean(np.abs(signal))
    shan_entropy = shannon_entropy(signal)
    
    # Envelope de Hilbert
    analytic_signal = hilbert(signal)
    amplitude_envelope = np.abs(analytic_signal)
    hilbert_mean = np.mean(amplitude_envelope)
    hilbert_rms = np.sqrt(np.mean(amplitude_envelope**2))
    hilbert_kurtosis = kurtosis(amplitude_envelope, fisher=False)
    
    # Teager-Kaiser Energy Operator (TKEO)
    tkeo_signal = teager_kaiser_energy(signal)
    tkeo_mean = np.mean(tkeo_signal)
    tkeo_std = np.std(tkeo_signal)
    
    time_features = [abs_mean, shan_entropy, hilbert_mean, hilbert_rms, 
                     hilbert_kurtosis, tkeo_mean, tkeo_std]

    # -------------------------------------------------------------
    # 2. DOMÍNIO DA FREQUÊNCIA (Baseado no PSD / Welch)
    # -------------------------------------------------------------
    # Usando Welch para um PSD mais limpo. Nperseg = fs/2 para boa resolução.
    nperseg = min(len(signal), int(fs/2))
    f, Pxx = welch(signal, fs, nperseg=nperseg)
    
    peak_idx = np.argmax(Pxx)
    peak_freq = f[peak_idx]
    peak_mag = Pxx[peak_idx]
    
    total_power = np.sum(Pxx)
    # Exemplo: Dividindo a banda em Low (0-1000Hz) e High (>1000Hz)
    low_freq_mask = f <= 1000
    high_freq_mask = f > 1000
    
    low_power = np.sum(Pxx[low_freq_mask])
    high_power = np.sum(Pxx[high_freq_mask])
    
    # Relativo e Taxa de Energia
    relative_band_power = low_power / (total_power + 1e-9)
    high_low_energy_rate = high_power / (low_power + 1e-9)
    legacy_band_power = total_power # Poderia ser de uma banda específica exigida
    
    freq_features = [peak_freq, peak_mag, total_power, relative_band_power, 
                     high_low_energy_rate, legacy_band_power]

    # -------------------------------------------------------------
    # 3. TEMPO-FREQUÊNCIA
    # -------------------------------------------------------------
    # STFT (Short-Time Fourier Transform)
    f_stft, t_stft, Zxx = stft(signal, fs, nperseg=256)
    stft_power = np.abs(Zxx)**2
    log_stft_power = np.log(stft_power + 1e-9)
    
    log_stft_mean = np.mean(log_stft_power)
    log_stft_std = np.std(log_stft_power)
    
    # PSD Peak Frequency in 1-200 Hz Band
    band_mask = (f >= 1) & (f <= 200)
    if np.any(band_mask):
        peak_idx_band = np.argmax(Pxx[band_mask])
        psd_peak_1_200 = f[band_mask][peak_idx_band]
    else:
        psd_peak_1_200 = 0.0
        
    time_freq_features = [log_stft_mean, log_stft_std, psd_peak_1_200]

    # Concatena tudo em um único vetor (7 + 6 + 3 = 16 Super Features)
    all_features = np.array(time_features + freq_features + time_freq_features)
    
    # Substitui NaNs ou Infs por 0 (segurança matemática)
    all_features = np.nan_to_num(all_features, nan=0.0, posinf=0.0, neginf=0.0)
    
    return all_features
