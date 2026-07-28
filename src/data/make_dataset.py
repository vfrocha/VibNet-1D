import os
import numpy as np
import pandas as pd
from tqdm import tqdm
import vibdata.raw as raw_datasets
from vibdata.deep.signal.transforms import Sequential, Transform
from scipy.signal import detrend

# --- CLASSES AUXILIARES (1D) ---
class SimpleSplit(Transform):
    def __init__(self, window_size=2048, overlap=0):
        super().__init__()
        self.window_size = window_size
        self.step = window_size - overlap
        
    def transform(self, data):
        data = data.copy()
        sig = data['signal']
        if isinstance(sig, list): sig = sig[0]
        if isinstance(sig, np.ndarray): sig = sig.flatten()
        windows = []
        if len(sig) >= self.window_size:
            for i in range(0, len(sig) - self.window_size + 1, self.step):
                windows.append(sig[i : i + self.window_size])
        data['signal'] = windows
        return data

class Detrend(Transform):
    def transform(self, data):
        data = data.copy()
        sig = data['signal']
        if isinstance(sig, np.ndarray):
            sig = sig.flatten()
            data['signal'] = detrend(sig, type='linear')
        elif isinstance(sig, list):
            data['signal'] = [detrend(s.flatten(), type='linear') if isinstance(s, np.ndarray) else s for s in sig]
        return data

# --- PIPELINES 1D (Janelamento de 1 Segundo Exato - BASEADO EM METADADOS) ---
PIPELINES = {
    "CWRU_12k": Sequential([Detrend(), SimpleSplit(window_size=12000)]), # fs = 12.000 Hz
    "CWRU_48k": Sequential([Detrend(), SimpleSplit(window_size=48000)]), # fs = 48.000 Hz
    "HUST": Sequential([Detrend(), SimpleSplit(window_size=51200)]),     # fs = 51.200 Hz (Corrigido)
    "HUST_Gearbox": Sequential([Detrend(), SimpleSplit(window_size=25600)]),
    "UORED": Sequential([Detrend(), SimpleSplit(window_size=42000)]),    # fs = 42.000 Hz (Corrigido) overlap de 90%
    "PU": Sequential([Detrend(), SimpleSplit(window_size=64000)]),        # fs = 64.000 Hz
    "UOEMD": Sequential([Detrend(), SimpleSplit(window_size=42000)]),
    "Mechanical_Gear": Sequential([Detrend(), SimpleSplit(window_size=5000)]),
    "Electric_Motor": Sequential([Detrend(), SimpleSplit(window_size=50000)]),
    "IMS": Sequential([Detrend(), SimpleSplit(window_size=20000)]),
    "MFPT": Sequential([Detrend(), SimpleSplit(window_size=48828)]), # Ou 97656 dependendo da condição, ajuste se necessário
    "UOC": Sequential([Detrend(), SimpleSplit(window_size=2048)])
}

def get_names(ds_name, meta):
    # 1. CWRU 12k -> 4 Condições (Apenas Carga)
    if ds_name == "CWRU_12k":
        load = meta.get('load', 0)
        try: load = int(load)
        except: load = 0
        cond = f"Load_{load}HP"

    # 2. CWRU 48k -> 12 Condições (Carga x Severidade)
    elif ds_name == "CWRU_48k":
        load = meta.get('load', 0)
        try: load = int(load)
        except: load = 0
        
        # Puxa o diâmetro da falha. Se não existir (ex: Normal), usa '0.000'
        sev = meta.get('fault_diameter', meta.get('severity', '0.000'))
        # Limpa o valor se ele vier como número solto (ex: 0.007)
        if isinstance(sev, (float, int)):
            sev = f"{sev:.3f}"
        cond = f"Load_{load}HP_Sev_{sev}"

    # 3. UOEMD -> 8 ou 16 Condições (Carga x Velocidade)
    elif ds_name == "UOEMD":
        speed = meta.get('speed', 'Unknown')
        load = meta.get('load', 'Unknown')
        # Cria exatamente o padrão do seu dicionário: "Load_Loaded_Speed_15Hz"
        cond = f"Load_{load}_Speed_{speed}"

    # 4. HUST Gearbox -> 30 Condições (Velocidade x Carga)
    elif "HUST" in ds_name:
        speed = meta.get('rotation_hz', meta.get('speed', '0'))
        load = meta.get('load_W', meta.get('load', '0'))
        # Cria exatamente o padrão do seu dicionário: "Cond_20_0" ou "Cond_L0_VS_0_40_0"
        cond = f"Cond_{speed}_{load}"

    # 5. Outros Datasets (PU, UORED, etc)
    else:
        val = meta.get('load', meta.get('rotation_hz', '0'))
        cond = f"Cond_{str(val).replace('.', '')}"

    # --- DEFINIÇÃO DA CLASSE (Rótulos) ---
    orig_label = meta.get('label', '')
    if isinstance(orig_label, pd.Series): orig_label = orig_label.item()
    
    # Padroniza a Classe Normal independentemente do dataset
    label_str = str(orig_label).lower()
    if 'normal' in label_str or 'healthy' in label_str or label_str == '0':
        label_name = "Class_Normal"
    else:
        label_name = f"Class_{orig_label}"
        
    return cond, label_name

def extract_signal(item):
    raw = item.get('signal')
    if isinstance(raw, np.ndarray) and raw.dtype == 'O' and raw.size > 0: return raw[0]
    if isinstance(raw, np.ndarray): return raw
    return None

# --- CONFIGURAÇÃO DE DIRETÓRIOS ---
RAW_DATA_DIR = "/home/vfrocha/VibNet_Project/raw_data"

# Salva os dados processados dentro do repositório atual (VibNet-1D)
#FINAL_1D_DIR = os.path.join(os.getcwd(), "data", "processed")
FINAL_1D_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))

if __name__ == "__main__":
    datasets = [
        # "UOEMD", "CWRU_12k", "CWRU_48k", "PU", "HUST", 
        # "HUST_Gearbox", "Mechanical_Gear", "Electric_Motor",
        # "UORED", "IMS", "MFPT", "UOC"
        "CWRU_48k", "UOEMD", "HUST_Gearbox"
    ]
    
    for ds_name in datasets:
        print(f"\n=== Processando {ds_name} (1D) ===")

        try:
            raw_cls = getattr(raw_datasets, f"{ds_name}_raw")
            # CRÍTICO: download=False impede que a biblioteca tente baixar novamente.
            # Ela vai ler os arquivos que já estão em /home/vfrocha/VibNet_Project/raw_data
            ds = raw_cls(RAW_DATA_DIR, download=False)
        except Exception as e: 
            print(f"Erro ao carregar {ds_name}: {e}")
            continue

        saved_count = {}

        for i in tqdm(range(len(ds))):
            try:
                item = ds[i]
                if not isinstance(item, dict): continue

                sig_array = extract_signal(item)
                if sig_array is None: continue

                meta = item['metainfo']
                if isinstance(meta, pd.DataFrame): meta = meta.iloc[0]

                target_ds_name = ds_name
                if ds_name == "CWRU":
                    sr = meta.get('sample_rate', 12000)
                    if sr > 20000:
                        target_ds_name = "CWRU_48k"
                    else:
                        target_ds_name = "CWRU_12k"

                current_transform = PIPELINES.get(target_ds_name)
                if not current_transform: continue

                save_path = os.path.join(FINAL_1D_DIR, target_ds_name)
                os.makedirs(save_path, exist_ok=True)

                sample = {"signal": sig_array, "metainfo": pd.DataFrame([meta])}
                processed = current_transform(sample)

                windows = processed["signal"]
                if isinstance(windows, list) and len(windows) > 0:
                    cond, lbl = get_names(target_ds_name, meta)
                    final_dir = os.path.join(save_path, cond, lbl)
                    os.makedirs(final_dir, exist_ok=True)

                    for idx, window in enumerate(windows):
                        if isinstance(window, np.ndarray):
                            # Salva como array NumPy (.npy) em vez de imagem (.png)
                            fname = f"s{i:05d}_w{idx:02d}.npy"
                            file_path = os.path.join(final_dir, fname)
                            np.save(file_path, window)

                    saved_count[target_ds_name] = saved_count.get(target_ds_name, 0) + len(windows)

            except Exception as e: 
                print(f"\n[AVISO] Erro silencioso no ficheiro {i} da base {target_ds_name}: {e}")
                continue

        print(f"--> Status de extração 1D: {saved_count}")
