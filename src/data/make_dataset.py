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
    "CWRU_12k_Severity": Sequential([Detrend(), SimpleSplit(window_size=12000)]), # NOVO PIPELINE PARALELO
    "CWRU_48k": Sequential([Detrend(), SimpleSplit(window_size=48000)]), # fs = 48.000 Hz
    "HUST": Sequential([Detrend(), SimpleSplit(window_size=51200)]),     
    "HUST_Gearbox": Sequential([Detrend(), SimpleSplit(window_size=25600)]),
    "UORED": Sequential([Detrend(), SimpleSplit(window_size=42000)]),    
    "PU": Sequential([Detrend(), SimpleSplit(window_size=64000)]),        
    "UOEMD": Sequential([Detrend(), SimpleSplit(window_size=42000)]),
    "Mechanical_Gear": Sequential([Detrend(), SimpleSplit(window_size=5000)]),
    "Electric_Motor": Sequential([Detrend(), SimpleSplit(window_size=50000)]),
    "IMS": Sequential([Detrend(), SimpleSplit(window_size=20000)]),
    "MFPT": Sequential([Detrend(), SimpleSplit(window_size=48828)]), 
    "UOC": Sequential([Detrend(), SimpleSplit(window_size=2048)])
}

# --- FUNÇÃO DE NOMES (Mantida para garantir Unbiased Split) ---
def get_names(ds_name, meta):   
    if ds_name == "CWRU_12k":
        load = meta.get('load', 0)
        try: load = int(load)
        except: load = 0
        cond = f"Load_{load}HP"

    # NOVO: CWRU 12k -> Condições Mistas (Carga x Severidade) para replicar experimento 48k
    elif ds_name == "CWRU_12k_Severity":
        load = meta.get('load', 0)
        try: load = int(load)
        except: load = 0
        
        sev = meta.get('fault_size', meta.get('severity', '0.000'))
        if isinstance(sev, (float, int)):
            sev = f"{sev:.3f}"
        cond = f"Load_{load}HP_Sev_{sev}"

    # 2. CWRU 48k -> 12 Condições (Carga x Severidade)
    elif ds_name == "CWRU_48k":
        load = meta.get('load', 0)
        try: load = int(load)
        except: load = 0
        
        sev = meta.get('fault_size', meta.get('severity', '0.000'))
        if isinstance(sev, (float, int)):
            sev = f"{sev:.3f}"
        cond = f"Load_{load}HP_Sev_{sev}"

    elif ds_name == "PU":
        fname = str(meta.get('file_name', ''))
        speed_code = fname[:3]
        torque = meta.get('load_nm', 0)
        radial = meta.get('radial_force_n', 0)
        if speed_code == "N15" and torque == 0.7 and radial == 1000: cond = "C1_1500rpm_0.7Nm_1000N"
        elif speed_code == "N09" and torque == 0.7 and radial == 1000: cond = "C2_900rpm_0.7Nm_1000N"
        elif speed_code == "N15" and torque == 0.1 and radial == 1000: cond = "C3_1500rpm_0.1Nm_1000N"
        elif speed_code == "N15" and torque == 0.7 and radial == 400:  cond = "C4_1500rpm_0.7Nm_400N"
        else: cond = f"Cx_Other_{speed_code}_{torque}Nm"

    elif ds_name == "HUST":
        load_w = meta.get('load_W', 0)
        cond = f"Load_{load_w}W"

    elif ds_name == "UORED":
        bid = meta.get('bearing_id', meta.get('bearing.id', 'Unknown'))
        cond = f"Bearing_{bid}"
        stage = meta.get('stage', 'unknown')
        if stage == 'healthy':
            return cond, "Class_Normal"

    elif ds_name == "UOEMD":
        load = meta.get('load', 'Unknown')
        speed = meta.get('speed', 'Unknown')
        cond = f"Load_{load}_Speed_{speed}"
        
    elif ds_name == "Mechanical_Gear":
        cond = f"Cond_{meta.get('condition', 'Unknown')}"

    elif ds_name == "Electric_Motor":
        cond = f"Cond_{meta.get('condition', 'Unknown')}"
    
    elif ds_name == "IMS":
        test_num = meta.get('test', 'Unknown')
        bearing_num = meta.get('bearing', 'Unknown')
        cond = f"Test_{test_num}_Bearing_{bearing_num}"
        
    elif ds_name == "MFPT":
        cond = f"Load_{meta.get('load', 'Unknown')}"
        
    elif ds_name == "UOC":
        cond = f"Cond_{meta.get('condition', 'Unknown')}"

    elif ds_name == "HUST_Gearbox":
        cond = f"Cond_{meta.get('condition', 'Unknown')}"
    
    else:
        val = meta.get('load', meta.get('rotation_hz', '0'))
        cond = f"Cond_{str(val).replace('.', '')}"

    orig_label = meta.get('label')
    if isinstance(orig_label, pd.Series): orig_label = orig_label.item()
    label_name = f"Class_{orig_label}"
    
    return cond, label_name

def extract_signal(item):
    raw = item.get('signal')
    if isinstance(raw, np.ndarray) and raw.dtype == 'O' and raw.size > 0: return raw[0]
    if isinstance(raw, np.ndarray): return raw
    return None

# --- CONFIGURAÇÃO DE DIRETÓRIOS ---
RAW_DATA_DIR = "/home/vfrocha/VibNet_Project/raw_data"
FINAL_1D_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))

if __name__ == "__main__":
    datasets = ["CWRU"]
    
    for ds_name in datasets:
        print(f"\n=== Processando {ds_name} (1D) ===")

        try:
            raw_cls = getattr(raw_datasets, f"{ds_name}_raw")
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

                # ROTEAMENTO INTELIGENTE (Salva 12k duas vezes em pastas diferentes)
                if ds_name == "CWRU":
                    sr = meta.get('sample_rate', 12000)
                    if sr > 20000:
                        target_ds_names = ["CWRU_48k"]
                    else:
                        # Processará o sinal bruto para ambas as estruturas simultaneamente
                        target_ds_names = ["CWRU_12k", "CWRU_12k_Severity"]
                else:
                    target_ds_names = [ds_name]

                # LOOP INTERNO PARA MÚLTIPLOS TARGETS
                for target_ds_name in target_ds_names:
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
                                fname = f"s{i:05d}_w{idx:02d}.npy"
                                file_path = os.path.join(final_dir, fname)
                                np.save(file_path, window)

                        saved_count[target_ds_name] = saved_count.get(target_ds_name, 0) + len(windows)

            except Exception as e: 
                print(f"\n[AVISO] Erro silencioso no ficheiro {i} da base {ds_name}: {e}")
                continue

        print(f"--> Status de extração 1D: {saved_count}")
