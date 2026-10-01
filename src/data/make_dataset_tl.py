import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import detrend, resample_poly

import vibdata.raw as raw_datasets
from vibdata.deep.signal.transforms import Sequential, Transform


# ============================================================
# TRANSFORMAÇÕES 1D
# ============================================================

class SimpleSplit(Transform):
    """
    Divide um sinal em janelas de tamanho fixo, sem sobreposição.
    """
    def __init__(self, window_size=2048, overlap=0):
        super().__init__()
        self.window_size = window_size
        self.step = window_size - overlap

    def transform(self, data):
        data = data.copy()
        sig = data["signal"]

        if isinstance(sig, list):
            sig = sig[0]

        if isinstance(sig, np.ndarray):
            sig = sig.flatten()

        windows = []

        if len(sig) >= self.window_size:
            for i in range(
                0,
                len(sig) - self.window_size + 1,
                self.step
            ):
                windows.append(sig[i:i + self.window_size])

        data["signal"] = windows
        return data


class Detrend(Transform):
    """
    Remove tendência linear do sinal antes do janelamento.
    """
    def transform(self, data):
        data = data.copy()
        sig = data["signal"]

        if isinstance(sig, np.ndarray):
            sig = sig.flatten()
            data["signal"] = detrend(sig, type="linear")

        elif isinstance(sig, list):
            data["signal"] = [
                detrend(s.flatten(), type="linear")
                if isinstance(s, np.ndarray)
                else s
                for s in sig
            ]

        return data


# ============================================================
# CONFIGURAÇÃO DOS EXPERIMENTOS
# ============================================================

# Dataset bruto usado pelo vibdata.
RAW_DATA_DIR = "/home/vfrocha/VibNet_Project/raw_data"

# Saída experimental exclusiva para Transfer Learning.
TL_OUTPUT_DIR = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "../../data/processed_tl"
    )
)

# Os datasets efetivamente usados no experimento atual.
SOURCE_DATASETS = ["CWRU", "MFPT"]

# Para o MFPT, todos os sinais são harmonizados para 48.828 Hz
# antes do janelamento, de modo que cada janela tenha 1 segundo.
# O baseline original é 97.656 Hz e é reduzido por fator 2.

# Para cada dataset, definimos o tamanho de janela.
WINDOW_SIZES = {
    "CWRU_12k": 12000,
    "CWRU_48k": 48000,
    "MFPT": 48828,
}


# ============================================================
# NOMES DE CONDIÇÃO / CLASSE
# ============================================================

def harmonize_mfpt_sample_rate(signal, meta):
    """
    Harmoniza o MFPT para 48.828 Hz.

    O MFPT possui:
        - baseline normal (Class_23): 97.656 Hz
        - falhas (Class_24/Class_25): 48.828 Hz

    Para manter 1 segundo = 48.828 amostras em todas as classes,
    os sinais de 97.656 Hz são reduzidos por fator 2 usando
    resample_poly(up=1, down=2), que inclui filtragem anti-aliasing.
    """
    try:
        sample_rate = float(meta.get("sample_rate"))
    except (TypeError, ValueError):
        return signal

    target_rate = 48828.0

    if np.isclose(sample_rate, target_rate):
        return signal

    if np.isclose(sample_rate, 2 * target_rate):
        signal = np.asarray(signal).reshape(-1)
        return resample_poly(signal, up=1, down=2)

    raise ValueError(
        f"Taxa de amostragem MFPT não suportada: "
        f"{sample_rate} Hz. Esperado 48828 ou 97656 Hz."
    )


def get_names(ds_name, meta):
    """
    Mantém a mesma convenção de diretórios do pipeline original.
    Retorna:
        condition_dir, class_dir
    """

    if ds_name == "CWRU_12k":
        load = meta.get("load", 0)
        try:
            load = int(load)
        except (TypeError, ValueError):
            load = 0

        cond = f"Load_{load}HP"

    elif ds_name == "CWRU_48k":
        load = meta.get("load", 0)
        try:
            load = int(load)
        except (TypeError, ValueError):
            load = 0

        sev = meta.get(
            "fault_size",
            meta.get("severity", "0.000")
        )

        if isinstance(sev, (float, int)):
            sev = f"{sev:.3f}"

        cond = f"Load_{load}HP_Sev_{sev}"

    elif ds_name == "MFPT":
        cond = f"Load_{meta.get('load', 'Unknown')}"

    else:
        val = meta.get(
            "load",
            meta.get("rotation_hz", "0")
        )
        cond = f"Cond_{str(val).replace('.', '')}"

    orig_label = meta.get("label")

    if isinstance(orig_label, pd.Series):
        orig_label = orig_label.item()

    label_name = f"Class_{orig_label}"

    return cond, label_name


# ============================================================
# AUXILIARES
# ============================================================

def extract_signal(item):
    """
    Extrai o sinal do item retornado pelo vibdata.
    """
    raw = item.get("signal")

    if (
        isinstance(raw, np.ndarray)
        and raw.dtype == "O"
        and raw.size > 0
    ):
        return raw[0]

    if isinstance(raw, np.ndarray):
        return raw

    return None


def build_pipeline(dataset_name, detrend_signal):
    """
    Monta o pipeline experimental.

    raw:
        SimpleSplit()

    detrended:
        Detrend() -> SimpleSplit()
    """
    window_size = WINDOW_SIZES[dataset_name]

    transforms = []

    if detrend_signal:
        transforms.append(Detrend())

    transforms.append(
        SimpleSplit(window_size=window_size)
    )

    return Sequential(transforms)


def save_windows(
    windows,
    output_dir,
    dataset_name,
    meta,
    sample_index
):
    """
    Salva as janelas mantendo a mesma estrutura:
        dataset / condição / classe / *.npy
    """
    if not isinstance(windows, list) or not windows:
        return 0

    cond, lbl = get_names(dataset_name, meta)

    final_dir = os.path.join(
        output_dir,
        dataset_name,
        cond,
        lbl
    )

    os.makedirs(final_dir, exist_ok=True)

    saved = 0

    for idx, window in enumerate(windows):
        if not isinstance(window, np.ndarray):
            continue

        filename = f"s{sample_index:05d}_w{idx:02d}.npy"
        file_path = os.path.join(final_dir, filename)

        np.save(file_path, window)
        saved += 1

    return saved


# ============================================================
# PROCESSAMENTO
# ============================================================

def process_dataset(ds_name, ds):
    """
    Gera as duas versões do dataset:
        processed_tl/raw/
        processed_tl/detrended/

    No MFPT, o baseline de 97.656 Hz é reduzido para 48.828 Hz
    antes do janelamento, mantendo 1 segundo por janela.
    """

    pipelines = {
        "raw": build_pipeline(
            ds_name,
            detrend_signal=False
        ),
        "detrended": build_pipeline(
            ds_name,
            detrend_signal=True
        ),
    }

    counts = {
        variant: 0
        for variant in pipelines
    }

    errors = 0

    for i in tqdm(
        range(len(ds)),
        desc=f"{ds_name}"
    ):
        try:
            item = ds[i]

            if not isinstance(item, dict):
                continue

            sig_array = extract_signal(item)

            if sig_array is None:
                continue

            meta = item["metainfo"]

            if isinstance(meta, pd.DataFrame):
                meta = meta.iloc[0]

            # MFPT: converte o baseline de 97.656 Hz para 48.828 Hz.
            if ds_name == "MFPT":
                sig_array = harmonize_mfpt_sample_rate(
                    sig_array,
                    meta,
                )

            # Gera as duas versões usando o MESMO sinal de origem.
            for variant, pipeline in pipelines.items():

                sample = {
                    "signal": sig_array,
                    "metainfo": pd.DataFrame([meta]),
                }

                processed = pipeline(sample)
                windows = processed["signal"]

                output_dir = os.path.join(
                    TL_OUTPUT_DIR,
                    variant
                )

                counts[variant] += save_windows(
                    windows=windows,
                    output_dir=output_dir,
                    dataset_name=ds_name,
                    meta=meta,
                    sample_index=i,
                )

        except Exception as exc:
            errors += 1
            print(
                f"\n[AVISO] Erro no item {i} "
                f"da base {ds_name}: {exc}"
            )

    print(
        f"--> {ds_name}: "
        f"raw={counts['raw']} janelas, "
        f"detrended={counts['detrended']} janelas, "
        f"erros={errors}"
    )

    return counts


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    os.makedirs(TL_OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("GERAÇÃO DOS DATASETS EXPERIMENTAIS DE TRANSFER LEARNING")
    print("=" * 60)
    print(f"Origem : {RAW_DATA_DIR}")
    print(f"Destino: {TL_OUTPUT_DIR}")
    print()
    print("Versões geradas:")
    print("  1. raw       = SimpleSplit")
    print("  2. detrended = Detrend + SimpleSplit")
    print()

    total_counts = {}

    for source_name in SOURCE_DATASETS:

        print(f"\n=== Carregando {source_name} ===")

        try:
            raw_cls = getattr(
                raw_datasets,
                f"{source_name}_raw"
            )

            ds = raw_cls(
                RAW_DATA_DIR,
                download=False
            )

        except Exception as exc:
            print(
                f"[ERRO] Não foi possível carregar "
                f"{source_name}: {exc}"
            )
            continue

        dataset_counts = {}

        # ----------------------------------------------------
        # CWRU:
        # o mesmo dataset é dividido em CWRU_12k ou CWRU_48k
        # de acordo com o sample_rate do registro.
        # ----------------------------------------------------
        if source_name == "CWRU":

            # Os registros são processados individualmente
            counts = {
                "CWRU_12k": {"raw": 0, "detrended": 0},
                "CWRU_48k": {"raw": 0, "detrended": 0},
            }

            for i in tqdm(
                range(len(ds)),
                desc="CWRU"
            ):
                try:
                    item = ds[i]

                    if not isinstance(item, dict):
                        continue

                    sig_array = extract_signal(item)

                    if sig_array is None:
                        continue

                    meta = item["metainfo"]

                    if isinstance(meta, pd.DataFrame):
                        meta = meta.iloc[0]

                    sr = meta.get("sample_rate", 12000)

                    try:
                        sr = float(sr)
                    except (TypeError, ValueError):
                        sr = 12000

                    if sr > 20000:
                        target_ds_name = "CWRU_48k"
                    else:
                        target_ds_name = "CWRU_12k"

                    for variant, detrend_signal in [
                        ("raw", False),
                        ("detrended", True),
                    ]:
                        pipeline = build_pipeline(
                            target_ds_name,
                            detrend_signal
                        )

                        sample = {
                            "signal": sig_array,
                            "metainfo": pd.DataFrame([meta]),
                        }

                        processed = pipeline(sample)

                        saved = save_windows(
                            windows=processed["signal"],
                            output_dir=os.path.join(
                                TL_OUTPUT_DIR,
                                variant
                            ),
                            dataset_name=target_ds_name,
                            meta=meta,
                            sample_index=i,
                        )

                        counts[target_ds_name][variant] += saved

                except Exception as exc:
                    print(
                        f"\n[AVISO] Erro no item {i} "
                        f"da base CWRU: {exc}"
                    )

            total_counts.update(counts)

            print(
                "--> CWRU_12k: "
                f"raw={counts['CWRU_12k']['raw']}, "
                f"detrended={counts['CWRU_12k']['detrended']}"
            )

            print(
                "--> CWRU_48k: "
                f"raw={counts['CWRU_48k']['raw']}, "
                f"detrended={counts['CWRU_48k']['detrended']}"
            )

        # ----------------------------------------------------
        # MFPT:
        # processa diretamente como MFPT.
        # ----------------------------------------------------
        elif source_name == "MFPT":

            counts = process_dataset(
                ds_name="MFPT",
                ds=ds
            )

            total_counts["MFPT"] = counts

    # ========================================================
    # RESUMO FINAL
    # ========================================================

    print("\n" + "=" * 60)
    print("RESUMO FINAL")
    print("=" * 60)

    for dataset_name, counts in total_counts.items():

        if isinstance(counts, dict) and "raw" in counts:
            print(
                f"{dataset_name}: "
                f"raw={counts['raw']}, "
                f"detrended={counts['detrended']}"
            )

        else:
            print(f"{dataset_name}: {counts}")

    print("\nConcluído.")
    print(
        f"Os datasets experimentais estão em:\n"
        f"  {TL_OUTPUT_DIR}"
    )
