#!/usr/bin/env python3
"""
normalize_dataset_tl.py

Geração das variantes normalizadas por RMS de referência para os
experimentos de Transfer Learning.

Entrada primária:
    data/processed_tl/raw/
    data/processed_tl/detrended/

Fallback opcional (somente-leitura para a variante detrended, gerada por
src/data/make_dataset.py com Detrend + SimpleSplit):
    data/processed/

Saída:
    data/processed_tl/raw_rms/
    data/processed_tl/detrended_rms/

Regra:
    RMS_ref = média dos RMS das janelas da classe normal
    x_normalized = x / RMS_ref

Referências por dataset:
    CWRU_12k:
        classe normal = Class_0 ou Class_Normal.
    CWRU_48k:
        referência cruzada = CWRU_12k / classe normal (pois CWRU_48k não
        possui classe normal própria).
    MFPT:
        classe normal = Class_23 ou Class_Normal.
    PU:
        classe normal = Class_Normal ou Class_26.
    HUST_Gearbox:
        classe normal = Class_Normal ou Class_Healthy.
    UOEMD:
        classe normal = Class_Normal ou Class_H (Healthy).
    UORED:
        classe normal = Class_Normal ou Class_Healthy.
    HUST:
        classe normal = Class_Normal ou Class_40.
    Mechanical_Gear:
        classe normal = Class_Normal ou Class_Healthy.
    Electric_Motor:
        classe normal = Class_Normal ou Class_Healthy.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm


# ============================================================
# CONFIGURAÇÃO
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_ROOT = PROJECT_ROOT / "data" / "processed_tl"
PROCESSED_FALLBACK_ROOT = PROJECT_ROOT / "data" / "processed"

VARIANTS = {
    "raw": {
        "input_dir": INPUT_ROOT / "raw",
        "output_dir": INPUT_ROOT / "raw_rms",
    },
    "detrended": {
        "input_dir": INPUT_ROOT / "detrended",
        "output_dir": INPUT_ROOT / "detrended_rms",
    },
}

DEFAULT_DATASETS = [
    "CWRU_12k",
    "MFPT",
    "CWRU_48k",
]

CROSS_DATASET_TL_DATASETS = [
    "CWRU_12k",
    "CWRU_48k",
    "PU",
    "HUST_Gearbox",
    "UOEMD",
    "UORED",
    "HUST",
    "Mechanical_Gear",
    "Electric_Motor",
]

ALL_SUPPORTED_DATASETS = [
    "CWRU_12k",
    "MFPT",
    "CWRU_48k",
    "PU",
    "HUST_Gearbox",
    "UOEMD",
    "UORED",
    "HUST",
    "Mechanical_Gear",
    "Electric_Motor",
]

CWRU_NORMAL_CLASS_ALIASES = (
    "Class_0",
    "Class_Normal",
    "Normal",
)

MFPT_NORMAL_CLASS_ALIASES = (
    "Class_23",
    "Class_Normal",
    "Normal",
)

COMMON_NORMAL_ALIASES = (
    "Class_Normal",
    "Class_Healthy",
    "Normal",
    "Healthy",
    "Class_0",
)


# ============================================================
# AUXILIARES
# ============================================================

def find_npy_files(root: Path):
    """Retorna todos os .npy de forma recursiva."""
    if not root.exists():
        return []
    return sorted(root.rglob("*.npy"))


def get_class_from_path(file_path: Path):
    """
    Estrutura esperada:

        dataset/
            condition/
                Class_xx/
                    janela.npy
    """
    return file_path.parent.name


def discover_normal_class(dataset_root: Path, aliases):
    """
    Procura a classe normal entre os aliases informados.

    Retorna:
        (nome_da_pasta_encontrada, lista_de_classes_disponiveis)
    """
    files = find_npy_files(dataset_root)

    available_classes = sorted(
        {
            get_class_from_path(path)
            for path in files
        }
    )

    for alias in aliases:
        if alias in available_classes:
            return alias, available_classes

    # Busca case-insensitive por substring "normal" ou "healthy"
    for cls_name in available_classes:
        lower = cls_name.lower()
        if "normal" in lower or "healthy" in lower:
            return cls_name, available_classes

    return None, available_classes


def calculate_rms(signal: np.ndarray) -> float:
    """
    RMS da janela:

        RMS(x) = sqrt(mean(x²))
    """
    signal = np.asarray(
        signal,
        dtype=np.float64,
    ).reshape(-1)

    if signal.size == 0:
        raise ValueError("Sinal vazio.")

    rms = float(
        np.sqrt(
            np.mean(
                np.square(signal)
            )
        )
    )

    if not np.isfinite(rms):
        raise ValueError("RMS não finito.")

    if rms <= 0:
        raise ValueError("RMS zero.")

    return rms


def calculate_reference_rms(
    dataset_root: Path,
    reference_classes,
    variant_name: str,
):
    """
    Calcula:

        RMS_ref = mean(RMS(x_i))

    usando somente as classes indicadas.
    """
    all_files = find_npy_files(dataset_root)

    reference_files = [
        path
        for path in all_files
        if get_class_from_path(path)
        in reference_classes
    ]

    if not reference_files:
        available_classes = sorted(
            {
                get_class_from_path(path)
                for path in all_files
            }
        )

        raise RuntimeError(
            "Nenhuma janela encontrada para a referência.\n"
            f"Dataset: {dataset_root.name}\n"
            f"Variante: {variant_name}\n"
            f"Classes solicitadas: {reference_classes}\n"
            f"Classes disponíveis: {available_classes}"
        )

    rms_values = []

    for file_path in tqdm(
        reference_files,
        desc=(
            f"RMS ref. {dataset_root.name} "
            f"{variant_name}"
        ),
    ):
        signal = np.load(
            file_path,
            allow_pickle=False,
        )

        rms_values.append(
            calculate_rms(signal)
        )

    reference_rms = float(
        np.mean(rms_values)
    )

    if not np.isfinite(reference_rms) or reference_rms <= 0:
        raise ValueError(
            f"RMS_ref inválido: {reference_rms}"
        )

    return reference_rms, len(reference_files)


def normalize_file(
    input_file: Path,
    dataset_input_root: Path,
    dataset_output_root: Path,
    reference_rms: float,
):
    """Normaliza uma janela e preserva a estrutura relativa dentro do dataset."""
    relative_path = input_file.relative_to(
        dataset_input_root
    )

    output_file = dataset_output_root / relative_path

    output_file.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    signal = np.load(
        input_file,
        allow_pickle=False,
    )

    if signal.size == 0:
        raise ValueError(
            f"Sinal vazio: {input_file}"
        )

    normalized = signal / reference_rms

    np.save(
        output_file,
        normalized,
    )


def resolve_dataset_input_dir(
    dataset_name: str,
    variant_name: str,
    primary_input_root: Path,
    fallback_processed_root: Path | None = None,
) -> Path:
    """Localiza o diretório de entrada do dataset sem modificar a origem."""
    primary = primary_input_root / dataset_name
    if primary.exists() and find_npy_files(primary):
        return primary

    if variant_name == "detrended" and fallback_processed_root is not None:
        fallback = fallback_processed_root / dataset_name
        if fallback.exists() and find_npy_files(fallback):
            return fallback

    return primary


def normalize_dataset(
    dataset_name: str,
    dataset_input: Path,
    output_root: Path,
    reference_rms: float,
):
    """
    Aplica um único RMS_ref ao dataset inteiro.
    """
    files = find_npy_files(
        dataset_input
    )

    if not files:
        raise RuntimeError(
            f"Nenhum .npy encontrado em {dataset_input}"
        )

    dataset_output = output_root / dataset_name
    errors = 0

    for input_file in tqdm(
        files,
        desc=f"Normalizando {dataset_name}",
    ):
        try:
            normalize_file(
                input_file=input_file,
                dataset_input_root=dataset_input,
                dataset_output_root=dataset_output,
                reference_rms=reference_rms,
            )
        except Exception as exc:
            errors += 1
            print(
                f"\n[AVISO] Erro em {input_file}: {exc}"
            )

    return len(files), errors


def write_manifest(
    rows,
    output_root: Path,
    merge_existing: bool = True,
):
    """Registra a referência e os resultados do processamento, preservando entradas existentes."""
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_file = (
        output_root /
        "normalization_manifest.csv"
    )

    fieldnames = [
        "dataset",
        "output_variant",
        "reference_dataset",
        "reference_variant",
        "reference_classes",
        "reference_rms",
        "reference_windows",
        "input_windows",
        "errors",
        "formula",
    ]

    merged = {}
    if merge_existing and manifest_file.exists():
        with manifest_file.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for existing_row in reader:
                key = (existing_row.get("dataset"), existing_row.get("output_variant"))
                merged[key] = {k: existing_row.get(k, "") for k in fieldnames}

    for row in rows:
        key = (row["dataset"], row["output_variant"])
        merged[key] = {k: row.get(k, "") for k in fieldnames}

    final_rows = list(merged.values())

    with manifest_file.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(final_rows)

    return manifest_file


# ============================================================
# REFERÊNCIAS
# ============================================================

def build_reference_map():
    """
    Define a referência oficial para cada dataset.

    - CWRU_12k: label 0 / Normal.
    - CWRU_48k: usa CWRU_12k como domínio de referência (pois CWRU_48k não tem classe normal).
    - MFPT: Class_23 / Class_Normal.
    - PU: Class_Normal / Class_26.
    - HUST_Gearbox: Class_Normal / Class_Healthy.
    - UOEMD: Class_Normal / Class_H.
    - UORED: Class_Normal / Class_Healthy.
    - HUST: Class_Normal / Class_40.
    - Mechanical_Gear: Class_Healthy / Class_Normal.
    - Electric_Motor: Class_Healthy / Class_Normal.
    """
    return {
        "CWRU_12k": {
            "reference_dataset": "CWRU_12k",
            "reference_class_aliases": CWRU_NORMAL_CLASS_ALIASES,
        },
        "MFPT": {
            "reference_dataset": "MFPT",
            "reference_class_aliases": MFPT_NORMAL_CLASS_ALIASES,
        },
        "CWRU_48k": {
            "reference_dataset": "CWRU_12k",
            "reference_class_aliases": CWRU_NORMAL_CLASS_ALIASES,
        },
        "PU": {
            "reference_dataset": "PU",
            "reference_class_aliases": ("Class_Normal", "Class_26", "Normal", "Healthy"),
        },
        "HUST_Gearbox": {
            "reference_dataset": "HUST_Gearbox",
            "reference_class_aliases": ("Class_Normal", "Class_Healthy", "Normal", "Healthy"),
        },
        "UOEMD": {
            "reference_dataset": "UOEMD",
            "reference_class_aliases": ("Class_Normal", "H_1", "Class_H", "Class_Healthy", "Normal", "Healthy"),
        },
        "UORED": {
            "reference_dataset": "UORED",
            "reference_class_aliases": ("Class_Normal", "H", "Class_Healthy", "Normal", "Healthy"),
        },
        "HUST": {
            "reference_dataset": "HUST",
            "reference_class_aliases": ("Class_Normal", "N", "Class_40", "Normal", "Healthy"),
        },
        "Mechanical_Gear": {
            "reference_dataset": "Mechanical_Gear",
            "reference_class_aliases": COMMON_NORMAL_ALIASES,
        },
        "Electric_Motor": {
            "reference_dataset": "Electric_Motor",
            "reference_class_aliases": COMMON_NORMAL_ALIASES,
        },
    }


# ============================================================
# PROCESSAMENTO
# ============================================================

def process_variant(
    variant_name: str,
    dataset_name: str,
    references: dict,
    manifest_rows: list,
    fallback_processed_root: Path | None = None,
):
    """Processa uma variante raw ou detrended."""
    variant = VARIANTS[variant_name]

    input_root = variant["input_dir"]
    output_root = variant["output_dir"]

    dataset_input = resolve_dataset_input_dir(
        dataset_name=dataset_name,
        variant_name=variant_name,
        primary_input_root=input_root,
        fallback_processed_root=fallback_processed_root,
    )

    if not dataset_input.exists():
        raise FileNotFoundError(
            f"Dataset de entrada não encontrado: "
            f"{dataset_input}"
        )

    info = references[dataset_name]

    reference_dataset = (
        info["reference_dataset"]
    )

    reference_root = resolve_dataset_input_dir(
        dataset_name=reference_dataset,
        variant_name=variant_name,
        primary_input_root=input_root,
        fallback_processed_root=fallback_processed_root,
    )

    if not reference_root.exists():
        raise FileNotFoundError(
            "Dataset de referência não encontrado:\n"
            f"{reference_root}"
        )

    # --------------------------------------------------------
    # Descobre a classe normal real presente no dataset.
    # --------------------------------------------------------
    normal_class, available_classes = (
        discover_normal_class(
            dataset_root=reference_root,
            aliases=info["reference_class_aliases"],
        )
    )

    if normal_class is None:
        raise RuntimeError(
            "Classe normal não encontrada.\n"
            f"Dataset de referência: {reference_dataset}\n"
            f"Variante: {variant_name}\n"
            f"Aliases procurados: "
            f"{list(info['reference_class_aliases'])}\n"
            f"Classes disponíveis: {available_classes}"
        )

    print()
    print("-" * 72)
    print(
        f"Dataset: {dataset_name} | "
        f"Variante: {variant_name}"
    )
    print(
        f"Entrada: {dataset_input}"
    )
    print(
        f"Referência: {reference_dataset} ({reference_root})"
    )
    print(
        f"Classe normal usada: {normal_class}"
    )

    reference_rms, reference_windows = (
        calculate_reference_rms(
            dataset_root=reference_root,
            reference_classes=[normal_class],
            variant_name=variant_name,
        )
    )

    print(
        f"RMS_ref = {reference_rms:.12g}"
    )
    print(
        f"Janelas usadas na referência = "
        f"{reference_windows}"
    )

    input_windows, errors = normalize_dataset(
        dataset_name=dataset_name,
        dataset_input=dataset_input,
        output_root=output_root,
        reference_rms=reference_rms,
    )

    output_variant = (
        f"{variant_name}_rms"
    )

    manifest_rows.append(
        {
            "dataset": dataset_name,
            "output_variant": output_variant,
            "reference_dataset": reference_dataset,
            "reference_variant": variant_name,
            "reference_classes": normal_class,
            "reference_rms": reference_rms,
            "reference_windows": reference_windows,
            "input_windows": input_windows,
            "errors": errors,
            "formula": "x_normalized = x / RMS_ref",
        }
    )

    print(
        f"--> {output_variant}: "
        f"{input_windows} janelas, "
        f"{errors} erros"
    )


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Gera datasets normalizados pelo RMS médio "
            "da classe normal de referência."
        )
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DEFAULT_DATASETS,
        choices=ALL_SUPPORTED_DATASETS + ["all", "cross_dataset"],
        help=(
            "Datasets que serão processados. Use 'cross_dataset' para os 9 "
            "datasets (5 targets + 4 sources) ou 'all' para todos os 10."
        ),
    )

    parser.add_argument(
        "--variants",
        nargs="+",
        default=[
            "raw",
            "detrended",
        ],
        choices=[
            "raw",
            "detrended",
        ],
        help="Variantes que serão normalizadas.",
    )

    parser.add_argument(
        "--use-processed-fallback",
        action="store_true",
        help=(
            "Para a variante 'detrended', caso o dataset não esteja em "
            "data/processed_tl/detrended/<ds>, lê de data/processed/<ds> "
            "(que já contém janelas após Detrend + SimpleSplit)."
        ),
    )

    args = parser.parse_args()

    selected_datasets = []
    for ds in args.datasets:
        if ds == "all":
            selected_datasets.extend(ALL_SUPPORTED_DATASETS)
        elif ds == "cross_dataset":
            selected_datasets.extend(CROSS_DATASET_TL_DATASETS)
        else:
            selected_datasets.append(ds)
    # Remove duplicatas preservando a ordem
    selected_datasets = list(dict.fromkeys(selected_datasets))

    fallback_root = PROCESSED_FALLBACK_ROOT if args.use_processed_fallback else None

    for variant_name in args.variants:
        VARIANTS[variant_name]["output_dir"].mkdir(
            parents=True,
            exist_ok=True,
        )

    print("=" * 72)
    print(
        "NORMALIZAÇÃO EXPERIMENTAL "
        "POR RMS DE REFERÊNCIA"
    )
    print("=" * 72)
    print(
        f"Projeto  : {PROJECT_ROOT}"
    )
    print(
        f"Entrada  : {INPUT_ROOT}"
    )
    if fallback_root is not None:
        print(
            f"Fallback : {fallback_root} (somente-leitura para detrended)"
        )
    print(
        f"Datasets : {', '.join(selected_datasets)}"
    )

    references = build_reference_map()

    manifest_rows = []

    for variant_name in args.variants:
        for dataset_name in selected_datasets:

            try:
                process_variant(
                    variant_name=variant_name,
                    dataset_name=dataset_name,
                    references=references,
                    manifest_rows=manifest_rows,
                    fallback_processed_root=fallback_root,
                )

            except Exception as exc:
                print()
                print(
                    f"[ERRO] {dataset_name} / "
                    f"{variant_name}:"
                )
                print(exc)
                print(
                    "O processamento desse dataset/variante "
                    "será ignorado e os demais continuarão."
                )

    raw_rows = [
        row for row in manifest_rows
        if row["output_variant"] == "raw_rms"
    ]
    detrended_rows = [
        row for row in manifest_rows
        if row["output_variant"] == "detrended_rms"
    ]

    if raw_rows or "raw" in args.variants:
        manifest_raw = write_manifest(
            raw_rows,
            VARIANTS["raw"]["output_dir"],
        )
        print(f"Manifest RAW       : {manifest_raw}")

    if detrended_rows or "detrended" in args.variants:
        manifest_detrended = write_manifest(
            detrended_rows,
            VARIANTS["detrended"]["output_dir"],
        )
        print(f"Manifest DETRENDED : {manifest_detrended}")

    print()
    print("=" * 72)
    print("CONCLUÍDO")
    print("=" * 72)

    if manifest_rows:
        print("Resumo:")
        for row in manifest_rows:
            print(
                f"{row['dataset']:16s} | "
                f"{row['output_variant']:16s} | "
                f"ref={row['reference_dataset']:16s} | "
                f"classe={row['reference_classes']:14s} | "
                f"RMS_ref={float(row['reference_rms']):.8g} | "
                f"janelas={row['input_windows']} | "
                f"erros={row['errors']}"
            )
    else:
        print(
            "Nenhum dataset foi normalizado."
        )


if __name__ == "__main__":
    main()
