#!/usr/bin/env python3
"""
normalize_dataset_tl.py

Geração das variantes normalizadas por RMS de referência para o
experimento de Transfer Learning.

Entrada:
    data/processed_tl/raw/
    data/processed_tl/detrended/

Saída:
    data/processed_tl/raw_rms/
    data/processed_tl/detrended_rms/

Regra:
    RMS_ref = média dos RMS das janelas da classe normal
    x_normalized = x / RMS_ref

Referências:
    CWRU_12k:
        classe normal = label 0.
        O script aceita tanto Class_0 quanto Class_Normal,
        para ser compatível com diferentes gerações dos dados.

    CWRU_48k:
        referência = CWRU_12k / classe normal.
        O mesmo escalar RMS_ref é aplicado ao alvo.

    MFPT:
        label 23 = Normal (Class_23), conforme vibdata/resources/labels.csv.
        label 24 = Inner Race.
        label 25 = Outer Race.

Importante:
    A referência é calculada na mesma variante que será normalizada:
        raw       -> referência calculada em raw
        detrended -> referência calculada em detrended

    O script não altera:
        data/processed/
        data/processed_tl/raw/
        data/processed_tl/detrended/
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

DATASETS = [
    "CWRU_12k",
    "MFPT",
    "CWRU_48k",
]

# O make_dataset_tl.py original que criamos usa Class_<label>.
# Portanto, label 0 do CWRU normalmente aparece como Class_0.
# Class_Normal é aceito para manter compatibilidade com outros
# datasets/processamentos.
CWRU_NORMAL_CLASS_ALIASES = (
    "Class_0",
    "Class_Normal",
)

MFPT_NORMAL_CLASS = "Class_23"


# ============================================================
# AUXILIARES
# ============================================================

def find_npy_files(root: Path):
    """Retorna todos os .npy de forma recursiva."""
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
        nome da pasta encontrada
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
    input_root: Path,
    output_root: Path,
    reference_rms: float,
):
    """Normaliza uma janela e preserva a estrutura relativa."""
    relative_path = input_file.relative_to(
        input_root
    )

    output_file = output_root / relative_path

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


def normalize_dataset(
    dataset_name: str,
    input_root: Path,
    output_root: Path,
    reference_rms: float,
):
    """
    Aplica um único RMS_ref ao dataset inteiro.
    """
    dataset_input = input_root / dataset_name

    files = find_npy_files(
        dataset_input
    )

    if not files:
        raise RuntimeError(
            f"Nenhum .npy encontrado em {dataset_input}"
        )

    errors = 0

    for input_file in tqdm(
        files,
        desc=f"Normalizando {dataset_name}",
    ):
        try:
            normalize_file(
                input_file=input_file,
                input_root=input_root,
                output_root=output_root,
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
):
    """Registra a referência e os resultados do processamento."""
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
        writer.writerows(rows)

    return manifest_file


# ============================================================
# REFERÊNCIAS
# ============================================================

def build_reference_map():
    """
    Define a referência oficial para cada dataset.

    CWRU:
        label 0 -> normal.

    MFPT:
        label 23 -> normal.

    CWRU_48k:
        usa CWRU_12k como domínio de referência.
    """
    return {
        "CWRU_12k": {
            "reference_dataset": "CWRU_12k",
            "reference_class_aliases": CWRU_NORMAL_CLASS_ALIASES,
        },
        "MFPT": {
            "reference_dataset": "MFPT",
            "reference_class_aliases": (
                MFPT_NORMAL_CLASS,
            ),
        },
        "CWRU_48k": {
            "reference_dataset": "CWRU_12k",
            "reference_class_aliases": CWRU_NORMAL_CLASS_ALIASES,
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
):
    """Processa uma variante raw ou detrended."""
    variant = VARIANTS[variant_name]

    input_root = variant["input_dir"]
    output_root = variant["output_dir"]

    dataset_input = input_root / dataset_name

    if not dataset_input.exists():
        raise FileNotFoundError(
            f"Dataset de entrada não encontrado: "
            f"{dataset_input}"
        )

    info = references[dataset_name]

    reference_dataset = (
        info["reference_dataset"]
    )

    reference_root = (
        input_root /
        reference_dataset
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
        f"Referência: {reference_dataset}"
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
        input_root=input_root,
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
        default=DATASETS,
        choices=DATASETS,
        help="Datasets que serão processados.",
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

    args = parser.parse_args()

    # Cria os diretórios de saída logo no início.
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
        f"Projeto : {PROJECT_ROOT}"
    )
    print(
        f"Entrada : {INPUT_ROOT}"
    )
    print(
        "CWRU normal: label 0 "
        "(Class_0 ou Class_Normal)"
    )
    print(
        "MFPT normal: label 23 (Class_23)"
    )
    print(
        "CWRU_48k: referência cruzada = CWRU_12k"
    )

    references = build_reference_map()

    manifest_rows = []

    for variant_name in args.variants:
        for dataset_name in args.datasets:

            try:
                process_variant(
                    variant_name=variant_name,
                    dataset_name=dataset_name,
                    references=references,
                    manifest_rows=manifest_rows,
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

    # Um manifesto em cada pasta de saída.
    manifest_raw = write_manifest(
        [
            row for row in manifest_rows
            if row["output_variant"] == "raw_rms"
        ],
        VARIANTS["raw"]["output_dir"],
    )

    manifest_detrended = write_manifest(
        [
            row for row in manifest_rows
            if row["output_variant"] == "detrended_rms"
        ],
        VARIANTS["detrended"]["output_dir"],
    )

    print()
    print("=" * 72)
    print("CONCLUÍDO")
    print("=" * 72)
    print(
        f"Manifest RAW       : {manifest_raw}"
    )
    print(
        f"Manifest DETRENDED : {manifest_detrended}"
    )

    print()
    if manifest_rows:
        print("Resumo:")
        for row in manifest_rows:
            print(
                f"{row['dataset']:12s} | "
                f"{row['output_variant']:20s} | "
                f"ref={row['reference_dataset']:12s} | "
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
