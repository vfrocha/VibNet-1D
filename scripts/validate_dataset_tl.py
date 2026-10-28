#!/usr/bin/env python3
"""
validate_dataset_tl.py

Validação dos datasets experimentais de Transfer Learning.

Compara:
    raw
    raw_rms
    detrended
    detrended_rms

Valida:
    1. quantidade de arquivos;
    2. estrutura relativa dos arquivos;
    3. shape/dtype;
    4. estatísticas de RMS;
    5. relação matemática:
           RMS(x_normalized) = RMS(x) / RMS_ref
    6. RMS médio das classes usadas como referência.

Saídas:
    data/processed_tl/validation/
        dataset_validation.csv
        file_comparison.csv
        reference_validation.csv
        rms_statistics.csv

Uso:
    python scripts/validate_dataset_tl.py

Opcional:
    python scripts/validate_dataset_tl.py --datasets CWRU_12k MFPT CWRU_48k
    python scripts/validate_dataset_tl.py --datasets cross_dataset --variants detrended --use-processed-fallback
    python scripts/validate_dataset_tl.py --sample-size 100
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from tqdm import tqdm


# ============================================================
# CONFIGURAÇÃO
# ============================================================

def _resolve_project_root() -> Path:
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / "src").is_dir():
            return parent
    return here.parents[1]


PROJECT_ROOT = _resolve_project_root()
ROOT = PROJECT_ROOT / "data" / "processed_tl"
PROCESSED_FALLBACK_ROOT = PROJECT_ROOT / "data" / "processed"
VALIDATION_ROOT = ROOT / "validation"

VARIANTS = {
    "raw": ROOT / "raw",
    "raw_rms": ROOT / "raw_rms",
    "detrended": ROOT / "detrended",
    "detrended_rms": ROOT / "detrended_rms",
}

DATASETS = [
    "CWRU_12k",
    "MFPT",
    "CWRU_48k",
]

CROSS_DATASET_TARGETS = [
    "CWRU_12k",
    "CWRU_48k",
    "PU",
    "HUST_Gearbox",
    "UOEMD",
]

CROSS_DATASET_SOURCES = [
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

# Referências oficiais utilizadas pelo normalize_dataset_tl.py
REFERENCE_CLASSES = {
    "CWRU_12k": "Class_0",
    "MFPT": "Class_23",
    "PU": "Normal",
    "HUST_Gearbox": "Normal",
    "UOEMD": "H_1",
    "UORED": "H",
    "HUST": "N",
    "Mechanical_Gear": "Normal",
    "Electric_Motor": "Normal",
}

CWRU_REFERENCE_DATASET = "CWRU_12k"

# Tolerâncias numéricas.
# A validação matemática trabalha com números float64.
REL_TOL = 1e-10
ABS_TOL = 1e-12


# ============================================================
# FUNÇÕES AUXILIARES
# ============================================================

def resolve_variant_dataset_dir(
    variant: str,
    dataset: str,
    use_processed_fallback: bool = False,
) -> Path:
    """
    Resolve o diretório de um dataset em uma variante.
    Quando use_processed_fallback=True e variant == 'detrended',
    permite usar data/processed/<dataset> caso data/processed_tl/detrended/<dataset>
    não exista.
    """
    primary = VARIANTS[variant] / dataset
    if primary.exists():
        return primary
    if use_processed_fallback and variant == "detrended":
        fallback = PROCESSED_FALLBACK_ROOT / dataset
        if fallback.exists():
            return fallback
    return primary


def find_npy_files(root: Path):
    """Lista recursivamente os .npy."""
    if not root.exists():
        return []

    return sorted(root.rglob("*.npy"))


def relative_files(root: Path):
    """Retorna o conjunto de caminhos relativos dos .npy."""
    return {
        p.relative_to(root)
        for p in find_npy_files(root)
    }


def calculate_rms(signal):
    """RMS = sqrt(mean(x²))."""
    signal = np.asarray(
        signal,
        dtype=np.float64,
    ).reshape(-1)

    return float(
        np.sqrt(
            np.mean(
                np.square(signal)
            )
        )
    )


def path_class(file_path: Path):
    """Classe = diretório pai do arquivo."""
    return file_path.parent.name


def path_condition(file_path: Path):
    """Condição = diretório imediatamente acima da classe."""
    return file_path.parent.parent.name


def load_manifest_rms(variant: str):
    """
    Recupera RMS_ref do normalization_manifest.csv.
    """
    if variant not in ("raw", "detrended"):
        raise ValueError(
            f"Variante inválida para manifest: {variant}"
        )

    output_dir = (
        VARIANTS[variant + "_rms"]
    )
    manifest = (
        output_dir /
        "normalization_manifest.csv"
    )

    if not manifest.exists():
        raise FileNotFoundError(
            f"Manifesto não encontrado: {manifest}"
        )

    refs = {}

    with manifest.open(
        "r",
        encoding="utf-8",
    ) as f:
        reader = csv.DictReader(f)

        for row in reader:
            dataset = row["dataset"]
            output_variant = row["output_variant"]

            if output_variant == f"{variant}_rms":
                refs[dataset] = {
                    "reference_dataset": row[
                        "reference_dataset"
                    ],
                    "reference_variant": row[
                        "reference_variant"
                    ],
                    "reference_class": row[
                        "reference_classes"
                    ],
                    "reference_rms": float(
                        row["reference_rms"]
                    ),
                    "reference_windows": int(
                        row["reference_windows"]
                    ),
                }

    return refs


def grouped_summary(
    dataset_root: Path,
    dataset_name: str = None,
):
    """
    Calcula estatísticas por:
        dataset
        condition
        class
    """
    files = find_npy_files(dataset_root)

    groups = {}
    resolved_dataset_name = dataset_name or dataset_root.name

    for file_path in tqdm(
        files,
        desc=f"Estatísticas {resolved_dataset_name}",
        leave=False,
    ):
        rel = file_path.relative_to(dataset_root)

        # rel esperado dentro de dataset_root:
        # condition/class/file.npy (3 partes)
        # ou, se chamado na raiz da variante:
        # dataset/condition/class/file.npy (4 partes)
        parts = rel.parts

        if len(parts) == 3:
            dataset = resolved_dataset_name
            condition = parts[0]
            class_name = parts[1]
        elif len(parts) >= 4:
            dataset = parts[0]
            condition = parts[1]
            class_name = parts[2]
        else:
            continue

        signal = np.load(
            file_path,
            allow_pickle=False,
        )

        rms = calculate_rms(signal)

        key = (
            dataset,
            condition,
            class_name,
        )

        groups.setdefault(
            key,
            [],
        ).append(rms)

    rows = []

    for (dataset, condition, class_name), values in sorted(
        groups.items()
    ):
        values = np.asarray(
            values,
            dtype=np.float64,
        )

        rows.append(
            {
                "dataset": dataset,
                "condition": condition,
                "class": class_name,
                "n": int(values.size),
                "mean_rms": float(np.mean(values)),
                "std_rms": float(np.std(values)),
                "min_rms": float(np.min(values)),
                "max_rms": float(np.max(values)),
            }
        )

    return rows


def compare_pair(
    dataset,
    original_variant,
    normalized_variant,
    sample_size=None,
    use_processed_fallback=False,
):
    """
    Compara arquivos equivalentes entre:
        original_variant
        normalized_variant

    Verifica:
        RMS_norm ≈ RMS_original / RMS_ref
    """
    original_root = resolve_variant_dataset_dir(
        original_variant,
        dataset,
        use_processed_fallback=use_processed_fallback,
    )

    normalized_root = resolve_variant_dataset_dir(
        normalized_variant,
        dataset,
        use_processed_fallback=False,
    )

    original_files = find_npy_files(
        original_root
    )
    normalized_files = find_npy_files(
        normalized_root
    )

    original_map = {
        p.relative_to(original_root): p
        for p in original_files
    }

    normalized_map = {
        p.relative_to(normalized_root): p
        for p in normalized_files
    }

    common = sorted(
        set(original_map) &
        set(normalized_map)
    )

    missing_normalized = sorted(
        set(original_map) -
        set(normalized_map)
    )

    extra_normalized = sorted(
        set(normalized_map) -
        set(original_map)
    )

    refs = load_manifest_rms(
        original_variant
    )

    if dataset not in refs:
        raise RuntimeError(
            f"RMS_ref não encontrado no manifest para {dataset}"
        )

    reference_rms = refs[dataset][
        "reference_rms"
    ]

    if sample_size is not None:
        common_for_test = common[:sample_size]
    else:
        common_for_test = common

    errors = []
    shape_mismatches = 0
    dtype_mismatches = 0

    comparisons = []

    for rel in tqdm(
        common_for_test,
        desc=(
            f"Validando {dataset}: "
            f"{original_variant} → "
            f"{normalized_variant}"
        ),
    ):
        original_file = original_map[rel]
        normalized_file = normalized_map[rel]

        original_signal = np.load(
            original_file,
            allow_pickle=False,
        )

        normalized_signal = np.load(
            normalized_file,
            allow_pickle=False,
        )

        if original_signal.shape != normalized_signal.shape:
            shape_mismatches += 1

        if original_signal.dtype != normalized_signal.dtype:
            dtype_mismatches += 1

        rms_original = calculate_rms(
            original_signal
        )

        rms_normalized = calculate_rms(
            normalized_signal
        )

        expected = (
            rms_original /
            reference_rms
        )

        abs_error = abs(
            rms_normalized -
            expected
        )

        rel_error = (
            abs_error /
            max(abs(expected), ABS_TOL)
        )

        passed = (
            abs_error <= (
                ABS_TOL +
                REL_TOL *
                abs(expected)
            )
        )

        if not passed:
            errors.append(
                rel
            )

        comparisons.append(
            {
                "dataset": dataset,
                "original_variant": original_variant,
                "normalized_variant": normalized_variant,
                "file": str(rel),
                "class": path_class(rel),
                "condition": path_condition(rel),
                "rms_original": rms_original,
                "rms_normalized": rms_normalized,
                "rms_expected": expected,
                "absolute_error": abs_error,
                "relative_error": rel_error,
                "passed": passed,
            }
        )

    return {
        "dataset": dataset,
        "original_variant": original_variant,
        "normalized_variant": normalized_variant,
        "reference_dataset": refs[dataset][
            "reference_dataset"
        ],
        "reference_variant": refs[dataset][
            "reference_variant"
        ],
        "reference_class": refs[dataset][
            "reference_class"
        ],
        "reference_rms": reference_rms,
        "reference_windows": refs[dataset][
            "reference_windows"
        ],
        "n_original": len(original_files),
        "n_normalized": len(normalized_files),
        "n_common": len(common),
        "n_checked": len(common_for_test),
        "missing_normalized": len(
            missing_normalized
        ),
        "extra_normalized": len(
            extra_normalized
        ),
        "shape_mismatches": shape_mismatches,
        "dtype_mismatches": dtype_mismatches,
        "mathematical_failures": len(errors),
        "math_pass": (
            len(errors) == 0
            and len(common_for_test) > 0
        ),
    }, comparisons


def validate_reference_class(
    dataset,
    variant,
    use_processed_fallback=False,
):
    """
    Verifica o RMS médio da classe de referência.
    """
    refs = load_manifest_rms(
        variant
    )

    if dataset in refs:
        reference_dataset = refs[dataset]["reference_dataset"]
        reference_class = refs[dataset]["reference_class"]
        reference_rms = refs[dataset]["reference_rms"]
    else:
        if dataset == "CWRU_48k":
            reference_dataset = CWRU_REFERENCE_DATASET
        else:
            reference_dataset = dataset
        reference_class = REFERENCE_CLASSES.get(reference_dataset)
        reference_rms = np.nan

    if reference_class is None:
        return None

    root = resolve_variant_dataset_dir(
        variant,
        reference_dataset,
        use_processed_fallback=use_processed_fallback,
    )

    files = [
        f
        for f in find_npy_files(root)
        if path_class(f) == reference_class
    ]

    if not files:
        return {
            "dataset": dataset,
            "variant": variant,
            "reference_dataset": reference_dataset,
            "reference_class": reference_class,
            "n": 0,
            "mean_rms": np.nan,
            "difference_from_manifest": np.nan,
            "pass": False,
        }

    values = []

    for file_path in files:
        signal = np.load(
            file_path,
            allow_pickle=False,
        )
        values.append(
            calculate_rms(signal)
        )

    mean_rms = float(
        np.mean(values)
    )

    difference = (
        mean_rms -
        reference_rms
    )

    passed = bool(
        np.isclose(
            mean_rms,
            reference_rms,
            rtol=REL_TOL,
            atol=ABS_TOL,
        )
    )

    return {
        "dataset": dataset,
        "variant": variant,
        "reference_dataset": reference_dataset,
        "reference_class": reference_class,
        "n": len(files),
        "mean_rms": mean_rms,
        "difference_from_manifest": difference,
        "pass": passed,
    }


def write_csv(
    path: Path,
    rows,
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    if not rows:
        return

    fieldnames = list(
        rows[0].keys()
    )

    with path.open(
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


def expand_dataset_selection(selected):
    """Expande atalhos 'cross_dataset' e 'all' preservando ordem sem duplicatas."""
    expanded = []
    for item in selected:
        if item == "cross_dataset":
            for ds in CROSS_DATASET_TARGETS + CROSS_DATASET_SOURCES:
                if ds not in expanded:
                    expanded.append(ds)
        elif item == "all":
            for ds in ALL_SUPPORTED_DATASETS:
                if ds not in expanded:
                    expanded.append(ds)
        else:
            if item not in expanded:
                expanded.append(item)
    return expanded


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Valida os datasets experimentais TL "
            "e a normalização por RMS."
        )
    )

    parser.add_argument(
        "--datasets",
        nargs="+",
        default=DATASETS,
        choices=ALL_SUPPORTED_DATASETS + ["cross_dataset", "all"],
        help=(
            "Datasets a validar. Aceita nomes individuais, "
            "'cross_dataset' (5 targets + 4 sources) ou 'all'."
        ),
    )

    parser.add_argument(
        "--variants",
        nargs="+",
        default=["raw", "detrended"],
        choices=["raw", "detrended"],
        help="Variantes base a validar contra suas versões _rms (default: raw detrended).",
    )

    parser.add_argument(
        "--use-processed-fallback",
        action="store_true",
        help=(
            "Para a variante 'detrended', aceita data/processed/<dataset> "
            "como origem quando data/processed_tl/detrended/<dataset> não existir."
        ),
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help=(
            "Número máximo de arquivos por comparação "
            "matemática. Sem essa opção, verifica todos."
        ),
    )

    args = parser.parse_args()
    selected_datasets = expand_dataset_selection(args.datasets)

    active_pairs = [
        (v, f"{v}_rms")
        for v in args.variants
    ]
    active_variants = []
    for orig_v, norm_v in active_pairs:
        active_variants.extend([orig_v, norm_v])

    print("=" * 72)
    print("VALIDAÇÃO DOS DATASETS EXPERIMENTAIS TL")
    print("=" * 72)
    print(f"Projeto: {PROJECT_ROOT}")
    print(f"Raiz   : {ROOT}")
    print(f"Datasets: {selected_datasets}")
    print(f"Variantes: {args.variants}")
    print()

    # --------------------------------------------------------
    # 1. Verifica existência das entradas/saídas.
    # --------------------------------------------------------
    missing_dirs = []

    for variant_name in active_variants:
        for dataset in selected_datasets:
            directory = resolve_variant_dataset_dir(
                variant_name,
                dataset,
                use_processed_fallback=args.use_processed_fallback,
            )

            if not directory.exists():
                missing_dirs.append(
                    directory
                )

    if missing_dirs:
        print(
            "[ERRO] Diretórios ausentes:"
        )
        for directory in missing_dirs:
            print(
                f"  {directory}"
            )
        raise SystemExit(1)

    # --------------------------------------------------------
    # 2. Compara quantidade e estrutura dos pares.
    # --------------------------------------------------------
    summary_rows = []
    comparison_rows = []

    for dataset in selected_datasets:
        for original_variant, normalized_variant in active_pairs:

            result, comparisons = compare_pair(
                dataset=dataset,
                original_variant=original_variant,
                normalized_variant=normalized_variant,
                sample_size=args.sample_size,
                use_processed_fallback=args.use_processed_fallback,
            )

            summary_rows.append(
                result
            )

            comparison_rows.extend(
                comparisons
            )

            print()
            print(
                f"{dataset} | "
                f"{original_variant} → "
                f"{normalized_variant}"
            )
            print(
                f"  referência RMS : "
                f"{result['reference_rms']:.10g}"
            )
            print(
                f"  arquivos       : "
                f"{result['n_original']} → "
                f"{result['n_normalized']}"
            )
            print(
                f"  comuns         : "
                f"{result['n_common']}"
            )
            print(
                f"  verificados    : "
                f"{result['n_checked']}"
            )
            print(
                f"  faltantes      : "
                f"{result['missing_normalized']}"
            )
            print(
                f"  extras         : "
                f"{result['extra_normalized']}"
            )
            print(
                f"  shape mismatch : "
                f"{result['shape_mismatches']}"
            )
            print(
                f"  dtype mismatch : "
                f"{result['dtype_mismatches']}"
            )
            print(
                f"  erros RMS      : "
                f"{result['mathematical_failures']}"
            )
            print(
                f"  STATUS         : "
                f"{'OK' if result['math_pass'] else 'FALHOU'}"
            )

    # --------------------------------------------------------
    # 3. Valida que a média da classe de referência bate com
    #    o RMS_ref registrado no manifesto.
    # --------------------------------------------------------
    print()
    print("=" * 72)
    print("VALIDAÇÃO DAS REFERÊNCIAS")
    print("=" * 72)

    reference_rows = []

    for dataset in selected_datasets:
        for variant in args.variants:

            row = validate_reference_class(
                dataset=dataset,
                variant=variant,
                use_processed_fallback=args.use_processed_fallback,
            )

            if row is None:
                continue

            reference_rows.append(
                row
            )

            print(
                f"{dataset} | {variant} | "
                f"{row['reference_dataset']} / "
                f"{row['reference_class']} | "
                f"N={row['n']} | "
                f"média RMS={row['mean_rms']:.10g} | "
                f"STATUS={'OK' if row['pass'] else 'FALHOU'}"
            )

    # --------------------------------------------------------
    # 4. Estatísticas agregadas por dataset/classe/condição.
    # --------------------------------------------------------
    stats_rows = []

    print()
    print("=" * 72)
    print("ESTATÍSTICAS DE RMS POR CLASSE")
    print("=" * 72)

    for variant in active_variants:
        for dataset in selected_datasets:
            root = resolve_variant_dataset_dir(
                variant,
                dataset,
                use_processed_fallback=args.use_processed_fallback,
            )

            rows = grouped_summary(
                root,
                dataset_name=dataset,
            )

            stats_rows.extend(
                [
                    {
                        "variant": variant,
                        **row,
                    }
                    for row in rows
                ]
            )

    # --------------------------------------------------------
    # 5. Salva relatórios.
    # --------------------------------------------------------
    validation_dir = VALIDATION_ROOT

    summary_file = (
        validation_dir /
        "dataset_validation.csv"
    )

    comparison_file = (
        validation_dir /
        "file_comparison.csv"
    )

    reference_file = (
        validation_dir /
        "reference_validation.csv"
    )

    stats_file = (
        validation_dir /
        "rms_statistics.csv"
    )

    write_csv(
        summary_file,
        summary_rows,
    )

    write_csv(
        comparison_file,
        comparison_rows,
    )

    write_csv(
        reference_file,
        reference_rows,
    )

    write_csv(
        stats_file,
        stats_rows,
    )

    # --------------------------------------------------------
    # 6. Resultado geral.
    # --------------------------------------------------------
    structural_ok = all(
        row["n_original"] == row["n_normalized"]
        and row["missing_normalized"] == 0
        and row["extra_normalized"] == 0
        and row["shape_mismatches"] == 0
        and row["dtype_mismatches"] == 0
        for row in summary_rows
    )

    math_ok = all(
        row["math_pass"]
        for row in summary_rows
    )

    reference_ok = all(
        row["pass"]
        for row in reference_rows
    )

    print()
    print("=" * 72)
    print("RESULTADO FINAL")
    print("=" * 72)
    print(
        f"Estrutura/arquivos : "
        f"{'OK' if structural_ok else 'FALHOU'}"
    )
    print(
        f"Normalização RMS   : "
        f"{'OK' if math_ok else 'FALHOU'}"
    )
    print(
        f"Referências RMS    : "
        f"{'OK' if reference_ok else 'FALHOU'}"
    )

    print()
    print("Relatórios:")
    print(
        f"  {summary_file}"
    )
    print(
        f"  {comparison_file}"
    )
    print(
        f"  {reference_file}"
    )
    print(
        f"  {stats_file}"
    )

    if not (
        structural_ok
        and math_ok
        and reference_ok
    ):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
