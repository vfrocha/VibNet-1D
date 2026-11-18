#!/usr/bin/env python3
"""Orquestrador de processamento e validação dos datasets do VibNet-1D (`scripts/setup/prepare_datasets.py`).

Reutiliza estritamente os pipelines oficiais já existentes no repositório:
  - `src.data.make_dataset` (`PIPELINES`, `Detrend`, `SimpleSplit`, `get_names`, `harmonize_mfpt_sample_rate`)
  - `src.data.make_dataset_tl` (`SimpleSplit`, `Detrend`)
  - `src.preprocessing.normalize_dataset_tl` (`build_reference_map`, `process_variant`, `write_manifest`)
  - `src.data.validate_dataset_tl` (`compare_pair`, `validate_reference_class`)

Respeita `VIBNET_RAW_DIR`, `VIBNET_PROCESSED_DIR`, `VIBNET_PROCESSED_TL_DIR` e `VIBNET_RESULTS_DIR`,
reutilizando dados processados já válidos e registrando um manifesto completo (`processing_manifest.json`).
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.setup.download_datasets import (
    _ensure_writable_dir,
    get_dataset_to_raw_source_map,
    load_setup_config,
    resolve_selected_datasets,
)
from src.config import (
    PROCESSED_DATA_DIR,
    PROCESSED_TL_DIR,
    RAW_DATA_DIR,
)
from src.data.make_dataset import (
    Detrend,
    SimpleSplit,
    extract_signal,
    get_names as get_make_dataset_names,
    harmonize_mfpt_sample_rate,
)
from src.preprocessing import normalize_dataset_tl as norm_tl
from src.vibnet.data.registry import CANONICAL_DATASETS, load_yaml_config
from vibdata.deep.signal.transforms import Sequential

SUPPORTED_VARIANTS = (
    "processed_1d",
    "raw",
    "detrended",
    "raw_rms",
    "detrended_rms",
)
PROCESSING_MANIFEST_FILENAME = "processing_manifest.json"


def compute_processing_code_hash() -> str:
    """Calcula um hash SHA-256 combinado dos módulos oficiais de processamento."""
    h = hashlib.sha256()
    tracked_files = [
        PROJECT_ROOT / "src" / "data" / "make_dataset.py",
        PROJECT_ROOT / "src" / "data" / "make_dataset_tl.py",
        PROJECT_ROOT / "src" / "preprocessing" / "normalize_dataset_tl.py",
        PROJECT_ROOT / "configs" / "datasets.yaml",
    ]
    for fp in tracked_files:
        if fp.is_file():
            h.update(fp.read_bytes())
    return h.hexdigest()[:16]


def compute_variant_config_signature(
    dataset_name: str,
    variant: str,
    fs_hz: int,
    window_length: int,
    code_hash: str,
) -> str:
    """Calcula a assinatura da configuração de processamento de `(dataset_name, variant)`."""
    spec = CANONICAL_DATASETS[dataset_name]
    payload = {
        "dataset": dataset_name,
        "variant": variant,
        "fs_hz": int(fs_hz),
        "window_length": int(window_length),
        "reference_dataset": spec.reference_dataset,
        "normal_class": spec.normal_class,
        "code_hash": code_hash,
    }
    raw_bytes = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw_bytes).hexdigest()[:16]


def resolve_variant_dir(
    dataset_name: str,
    variant: str,
    processed_dir: Path,
    processed_tl_dir: Path,
) -> Path:
    """Resolve o diretório de saída de uma variante para um dataset."""
    if variant == "processed_1d":
        return processed_dir / dataset_name
    if variant in ("raw", "detrended", "raw_rms", "detrended_rms"):
        return processed_tl_dir / variant / dataset_name
    raise ValueError(f"Variante desconhecida: '{variant}'. Esperado uma de {SUPPORTED_VARIANTS}.")


def inspect_existing_processed_variant(
    dataset_name: str,
    variant: str,
    variant_dir: Path,
    expected_window_len: int,
) -> Dict[str, Any]:
    """Verifica se uma variante já processada existe, possui janelas `.npy` íntegras e shape 1s correto."""
    if not variant_dir.exists():
        return {
            "exists": False,
            "valid": False,
            "window_count": 0,
            "reason": f"Diretório ausente: {variant_dir}",
        }
    npy_files = sorted(variant_dir.rglob("*.npy"))
    if not npy_files:
        return {
            "exists": True,
            "valid": False,
            "window_count": 0,
            "reason": f"Nenhum arquivo .npy encontrado em {variant_dir}",
        }

    # Inspeciona amostra de arquivos (início, meio e fim) para verificar shape 1s e finitude
    sample_indices = sorted({0, len(npy_files) // 2, len(npy_files) - 1})
    conditions = sorted({p.parent.parent.name for p in npy_files})
    classes = sorted({p.parent.name for p in npy_files})

    for idx in sample_indices:
        fp = npy_files[idx]
        try:
            arr = np.load(fp, allow_pickle=False)
        except Exception as exc:
            return {
                "exists": True,
                "valid": False,
                "window_count": len(npy_files),
                "reason": f"Arquivo .npy corrompido ({fp.name}): {exc}",
            }
        flat = np.asarray(arr).reshape(-1)
        if flat.size != expected_window_len:
            return {
                "exists": True,
                "valid": False,
                "window_count": len(npy_files),
                "reason": (
                    f"Comprimento de janela divergente em {fp.name}: "
                    f"obtido {flat.size} != esperado {expected_window_len} (1s)."
                ),
            }
        if not np.all(np.isfinite(flat)):
            return {
                "exists": True,
                "valid": False,
                "window_count": len(npy_files),
                "reason": f"Valores NaN/Inf detectados em {fp.name}.",
            }

    return {
        "exists": True,
        "valid": True,
        "window_count": len(npy_files),
        "conditions": conditions,
        "classes": classes,
        "reason": f"{len(npy_files)} janelas de {expected_window_len} amostras verificadas.",
    }


def _load_raw_iterable_for_source(
    raw_source: str,
    raw_dir: Path,
) -> Iterable[Dict[str, Any]]:
    """Carrega registros brutos de fixtures sintéticas (para testes) ou via `vibdata.raw.<raw_source>_raw`."""
    fixture_manifest = raw_dir / f"{raw_source}_raw" / "fixture_records.json"
    if fixture_manifest.is_file():
        records_meta = json.loads(fixture_manifest.read_text(encoding="utf-8"))
        items: List[Dict[str, Any]] = []
        for rec in records_meta:
            sig_path = raw_dir / f"{raw_source}_raw" / rec["signal_file"]
            sig = np.load(sig_path, allow_pickle=False)
            meta_df = pd.DataFrame([rec["metainfo"]])
            items.append({"signal": sig, "metainfo": meta_df})
        return items

    try:
        import vibdata.raw as raw_datasets  # type: ignore

        cls_name = f"{raw_source}_raw"
        if hasattr(raw_datasets, cls_name):
            raw_cls = getattr(raw_datasets, cls_name)
            return raw_cls(str(raw_dir), download=False)
    except Exception:
        pass

    raise FileNotFoundError(
        f"Não foi possível carregar dados brutos para a fonte '{raw_source}' em '{raw_dir}'. "
        "Execute 'python scripts/setup/download_datasets.py' primeiro."
    )


def generate_base_windows_for_raw_source(
    raw_source: str,
    target_datasets: Sequence[str],
    base_variants: Sequence[str],
    raw_dir: Path,
    processed_dir: Path,
    processed_tl_dir: Path,
) -> Dict[Tuple[str, str], int]:
    """Gera as janelas de 1 segundo (`processed_1d`, `raw`, `detrended`) para `target_datasets`."""
    ds_iterable = _load_raw_iterable_for_source(raw_source, raw_dir)
    datasets_yaml = load_yaml_config(PROJECT_ROOT / "configs" / "datasets.yaml").get("datasets", {})

    counts: Dict[Tuple[str, str], int] = {
        (ds, var): 0 for ds in target_datasets for var in base_variants
    }

    n_items = len(ds_iterable) if hasattr(ds_iterable, "__len__") else None
    iterator = range(n_items) if n_items is not None else enumerate(ds_iterable)

    for idx_entry in iterator:
        if n_items is not None:
            i = int(idx_entry)
            item = ds_iterable[i]  # type: ignore[index]
        else:
            i, item = idx_entry  # type: ignore[misc]

        if not isinstance(item, dict):
            continue
        sig_array = extract_signal(item)
        if sig_array is None:
            continue

        meta = item["metainfo"]
        if isinstance(meta, pd.DataFrame):
            meta = meta.iloc[0]

        if raw_source == "MFPT":
            sig_array = harmonize_mfpt_sample_rate(sig_array, meta)

        if raw_source == "CWRU":
            sr = meta.get("sample_rate", 12000)
            try:
                sr_val = float(sr)
            except (TypeError, ValueError):
                sr_val = 12000.0
            routed_ds = "CWRU_48k" if sr_val > 20000.0 else "CWRU_12k"
            applicable_targets = [d for d in target_datasets if d == routed_ds]
        else:
            applicable_targets = [d for d in target_datasets if d == raw_source]

        for ds_name in applicable_targets:
            fs_hz = int(
                datasets_yaml.get(ds_name, {}).get(
                    "window_length", CANONICAL_DATASETS[ds_name].sampling_rate_hz
                )
            )
            cond, lbl = get_make_dataset_names(ds_name, meta)
            if lbl.startswith("Class_") and lbl.endswith(".0") and lbl[6:-2].isdigit():
                lbl = f"Class_{lbl[6:-2]}"
            if cond.startswith("Load_") and cond.endswith(".0"):
                cond = cond[:-2]

            for var in base_variants:
                if var == "raw":
                    pipeline = Sequential([SimpleSplit(window_size=fs_hz)])
                else:
                    # tanto 'detrended' quanto 'processed_1d' usam Detrend() + SimpleSplit(window_size=fs_hz)
                    pipeline = Sequential([Detrend(), SimpleSplit(window_size=fs_hz)])

                sample = {"signal": sig_array, "metainfo": pd.DataFrame([meta])}
                processed = pipeline(sample)
                windows = processed["signal"]
                if not isinstance(windows, list) or not windows:
                    continue

                out_dir = (
                    resolve_variant_dir(ds_name, var, processed_dir, processed_tl_dir)
                    / cond
                    / lbl
                )
                out_dir.mkdir(parents=True, exist_ok=True)
                for w_idx, window in enumerate(windows):
                    if isinstance(window, np.ndarray):
                        fname = f"s{i:05d}_w{w_idx:02d}.npy"
                        np.save(out_dir / fname, window)
                        counts[(ds_name, var)] += 1

    return counts


def run_rms_normalization_for_datasets(
    datasets_to_normalize: Sequence[str],
    rms_variants: Sequence[str],
    processed_dir: Path,
    processed_tl_dir: Path,
) -> List[Dict[str, Any]]:
    """Executa a normalização oficial por RMS saudável (`raw_rms` e/ou `detrended_rms`) via `normalize_dataset_tl`."""
    # Configura os diretórios no módulo oficial normalize_dataset_tl
    norm_tl.INPUT_ROOT = processed_tl_dir
    norm_tl.PROCESSED_FALLBACK_ROOT = processed_dir
    norm_tl.VARIANTS = {
        "raw": {
            "input_dir": processed_tl_dir / "raw",
            "output_dir": processed_tl_dir / "raw_rms",
        },
        "detrended": {
            "input_dir": processed_tl_dir / "detrended",
            "output_dir": processed_tl_dir / "detrended_rms",
        },
    }

    references = norm_tl.build_reference_map()
    manifest_rows: List[Dict[str, Any]] = []

    for rms_var in rms_variants:
        base_var = "raw" if rms_var == "raw_rms" else "detrended"
        out_root = norm_tl.VARIANTS[base_var]["output_dir"]
        out_root.mkdir(parents=True, exist_ok=True)

        # Carrega linhas prévias do normalization_manifest.csv para não perder outros datasets já processados
        existing_rows_by_ds: Dict[str, Dict[str, Any]] = {}
        manifest_csv = out_root / "normalization_manifest.csv"
        if manifest_csv.is_file():
            try:
                df_prev = pd.read_csv(manifest_csv)
                for r in df_prev.to_dict("records"):
                    existing_rows_by_ds[str(r["dataset"])] = r
            except Exception:
                pass

        for ds_name in datasets_to_normalize:
            norm_tl.process_variant(
                variant_name=base_var,
                dataset_name=ds_name,
                references=references,
                manifest_rows=manifest_rows,
                fallback_processed_root=processed_dir,
            )

        for r in manifest_rows:
            if r["output_variant"] == rms_var:
                existing_rows_by_ds[str(r["dataset"])] = r

        if existing_rows_by_ds:
            norm_tl.write_manifest(list(existing_rows_by_ds.values()), out_root)

    return manifest_rows


def prepare_datasets(
    datasets: Optional[Sequence[str]] = None,
    variants: Optional[Sequence[str]] = None,
    raw_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
    processed_tl_dir: Optional[Path] = None,
    force: bool = False,
) -> Dict[str, Any]:
    """Orquestra a preparação incremental e validação dos datasets e variantes solicitados."""
    resolved_raw_dir = Path(raw_dir or RAW_DATA_DIR).expanduser().resolve()
    resolved_proc_dir = _ensure_writable_dir(
        Path(processed_dir or PROCESSED_DATA_DIR), "VIBNET_PROCESSED_DIR"
    )
    resolved_proc_tl_dir = _ensure_writable_dir(
        Path(processed_tl_dir or PROCESSED_TL_DIR), "VIBNET_PROCESSED_TL_DIR"
    )

    selected_datasets = resolve_selected_datasets(datasets)
    req_variants = list(variants) if variants else list(SUPPORTED_VARIANTS)
    if "all" in req_variants:
        req_variants = list(SUPPORTED_VARIANTS)
    for v in req_variants:
        if v not in SUPPORTED_VARIANTS:
            raise ValueError(
                f"Variante inválida '{v}'. Variantes válidas: {list(SUPPORTED_VARIANTS)}"
            )

    # Se raw_rms ou detrended_rms foi solicitado, garante que a variante base correspondente também exista
    effective_datasets = list(selected_datasets)
    if "CWRU_48k" in effective_datasets and "CWRU_12k" not in effective_datasets:
        if any(v in req_variants for v in ("raw_rms", "detrended_rms")):
            effective_datasets.insert(0, "CWRU_12k")

    datasets_yaml = load_yaml_config(PROJECT_ROOT / "configs" / "datasets.yaml").get("datasets", {})
    code_hash = compute_processing_code_hash()

    manifest_path = resolved_proc_tl_dir / PROCESSING_MANIFEST_FILENAME
    prev_manifest: Dict[str, Any] = {}
    if manifest_path.is_file():
        try:
            prev_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            prev_manifest = {}

    prev_entries: Dict[str, Dict[str, Any]] = dict(prev_manifest.get("entries", {}))

    # Determina quais (dataset, base_variant) precisam ser gerados a partir do bruto
    base_variants_needed_by_ds: Dict[str, List[str]] = {ds: [] for ds in effective_datasets}
    rms_variants_needed_by_ds: Dict[str, List[str]] = {ds: [] for ds in selected_datasets}

    reused_entries: List[str] = []

    for ds_name in effective_datasets:
        fs_hz = int(
            datasets_yaml.get(ds_name, {}).get(
                "window_length", CANONICAL_DATASETS[ds_name].sampling_rate_hz
            )
        )
        # Quais variantes base precisamos para este dataset?
        candidate_base_vars: List[str] = []
        if ds_name in selected_datasets:
            for v in req_variants:
                if v in ("processed_1d", "raw", "detrended") and v not in candidate_base_vars:
                    candidate_base_vars.append(v)
        if any(v == "raw_rms" for v in req_variants) and "raw" not in candidate_base_vars:
            candidate_base_vars.append("raw")
        if any(v == "detrended_rms" for v in req_variants) and "detrended" not in candidate_base_vars:
            candidate_base_vars.append("detrended")

        for b_var in candidate_base_vars:
            entry_key = f"{ds_name}::{b_var}"
            cfg_sig = compute_variant_config_signature(ds_name, b_var, fs_hz, fs_hz, code_hash)
            v_dir = resolve_variant_dir(ds_name, b_var, resolved_proc_dir, resolved_proc_tl_dir)
            check = inspect_existing_processed_variant(ds_name, b_var, v_dir, fs_hz)
            prev_sig = prev_entries.get(entry_key, {}).get("config_signature")

            if check["valid"] and not force and (prev_sig in (None, cfg_sig)):
                reused_entries.append(entry_key)
                prev_entries[entry_key] = {
                    "dataset": ds_name,
                    "variant": b_var,
                    "output_dir": str(v_dir),
                    "fs_hz": fs_hz,
                    "window_length": fs_hz,
                    "window_count": check["window_count"],
                    "code_hash": code_hash,
                    "config_signature": cfg_sig,
                    "status": "REUSED_VALID",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
            else:
                base_variants_needed_by_ds[ds_name].append(b_var)

    # Agrupa geração de janelas base por fonte bruta (CWRU gera CWRU_12k e CWRU_48k juntos)
    ds_to_raw = get_dataset_to_raw_source_map()
    raw_to_targets: Dict[str, List[str]] = {}
    for ds_name, b_vars in base_variants_needed_by_ds.items():
        if b_vars:
            r_src = ds_to_raw[ds_name]
            raw_to_targets.setdefault(r_src, []).append(ds_name)

    errors_by_ds: Dict[str, str] = {}

    for r_src, t_datasets in raw_to_targets.items():
        union_base_vars: List[str] = []
        for d in t_datasets:
            for bv in base_variants_needed_by_ds[d]:
                if bv not in union_base_vars:
                    union_base_vars.append(bv)
        try:
            gen_counts = generate_base_windows_for_raw_source(
                raw_source=r_src,
                target_datasets=t_datasets,
                base_variants=union_base_vars,
                raw_dir=resolved_raw_dir,
                processed_dir=resolved_proc_dir,
                processed_tl_dir=resolved_proc_tl_dir,
            )
            for d in t_datasets:
                fs_hz = int(
                    datasets_yaml.get(d, {}).get(
                        "window_length", CANONICAL_DATASETS[d].sampling_rate_hz
                    )
                )
                for bv in base_variants_needed_by_ds[d]:
                    entry_key = f"{d}::{bv}"
                    cfg_sig = compute_variant_config_signature(d, bv, fs_hz, fs_hz, code_hash)
                    v_dir = resolve_variant_dir(d, bv, resolved_proc_dir, resolved_proc_tl_dir)
                    check = inspect_existing_processed_variant(d, bv, v_dir, fs_hz)
                    prev_entries[entry_key] = {
                        "dataset": d,
                        "raw_source": r_src,
                        "variant": bv,
                        "output_dir": str(v_dir),
                        "fs_hz": fs_hz,
                        "window_length": fs_hz,
                        "window_count": int(gen_counts.get((d, bv), check["window_count"])),
                        "code_hash": code_hash,
                        "config_signature": cfg_sig,
                        "status": "PROCESSED_OK" if check["valid"] else "VALIDATION_FAILED",
                        "validation_detail": check["reason"],
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    }
        except Exception as exc:
            for d in t_datasets:
                errors_by_ds[d] = str(exc)
                for bv in base_variants_needed_by_ds[d]:
                    entry_key = f"{d}::{bv}"
                    prev_entries[entry_key] = {
                        "dataset": d,
                        "raw_source": r_src,
                        "variant": bv,
                        "status": "SKIPPED_MISSING_RAW",
                        "validation_detail": str(exc),
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    }

    # Agora verifica e executa variantes normalizadas por RMS (raw_rms e detrended_rms)
    for ds_name in selected_datasets:
        if ds_name in errors_by_ds:
            continue
        fs_hz = int(
            datasets_yaml.get(ds_name, {}).get(
                "window_length", CANONICAL_DATASETS[ds_name].sampling_rate_hz
            )
        )
        for rms_var in [v for v in req_variants if v in ("raw_rms", "detrended_rms")]:
            entry_key = f"{ds_name}::{rms_var}"
            cfg_sig = compute_variant_config_signature(ds_name, rms_var, fs_hz, fs_hz, code_hash)
            v_dir = resolve_variant_dir(ds_name, rms_var, resolved_proc_dir, resolved_proc_tl_dir)
            check = inspect_existing_processed_variant(ds_name, rms_var, v_dir, fs_hz)
            prev_sig = prev_entries.get(entry_key, {}).get("config_signature")
            base_needed = "raw" if rms_var == "raw_rms" else "detrended"
            base_rebuilt = base_needed in base_variants_needed_by_ds.get(ds_name, [])

            if check["valid"] and not force and not base_rebuilt and (prev_sig in (None, cfg_sig)):
                reused_entries.append(entry_key)
                prev_entries[entry_key] = {
                    "dataset": ds_name,
                    "variant": rms_var,
                    "output_dir": str(v_dir),
                    "fs_hz": fs_hz,
                    "window_length": fs_hz,
                    "window_count": check["window_count"],
                    "code_hash": code_hash,
                    "config_signature": cfg_sig,
                    "status": "REUSED_VALID",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
            else:
                rms_variants_needed_by_ds[ds_name].append(rms_var)

    for rms_var in ("raw_rms", "detrended_rms"):
        ds_list_for_rms = [
            d for d, rvars in rms_variants_needed_by_ds.items() if rms_var in rvars
        ]
        if not ds_list_for_rms:
            continue
        try:
            norm_rows = run_rms_normalization_for_datasets(
                datasets_to_normalize=ds_list_for_rms,
                rms_variants=[rms_var],
                processed_dir=resolved_proc_dir,
                processed_tl_dir=resolved_proc_tl_dir,
            )
            norm_by_ds = {str(r["dataset"]): r for r in norm_rows if r["output_variant"] == rms_var}
            for d in ds_list_for_rms:
                fs_hz = int(
                    datasets_yaml.get(d, {}).get(
                        "window_length", CANONICAL_DATASETS[d].sampling_rate_hz
                    )
                )
                entry_key = f"{d}::{rms_var}"
                cfg_sig = compute_variant_config_signature(d, rms_var, fs_hz, fs_hz, code_hash)
                v_dir = resolve_variant_dir(d, rms_var, resolved_proc_dir, resolved_proc_tl_dir)
                check = inspect_existing_processed_variant(d, rms_var, v_dir, fs_hz)
                n_row = norm_by_ds.get(d, {})
                prev_entries[entry_key] = {
                    "dataset": d,
                    "variant": rms_var,
                    "output_dir": str(v_dir),
                    "fs_hz": fs_hz,
                    "window_length": fs_hz,
                    "window_count": check["window_count"],
                    "reference_dataset": n_row.get("reference_dataset"),
                    "reference_class": n_row.get("reference_classes"),
                    "reference_rms": n_row.get("reference_rms"),
                    "code_hash": code_hash,
                    "config_signature": cfg_sig,
                    "status": "PROCESSED_OK" if check["valid"] else "VALIDATION_FAILED",
                    "validation_detail": check["reason"],
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
        except Exception as exc:
            for d in ds_list_for_rms:
                entry_key = f"{d}::{rms_var}"
                prev_entries[entry_key] = {
                    "dataset": d,
                    "variant": rms_var,
                    "status": "NORMALIZATION_FAILED",
                    "validation_detail": str(exc),
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }

    manifest_out = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_data_dir": str(resolved_raw_dir),
        "processed_data_dir": str(resolved_proc_dir),
        "processed_tl_dir": str(resolved_proc_tl_dir),
        "code_hash": code_hash,
        "selected_datasets": selected_datasets,
        "requested_variants": req_variants,
        "reused_entries": reused_entries,
        "entries": prev_entries,
    }
    manifest_path.write_text(
        json.dumps(manifest_out, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest_out


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Prepara e valida os datasets selecionados do VibNet-1D."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Datasets canônicos a preparar (default: todos os 10 datasets).",
    )
    parser.add_argument(
        "--variants",
        nargs="+",
        default=["all"],
        choices=list(SUPPORTED_VARIANTS) + ["all"],
        help="Variantes a preparar (default: all -> processed_1d, raw, detrended, raw_rms, detrended_rms).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Diretório de entrada dos dados brutos (default: VIBNET_RAW_DIR).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Diretório de saída para processed_1d (default: VIBNET_PROCESSED_DIR).",
    )
    parser.add_argument(
        "--processed-tl-dir",
        type=Path,
        default=None,
        help="Diretório de saída para variantes TL (default: VIBNET_PROCESSED_TL_DIR).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Força o reprocessamento mesmo quando os dados processados já existem e são válidos.",
    )
    args = parser.parse_args(argv)

    try:
        selected = resolve_selected_datasets(args.datasets)
    except ValueError as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        return 2

    manifest = prepare_datasets(
        datasets=selected,
        variants=args.variants,
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        processed_tl_dir=args.processed_tl_dir,
        force=args.force,
    )

    print("=" * 72)
    print(" RESUMO DO PROCESSAMENTO DE DATASETS — VIBNET-1D")
    print("=" * 72)
    for key, entry in sorted(manifest["entries"].items()):
        if entry.get("dataset") in selected:
            print(
                f"  - {key:32s}: status={entry.get('status'):20s} | "
                f"janelas={entry.get('window_count', 0)}"
            )
    print(f"\nManifesto salvo em: {Path(manifest['processed_tl_dir']) / PROCESSING_MANIFEST_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
