#!/usr/bin/env python3
"""Gerenciador reproduzível de download e verificação de datasets brutos (`scripts/setup/download_datasets.py`).

Baseia a seleção exclusivamente no registro canônico `src.vibnet.data.registry.CANONICAL_DATASETS`
e nos metadados de `scripts/setup/setup_config.yaml`, respeitando `VIBNET_RAW_DIR`.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import urllib.request
import zipfile
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import PROCESSED_TL_DIR, RAW_DATA_DIR
from src.vibnet.data.registry import (
    CANONICAL_DATASETS,
    list_all_datasets,
    list_sources,
    list_targets,
    load_yaml_config,
)

SETUP_CONFIG_PATH = Path(__file__).resolve().parent / "setup_config.yaml"
DOWNLOAD_MANIFEST_FILENAME = "download_manifest.json"


def load_setup_config(config_path: Path = SETUP_CONFIG_PATH) -> Dict[str, Any]:
    """Carrega `scripts/setup/setup_config.yaml`."""
    return load_yaml_config(config_path)


def get_dataset_to_raw_source_map(setup_cfg: Optional[Mapping[str, Any]] = None) -> Dict[str, str]:
    """Mapeia cada um dos 10 datasets canônicos para sua fonte bruta (`CWRU_12k` e `CWRU_48k` -> `CWRU`)."""
    cfg = setup_cfg or load_setup_config()
    raw_sources = cfg.get("raw_sources", {})
    mapping: Dict[str, str] = {}
    for raw_name, meta in raw_sources.items():
        for ds_name in meta.get("provides_datasets", []):
            mapping[str(ds_name)] = str(raw_name)
    for ds_name in list_all_datasets():
        if ds_name not in mapping:
            mapping[ds_name] = "CWRU" if ds_name.startswith("CWRU") else ds_name
    return mapping


def resolve_selected_datasets(requested: Optional[Sequence[str]] = None) -> List[str]:
    """Valida e expande a seleção de datasets contra `CANONICAL_DATASETS`.

    Lança `ValueError` com a lista de nomes válidos caso algum nome seja inválido.
    """
    canonical = list_all_datasets()
    if not requested:
        return list(canonical)

    selected: List[str] = []
    invalid: List[str] = []
    for item in requested:
        token = str(item).strip()
        if not token:
            continue
        if token.lower() == "all":
            for ds in canonical:
                if ds not in selected:
                    selected.append(ds)
        elif token.lower() == "cross_dataset":
            for ds in list_targets(True) + list_sources(True):
                if ds not in selected:
                    selected.append(ds)
        elif token in CANONICAL_DATASETS:
            if token not in selected:
                selected.append(token)
        else:
            invalid.append(token)

    if invalid:
        valid_str = ", ".join(canonical)
        raise ValueError(
            f"Dataset(s) inválido(s): {invalid}. "
            f"Datasets canônicos válidos no registro do projeto: [{valid_str}] "
            "(ou atalhos: 'all', 'cross_dataset')."
        )
    return selected


def compute_md5(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calcula o hash MD5 de um arquivo em disco."""
    h = hashlib.md5()
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def compute_sha256(file_path: Path, chunk_size: int = 1024 * 1024) -> str:
    """Calcula o hash SHA-256 de um arquivo em disco."""
    h = hashlib.sha256()
    with file_path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def estimate_disk_requirements(
    selected_datasets: Sequence[str],
    raw_dir: Path,
    processed_dir: Optional[Path] = None,
    setup_cfg: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Estima o espaço em disco bruto e processado para os datasets selecionados."""
    cfg = setup_cfg or load_setup_config()
    raw_sources_cfg = cfg.get("raw_sources", {})
    ds_to_raw = get_dataset_to_raw_source_map(cfg)

    needed_raw_sources: List[str] = []
    for ds in selected_datasets:
        r_src = ds_to_raw[ds]
        if r_src not in needed_raw_sources:
            needed_raw_sources.append(r_src)

    total_raw_mb = 0
    total_proc_mb = 0
    unknown_size_sources: List[str] = []
    per_source: Dict[str, Dict[str, Any]] = {}

    for r_src in needed_raw_sources:
        s_meta = raw_sources_cfg.get(r_src, {})
        raw_mb = s_meta.get("estimated_raw_mb")
        proc_mb = s_meta.get("estimated_processed_mb")
        if raw_mb is None or proc_mb is None:
            unknown_size_sources.append(r_src)
        else:
            total_raw_mb += int(raw_mb)
            total_proc_mb += int(proc_mb)
        per_source[r_src] = {
            "provides_datasets": [
                d for d in selected_datasets if ds_to_raw.get(d) == r_src
            ],
            "estimated_raw_mb": raw_mb,
            "estimated_processed_mb": proc_mb,
        }

    probe_raw = raw_dir.resolve()
    while not probe_raw.exists() and probe_raw.parent != probe_raw:
        probe_raw = probe_raw.parent
    free_raw_mb = int(shutil.disk_usage(probe_raw).free // (1024 * 1024))

    proc_target = (processed_dir or PROCESSED_TL_DIR).resolve()
    probe_proc = proc_target
    while not probe_proc.exists() and probe_proc.parent != probe_proc:
        probe_proc = probe_proc.parent
    free_proc_mb = int(shutil.disk_usage(probe_proc).free // (1024 * 1024))

    same_mount = probe_raw == probe_proc
    required_combined_mb = total_raw_mb + total_proc_mb if same_mount else total_raw_mb
    sufficient_space = free_raw_mb >= required_combined_mb

    return {
        "selected_datasets": list(selected_datasets),
        "raw_sources_required": needed_raw_sources,
        "estimated_raw_mb": total_raw_mb,
        "estimated_processed_mb": total_proc_mb,
        "estimated_total_mb": total_raw_mb + total_proc_mb,
        "unknown_size_sources": unknown_size_sources,
        "free_raw_dir_mb": free_raw_mb,
        "free_processed_dir_mb": free_proc_mb,
        "sufficient_space": bool(sufficient_space),
        "per_source": per_source,
    }


def verify_existing_raw_source(
    raw_source: str,
    raw_dir: Path,
    source_meta: Mapping[str, Any],
) -> Dict[str, Any]:
    """Verifica se uma fonte bruta já está baixada/extraída e íntegra em `raw_dir`."""
    archive_name = source_meta.get("archive_name") or f"{raw_source}.zip"
    expected_md5 = source_meta.get("expected_md5")

    candidate_dirs = [
        raw_dir / f"{raw_source}_raw",
        raw_dir / raw_source,
        raw_dir / "download" / raw_source,
    ]
    extracted_files: List[Path] = []
    extracted_dir_found: Optional[Path] = None
    for c_dir in candidate_dirs:
        if c_dir.is_dir():
            files = [
                p
                for p in c_dir.rglob("*")
                if p.is_file() and not p.name.endswith(".part")
            ]
            if files:
                extracted_dir_found = c_dir
                extracted_files = files
                break

    candidate_archives = [
        raw_dir / archive_name,
        raw_dir / "download" / archive_name,
        raw_dir / f"{raw_source}_raw" / archive_name,
    ]
    archive_found: Optional[Path] = None
    for c_arc in candidate_archives:
        if c_arc.is_file():
            archive_found = c_arc
            break

    part_files = list(raw_dir.rglob(f"{archive_name}.part"))

    if archive_found is not None:
        size_b = archive_found.stat().st_size
        if size_b == 0:
            return {
                "status": "CORRUPTED_EMPTY_ARCHIVE",
                "valid": False,
                "archive_path": str(archive_found),
                "extracted_dir": str(extracted_dir_found) if extracted_dir_found else None,
                "reason": f"Arquivo '{archive_found.name}' está vazio (0 bytes).",
            }
        if archive_found.suffix.lower() == ".zip" and not zipfile.is_zipfile(archive_found):
            return {
                "status": "CORRUPTED_INVALID_ZIP",
                "valid": False,
                "archive_path": str(archive_found),
                "extracted_dir": str(extracted_dir_found) if extracted_dir_found else None,
                "reason": f"Arquivo '{archive_found.name}' não é um arquivo ZIP válido (possível download truncado).",
            }
        actual_md5 = compute_md5(archive_found)
        if expected_md5 and actual_md5.lower() != str(expected_md5).lower():
            return {
                "status": "CHECKSUM_MISMATCH",
                "valid": False,
                "archive_path": str(archive_found),
                "actual_md5": actual_md5,
                "expected_md5": expected_md5,
                "reason": (
                    f"MD5 divergente para '{archive_found.name}': "
                    f"obtido {actual_md5} != esperado {expected_md5}."
                ),
            }
        return {
            "status": "ALREADY_PRESENT",
            "valid": True,
            "archive_path": str(archive_found),
            "archive_size_bytes": int(size_b),
            "md5": actual_md5,
            "extracted_dir": str(extracted_dir_found) if extracted_dir_found else None,
            "extracted_file_count": len(extracted_files),
            "reason": "Arquivo bruto existente e verificado com sucesso.",
        }

    if extracted_dir_found is not None and len(extracted_files) > 0:
        total_bytes = sum(f.stat().st_size for f in extracted_files)
        return {
            "status": "ALREADY_PRESENT_EXTRACTED",
            "valid": True,
            "archive_path": None,
            "archive_size_bytes": int(total_bytes),
            "md5": None,
            "extracted_dir": str(extracted_dir_found),
            "extracted_file_count": len(extracted_files),
            "reason": (
                f"Diretório bruto extraído encontrado em '{extracted_dir_found}' "
                f"({len(extracted_files)} arquivos)."
            ),
        }

    if part_files:
        return {
            "status": "PARTIAL_DOWNLOAD_DETECTED",
            "valid": False,
            "part_file": str(part_files[0]),
            "part_size_bytes": int(part_files[0].stat().st_size),
            "reason": f"Download interrompido detectado ({part_files[0].name}); pronto para retomada.",
        }

    return {
        "status": "MISSING",
        "valid": False,
        "archive_path": None,
        "extracted_dir": None,
        "reason": "Dados brutos ainda não baixados.",
    }


def download_url_with_resume(
    url: str,
    dest_path: Path,
    expected_md5: Optional[str] = None,
    opener: Optional[Callable[..., Any]] = None,
) -> Dict[str, Any]:
    """Baixa um arquivo via HTTP/HTTPS com suporte a retomada (`Range: bytes=...`) em `.part`."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_name(dest_path.name + ".part")

    existing_bytes = part_path.stat().st_size if part_path.exists() else 0
    headers: Dict[str, str] = {"User-Agent": "VibNet-1D-Installer/1.0"}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"

    req = urllib.request.Request(url, headers=headers)
    open_fn = opener or urllib.request.urlopen

    with open_fn(req, timeout=60) as resp:
        status_code = getattr(resp, "status", 200)
        mode = "ab" if (existing_bytes > 0 and status_code == 206) else "wb"
        with part_path.open(mode) as out_f:
            while True:
                chunk = resp.read(1024 * 256)
                if not chunk:
                    break
                out_f.write(chunk)

    actual_md5 = compute_md5(part_path)
    if expected_md5 and actual_md5.lower() != str(expected_md5).lower():
        raise RuntimeError(
            f"Falha de checksum MD5 após download de '{url}': "
            f"obtido {actual_md5} != esperado {expected_md5}."
        )

    part_path.replace(dest_path)
    return {
        "archive_path": str(dest_path),
        "archive_size_bytes": int(dest_path.stat().st_size),
        "md5": actual_md5,
        "resumed_from_bytes": int(existing_bytes),
    }


def extract_zip_safely(zip_path: Path, extract_to: Path) -> int:
    """Extrai um arquivo `.zip` validando antes sua integridade."""
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"O arquivo '{zip_path}' não é um arquivo ZIP válido.")
    extract_to.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        bad_member = zf.testzip()
        if bad_member is not None:
            raise ValueError(f"Membro corrompido '{bad_member}' dentro de '{zip_path}'.")
        zf.extractall(extract_to)
        return len(zf.namelist())


def download_single_raw_source(
    raw_source: str,
    raw_dir: Path,
    source_meta: Mapping[str, Any],
    force: bool = False,
    custom_downloader: Optional[Callable[[str, Path, Mapping[str, Any]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Executa a verificação e o download idempotente de uma fonte bruta."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    check = verify_existing_raw_source(raw_source, raw_dir, source_meta)

    if check["valid"] and not force:
        return {
            "raw_source": raw_source,
            "provides_datasets": list(source_meta.get("provides_datasets", [raw_source])),
            "status": check["status"],
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **check,
        }

    if custom_downloader is not None:
        dl_res = custom_downloader(raw_source, raw_dir, source_meta)
        return {
            "raw_source": raw_source,
            "provides_datasets": list(source_meta.get("provides_datasets", [raw_source])),
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            **dl_res,
        }

    # 1. Tenta via classe oficial da biblioteca vibdata (se disponível e se não possuir mirrors=[""])
    vib_cls_name = str(source_meta.get("vibdata_class", f"{raw_source}_raw"))
    direct_url = source_meta.get("direct_url")
    gdrive_id = source_meta.get("gdrive_file_id")
    archive_name = str(source_meta.get("archive_name") or f"{raw_source}.zip")
    expected_md5 = source_meta.get("expected_md5")

    try:
        import vibdata.raw as raw_datasets  # type: ignore

        if hasattr(raw_datasets, vib_cls_name):
            raw_cls = getattr(raw_datasets, vib_cls_name)
            mirrors = getattr(raw_cls, "mirrors", None) or getattr(raw_cls, "urls", None)
            has_non_empty_mirror = bool(mirrors and any(str(m).strip() for m in mirrors))
            if has_non_empty_mirror:
                raw_cls(str(raw_dir), download=True)
                post = verify_existing_raw_source(raw_source, raw_dir, source_meta)
                if post["valid"]:
                    return {
                        "raw_source": raw_source,
                        "provides_datasets": list(source_meta.get("provides_datasets", [raw_source])),
                        "status": "DOWNLOADED_VIA_VIBDATA",
                        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                        **post,
                    }
    except Exception as vib_exc:
        last_vib_error = str(vib_exc)
    else:
        last_vib_error = None

    # 2. Caso possua direct_url (ex.: HUST e UORED no Mendeley Data S3 onde vibdata 1.1.1 tem mirrors=[""])
    dest_archive = raw_dir / archive_name
    if direct_url:
        try:
            dl_info = download_url_with_resume(
                url=str(direct_url),
                dest_path=dest_archive,
                expected_md5=expected_md5,
            )
            extracted_target = raw_dir / f"{raw_source}_raw"
            n_extracted = extract_zip_safely(dest_archive, extracted_target)
            return {
                "raw_source": raw_source,
                "provides_datasets": list(source_meta.get("provides_datasets", [raw_source])),
                "status": "DOWNLOADED_DIRECT_URL",
                "valid": True,
                "extracted_dir": str(extracted_target),
                "extracted_file_count": n_extracted,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                **dl_info,
            }
        except Exception as url_exc:
            return {
                "raw_source": raw_source,
                "provides_datasets": list(source_meta.get("provides_datasets", [raw_source])),
                "status": "FAILED",
                "valid": False,
                "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "reason": f"Falha no download direto de {direct_url}: {url_exc}",
            }

    # 3. Caso possua gdrive_file_id (ex.: CWRU, MFPT, PU via gdown com retomada)
    if gdrive_id:
        try:
            import gdown  # type: ignore

            out_file = gdown.download(
                id=str(gdrive_id),
                output=str(dest_archive),
                quiet=False,
                resume=True,
            )
            if out_file and dest_archive.exists():
                actual_md5 = compute_md5(dest_archive)
                if expected_md5 and actual_md5.lower() != str(expected_md5).lower():
                    raise RuntimeError(
                        f"MD5 divergente para {dest_archive.name}: {actual_md5} != {expected_md5}"
                    )
                extracted_target = raw_dir / f"{raw_source}_raw"
                n_extracted = extract_zip_safely(dest_archive, extracted_target)
                return {
                    "raw_source": raw_source,
                    "provides_datasets": list(source_meta.get("provides_datasets", [raw_source])),
                    "status": "DOWNLOADED_GDRIVE",
                    "valid": True,
                    "archive_path": str(dest_archive),
                    "archive_size_bytes": int(dest_archive.stat().st_size),
                    "md5": actual_md5,
                    "extracted_dir": str(extracted_target),
                    "extracted_file_count": n_extracted,
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                }
        except Exception as gd_exc:
            last_vib_error = f"{last_vib_error or ''}; gdown: {gd_exc}".strip("; ")

    # 4. Caso exija o fork customizado da vibdata ou disponibilização manual
    requires_fork = bool(source_meta.get("requires_custom_vibdata_fork", False))
    notes = str(source_meta.get("license_and_notes", ""))
    return {
        "raw_source": raw_source,
        "provides_datasets": list(source_meta.get("provides_datasets", [raw_source])),
        "status": "MANUAL_OR_FORK_REQUIRED" if requires_fork else "FAILED",
        "valid": False,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "reason": (
            f"Não foi possível baixar '{raw_source}' automaticamente no ambiente atual. "
            f"{notes} "
            + (f"(Detalhe técnico: {last_vib_error})" if last_vib_error else "")
        ).strip(),
    }


def download_datasets(
    datasets: Optional[Sequence[str]] = None,
    raw_dir: Optional[Path] = None,
    force: bool = False,
    dry_run: bool = False,
    custom_downloader: Optional[Callable[[str, Path, Mapping[str, Any]], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """Orquestra a estimativa de espaço, download e atualização de `download_manifest.json`."""
    resolved_raw_dir = Path(raw_dir or RAW_DATA_DIR).expanduser().resolve()
    resolved_raw_dir.mkdir(parents=True, exist_ok=True)

    setup_cfg = load_setup_config()
    selected = resolve_selected_datasets(datasets)
    disk_est = estimate_disk_requirements(selected, resolved_raw_dir, setup_cfg=setup_cfg)

    raw_sources_cfg = setup_cfg.get("raw_sources", {})
    ds_to_raw = get_dataset_to_raw_source_map(setup_cfg)

    needed_raw_sources: List[str] = []
    for ds in selected:
        r_src = ds_to_raw[ds]
        if r_src not in needed_raw_sources:
            needed_raw_sources.append(r_src)

    manifest_path = resolved_raw_dir / DOWNLOAD_MANIFEST_FILENAME
    existing_manifest: Dict[str, Any] = {}
    if manifest_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            existing_manifest = {}

    sources_results: Dict[str, Any] = dict(existing_manifest.get("raw_sources", {}))
    dataset_status: Dict[str, Any] = dict(existing_manifest.get("datasets", {}))

    if not dry_run:
        for r_src in needed_raw_sources:
            s_meta = raw_sources_cfg.get(
                r_src,
                {
                    "provides_datasets": [d for d in selected if ds_to_raw.get(d) == r_src],
                    "vibdata_class": f"{r_src}_raw",
                    "archive_name": f"{r_src}.zip",
                },
            )
            res = download_single_raw_source(
                raw_source=r_src,
                raw_dir=resolved_raw_dir,
                source_meta=s_meta,
                force=force,
                custom_downloader=custom_downloader,
            )
            sources_results[r_src] = res
            for ds_name in s_meta.get("provides_datasets", []):
                if ds_name in selected:
                    spec = CANONICAL_DATASETS[ds_name]
                    dataset_status[ds_name] = {
                        "dataset": ds_name,
                        "raw_source": r_src,
                        "sampling_rate_hz": spec.sampling_rate_hz,
                        "status": res["status"],
                        "valid": bool(res.get("valid", False)),
                        "archive_path": res.get("archive_path"),
                        "extracted_dir": res.get("extracted_dir"),
                        "md5": res.get("md5"),
                        "timestamp_utc": res.get("timestamp_utc"),
                        "reason": res.get("reason"),
                    }

    manifest_payload = {
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "raw_data_dir": str(resolved_raw_dir),
        "selected_datasets": selected,
        "disk_estimate": disk_est,
        "raw_sources": sources_results,
        "datasets": dataset_status,
    }
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest_payload


def format_dataset_catalog_table(setup_cfg: Optional[Mapping[str, Any]] = None) -> str:
    """Formata a tabela de todos os datasets canônicos para `--list`."""
    cfg = setup_cfg or load_setup_config()
    raw_sources_cfg = cfg.get("raw_sources", {})
    ds_to_raw = get_dataset_to_raw_source_map(cfg)

    lines: List[str] = []
    header = (
        f"{'Dataset Canônico':16s} | {'Fonte Bruta':15s} | {'Fs (Hz)':>8s} | "
        f"{'Papel Padrão':16s} | {'Raw (MB)':>8s} | {'Auto-DL':7s} | {'Origem / Notas'}"
    )
    lines.append("=" * len(header))
    lines.append(header)
    lines.append("-" * len(header))

    for ds_name in list_all_datasets():
        spec = CANONICAL_DATASETS[ds_name]
        r_src = ds_to_raw[ds_name]
        s_meta = raw_sources_cfg.get(r_src, {})
        raw_mb = s_meta.get("estimated_raw_mb")
        raw_mb_str = f"~{raw_mb}" if raw_mb is not None else "N/D"
        auto_dl = "Sim*" if s_meta.get("requires_custom_vibdata_fork") else (
            "Sim" if s_meta.get("auto_downloadable", True) else "Manual"
        )
        notes = str(s_meta.get("license_and_notes", spec.notes))
        lines.append(
            f"{ds_name:16s} | {r_src:15s} | {spec.sampling_rate_hz:8d} | "
            f"{spec.role_default:16s} | {raw_mb_str:>8s} | {auto_dl:7s} | {notes}"
        )

    lines.append("=" * len(header))
    lines.append(
        "* Nota: 'Sim*' indica datasets suportados pelo fork oficial do projeto "
        "(git+https://github.com/vfrocha/vibdata.git) ou via diretório local em VIBNET_RAW_DIR."
    )
    return "\n".join(lines)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Gerenciador de download e inventário dos datasets do VibNet-1D."
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Lista os 10 datasets canônicos, fontes brutas, tamanhos estimados e requisitos.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="Lista de datasets canônicos a baixar (default: todos os 10 datasets).",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Diretório de armazenamento dos dados brutos (default: VIBNET_RAW_DIR).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Força novo download mesmo se o dataset bruto já existir e estiver íntegro.",
    )
    parser.add_argument(
        "--estimate-only",
        action="store_true",
        help="Apenas calcula e exibe a estimativa de espaço em disco sem iniciar downloads.",
    )
    args = parser.parse_args(argv)

    setup_cfg = load_setup_config()
    if args.list:
        print(format_dataset_catalog_table(setup_cfg))
        return 0

    try:
        selected = resolve_selected_datasets(args.datasets)
    except ValueError as exc:
        print(f"[ERRO] {exc}", file=sys.stderr)
        return 2

    target_raw_dir = Path(args.raw_dir or RAW_DATA_DIR).expanduser().resolve()
    est = estimate_disk_requirements(selected, target_raw_dir, setup_cfg=setup_cfg)

    print("=" * 72)
    print(" ESTIMATIVA DE ESPAÇO E PREPARAÇÃO DE DOWNLOAD — VIBNET-1D")
    print("=" * 72)
    print(f"Diretório Bruto (VIBNET_RAW_DIR) : {target_raw_dir}")
    print(f"Datasets Selecionados ({len(selected)})       : {', '.join(selected)}")
    print(f"Fontes Brutas Necessárias ({len(est['raw_sources_required'])})   : {', '.join(est['raw_sources_required'])}")
    print(f"Tamanho Bruto Estimado           : ~{est['estimated_raw_mb']} MB")
    print(f"Tamanho Processado Estimado      : ~{est['estimated_processed_mb']} MB")
    print(f"Espaço Livre no Disco Alvo       : {est['free_raw_dir_mb']} MB")
    if est["unknown_size_sources"]:
        print(
            f"[AVISO] Tamanho exato não catalogado para: {', '.join(est['unknown_size_sources'])}"
        )
    if not est["sufficient_space"]:
        print(
            "[ERRO] Espaço em disco insuficiente para o download e processamento estimados!",
            file=sys.stderr,
        )
        return 3

    if args.estimate_only:
        return 0

    manifest = download_datasets(
        datasets=selected,
        raw_dir=target_raw_dir,
        force=args.force,
        dry_run=False,
    )

    print("\nResumo do Download / Verificação:")
    for ds_name in selected:
        d_info = manifest["datasets"].get(ds_name, {})
        print(
            f"  - {ds_name:16s}: status={d_info.get('status')} | "
            f"válido={d_info.get('valid')} | {d_info.get('reason', '')}"
        )
    print(f"\nManifesto salvo em: {target_raw_dir / DOWNLOAD_MANIFEST_FILENAME}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

