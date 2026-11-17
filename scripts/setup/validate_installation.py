#!/usr/bin/env python3
"""Validação final automatizada da instalação e geração de relatório diagnóstico (`scripts/setup/validate_installation.py`).

Executa as 9 verificações obrigatórias da Seção 9:
  1. Ambiente Python ativo e isolamento (.venv).
  2. Importação dos módulos principais do projeto.
  3. Versões das dependências instaladas (obrigatórias e opcionais).
  4. Disponibilidade e teste funcional real de CPU e GPU/CUDA (`x @ x`).
  5. Acesso de leitura/escrita aos diretórios configurados (`VIBNET_*_DIR`).
  6. Inventário dos datasets baixados (`download_manifest.json` + pastas brutas).
  7. Integridade dos dados processados e variantes (`processed_1d`, `raw`, `detrended`, `raw_rms`, `detrended_rms`).
  8. Execução de testes unitários relevantes do projeto.
  9. Verificação de importação/compilação sintática dos scripts de experimentos.

Gera:
  - `results/setup/installation_report.json`
  - `results/setup/installation_report.md`
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import py_compile
import subprocess
import sys
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.setup.download_datasets import (
    get_dataset_to_raw_source_map,
    load_setup_config,
    verify_existing_raw_source,
)
from scripts.setup.prepare_datasets import (
    SUPPORTED_VARIANTS,
    inspect_existing_processed_variant,
    resolve_variant_dir,
)
from scripts.setup.system_info import collect_system_info
from src.config import (
    PROCESSED_DATA_DIR,
    PROCESSED_TL_DIR,
    RAW_DATA_DIR,
    RESULTS_DIR,
)
from src.vibnet.data.registry import CANONICAL_DATASETS, list_all_datasets, load_yaml_config


REQUIRED_PACKAGES_BY_GROUP: Dict[str, List[str]] = {
    "scientific_base": ["numpy", "scipy", "pandas", "scikit-learn"],
    "signal_processing": ["PyWavelets"],
    "deep_learning": ["torch", "pytorch-tabnet"],
    "classical_ml": ["xgboost", "imbalanced-learn"],
    "download_and_utils": ["gdown", "tqdm", "PyYAML", "rarfile"],
    "visualization": ["matplotlib", "seaborn"],
    "vibration_ecosystem": ["vibdata"],
}

CORE_PROJECT_MODULES: List[str] = [
    "src.config",
    "src.vibnet.data.registry",
    "src.vibnet.evaluation.provenance",
    "src.data.make_dataset",
    "src.data.make_dataset_tl",
    "src.data.validate_dataset_tl",
    "src.data.transforms_1d",
    "src.data.time_resampler",
    "src.preprocessing.normalize_dataset_tl",
    "src.models.resnet1d_time",
    "src.models.build_1d_time",
]

EXPERIMENT_SCRIPTS_TO_CHECK: List[str] = [
    "src/experiments/run_baseline_loco_v2.py",
    "src/experiments/run_scratch_vs_tl_benchmark.py",
    "src/experiments/run_scratch_vs_tl_classical_baselines.py",
    "src/experiments/analyze_scratch_tl_classical_comparison.py",
    "src/experiments/run_tabvibnet_linear_probing.py",
    "src/experiments/run_time_scratch_ablation.py",
    "scripts/exp_tl_cross_dataset_mlp_tabnet.py",
    "scripts/exp_tl_contrastive_mlp.py",
    "scripts/exp_tl_balance_strategies.py",
    "scripts/validate_dataset_tl.py",
]


def check_dependencies() -> Dict[str, Any]:
    """Verifica versões instaladas de todas as dependências agrupadas."""
    installed: Dict[str, Optional[str]] = {}
    missing_mandatory: List[str] = []
    missing_optional: List[str] = []
    by_group: Dict[str, Dict[str, Optional[str]]] = {}

    optional_groups = {"visualization", "vibration_ecosystem"}

    for group, pkgs in REQUIRED_PACKAGES_BY_GROUP.items():
        by_group[group] = {}
        for pkg in pkgs:
            ver: Optional[str] = None
            try:
                ver = importlib.metadata.version(pkg)
            except importlib.metadata.PackageNotFoundError:
                # Fallback via import direto
                mod_map = {
                    "scikit-learn": "sklearn",
                    "imbalanced-learn": "imblearn",
                    "pytorch-tabnet": "pytorch_tabnet",
                    "PyWavelets": "pywt",
                    "PyYAML": "yaml",
                }
                mod_name = mod_map.get(pkg, pkg)
                try:
                    mod = importlib.import_module(mod_name)
                    ver = str(getattr(mod, "__version__", "installed"))
                except Exception:
                    ver = None
            installed[pkg] = ver
            by_group[group][pkg] = ver
            if ver is None:
                if group in optional_groups or pkg in ("rarfile", "PyYAML"):
                    missing_optional.append(pkg)
                else:
                    missing_mandatory.append(pkg)

    return {
        "all_mandatory_installed": len(missing_mandatory) == 0,
        "missing_mandatory": missing_mandatory,
        "missing_optional": missing_optional,
        "by_group": by_group,
        "flat_versions": installed,
    }


def check_core_module_imports() -> Dict[str, Any]:
    """Tenta importar os módulos centrais do VibNet-1D."""
    results: Dict[str, Dict[str, Any]] = {}
    failed_mandatory: List[str] = []

    for mod_name in CORE_PROJECT_MODULES:
        try:
            importlib.import_module(mod_name)
            results[mod_name] = {"imported": True, "error": None}
        except Exception as exc:
            err_str = f"{type(exc).__name__}: {exc}"
            results[mod_name] = {"imported": False, "error": err_str}
            # make_dataset / make_dataset_tl importam vibdata no topo; se vibdata for opcional, anota aviso
            if "vibdata" in err_str and mod_name in ("src.data.make_dataset", "src.data.make_dataset_tl"):
                results[mod_name]["note"] = "Requer pacote 'vibdata' instalado."
            else:
                failed_mandatory.append(mod_name)

    return {
        "all_core_imported": len(failed_mandatory) == 0,
        "failed_modules": failed_mandatory,
        "modules": results,
    }


def check_configured_directories(
    raw_dir: Path,
    processed_dir: Path,
    processed_tl_dir: Path,
    results_dir: Path,
) -> Dict[str, Any]:
    """Verifica acesso de criação, leitura e escrita nos 4 diretórios configurados."""
    dirs = {
        "VIBNET_RAW_DIR": raw_dir,
        "VIBNET_PROCESSED_DIR": processed_dir,
        "VIBNET_PROCESSED_TL_DIR": processed_tl_dir,
        "VIBNET_RESULTS_DIR": results_dir,
    }
    status: Dict[str, Dict[str, Any]] = {}
    all_ok = True

    for env_name, d_path in dirs.items():
        resolved = Path(d_path).expanduser().resolve()
        try:
            resolved.mkdir(parents=True, exist_ok=True)
            probe_file = resolved / ".vibnet_write_probe"
            probe_file.write_text("ok", encoding="utf-8")
            probe_file.unlink()
            status[env_name] = {
                "path": str(resolved),
                "exists": True,
                "writable": True,
                "error": None,
            }
        except Exception as exc:
            all_ok = False
            status[env_name] = {
                "path": str(resolved),
                "exists": resolved.exists(),
                "writable": False,
                "error": str(exc),
            }

    return {
        "all_accessible": all_ok,
        "directories": status,
    }


def inventory_datasets(
    raw_dir: Path,
    processed_dir: Path,
    processed_tl_dir: Path,
) -> Dict[str, Any]:
    """Inventaria todos os 10 datasets canônicos (brutos e variantes processadas)."""
    setup_cfg = load_setup_config()
    raw_sources_cfg = setup_cfg.get("raw_sources", {})
    ds_to_raw = get_dataset_to_raw_source_map(setup_cfg)
    datasets_yaml = load_yaml_config(PROJECT_ROOT / "configs" / "datasets.yaml").get("datasets", {})

    raw_inventory: Dict[str, Any] = {}
    processed_inventory: Dict[str, Dict[str, Any]] = {}
    available_variants_summary: Dict[str, List[str]] = {v: [] for v in SUPPORTED_VARIANTS}

    for ds_name in list_all_datasets():
        r_src = ds_to_raw[ds_name]
        s_meta = raw_sources_cfg.get(r_src, {})
        r_check = verify_existing_raw_source(r_src, raw_dir, s_meta)
        raw_inventory[ds_name] = {
            "raw_source": r_src,
            "present": bool(r_check["valid"]),
            "status": r_check["status"],
            "detail": r_check["reason"],
        }

        fs_hz = int(
            datasets_yaml.get(ds_name, {}).get(
                "window_length", CANONICAL_DATASETS[ds_name].sampling_rate_hz
            )
        )
        processed_inventory[ds_name] = {}
        for var in SUPPORTED_VARIANTS:
            v_dir = resolve_variant_dir(ds_name, var, processed_dir, processed_tl_dir)
            v_check = inspect_existing_processed_variant(ds_name, var, v_dir, fs_hz)
            processed_inventory[ds_name][var] = v_check
            if v_check["valid"]:
                available_variants_summary[var].append(ds_name)

    return {
        "canonical_datasets": list_all_datasets(),
        "raw_inventory": raw_inventory,
        "processed_inventory": processed_inventory,
        "available_variants_summary": available_variants_summary,
        "downloaded_datasets": [
            ds for ds, info in raw_inventory.items() if info["present"]
        ],
        "processed_datasets_any_variant": [
            ds
            for ds, vmap in processed_inventory.items()
            if any(v_info["valid"] for v_info in vmap.values())
        ],
    }


def check_experiment_scripts() -> Dict[str, Any]:
    """Verifica que todos os scripts principais de experimento compilam sem erros de sintaxe."""
    script_status: Dict[str, Dict[str, Any]] = {}
    all_ok = True
    for rel_path in EXPERIMENT_SCRIPTS_TO_CHECK:
        fp = PROJECT_ROOT / rel_path
        if not fp.is_file():
            all_ok = False
            script_status[rel_path] = {"valid": False, "error": "Arquivo não encontrado"}
            continue
        try:
            py_compile.compile(str(fp), doraise=True)
            script_status[rel_path] = {"valid": True, "error": None}
        except Exception as exc:
            all_ok = False
            script_status[rel_path] = {"valid": False, "error": str(exc)}

    return {
        "all_scripts_valid": all_ok,
        "scripts": script_status,
    }


def run_relevant_unit_tests(run_tests: bool = True) -> Dict[str, Any]:
    """Executa os testes unitários rápidos de verificação da infraestrutura."""
    if not run_tests:
        return {
            "executed": False,
            "passed": True,
            "detail": "Execução de testes unitários desativada (--skip-tests).",
        }

    test_files = [
        "tests/test_setup_installer.py",
        "tests/test_vibnet_modular_architecture.py",
    ]

    results: Dict[str, Any] = {}
    all_passed = True
    sub_env = dict(os.environ)
    sub_env["PYTHONPATH"] = str(PROJECT_ROOT)

    for tf in test_files:
        full_tf = PROJECT_ROOT / tf
        if not full_tf.is_file():
            continue
        cmd = [sys.executable, tf]
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            env=sub_env,
            capture_output=True,
            text=True,
            timeout=120,
            check=False,
        )
        passed = proc.returncode == 0
        if not passed:
            all_passed = False
        results[tf] = {
            "passed": passed,
            "returncode": proc.returncode,
            "output_tail": (proc.stdout + "\n" + proc.stderr).strip()[-600:],
        }

    return {
        "executed": True,
        "passed": all_passed,
        "suites": results,
    }


def format_markdown_report(report: Mapping[str, Any]) -> str:
    """Formata o relatório de diagnóstico da instalação em Markdown (`installation_report.md`)."""
    sys_info = report["system_info"]
    os_info = sys_info["os"]
    py_info = sys_info["python"]
    hw_info = sys_info["hardware"]
    gpu_info = sys_info["gpu_driver"]
    pt_info = sys_info["pytorch_runtime"]
    deps = report["dependencies"]
    dirs = report["directories"]["directories"]
    inv = report["datasets_inventory"]
    tests = report["unit_tests"]
    scripts = report["experiment_scripts"]

    lines: List[str] = [
        "# Relatório de Diagnóstico da Instalação — VibNet-1D",
        "",
        f"- **Data/Hora (UTC)**: `{report['timestamp_utc']}`",
        f"- **Status Geral do Ambiente**: **`{'APROVADO (OK)' if report['overall_environment_ok'] else 'PENDÊNCIAS DETECTADAS'}`**",
        "",
        "---",
        "",
        "## 1. Sistema Operacional, Python e Hardware",
        "",
        f"- **Sistema Operacional**: `{os_info['platform']}` (`{os_info.get('distro_name') or os_info['system']}`)",
        f"- **Versão do Python**: `{py_info['version']}` (`{py_info['executable']}`)",
        f"- **Ambiente Virtual Isolado (`.venv`)**: `{'Sim' if py_info['in_virtualenv'] else 'Não'}` (`{py_info.get('virtual_env_path')}`)",
        f"- **Processador (CPU)**: `{hw_info['cpu_model']}` (`{hw_info['logical_cores']}` núcleos lógicos)",
        f"- **Memória RAM**: Total `{hw_info['total_ram_gb']} GB` | Disponível `{hw_info['available_ram_gb']} GB`",
        "",
        "## 2. Aceleração por GPU e Runtime PyTorch / CUDA",
        "",
    ]

    if gpu_info.get("gpus"):
        for g in gpu_info["gpus"]:
            lines.append(
                f"- **GPU Detectada [{g['index']}]**: `{g['name']}` | Driver NVIDIA `{g['driver_version']}` | "
                f"CUDA máx. do driver `{gpu_info.get('driver_cuda_version')}` | VRAM `{g['vram_mb']} MB`"
            )
    else:
        lines.append("- **GPU NVIDIA**: Nenhuma GPU detectada via `nvidia-smi` (modo CPU).")

    lines.extend(
        [
            f"- **PyTorch Instalado**: `{pt_info['torch_installed']}` (versão `{pt_info['torch_version']}`, CUDA compilado `{pt_info['torch_cuda_compiled_version']}`)",
            f"- **`torch.cuda.is_available()`**: `{pt_info['cuda_available']}`",
            f"- **Teste Multiplicação Matricial CPU (`100x100`)**: `{'OK' if pt_info['cpu_matmul_ok'] else 'FALHOU'}`",
            f"- **Teste Multiplicação Matricial CUDA (`100x100`)**: `{'OK' if pt_info['cuda_matmul_ok'] else 'N/A (CPU)'}`",
            f"- **Diagnóstico PyTorch**: {pt_info['diagnostic']}",
            "",
            "## 3. Diretórios Configurados",
            "",
            "| Variável | Caminho Resolvido | Gravável |",
            "|---|---|:---:|",
        ]
    )
    for k, d in dirs.items():
        lines.append(f"| `{k}` | `{d['path']}` | `{'Sim' if d['writable'] else 'Não'}` |")

    lines.extend(
        [
            "",
            "## 4. Dependências Instaladas por Grupo",
            "",
            "| Grupo | Pacote | Versão Instalada |",
            "|---|---|---|",
        ]
    )
    for grp, pkg_map in deps["by_group"].items():
        for pkg, ver in pkg_map.items():
            lines.append(f"| `{grp}` | `{pkg}` | `{ver or 'AUSENTE'}` |")

    lines.extend(
        [
            "",
            "## 5. Inventário de Datasets (Brutos e Processados)",
            "",
            f"- **Datasets Canônicos no Registro ({len(inv['canonical_datasets'])})**: `{', '.join(inv['canonical_datasets'])}`",
            f"- **Datasets Brutos Presentes ({len(inv['downloaded_datasets'])})**: `{', '.join(inv['downloaded_datasets']) or 'Nenhum'}`",
            f"- **Datasets Processados em Alguma Variante ({len(inv['processed_datasets_any_variant'])})**: `{', '.join(inv['processed_datasets_any_variant']) or 'Nenhum'}`",
            "",
            "| Dataset | Bruto Presente | `processed_1d` | `raw` | `detrended` | `raw_rms` | `detrended_rms` |",
            "|---|:---:|:---:|:---:|:---:|:---:|:---:|",
        ]
    )
    for ds_name in inv["canonical_datasets"]:
        r_pres = "Sim" if inv["raw_inventory"][ds_name]["present"] else "Não"
        p_map = inv["processed_inventory"][ds_name]
        cols = [
            f"{p_map[v]['window_count']} win" if p_map[v]["valid"] else "—"
            for v in SUPPORTED_VARIANTS
        ]
        lines.append(
            f"| `{ds_name}` | {r_pres} | {cols[0]} | {cols[1]} | {cols[2]} | {cols[3]} | {cols[4]} |"
        )

    lines.extend(
        [
            "",
            "## 6. Verificação de Scripts e Testes Automatizados",
            "",
            f"- **Compilação Sintática dos Scripts de Experimento**: `{'OK' if scripts['all_scripts_valid'] else 'FALHOU'}`",
            f"- **Suítes de Testes Unitários Executadas**: `{'OK' if tests['passed'] else 'FALHOU'}`",
            "",
            "## 7. Avisos e Recomendações",
            "",
        ]
    )
    if report["warnings"]:
        for w in report["warnings"]:
            lines.append(f"- **Aviso**: {w}")
    else:
        lines.append("- Nenhum aviso registrado.")

    if report["failures"]:
        lines.append("")
        lines.append("## 8. Falhas Críticas Detectadas")
        lines.append("")
        for f_msg in report["failures"]:
            lines.append(f"- **FALHA**: {f_msg}")

    return "\n".join(lines) + "\n"


def validate_installation(
    raw_dir: Optional[Path] = None,
    processed_dir: Optional[Path] = None,
    processed_tl_dir: Optional[Path] = None,
    results_dir: Optional[Path] = None,
    report_dir: Optional[Path] = None,
    run_tests: bool = True,
) -> Dict[str, Any]:
    """Executa todas as verificações e persiste `installation_report.json` e `installation_report.md`."""
    res_raw = Path(raw_dir or RAW_DATA_DIR).expanduser().resolve()
    res_proc = Path(processed_dir or PROCESSED_DATA_DIR).expanduser().resolve()
    res_proc_tl = Path(processed_tl_dir or PROCESSED_TL_DIR).expanduser().resolve()
    res_results = Path(results_dir or RESULTS_DIR).expanduser().resolve()
    out_report_dir = Path(report_dir or (res_results / "setup")).expanduser().resolve()
    out_report_dir.mkdir(parents=True, exist_ok=True)

    sys_info = collect_system_info()
    deps_info = check_dependencies()
    mods_info = check_core_module_imports()
    dirs_info = check_configured_directories(res_raw, res_proc, res_proc_tl, res_results)
    inv_info = inventory_datasets(res_raw, res_proc, res_proc_tl)
    scripts_info = check_experiment_scripts()
    tests_info = run_relevant_unit_tests(run_tests=run_tests)

    failures: List[str] = []
    warnings: List[str] = list(sys_info.get("warnings", []))

    if not deps_info["all_mandatory_installed"]:
        failures.append(
            f"Dependências obrigatórias ausentes: {deps_info['missing_mandatory']}"
        )
    if deps_info["missing_optional"]:
        warnings.append(
            f"Dependências opcionais/ecossistema ausentes: {deps_info['missing_optional']}"
        )
    if not mods_info["all_core_imported"]:
        failures.append(
            f"Falha ao importar módulos centrais: {mods_info['failed_modules']}"
        )
    if not dirs_info["all_accessible"]:
        failures.append("Um ou mais diretórios configurados não possuem permissão de escrita.")
    if not scripts_info["all_scripts_valid"]:
        failures.append("Um ou mais scripts de experimento falharam na compilação sintática.")
    if not tests_info["passed"]:
        failures.append("Uma ou mais suítes de testes unitários falharam.")
    if not sys_info["python"]["in_virtualenv"]:
        warnings.append(
            "O Python ativo não está em um ambiente virtual isolado (.venv)."
        )

    overall_ok = len(failures) == 0

    report_payload: Dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "overall_environment_ok": overall_ok,
        "system_info": sys_info,
        "dependencies": deps_info,
        "core_modules": mods_info,
        "directories": dirs_info,
        "datasets_inventory": inv_info,
        "experiment_scripts": scripts_info,
        "unit_tests": tests_info,
        "warnings": warnings,
        "failures": failures,
        "report_paths": {
            "json": str(out_report_dir / "installation_report.json"),
            "markdown": str(out_report_dir / "installation_report.md"),
        },
    }

    json_path = out_report_dir / "installation_report.json"
    md_path = out_report_dir / "installation_report.md"

    json_path.write_text(
        json.dumps(report_payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    md_path.write_text(format_markdown_report(report_payload), encoding="utf-8")

    return report_payload


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Valida a instalação do VibNet-1D e gera installation_report.{json,md}."
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=None,
        help="Diretório de dados brutos (default: VIBNET_RAW_DIR).",
    )
    parser.add_argument(
        "--processed-dir",
        type=Path,
        default=None,
        help="Diretório de dados processados 1D (default: VIBNET_PROCESSED_DIR).",
    )
    parser.add_argument(
        "--processed-tl-dir",
        type=Path,
        default=None,
        help="Diretório de dados processados TL (default: VIBNET_PROCESSED_TL_DIR).",
    )
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=None,
        help="Diretório de resultados (default: VIBNET_RESULTS_DIR).",
    )
    parser.add_argument(
        "--report-dir",
        type=Path,
        default=None,
        help="Diretório onde salvar installation_report.{json,md} (default: <RESULTS_DIR>/setup).",
    )
    parser.add_argument(
        "--skip-tests",
        action="store_true",
        help="Pula a execução das suítes de testes unitários.",
    )
    args = parser.parse_args(argv)

    report = validate_installation(
        raw_dir=args.raw_dir,
        processed_dir=args.processed_dir,
        processed_tl_dir=args.processed_tl_dir,
        results_dir=args.results_dir,
        report_dir=args.report_dir,
        run_tests=not args.skip_tests,
    )

    print("=" * 72)
    print(" VALIDAÇÃO DA INSTALAÇÃO — VIBNET-1D")
    print("=" * 72)
    print(f"Status Geral do Ambiente : {'APROVADO (OK)' if report['overall_environment_ok'] else 'COM FALHAS'}")
    print(f"Relatório JSON           : {report['report_paths']['json']}")
    print(f"Relatório Markdown       : {report['report_paths']['markdown']}")
    if report["failures"]:
        for f_msg in report["failures"]:
            print(f"  [FALHA] {f_msg}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
