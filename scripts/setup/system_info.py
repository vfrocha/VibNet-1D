#!/usr/bin/env python3
"""Inspeção de hardware, sistema operacional, espaço em disco e GPU/CUDA (`scripts/setup/system_info.py`).

Funciona tanto em ambiente Python mínimo (stdlib pura) quanto com `torch` instalado,
sem instalar nem modificar drivers NVIDIA do sistema.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
from typing import Any, Dict, List, Optional

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    PROCESSED_DATA_DIR,
    PROCESSED_TL_DIR,
    RAW_DATA_DIR,
    RESULTS_DIR,
)


def _detect_linux_distro() -> Dict[str, Optional[str]]:
    os_release = Path("/etc/os-release")
    info: Dict[str, Optional[str]] = {
        "distro_id": None,
        "distro_name": None,
        "distro_version": None,
    }
    if not os_release.exists():
        return info
    try:
        kv: Dict[str, str] = {}
        for line in os_release.read_text(encoding="utf-8", errors="ignore").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                kv[k.strip()] = v.strip().strip('"').strip("'")
        info["distro_id"] = kv.get("ID")
        info["distro_name"] = kv.get("PRETTY_NAME") or kv.get("NAME")
        info["distro_version"] = kv.get("VERSION_ID")
    except Exception:
        pass
    return info


def detect_cpu_and_ram() -> Dict[str, Any]:
    """Detecta modelo de CPU, núcleos lógicos e memória RAM total/disponível."""
    cpu_model = platform.processor() or "Unknown"
    logical_cores = os.cpu_count() or 1
    total_ram_gb: Optional[float] = None
    avail_ram_gb: Optional[float] = None

    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.exists():
        try:
            for line in cpuinfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                if "model name" in line and ":" in line:
                    cpu_model = line.split(":", 1)[1].strip()
                    break
        except Exception:
            pass

    meminfo = Path("/proc/meminfo")
    if meminfo.exists():
        try:
            mem_kv: Dict[str, int] = {}
            for line in meminfo.read_text(encoding="utf-8", errors="ignore").splitlines():
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    if parts[1].isdigit():
                        mem_kv[key] = int(parts[1])  # kB
            if "MemTotal" in mem_kv:
                total_ram_gb = round(mem_kv["MemTotal"] / (1024.0 * 1024.0), 2)
            if "MemAvailable" in mem_kv:
                avail_ram_gb = round(mem_kv["MemAvailable"] / (1024.0 * 1024.0), 2)
        except Exception:
            pass
    elif sys.platform == "win32":
        try:
            import ctypes

            class MEMORYSTATUSEX(ctypes.Structure):
                _fields_ = [
                    ("dwLength", ctypes.c_ulong),
                    ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong),
                    ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong),
                    ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong),
                    ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                ]

            stat = MEMORYSTATUSEX()
            stat.dwLength = ctypes.sizeof(stat)
            if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat)):
                total_ram_gb = round(stat.ullTotalPhys / (1024.0**3), 2)
                avail_ram_gb = round(stat.ullAvailPhys / (1024.0**3), 2)
        except Exception:
            pass

    return {
        "cpu_model": cpu_model,
        "logical_cores": int(logical_cores),
        "total_ram_gb": total_ram_gb,
        "available_ram_gb": avail_ram_gb,
    }


def _nearest_existing_parent(path: Path) -> Path:
    curr = path.resolve()
    while not curr.exists() and curr.parent != curr:
        curr = curr.parent
    return curr


def detect_disk_space(paths: Optional[Dict[str, Path]] = None) -> Dict[str, Dict[str, Any]]:
    """Inspeciona espaço total, usado e livre para os diretórios configurados."""
    target_paths = paths or {
        "project_root": PROJECT_ROOT,
        "raw_data_dir": RAW_DATA_DIR,
        "processed_data_dir": PROCESSED_DATA_DIR,
        "processed_tl_dir": PROCESSED_TL_DIR,
        "results_dir": RESULTS_DIR,
    }
    report: Dict[str, Dict[str, Any]] = {}
    for label, p in target_paths.items():
        probe = _nearest_existing_parent(Path(p))
        try:
            usage = shutil.disk_usage(probe)
            report[label] = {
                "configured_path": str(Path(p).resolve()),
                "existing_mount_probe": str(probe),
                "total_gb": round(usage.total / (1024.0**3), 2),
                "used_gb": round(usage.used / (1024.0**3), 2),
                "free_gb": round(usage.free / (1024.0**3), 2),
            }
        except Exception as exc:
            report[label] = {
                "configured_path": str(p),
                "existing_mount_probe": str(probe),
                "error": str(exc),
            }
    return report


def detect_nvidia_driver_and_gpus(
    nvidia_smi_cmd: str = "nvidia-smi",
) -> Dict[str, Any]:
    """Detecta GPUs NVIDIA, versão do driver e versão CUDA máxima reportada pelo driver."""
    smi_path = shutil.which(nvidia_smi_cmd)
    result: Dict[str, Any] = {
        "nvidia_smi_found": bool(smi_path),
        "nvidia_smi_path": smi_path,
        "driver_version": None,
        "driver_cuda_version": None,
        "gpus": [],
    }
    if not smi_path:
        return result

    try:
        q_proc = subprocess.run(
            [
                smi_path,
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if q_proc.returncode == 0:
            gpus: List[Dict[str, Any]] = []
            for line in q_proc.stdout.strip().splitlines():
                parts = [x.strip() for x in line.split(",")]
                if len(parts) >= 4:
                    result["driver_version"] = parts[2]
                    gpus.append(
                        {
                            "index": int(parts[0]) if parts[0].isdigit() else 0,
                            "name": parts[1],
                            "driver_version": parts[2],
                            "vram_mb": float(parts[3]) if parts[3].replace(".", "", 1).isdigit() else None,
                        }
                    )
            result["gpus"] = gpus

        hdr_proc = subprocess.run(
            [smi_path],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
        if hdr_proc.returncode == 0:
            m = re.search(r"CUDA Version:\s*([0-9]+\.[0-9]+)", hdr_proc.stdout)
            if m:
                result["driver_cuda_version"] = m.group(1)
    except Exception as exc:
        result["error"] = str(exc)

    return result


def recommend_pytorch_index_url(
    gpu_info: Dict[str, Any],
    force_cpu: bool = False,
) -> Dict[str, str]:
    """Recomenda o índice oficial de rodas do PyTorch compatível com o driver NVIDIA detectado."""
    if force_cpu or not gpu_info.get("nvidia_smi_found") or not gpu_info.get("gpus"):
        return {
            "compute_target": "cpu",
            "index_url": "https://download.pytorch.org/whl/cpu",
            "reason": (
                "Modo CPU solicitado explicitamente (--cpu-only)."
                if force_cpu
                else "Nenhuma GPU NVIDIA detectada via nvidia-smi; selecionando build CPU do PyTorch."
            ),
        }

    cuda_str = gpu_info.get("driver_cuda_version") or ""
    try:
        major_minor = float(cuda_str)
    except ValueError:
        major_minor = 0.0

    if major_minor >= 12.4:
        return {
            "compute_target": "cu124",
            "index_url": "https://download.pytorch.org/whl/cu124",
            "reason": f"GPU NVIDIA detectada com suporte de driver até CUDA {cuda_str} (>= 12.4).",
        }
    if major_minor >= 12.1:
        return {
            "compute_target": "cu121",
            "index_url": "https://download.pytorch.org/whl/cu121",
            "reason": f"GPU NVIDIA detectada com suporte de driver até CUDA {cuda_str} (>= 12.1).",
        }
    if major_minor >= 11.8:
        return {
            "compute_target": "cu118",
            "index_url": "https://download.pytorch.org/whl/cu118",
            "reason": f"GPU NVIDIA detectada com suporte de driver até CUDA {cuda_str} (>= 11.8).",
        }

    return {
        "compute_target": "cu121",
        "index_url": "https://download.pytorch.org/whl/cu121",
        "reason": "GPU NVIDIA detectada (versão CUDA do driver não identificada); usando build padrão cu121.",
    }


def verify_pytorch_runtime() -> Dict[str, Any]:
    """Executa teste funcional real do PyTorch em CPU e (quando disponível) em GPU CUDA (`x @ x`)."""
    info: Dict[str, Any] = {
        "torch_installed": False,
        "torch_version": None,
        "torch_cuda_compiled_version": None,
        "cuda_available": False,
        "device_count": 0,
        "device_name": None,
        "cpu_matmul_ok": False,
        "cuda_matmul_ok": False,
        "cuda_matmul_shape": None,
        "diagnostic": None,
    }
    try:
        import torch  # type: ignore

        info["torch_installed"] = True
        info["torch_version"] = str(torch.__version__)
        info["torch_cuda_compiled_version"] = getattr(torch.version, "cuda", None)

        x_cpu = torch.randn(100, 100, device="cpu")
        y_cpu = x_cpu @ x_cpu
        info["cpu_matmul_ok"] = tuple(y_cpu.shape) == (100, 100)

        cuda_avail = bool(torch.cuda.is_available())
        info["cuda_available"] = cuda_avail
        if cuda_avail:
            info["device_count"] = int(torch.cuda.device_count())
            info["device_name"] = str(torch.cuda.get_device_name(0))
            x_gpu = torch.randn(100, 100, device="cuda")
            y_gpu = x_gpu @ x_gpu
            torch.cuda.synchronize()
            info["cuda_matmul_shape"] = list(y_gpu.shape)
            info["cuda_matmul_ok"] = tuple(y_gpu.shape) == (100, 100)
            info["diagnostic"] = (
                f"CUDA ativo e verificado com multiplicação matricial 100x100 em {info['device_name']}."
            )
        else:
            info["diagnostic"] = (
                "PyTorch operando em modo CPU (torch.cuda.is_available() == False)."
            )
    except ImportError:
        info["diagnostic"] = "Pacote 'torch' ainda não está instalado no ambiente Python atual."
    except Exception as exc:
        info["diagnostic"] = f"Falha durante execução do teste funcional do PyTorch: {exc}"

    return info


def collect_system_info(force_cpu: bool = False) -> Dict[str, Any]:
    """Coleta o relatório completo de hardware, sistema operacional, diretórios e PyTorch."""
    in_venv = (
        hasattr(sys, "real_prefix")
        or (hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix)
        or bool(os.environ.get("VIRTUAL_ENV"))
    )
    gpu_info = detect_nvidia_driver_and_gpus()
    torch_info = verify_pytorch_runtime()
    pytorch_rec = recommend_pytorch_index_url(gpu_info, force_cpu=force_cpu)

    warnings: List[str] = []
    if (
        gpu_info.get("gpus")
        and torch_info.get("torch_installed")
        and not torch_info.get("cuda_available")
    ):
        warnings.append(
            "GPU NVIDIA detectada pelo sistema (nvidia-smi), mas torch.cuda.is_available() "
            "retornou False. Verifique se o PyTorch foi instalado com suporte CUDA compatível "
            f"(ex.: pip install --upgrade torch --index-url {pytorch_rec['index_url']})."
        )

    return {
        "os": {
            "system": platform.system(),
            "release": platform.release(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            **_detect_linux_distro(),
        },
        "python": {
            "version": platform.python_version(),
            "executable": sys.executable,
            "in_virtualenv": bool(in_venv),
            "virtual_env_path": os.environ.get("VIRTUAL_ENV") or (sys.prefix if in_venv else None),
        },
        "hardware": detect_cpu_and_ram(),
        "gpu_driver": gpu_info,
        "pytorch_recommendation": pytorch_rec,
        "pytorch_runtime": torch_info,
        "disks": detect_disk_space(),
        "warnings": warnings,
    }


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Inspeciona hardware, SO, disco, GPU NVIDIA e disponibilidade de CUDA."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Imprime a saída exclusivamente em formato JSON.",
    )
    parser.add_argument(
        "--recommend-torch-index",
        action="store_true",
        help="Imprime apenas a URL do índice de pacotes PyTorch recomendada para a máquina.",
    )
    parser.add_argument(
        "--cpu-only",
        action="store_true",
        help="Força recomendação do índice CPU do PyTorch.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Salva o relatório JSON no caminho indicado.",
    )
    args = parser.parse_args(argv)

    info = collect_system_info(force_cpu=args.cpu_only)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    if args.recommend_torch_index:
        print(info["pytorch_recommendation"]["index_url"])
        return 0

    if args.json:
        print(json.dumps(info, indent=2, ensure_ascii=False))
        return 0

    print("=" * 72)
    print(" DIAGNÓSTICO DE SISTEMA E HARDWARE — VIBNET-1D")
    print("=" * 72)
    os_info = info["os"]
    py_info = info["python"]
    hw_info = info["hardware"]
    gpu_info = info["gpu_driver"]
    pt_info = info["pytorch_runtime"]

    print(f"Sistema Operacional : {os_info['platform']} ({os_info.get('distro_name') or os_info['system']})")
    print(f"Python              : {py_info['version']} ({py_info['executable']})")
    print(f"Ambiente Virtual    : {'Sim' if py_info['in_virtualenv'] else 'Não'} ({py_info['virtual_env_path']})")
    print(f"CPU                 : {hw_info['cpu_model']} ({hw_info['logical_cores']} núcleos lógicos)")
    print(f"Memória RAM         : Total={hw_info['total_ram_gb']} GB | Disponível={hw_info['available_ram_gb']} GB")

    if gpu_info["gpus"]:
        for g in gpu_info["gpus"]:
            print(
                f"GPU NVIDIA [{g['index']}]     : {g['name']} | Driver={g['driver_version']} | "
                f"CUDA Driver={gpu_info['driver_cuda_version']} | VRAM={g['vram_mb']} MB"
            )
    else:
        print("GPU NVIDIA          : Nenhuma GPU detectada via nvidia-smi")

    print(
        f"PyTorch Runtime     : instalado={pt_info['torch_installed']} | "
        f"versão={pt_info['torch_version']} | cuda_available={pt_info['cuda_available']}"
    )
    if pt_info[" diagnostic" if " diagnostic" in pt_info else "diagnostic"]:
        print(f"Diagnóstico PyTorch : {pt_info['diagnostic']}")

    print("\nEspaço em Disco nos Diretórios Configurados:")
    for key, d in info["disks"].items():
        if "free_gb" in d:
            print(f"  - {key:20s}: {d['configured_path']} (Livre: {d['free_gb']} GB / Total: {d['total_gb']} GB)")
        else:
            print(f"  - {key:20s}: {d['configured_path']} (Erro: {d.get('error')})")

    if info["warnings"]:
        print("\nAvisos:")
        for w in info["warnings"]:
            print(f"  [AVISO] {w}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

