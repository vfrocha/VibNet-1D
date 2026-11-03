"""Gerenciador central de configurações e caminhos do VibNet-1D.

Centraliza a resolução de diretórios do projeto utilizando `pathlib.Path` e
variáveis de ambiente para garantir portabilidade entre servidores Linux
(Andromeda/Hydra), máquinas locais e Google Colab:

- `VIBNET_RAW_DIR`: Diretório dos datasets brutos (`vibdata`).
  Fallback: `/home/vfrocha/VibNet_Project/raw_data` (se existir no servidor)
  ou `<raiz_do_repo>/data/raw`.
- `VIBNET_PROCESSED_DIR`: Diretório das janelas 1D processadas (`.npy`).
  Fallback: `<raiz_do_repo>/data/processed`.
- `VIBNET_PROCESSED_TL_DIR`: Diretório das variantes de Transfer Learning.
  Fallback: `<raiz_do_repo>/data/processed_tl`.
- `VIBNET_RESULTS_DIR`: Diretório de saída de logs, métricas e figuras.
  Fallback: `<raiz_do_repo>/results`.
"""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent

_LEGACY_SERVER_RAW_DIR: Path = Path("/home/vfrocha/VibNet_Project/raw_data")


def _resolve_raw_data_dir() -> Path:
    env_raw = os.environ.get("VIBNET_RAW_DIR")
    if env_raw:
        return Path(env_raw).expanduser().resolve()
    if _LEGACY_SERVER_RAW_DIR.exists():
        return _LEGACY_SERVER_RAW_DIR
    return PROJECT_ROOT / "data" / "raw"


def _resolve_env_or_default(env_var: str, default_path: Path) -> Path:
    env_val = os.environ.get(env_var)
    if env_val:
        return Path(env_val).expanduser().resolve()
    return default_path


RAW_DATA_DIR: Path = _resolve_raw_data_dir()
PROCESSED_DATA_DIR: Path = _resolve_env_or_default(
    "VIBNET_PROCESSED_DIR", PROJECT_ROOT / "data" / "processed"
)
DATA_ROOT: Path = PROCESSED_DATA_DIR
PROCESSED_TL_DIR: Path = _resolve_env_or_default(
    "VIBNET_PROCESSED_TL_DIR", PROJECT_ROOT / "data" / "processed_tl"
)
RESULTS_DIR: Path = _resolve_env_or_default(
    "VIBNET_RESULTS_DIR", PROJECT_ROOT / "results"
)

__all__ = [
    "PROJECT_ROOT",
    "RAW_DATA_DIR",
    "PROCESSED_DATA_DIR",
    "DATA_ROOT",
    "PROCESSED_TL_DIR",
    "RESULTS_DIR",
]

