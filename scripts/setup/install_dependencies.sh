#!/usr/bin/env bash
# ==============================================================================
# VibNet-1D — Instalador de Ambiente Virtual e Dependências Python (Linux)
# ==============================================================================
# Cria/atualiza o ambiente virtual isolado (.venv), seleciona automaticamente a
# versão compatível do PyTorch (CUDA vs CPU) sem tocar nos drivers NVIDIA do
# sistema, instala os grupos de dependências e o pacote vibnet1d em modo editável.
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VENV_DIR="${VIBNET_VENV_DIR:-${PROJECT_ROOT}/.venv}"
PYTHON_BIN="${VIBNET_PYTHON_BIN:-python3}"
CPU_ONLY=0
UPDATE_ENV=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --venv-dir)
      VENV_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --cpu-only)
      CPU_ONLY=1
      shift
      ;;
    --update)
      UPDATE_ENV=1
      shift
      ;;
    *)
      echo "[ERRO] Opção desconhecida em install_dependencies.sh: $1" >&2
      exit 2
      ;;
  esac
done

echo "========================================================================"
echo " [1/4] Verificando pré-requisitos do sistema (Linux)"
echo "========================================================================"

if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  echo "[ERRO] Interpretador '${PYTHON_BIN}' não encontrado no PATH." >&2
  echo "       No Ubuntu 22.04/24.04, instale com: sudo apt-get update && sudo apt-get install -y python3 python3-venv python3-pip" >&2
  exit 1
fi

PY_VER="$("${PYTHON_BIN}" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
PY_OK="$("${PYTHON_BIN}" -c 'import sys; print(int(sys.version_info >= (3, 9)))')"
if [[ "${PY_OK}" != "1" ]]; then
  echo "[ERRO] Python >= 3.9 é necessário (encontrado: ${PY_VER})." >&2
  exit 1
fi
echo "  -> Python detectado : ${PYTHON_BIN} (versão ${PY_VER})"

if ! "${PYTHON_BIN}" -c "import venv" >/dev/null 2>&1; then
  echo "[ERRO] O módulo 'venv' do Python não está disponível." >&2
  echo "       No Ubuntu/Debian, instale com: sudo apt-get install -y python3-venv" >&2
  exit 1
fi

for tool in git; do
  if ! command -v "${tool}" >/dev/null 2>&1; then
    echo "[AVISO] Ferramenta '${tool}' não encontrada no PATH (necessária para instalar o fork git da vibdata)."
  fi
done

echo "========================================================================"
echo " [2/4] Configurando ambiente virtual isolado: ${VENV_DIR}"
echo "========================================================================"

if [[ ! -f "${VENV_DIR}/bin/python" ]]; then
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  echo "  -> Ambiente virtual criado em: ${VENV_DIR}"
else
  echo "  -> Ambiente virtual existente reutilizado: ${VENV_DIR}"
fi

VENV_PY="${VENV_DIR}/bin/python"
VENV_PIP="${VENV_DIR}/bin/pip"

if [[ "${UPDATE_ENV}" -eq 1 ]]; then
  "${VENV_PY}" -m pip install --upgrade pip setuptools wheel
else
  "${VENV_PY}" -m pip install --quiet --upgrade pip setuptools wheel
fi

echo "========================================================================"
echo " [3/4] Detectando CPU / GPU NVIDIA e instalando PyTorch compatível"
echo "========================================================================"

SYS_FLAGS=()
if [[ "${CPU_ONLY}" -eq 1 ]]; then
  SYS_FLAGS+=("--cpu-only")
fi

TORCH_INDEX_URL="$("${VENV_PY}" "${SCRIPT_DIR}/system_info.py" --recommend-torch-index "${SYS_FLAGS[@]}")"
echo "  -> Índice PyTorch selecionado: ${TORCH_INDEX_URL}"

NEED_TORCH_INSTALL=0
if ! "${VENV_PY}" -c "import torch" >/dev/null 2>&1; then
  NEED_TORCH_INSTALL=1
elif [[ "${UPDATE_ENV}" -eq 1 ]]; then
  NEED_TORCH_INSTALL=1
fi

if [[ "${NEED_TORCH_INSTALL}" -eq 1 ]]; then
  PIP_UPGRADE_FLAG=()
  if [[ "${UPDATE_ENV}" -eq 1 ]]; then
    PIP_UPGRADE_FLAG+=("--upgrade")
  fi
  "${VENV_PIP}" install "${PIP_UPGRADE_FLAG[@]}" torch --index-url "${TORCH_INDEX_URL}"
else
  echo "  -> PyTorch já instalado no ambiente virtual (use --update para atualizar)."
fi

echo "========================================================================"
echo " [4/4] Instalando grupos de dependências e pacote VibNet-1D (-e .)"
echo "========================================================================"

PIP_FLAGS=()
if [[ "${UPDATE_ENV}" -eq 1 ]]; then
  PIP_FLAGS+=("--upgrade")
fi

# Instala o projeto em modo editável junto com todos os grupos declarados no pyproject.toml
"${VENV_PIP}" install "${PIP_FLAGS[@]}" -e "${PROJECT_ROOT}[scientific,signal,dl,classical,download,viz,dev]"

# Instala o fork oficial da vibdata caso ainda não esteja presente ou em modo --update
if ! "${VENV_PY}" -c "import vibdata" >/dev/null 2>&1 || [[ "${UPDATE_ENV}" -eq 1 ]]; then
  if command -v git >/dev/null 2>&1; then
    echo "  -> Instalando fork customizado da vibdata (git+https://github.com/vfrocha/vibdata.git)..."
    if ! "${VENV_PIP}" install "${PIP_FLAGS[@]}" "git+https://github.com/vfrocha/vibdata.git"; then
      echo "  [AVISO] Não foi possível instalar o fork git da vibdata; tentando pacote PyPI padrão..."
      "${VENV_PIP}" install "${PIP_FLAGS[@]}" vibdata || echo "  [AVISO] Falha ao instalar vibdata (verifique conectividade de rede)."
    fi
  fi
else
  echo "  -> Biblioteca vibdata já instalada no ambiente virtual."
fi

echo "  -> Verificando runtime PyTorch / CUDA no ambiente virtual:"
"${VENV_PY}" -c '
import torch
print("     torch.__version__       :", torch.__version__)
print("     torch.cuda.is_available :", torch.cuda.is_available())
if torch.cuda.is_available():
    print("     GPU Device 0            :", torch.cuda.get_device_name(0))
    x = torch.randn(100, 100, device="cuda")
    print("     Teste MatMul CUDA (x@x) :", tuple((x @ x).shape))
'

