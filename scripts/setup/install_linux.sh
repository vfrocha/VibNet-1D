#!/usr/bin/env bash
# ==============================================================================
# VibNet-1D — Instalador Reproduzível Principal para Linux (`install_linux.sh`)
# ==============================================================================
# Uso padrão (instala ambiente + baixa e prepara todos os datasets disponíveis):
#   bash scripts/setup/install_linux.sh
#
# Exemplos adicionais:
#   bash scripts/setup/install_linux.sh --datasets CWRU_12k MFPT CWRU_48k
#   bash scripts/setup/install_linux.sh --skip-datasets
#   bash scripts/setup/install_linux.sh --download-only
#   bash scripts/setup/install_linux.sh --verify-only
#   bash scripts/setup/install_linux.sh --update
# ==============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

VENV_DIR="${VIBNET_VENV_DIR:-${PROJECT_ROOT}/.venv}"
PYTHON_BIN="${VIBNET_PYTHON_BIN:-python3}"

SKIP_DATASETS=0
DOWNLOAD_ONLY=0
VERIFY_ONLY=0
UPDATE_ENV=0
CPU_ONLY=0
FORCE_DATA=0
SKIP_TESTS=0

DATASETS=()
VARIANTS=("all")

usage() {
  cat <<EOF
Uso: bash scripts/setup/install_linux.sh [OPÇÕES]

Opções principais:
  --datasets <DS1> [DS2 ...]   Seleciona os datasets canônicos a baixar e preparar
                               (default: todos os 10 datasets canônicos).
  --variants <V1> [V2 ...]     Seleciona as variantes a preparar:
                               processed_1d, raw, detrended, raw_rms, detrended_rms, all
                               (default: all).
  --skip-datasets              Instala e valida o ambiente Python sem baixar/processar datasets.
  --download-only              Instala o ambiente e baixa os datasets brutos sem processá-los.
  --verify-only                Apenas executa a validação da instalação existente e gera relatórios.
  --update                     Atualiza pacotes do ambiente virtual (.venv).
  --cpu-only                   Força instalação da versão CPU do PyTorch mesmo se houver GPU NVIDIA.
  --force-data                 Força novo download/reprocessamento de datasets já existentes.
  --skip-tests                 Pula a execução das suítes de testes unitários na validação final.

Opções de diretórios (também configuráveis via variáveis de ambiente VIBNET_*_DIR):
  --venv-dir <PATH>            Caminho do ambiente virtual (default: .venv).
  --raw-dir <PATH>             Define VIBNET_RAW_DIR (dados brutos).
  --processed-dir <PATH>       Define VIBNET_PROCESSED_DIR (janelas 1s em data/processed).
  --processed-tl-dir <PATH>    Define VIBNET_PROCESSED_TL_DIR (variantes TL em data/processed_tl).
  --results-dir <PATH>         Define VIBNET_RESULTS_DIR (saídas e relatórios em results).
  -h, --help                   Exibe esta mensagem de ajuda.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --datasets)
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        DATASETS+=("$1")
        shift
      done
      ;;
    --variants)
      VARIANTS=()
      shift
      while [[ $# -gt 0 && ! "$1" =~ ^-- ]]; do
        VARIANTS+=("$1")
        shift
      done
      ;;
    --skip-datasets)
      SKIP_DATASETS=1
      shift
      ;;
    --download-only)
      DOWNLOAD_ONLY=1
      shift
      ;;
    --verify-only)
      VERIFY_ONLY=1
      shift
      ;;
    --update)
      UPDATE_ENV=1
      shift
      ;;
    --cpu-only)
      CPU_ONLY=1
      shift
      ;;
    --force-data)
      FORCE_DATA=1
      shift
      ;;
    --skip-tests)
      SKIP_TESTS=1
      shift
      ;;
    --venv-dir)
      VENV_DIR="$2"
      shift 2
      ;;
    --raw-dir)
      export VIBNET_RAW_DIR="$2"
      shift 2
      ;;
    --processed-dir)
      export VIBNET_PROCESSED_DIR="$2"
      shift 2
      ;;
    --processed-tl-dir)
      export VIBNET_PROCESSED_TL_DIR="$2"
      shift 2
      ;;
    --results-dir)
      export VIBNET_RESULTS_DIR="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "[ERRO] Opção desconhecida: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

echo "========================================================================"
echo " INSTALADOR REPRODUZÍVEL DO VIBNET-1D (Linux)"
echo "========================================================================"
echo "Raiz do repositório : ${PROJECT_ROOT}"
echo "Ambiente virtual    : ${VENV_DIR}"

if [[ "${VERIFY_ONLY}" -eq 0 ]]; then
  DEP_ARGS=("--venv-dir" "${VENV_DIR}" "--python" "${PYTHON_BIN}")
  if [[ "${CPU_ONLY}" -eq 1 ]]; then
    DEP_ARGS+=("--cpu-only")
  fi
  if [[ "${UPDATE_ENV}" -eq 1 ]]; then
    DEP_ARGS+=("--update")
  fi
  bash "${SCRIPT_DIR}/install_dependencies.sh" "${DEP_ARGS[@]}"
fi

VENV_PY="${VENV_DIR}/bin/python"
if [[ ! -x "${VENV_PY}" ]]; then
  echo "[ERRO] Ambiente virtual não encontrado em ${VENV_PY}." >&2
  exit 1
fi

# Valida antecipadamente os nomes de datasets caso tenham sido informados
if [[ ${#DATASETS[@]} -gt 0 ]]; then
  "${VENV_PY}" "${SCRIPT_DIR}/download_datasets.py" --estimate-only --datasets "${DATASETS[@]}"
fi

if [[ "${VERIFY_ONLY}" -eq 0 && "${SKIP_DATASETS}" -eq 0 ]]; then
  echo "========================================================================"
  echo " Etapa de Dados 1/2: Download e Verificação de Datasets Brutos"
  echo "========================================================================"
  DL_ARGS=()
  if [[ ${#DATASETS[@]} -gt 0 ]]; then
    DL_ARGS+=("--datasets" "${DATASETS[@]}")
  fi
  if [[ "${FORCE_DATA}" -eq 1 ]]; then
    DL_ARGS+=("--force")
  fi
  "${VENV_PY}" "${SCRIPT_DIR}/download_datasets.py" "${DL_ARGS[@]}"

  if [[ "${DOWNLOAD_ONLY}" -eq 0 ]]; then
    echo "========================================================================"
    echo " Etapa de Dados 2/2: Processamento e Normalização de Datasets"
    echo "========================================================================"
    PREP_ARGS=("--variants" "${VARIANTS[@]}")
    if [[ ${#DATASETS[@]} -gt 0 ]]; then
      PREP_ARGS+=("--datasets" "${DATASETS[@]}")
    fi
    if [[ "${FORCE_DATA}" -eq 1 ]]; then
      PREP_ARGS+=("--force")
    fi
    "${VENV_PY}" "${SCRIPT_DIR}/prepare_datasets.py" "${PREP_ARGS[@]}"
  else
    echo "  -> Modo --download-only ativo: pulando etapa de processamento."
  fi
else
  echo "  -> Etapa de download/processamento de datasets pulada (--skip-datasets ou --verify-only)."
fi

echo "========================================================================"
echo " Etapa Final: Validação da Instalação e Relatório de Diagnóstico"
echo "========================================================================"
VAL_ARGS=()
if [[ "${SKIP_TESTS}" -eq 1 ]]; then
  VAL_ARGS+=("--skip-tests")
fi
"${VENV_PY}" "${SCRIPT_DIR}/validate_installation.py" "${VAL_ARGS[@]}"

echo "========================================================================"
echo " INSTALAÇÃO CONCLUÍDA COM SUCESSO!"
echo " Para ativar o ambiente virtual no seu terminal:"
echo "   source ${VENV_DIR}/bin/activate"
echo "========================================================================"

