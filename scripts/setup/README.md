# Guia de Instalação Reproduzível e Preparação de Dados — VibNet-1D

Este diretório (`scripts/setup/`) contém o sistema automatizado e idempotente de instalação, download, preparação de datasets e validação de ambiente do projeto **VibNet-1D**, projetado prioritariamente para estações e servidores **Linux** de laboratório (Ubuntu 22.04 LTS e 24.04 LTS).

---

## Sumário

1. [Como instalar em uma máquina Linux nova](#1-como-instalar-em-uma-máquina-linux-nova)
2. [Como escolher os datasets](#2-como-escolher-os-datasets)
3. [Como configurar os diretórios de armazenamento externos](#3-como-configurar-os-diretórios-de-armazenamento-externos)
4. [Como verificar a GPU e o suporte CUDA](#4-como-verificar-a-gpu-e-o-suporte-cuda)
5. [Como atualizar o ambiente](#5-como-atualizar-o-ambiente)
6. [Como baixar datasets posteriormente](#6-como-baixar-datasets-posteriormente)
7. [Como processar datasets posteriormente](#7-como-processar-datasets-posteriormente)
8. [Como verificar a integridade dos dados](#8-como-verificar-a-integridade-dos-dados)
9. [Como executar os experimentos](#9-como-executar-os-experimentos)
10. [Como transferir a instalação para outra máquina do laboratório](#10-como-transferir-a-instalação-para-outra-máquina-do-laboratório)

---

## 1. Como instalar em uma máquina Linux nova

### Pré-requisitos mínimos do sistema operacional (Ubuntu 22.04 / 24.04)
Caso a máquina ainda não possua `git` ou `python3-venv`:
```bash
sudo apt-get update
sudo apt-get install -y git python3 python3-venv python3-pip unrar p7zip-full
```
*(O instalador do VibNet-1D **não utiliza `sudo`** para instalar pacotes Python; tudo é instalado de forma isolada dentro de `.venv`).*

### Instalação completa padrão (ambiente + todos os datasets disponíveis)
```bash
git clone https://github.com/vfrocha/VibNet-1D.git
cd VibNet-1D
bash scripts/setup/install_linux.sh
```

O comando acima executa de forma **idempotente**:
1. Verifica `python3 >= 3.9` e cria o ambiente virtual `.venv`.
2. Detecta automaticamente se há GPU NVIDIA (`nvidia-smi`) e instala a build compatível do PyTorch (`cu124`, `cu121`, `cu118` ou `cpu`).
3. Instala todas as dependências científicas (`numpy`, `scipy`, `pandas`, `scikit-learn`, `PyWavelets`, `xgboost`, `imbalanced-learn`, `pytorch-tabnet`, `vibdata`) e registra o projeto em modo editável (`pip install -e .`).
4. Baixa os datasets brutos disponíveis, gera as variantes de janelas de 1 segundo (`processed_1d`, `raw`, `detrended`, `raw_rms`, `detrended_rms`) e valida os invariantes físicos de RMS.
5. Emite os relatórios diagnósticos em `results/setup/installation_report.json` e `results/setup/installation_report.md`.

---

## 2. Como escolher os datasets

A lista de datasets é lida diretamente do registro canônico (`src/vibnet/data/registry.py` e `configs/datasets.yaml`):
- `CWRU_12k`, `CWRU_48k`, `PU`, `HUST_Gearbox`, `UOEMD`, `UORED`, `HUST`, `MFPT`, `Mechanical_Gear`, `Electric_Motor`.

### Consultar catálogo, tamanhos estimados e requisitos de cada dataset
```bash
python3 scripts/setup/download_datasets.py --list
```

### Instalar preparando apenas um subconjunto de datasets
```bash
bash scripts/setup/install_linux.sh \
  --datasets CWRU_12k MFPT CWRU_48k
```

### Instalar apenas o ambiente Python (sem baixar datasets)
```bash
bash scripts/setup/install_linux.sh --skip-datasets
```

### Instalar o ambiente e apenas baixar os dados brutos (sem processar)
```bash
bash scripts/setup/install_linux.sh --download-only \
  --datasets CWRU_12k MFPT
```

---

## 3. Como configurar os diretórios de armazenamento externos

Para evitar armazenar gigabytes de sinais brutos e processados dentro da árvore do Git, você pode definir as variáveis de ambiente suportadas por `src/config.py` ou passar flags diretamente ao instalador:

### Via variáveis de ambiente (recomendado adicionar ao `~/.bashrc` ou `.env`)
```bash
export VIBNET_RAW_DIR="/home/lab/datasets/vibnet/raw"
export VIBNET_PROCESSED_DIR="/home/lab/datasets/vibnet/processed"
export VIBNET_PROCESSED_TL_DIR="/home/lab/datasets/vibnet/processed_tl"
export VIBNET_RESULTS_DIR="/home/lab/experiments/vibnet/results"

bash scripts/setup/install_linux.sh
```

### Ou via argumentos de linha de comando
```bash
bash scripts/setup/install_linux.sh \
  --raw-dir /home/lab/datasets/vibnet/raw \
  --processed-dir /home/lab/datasets/vibnet/processed \
  --processed-tl-dir /home/lab/datasets/vibnet/processed_tl \
  --results-dir /home/lab/experiments/vibnet/results
```

---

## 4. Como verificar a GPU e o suporte CUDA

Para inspecionar CPU, RAM, espaço em disco, driver NVIDIA e executar um teste real de multiplicação matricial em GPU (`torch.randn(100, 100, device="cuda")`):

```bash
.venv/bin/python scripts/setup/system_info.py
```

Para forçar uma instalação somente CPU (mesmo em máquinas com GPU):
```bash
bash scripts/setup/install_linux.sh --cpu-only --skip-datasets
```

---

## 5. Como atualizar o ambiente

Após realizar `git pull` com novas dependências ou atualizações do código:
```bash
bash scripts/setup/install_linux.sh --update --skip-datasets
```

Para apenas reexecutar a verificação e atualizar `results/setup/installation_report.{json,md}`:
```bash
bash scripts/setup/install_linux.sh --verify-only
```

---

## 6. Como baixar datasets posteriormente

Com o ambiente virtual ativado (`source .venv/bin/activate`):

```bash
# Estimar espaço em disco necessário antes de baixar:
python scripts/setup/download_datasets.py --estimate-only --datasets PU HUST_Gearbox

# Baixar datasets específicos (retoma downloads interrompidos e pula arquivos já íntegros):
python scripts/setup/download_datasets.py --datasets CWRU_12k MFPT HUST UORED
```

---

## 7. Como processar datasets posteriormente

O script `prepare_datasets.py` invoca os pipelines oficiais (`src/data/make_dataset.py` e `src/preprocessing/normalize_dataset_tl.py`), reutilizando variantes já processadas e válidas:

```bash
source .venv/bin/activate

# Preparar todas as variantes (processed_1d, raw, detrended, raw_rms, detrended_rms) para CWRU_12k e MFPT:
python scripts/setup/prepare_datasets.py --datasets CWRU_12k MFPT

# Preparar apenas a variante detrended_rms (usada no benchmark TabVibNet) e processed_1d (domínio do tempo):
python scripts/setup/prepare_datasets.py \
  --datasets CWRU_12k CWRU_48k \
  --variants processed_1d detrended detrended_rms
```
*(Nota: Sempre que `CWRU_48k` é normalizado por RMS de referência, `CWRU_12k` é incluído automaticamente como referência saudável conforme `ADR-001`).*

---

## 8. Como verificar a integridade dos dados

Para validar matematicamente as variantes normalizadas por RMS ($\text{RMS}(\tilde{x}) = \text{RMS}(x)/\text{RMS}_{\text{ref}}$) e verificar toda a instalação:

```bash
source .venv/bin/activate

# Validação matemática oficial das variantes de Transfer Learning:
python scripts/validate_dataset_tl.py --datasets CWRU_12k MFPT CWRU_48k --variants raw detrended

# Validação completa da instalação e regeneração de results/setup/installation_report.{json,md}:
python scripts/setup/validate_installation.py
```

---

## 9. Como executar os experimentos

Com o ambiente ativado (`source .venv/bin/activate`):

```bash
# 1. Benchmark Scratch vs. Transfer Learning (MLP e TabNet em detrended_rms):
python -m src.experiments.run_scratch_vs_tl_benchmark --config configs/scratch_vs_tl.yaml

# 2. Baselines Clássicos pareados (RandomForest, SVM, XGBoost com otimização interna por fold):
python -m src.experiments.run_scratch_vs_tl_classical_baselines --config configs/scratch_vs_tl_classical_baselines.yaml

# 3. Ablação From Scratch no Domínio do Tempo (ResNet18-1D em data/processed):
python -m src.experiments.run_time_scratch_ablation --k_factors 2 4 6 8 --resample_modes uniform fft poly linear
```

---

## 10. Como transferir a instalação para outra máquina do laboratório

Para replicar exatamente o ambiente e aproveitar os datasets já baixados/processados em uma segunda máquina Linux da rede do laboratório:

1. **Clonar o repositório na máquina de destino no mesmo commit**:
   ```bash
   git clone https://github.com/vfrocha/VibNet-1D.git
   cd VibNet-1D
   ```
2. **Copiar ou montar via NFS/rsync os diretórios de dados já baixados/processados** da máquina de origem:
   ```bash
   rsync -avh --progress usuario@maquina-origem:/caminho/raw/ /home/lab/datasets/vibnet/raw/
   rsync -avh --progress usuario@maquina-origem:/caminho/processed/ /home/lab/datasets/vibnet/processed/
   rsync -avh --progress usuario@maquina-origem:/caminho/processed_tl/ /home/lab/datasets/vibnet/processed_tl/
   ```
3. **Executar o instalador apontando para os diretórios transferidos**:
   ```bash
   bash scripts/setup/install_linux.sh \
     --raw-dir /home/lab/datasets/vibnet/raw \
     --processed-dir /home/lab/datasets/vibnet/processed \
     --processed-tl-dir /home/lab/datasets/vibnet/processed_tl
   ```
   O instalador detectará automaticamente que os arquivos brutos e as variantes processadas já existem e estão íntegros (`ALREADY_PRESENT` / `REUSED_VALID`), instalando apenas o `.venv` adequado à GPU/CPU da nova máquina e validando a integridade final em `results/setup/installation_report.md`.

