# VibNet-1D — Diagnóstico de Falhas em Máquinas Rotativas

Repositório de pesquisa para detecção e diagnóstico de falhas em máquinas rotativas com sinais de vibração (janelas físicas de 1 segundo), combinando:
- **Pipeline Tabular (`tabvibnet`)**: Extração de características no tempo, frequência e tempo-frequência (`VibNet-1D` + `SignAI-Framework`) com modelos clássicos, MLP, TabNet e Transfer Learning Cross-Dataset sob avaliação estrita *Leave-One-Condition-Out* (LOCO).
- **Pipeline 1D no Domínio do Tempo**: Sinais brutos 1D com *detrend*, *resampling* polifásico e normalização (*Z-score* / *RMS*) em `src/data/transforms_1d.py` e `src/models/build_1d_time.py`.

---

## Instalação Rápida (Linux / macOS / Servidores)

Crie um ambiente virtual, instale as dependências e registre o pacote `vibnet1d` (`src`) em modo editável (eliminando a necessidade de `sys.path.append`):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

---

## Execução no Google Colab

Para montar o Google Drive, configurar os caminhos de dados/resultados e instalar o projeto no Google Colab, execute o snippet de 3 linhas abaixo em uma célula:

```python
from google.colab import drive; drive.mount('/content/drive')
%env VIBNET_PROCESSED_DIR=/content/drive/MyDrive/VibNet_Project/data/processed
!pip install -q -r requirements.txt && pip install -q -e .
```

> **Variáveis de ambiente suportadas (`src/config.py`):**
> - `VIBNET_RAW_DIR`: caminho para os datasets brutos da `vibdata` (fallback automático para `/home/vfrocha/VibNet_Project/raw_data` no servidor ou `<raiz_do_repo>/data/raw`).
> - `VIBNET_PROCESSED_DIR`: caminho para as janelas `.npy` processadas (padrão: `<raiz_do_repo>/data/processed`).
> - `VIBNET_PROCESSED_TL_DIR`: caminho para as variantes de Transfer Learning (padrão: `<raiz_do_repo>/data/processed_tl`).
> - `VIBNET_RESULTS_DIR`: caminho para salvar logs e CSVs de resultados (padrão: `<raiz_do_repo>/results`).

---

## Estrutura do Projeto

- `src/config.py`: Gerenciador central de diretórios e variáveis de ambiente (`RAW_DATA_DIR`, `DATA_ROOT`, `PROCESSED_TL_DIR`, `RESULTS_DIR`).
- `src/data/`: Geração de datasets (`make_dataset.py`, `make_dataset_tl.py`), carregamento LOCO (`dataloader.py`), validação (`validate_dataset_tl.py`) e transformações temporais 1D (`transforms_1d.py`).
- `src/features/`: Extratores estatísticos e espectrais (`extractors.py`, `extractors_v2.py`, `signalai_wrapper.py`).
- `src/models/`: Arquiteturas tabulares (`build_sklearn.py`, `build_tabular.py`, `build_tabnet_resnet.py`, `build_tabnet_resnet_balance.py`, `build_mlp_contrastive.py`) e no domínio do tempo (`build_1d_time.py`).
- `src/experiments/` e `scripts/`: Runners experimentais de baseline, ablação de normalização e Transfer Learning.

---

## Experimento de Balanceamento — VibNet-1D

### Arquivos

- `scripts/tl/exp_tl_balance_strategies.py`: runner LOCO baseado no experimento existente, configurado por padrão para `detrended_rms` e para o target `CWRU_48k`.
- `src/models/build_tabnet_resnet_balance.py`: versão do módulo de treinamento com balanceamento opcional no treino supervisionado do target.

### Estratégias

- `none`: sem balanceamento (baseline).
- `undersample`: reduz cada classe do treino target à contagem da classe menos frequente.
- `smote`: aumenta as classes minoritárias até a contagem da classe mais frequente.
- `hybrid`: define como alvo comum a mediana das contagens originais; aplica SMOTE nas classes abaixo do alvo e undersampling nas classes acima.

O balanceamento ocorre **depois do `StandardScaler.fit_transform()` ajustado exclusivamente nos dados originais de treino do target**. As sources (`CWRU_12k` e `MFPT`) e o conjunto de teste LOCO não são balanceados nem usados para ajustar o scaler do target. A avaliação e a seleção de folds seguem a regra de validade do script original.

### Execução

Na raiz do repositório, execute cada estratégia com a mesma seed. Por padrão, o experimento usa `detrended_rms`, MLP e TabNet:

```bash
python scripts/tl/exp_tl_balance_strategies.py --balance-strategy none --seed 42
python scripts/tl/exp_tl_balance_strategies.py --balance-strategy undersample --seed 42
python scripts/tl/exp_tl_balance_strategies.py --balance-strategy smote --seed 42
python scripts/tl/exp_tl_balance_strategies.py --balance-strategy hybrid --seed 42
```

Para um teste rápido antes da execução completa:

```bash
python scripts/tl/exp_tl_balance_strategies.py --balance-strategy hybrid --models mlp --seed 42
```
