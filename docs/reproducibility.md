# Guia de Reprodutibilidade e Rastreabilidade Experimental — VibNet-1D

Este documento especifica o protocolo de rastreabilidade de execuções experimentais do **VibNet-1D**, audita o nível de cobertura dos runners existentes e documenta os comandos exatos para regenerar relatórios, validar datasets e reproduzir cada experimento da etapa **TabVibNet**.

---

## 1. Especificação de Metadados por Execução Experimental (Fase 5)

Toda bateria experimental no repositório deve criar um diretório próprio com carimbo de tempo (`results/<familia>/run_YYYYMMDD_HHMMSS/`) e persistir um arquivo `config.json` (além de `run.log`, `results.csv` e `summary*.csv`) contendo os 7 blocos obrigatórios de proveniência:

1. **Identificação de Código e Estado Git (`code_identification`)**:
   - `git_commit`: hash SHA completo de 40 caracteres obtido via `git rev-parse HEAD`.
   - `git_branch`: branch ativa obtida via `git rev-parse --abbrev-ref HEAD`.
   - `git_dirty`: booleano obtido verificando se `git status --porcelain` retorna saída não vazia.
2. **Hashes Criptográficos dos Scripts (`script_hashes_sha256`)**:
   - Dicionário mapeando o caminho relativo do script principal e dos módulos de modelo/extração de atributos utilizados para o respectivo hash `SHA-256` (`hashlib.sha256(path.read_bytes()).hexdigest()`). Garante rastreabilidade exata mesmo quando `git_dirty == True`.
3. **Comando Executado (`command_line`)**:
   - Lista de argumentos `sys.argv` e string reconstruída (`" ".join(shlex.quote(a) for a in sys.argv)`), incluindo o executável Python (`sys.executable`) e diretório de trabalho (`os.getcwd()`).
4. **Configuração Completa (`experiment_config`)**:
   - Domínios fonte (`sources`), domínios alvo (`targets`), variante de pré-processamento (`variant`), dimensão e flags de features (`n_features`, `include_window_rms_feature`), estratégia de balanceamento (`balance_strategy`), hiperparâmetros de modelo (`epochs`, `batch_size`, `lr`, arquitetura do encoder e da cabeça).
5. **Controle de Sementes Pseudoaleatórias (`seeds`)**:
   - Lista de seeds avaliadas (ex.: `[42, 43, 44]`) e aplicação determinística em cada fold/modelo:
     - `random.seed(seed)`
     - `np.random.seed(seed)`
     - `torch.manual_seed(seed)` e `torch.cuda.manual_seed_all(seed)`
     - `SMOTE(random_state=seed, k_neighbors=k)`
6. **Ambiente e Versões das Dependências (`environment`)**:
   - Versão do Python (`platform.python_version()`), sistema operacional (`platform.platform()`), dispositivo (`cpu` / `cuda` + nome da GPU) e versões instaladas de `numpy`, `scipy`, `pandas`, `scikit-learn`, `imbalanced-learn`, `torch`, `pytorch-tabnet`, `PyWavelets`, `EMD-signal` via `importlib.metadata.version()`.
7. **Contabilidade de Folds e Status de Execução (`fold_accounting`)**:
   - Em `results.csv`: colunas `Fold_Status` (`OK`, `SKIPPED`, `ERROR_MODEL_EXECUTION`), `N_Train_Before`, `N_Train_After`, `N_Test`, `N_Test_Classes_Present`, `ROC_AUC_Valid` e `Error_Message`.
   - Em `summary*.csv`: colunas `n_total_fold_evals`, `n_valid_fold_evals`, `n_failed_fold_evals` e `n_roc_auc_valid_evals`.

### 1.1. Cobertura Atual de Proveniência nos Runners do Repositório

A auditoria verificou como cada geração de scripts implementa esses requisitos:

| Runner / Script | `git_commit` & `git_dirty` | Hash SHA-256 do Script | `config.json` Completo | Comando CLI (`sys.argv`) | Versões de Pacotes | Contagem de Folds Válidos/Falhos |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| **`scripts/exp_tl_cross_dataset_mlp_tabnet.py`** | **Sim** (`get_code_identification` L273–296) | **Sim** (`script_sha256`) | **Sim** | Parcial (via `args`) | Parcial | **Sim** (`n_total_fold_evals`, `n_valid_fold_evals`, `n_failed_fold_evals`, `n_roc_auc_valid_evals`) |
| **`scripts/exp_tl_contrastive_mlp.py`** | Não | Não | **Sim** | Parcial | Não | **Sim** (`Valid_Folds`, `N_ROC_AUC_Folds`) |
| **`scripts/exp_tl_balance_strategies.py`** | Não | Não | **Sim** | Parcial | Não | **Sim** (`Valid_Folds`, `Total_Folds`, `Skipped_Folds`, `Error_Folds`) |
| **`scripts/tl/exp_tl_four_variants.py`** | Não | Não | **Sim** | Parcial | Não | **Sim** (`Valid_Folds`, `Total_Folds`) |
| **`scripts/tl/exp_tl_eight_configs.py`** | Não | Não | **Sim** | Parcial | Não | **Sim** (`Valid_Folds`, `Total_Folds`) |
| **`src/experiments/run_*.py` (Geração 2)** | Não | Não | Não | Não | Não | Não (apenas CSV bruto e log `.txt`) |

### 1.2. Implementação de Referência para Registro de Proveniência (Stdlib Pura)

A função abaixo expande `get_code_identification()` (já presente em `scripts/exp_tl_cross_dataset_mlp_tabnet.py`, linhas 273–296) para cobrir todos os 7 itens usando exclusivamente a biblioteca padrão do Python:

```python
import hashlib
import importlib.metadata
import os
import platform
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable

TRACKED_PACKAGES = (
    "numpy", "scipy", "pandas", "scikit-learn", "imbalanced-learn",
    "torch", "pytorch-tabnet", "xgboost", "PyWavelets", "EMD-signal", "pyarrow",
)

def collect_run_provenance(
    project_root: Path,
    script_paths: Iterable[Path],
    seeds: Iterable[int],
    extra_config: Dict[str, Any],
) -> Dict[str, Any]:
    def _git(args: list[str]) -> str:
        try:
            r = subprocess.run(
                ["git"] + args, cwd=str(project_root),
                capture_output=True, text=True, timeout=5, check=False,
            )
            return r.stdout.strip() if r.returncode == 0 else "UNKNOWN"
        except Exception:
            return "UNKNOWN"

    git_commit = _git(["rev-parse", "HEAD"])
    git_branch = _git(["rev-parse", "--abbrev-ref", "HEAD"])
    git_status = _git(["status", "--porcelain"])
    git_dirty = bool(git_status and git_status != "UNKNOWN")

    script_hashes = {}
    for sp in script_paths:
        p = Path(sp).resolve()
        if p.is_file():
            rel = p.relative_to(project_root.resolve()).as_posix()
            script_hashes[rel] = hashlib.sha256(p.read_bytes()).hexdigest()

    pkg_versions = {}
    for pkg in TRACKED_PACKAGES:
        try:
            pkg_versions[pkg] = importlib.metadata.version(pkg)
        except importlib.metadata.PackageNotFoundError:
            pkg_versions[pkg] = None

    return {
        "git_commit": git_commit,
        "git_branch": git_branch,
        "git_dirty": git_dirty,
        "command_argv": list(sys.argv),
        "command_str": " ".join(shlex.quote(a) for a in sys.argv),
        "working_directory": os.getcwd(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "seeds": list(seeds),
        "script_hashes_sha256": script_hashes,
        "dependency_versions": pkg_versions,
        "config": extra_config,
    }
```

---

## 2. Como Regenerar os Relatórios de Auditoria e Validação

### 2.1. Relatório Automatizado de Resultados Experimentais (Fase 2)

O script `scripts/audit_experimental_results.py` é escrito em **Python padrão puro** (não exige instalação de `pandas` ou `numpy`) e inventaria todos os 220 artefatos em `results/`:

```bash
# No Linux/macOS:
python scripts/audit_experimental_results.py --results-dir results --output-dir reports/reproducibility_audit

# No Windows (PowerShell):
py scripts/audit_experimental_results.py --results-dir results --output-dir reports/reproducibility_audit
```

**Saídas geradas em `reports/reproducibility_audit/`**:
- `results_inventory.csv`: inventário dos 220 arquivos, tipo, granularidade (`fold_level`, `summary`, `dataset_audit`), colunas de contexto e colunas não reconhecidas.
- `metrics_long.csv`: 89.445 registros de métricas com referência ao arquivo de origem (`relative_path`), `dataset`, `model`, `sources`, `strategy`, `seed`, `fold`, `status`, `is_valid` e `anomaly_flag`.
- `metrics_summary.csv`: 9.984 grupos descritivos (`count`, `invalid_count`, `n_seeds`, `n_folds`, `mean`, `std`, `min`, `max`) calculados estritamente dentro de cada arquivo e protocolo.
- `anomalies_and_warnings.csv`: 1.025 ocorrências auditadas (`SHIFTED_COLUMN_CORRUPTION`, `NON_OK_STATUS`, `ZERO_METRIC_FAILURE`, `ROC_AUC_ZERO_FALLBACK`, `ROC_AUC_NAN_MISSING_CLASS`).
- `EXPERIMENTAL_RESULTS_REPORT.md`: relatório consolidado em Markdown.

### 2.2. Execução dos Scripts de Verificação e Testes

Os scripts de verificação sintática, combinatória e de consistência dos artefatos publicados também funcionam tanto em ambiente Python mínimo quanto no ambiente completo:

```bash
# Verificação do experimento Cross-Dataset MLP vs TabNet e seus CSVs publicados:
py scripts/verify_cross_dataset_mlp_tabnet.py

# Verificação do experimento Contrastive MLP (SupCon vs CE):
py scripts/verify_mlp_contrastive.py

# Testes unitários do script de auditoria de resultados:
py -m unittest tests/test_audit_experimental_results.py
```

---

## 3. Como Reproduzir o Pipeline de Dados e Cada Experimento TabVibNet

> **Pré-requisito de Ambiente para Execução Completa de Treinamento**:
> Para reexecutar a extração de features ou o treinamento de modelos (`PyTorch`, `TabNet`, `scikit-learn`, `SMOTE`), utilize um ambiente virtual com as dependências de `requirements.txt` instaladas (`python -m pip install -r requirements.txt`) e os arquivos Parquet em `data/processed_tl/`.

### Passo 0 — Preparação e Validação dos Datasets Normalizados (`data/processed_tl/`)

```bash
# 1. Ingerir os arquivos brutos para janelas de 1s em data/processed_tl/{raw,detrended}:
python src/data/make_dataset_tl.py --raw-dir /caminho/para/dados_brutos --output-dir data/processed_tl

# 2. Gerar as variantes normalizadas por RMS saudável de referência (raw_rms e detrended_rms):
python src/preprocessing/normalize_dataset_tl.py --data-dir data/processed_tl

# 3. Validar integridade física e estatística das 4 variantes nos 10 datasets:
python scripts/validate_dataset_tl.py --data-dir data/processed_tl --output-dir results/dataset_validation_tl
```

### Experimento A — Comparação das 4 Variantes de Pré-processamento (`tl_four_variants`)
- **Artefato original de referência**: `results/tl_four_variants/run_20261005_163159/`
- **Comando de reprodução**:
```bash
python scripts/tl/exp_tl_four_variants.py \
    --data-dir data/processed_tl \
    --output-dir results/tl_four_variants \
    --target CWRU_48k \
    --variants raw detrended raw_rms detrended_rms \
    --models MLP TABNET \
    --seeds 42
```

### Experimento B — Ablação das Features Auxiliares (`tl_aux_features`)
- **Artefato original de referência**: `results/tl_aux_features/run_20261012_093659/`
- **Comando de reprodução**:
```bash
python scripts/tl/exp_tl_eight_configs.py \
    --data-dir data/processed_tl \
    --output-dir results/tl_aux_features \
    --target CWRU_48k \
    --variants raw detrended raw_rms detrended_rms \
    --aux-modes base rms_b1_b2 \
    --models MLP TABNET \
    --seeds 42
```

### Experimento C — Estratégias de Balanceamento no Domínio Alvo (`tl_balance_strategies`)
- **Artefatos originais de referência**: `results/tl_balance_strategies/run_20261012_145854` (`none`), `150217` (`class_weights`), `150540` (`smote`), `150855` (`hybrid`).
- **Comandos de reprodução**:
```bash
# Para none e class_weights nas 4 variantes:
python scripts/exp_tl_balance_strategies.py --balance-strategies none --variants raw detrended raw_rms detrended_rms --seeds 42
python scripts/exp_tl_balance_strategies.py --balance-strategies class_weights --variants raw detrended raw_rms detrended_rms --seeds 42

# Para smote e hybrid na variante vencedora detrended_rms:
python scripts/exp_tl_balance_strategies.py --balance-strategies smote --variants detrended_rms --seeds 42
python scripts/exp_tl_balance_strategies.py --balance-strategies hybrid --variants detrended_rms --seeds 42
```

### Experimento D — Pré-treinamento Contrastivo Supervisionado (`tl_contrastive_mlp`)
- **Artefatos originais de referência**: `results/tl_contrastive_mlp/run_20261027_112749` (modos `baseline`, `joint`, `pretrain`), `run_20261027_113203` (encoders `baseline` vs. `expanded`) e `run_20261027_113503` (cabeças `asymmetric` vs. `symmetric`) em `CWRU_48k` com 3 seeds (`42, 123, 2026`).
- **Comandos de reprodução**:
```bash
# Ablação de modo de pré-treinamento (baseline, joint, pretrain) com 3 seeds (42, 123, 2026):
python scripts/exp_tl_contrastive_mlp.py \
    --modes baseline joint pretrain \
    --encoder-profiles baseline \
    --head-shapes asymmetric \
    --balance-strategies none \
    --seeds 42 123 2026

# Ablação de perfil de encoder (baseline vs expanded) no modo joint:
python scripts/exp_tl_contrastive_mlp.py \
    --modes joint \
    --encoder-profiles baseline expanded \
    --head-shapes asymmetric \
    --balance-strategies none \
    --seeds 42 123 2026

# Ablação de formato de cabeça (asymmetric vs symmetric) no modo joint + expanded:
python scripts/exp_tl_contrastive_mlp.py \
    --modes joint \
    --encoder-profiles expanded \
    --head-shapes asymmetric symmetric \
    --balance-strategies none \
    --seeds 42 123 2026
```

### Experimento E — Estudo Fatorial Cross-Dataset `MLP` vs. `TabNet` (`tl_cross_dataset_mlp_tabnet`)
- **Artefatos originais de referência**:
  - `results/tl_cross_dataset_mlp_tabnet/run_20261028_105603/` (`all_sources` + 4 `single_source`, 1.980 avaliações, `git_commit: b07c70263638fb0981fd33f7510b51695b3a0542`)
  - `results/tl_cross_dataset_mlp_tabnet/run_20261028_173310/` (todas as 15 combinações não vazias de `{UORED, HUST, Mechanical_Gear, Electric_Motor}` sobre os 5 alvos, 5.940 avaliações, `git_commit: 69ca186049bd2785826dfcb2fc950702a38a7c77`)
- **Comandos de reprodução**:
```bash
# Run 1: Todas as 4 fontes juntas (all) + cada fonte individualmente (single) — 1.980 avaliações:
python scripts/exp_tl_cross_dataset_mlp_tabnet.py \
    --targets CWRU_12k CWRU_48k HUST_Gearbox PU UOEMD \
    --sources UORED HUST Mechanical_Gear Electric_Motor \
    --source-modes all single \
    --models MLP TabNet \
    --seeds 42 123 2026

# Run 2: Todas as 15 combinações não vazias das 4 fontes (all, single, combinations) — 5.940 avaliações:
python scripts/exp_tl_cross_dataset_mlp_tabnet.py \
    --targets CWRU_12k CWRU_48k HUST_Gearbox PU UOEMD \
    --sources UORED HUST Mechanical_Gear Electric_Motor \
    --source-modes all single combinations \
    --models MLP TabNet \
    --seeds 42 123 2026
```
