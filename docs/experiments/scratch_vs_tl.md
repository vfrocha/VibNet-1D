# Benchmark Experimental Independente: Scratch vs. Transfer Learning (`scratch_vs_tl`)

## 1. Objetivo Científico

Este documento descreve o protocolo experimental, a auditoria de compatibilidade e o uso do benchmark independente **`scratch_vs_tl`** ([`src/experiments/run_scratch_vs_tl_benchmark.py`](../../src/experiments/run_scratch_vs_tl_benchmark.py)), projetado para comparar o treinamento supervisionado convencional a partir do zero (**`Scratch`**) contra aprendizado por transferência (**`Fine-Tuning`** e **`Linear Probing`**) sob condições estritamente pareadas e livres de vazamento de dados (*Leave-One-Condition-Out* — LOCO).

- **Identificador do Experimento**: `scratch_vs_tl`
- **Tarefa**: Diagnóstico de falhas (`diagnosis`)
- **Arquiteturas Avaliadas**: `MLP` (`MLPEncoder` + `Bottleneck` + `Linear`) e `TabNet` (`TabNetPretrainer` / `TabNetClassifier`)
- **Métodos Comparados**:
  1. `Scratch`: Inicialização aleatória dos pesos; treinamento exclusivo nas $K_T - 1$ condições operacionais de treino do dataset alvo.
  2. `Fine-Tuning`: Pré-treinamento do encoder exclusivamente no pool de domínios fonte disjunto (`UORED`, `HUST`, `Mechanical_Gear`, `Electric_Motor`), seguido de adaptação supervisionada completa (encoder + nova cabeça linear do alvo) nas $K_T - 1$ condições de treino do dataset alvo.
  3. `Linear Probing`: Inicialização a partir do mesmo encoder pré-treinado nos domínios fonte, com congelamento estrito de todos os parâmetros e buffers do encoder (`requires_grad=False`, `.eval()`, verificado programaticamente após cada fold com `backbone_param_max_abs_diff == 0.0` e `backbone_buffer_max_abs_diff == 0.0`), otimizando apenas a cabeça linear do domínio alvo.

---

## 2. Datasets Alvo e Pool de Domínios Fonte

| Dataset Alvo | Folds LOCO ($K_T$) | Taxa de Amostragem Canônica (`fs`) | Pool de Domínios Fonte Disjunto (`source_pools_by_target`) |
|---|---:|---:|---|
| `CWRU_12k` | 4 | 12 000 Hz | `UORED`, `HUST`, `Mechanical_Gear`, `Electric_Motor` |
| `CWRU_48k` | 12 | 48 000 Hz | `UORED`, `HUST`, `Mechanical_Gear`, `Electric_Motor` |
| `HUST_Gearbox` | 30 | 25 600 Hz | `UORED`, `HUST`, `Mechanical_Gear`, `Electric_Motor` |
| `UOEMD` | 16 | 42 000 Hz | `UORED`, `HUST`, `Mechanical_Gear`, `Electric_Motor` |
| `PU` | 4 | 64 000 Hz | `UORED`, `HUST`, `Mechanical_Gear`, `Electric_Motor` |
| **Total** | **66** | — | Disjunto de todos os 5 datasets alvo |

---

## 3. Garantias Metodológicas e Anti-Vazamento

1. **Mesma Representação e Pré-processamento**:
   - Todos os modelos consomem as janelas pré-processadas de `data/processed_tl/detrended_rms/` e o vetor canônico de **141 atributos** extraído por [`extract_141_features_from_windows`](../experiments/baseline_loco_v2.md) com as taxas de amostragem canônicas de `configs/datasets.yaml`.
2. **Isolamento LOCO e Assinaturas Criptográficas**:
   - Em cada fold LOCO, a condição operacional de teste permanece 100% isolada.
   - O `StandardScaler` do alvo é ajustado exclusivamente nas $K_T - 1$ condições de treino do fold (`scale_loco_fold_strict`).
   - O balanceamento `SMOTE` (`balance_strategy = "smote"`) é aplicado exclusivamente aos dados de treino escalados do fold.
   - Cada fold registra `split_signature` (SHA-256 dos IDs de treino e teste) e `preprocessing_signature` (SHA-256 dos parâmetros do scaler e SMOTE) em `fold_manifest.csv` e `results.csv`.
3. **Verificação Programática de Linear Probing vs. Fine-Tuning**:
   - Para `MLP` e `TabNet`, os tensores de parâmetros e buffers do encoder são copiados antes e depois do treinamento no alvo:
     - Em `Linear Probing`: exige-se `backbone_param_max_abs_diff == 0.0` e `backbone_buffer_max_abs_diff == 0.0`.
     - Em `Fine-Tuning`: exige-se `backbone_param_max_abs_diff > 0.0`.
4. **Unidade Estatística Primária e Correção de Holm**:
   - Em `statistical_tests.csv`, as seeds repetidas (`42, 123, 2026`) são primeiramente agregadas pela média dentro de cada fold LOCO (`statistical_unit = "loco_fold_seed_averaged"`), evitando pseudorreplicação no teste pareado de Wilcoxon signed-rank, seguido de correção *step-down* de Holm-Bonferroni (`p_value_holm`).

---

## 4. Artefatos Gerados

### 4.1 Auditoria de Compatibilidade (`results/scratch_vs_tl/audit/`)
Gerada por:
```bash
python -m src.experiments.audit_scratch_vs_tl
```
Artefatos:
- `results/scratch_vs_tl/audit/experiment_inventory.csv`
- `results/scratch_vs_tl/audit/compatibility_audit.csv`
- `results/scratch_vs_tl/audit/audit_report.md`

### 4.2 Execução do Benchmark (`results/scratch_vs_tl/run_TIMESTAMP/`)
Cada execução cria um diretório único `results/scratch_vs_tl/run_TIMESTAMP/` contendo os 13 artefatos obrigatórios:
1. `comparison_config.json`
2. `environment.json`
3. `git_metadata.json`
4. `compatibility_audit.csv`
5. `fold_manifest.csv`
6. `results.csv`
7. `predictions.csv`
8. `summary_by_dataset.csv`
9. `summary_by_architecture.csv`
10. `summary_by_method.csv`
11. `statistical_tests.csv`
12. `rankings.csv`
13. `report.md`

---

## 5. Comandos de Execução

### 5.1 Suíte de Testes Unitários (11 testes cobrindo os 10 invariantes + validação piloto)
```bash
python -m unittest tests/test_scratch_vs_tl_benchmark.py -v
```

### 5.2 Execução Piloto em Dados Reais (Ex.: 1 Dataset Alvo, 1 Seed)
```bash
python -m src.experiments.run_scratch_vs_tl_benchmark \
    --targets CWRU_12k \
    --seeds 42
```

### 5.3 Execução Completa do Benchmark em Dados Reais (5 Datasets Alvo, 66 Folds LOCO, 2 Arquiteturas, 3 Métodos, 2 Balanceamentos, 3 Seeds = 2.376 Avaliações)
```bash
python -m src.experiments.run_scratch_vs_tl_benchmark \
    --config configs/scratch_vs_tl.yaml \
    --data-root data/processed_tl/detrended_rms \
    --output-root results/scratch_vs_tl
```

### 5.4 Retomada a partir de Checkpoint (`--resume-dir`)
Caso uma execução longa seja interrompida no servidor, retome exatamente do ponto onde parou sem sobrescrever nem repetir avaliações concluídas:
```bash
python -m src.experiments.run_scratch_vs_tl_benchmark \
    --config configs/scratch_vs_tl.yaml \
    --data-root data/processed_tl/detrended_rms \
    --resume-dir results/scratch_vs_tl/run_YYYYMMDD_HHMMSS
```

