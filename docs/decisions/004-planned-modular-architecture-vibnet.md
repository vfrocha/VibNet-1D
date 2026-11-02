# ADR 004: Plano de Arquitetura Modular (`src/vibnet/`) para Separação entre `TabVibNet` (Tabular) e `VibNet-1D` (Domínio do Tempo)

- **Status**: Proposto / Planejado (Fase 4 — Aguardando aprovação antes de mover módulos existentes)
- **Contexto**: Preparação do repositório `vfrocha/VibNet-1D` para a etapa de classificação diretamente a partir do sinal no domínio do tempo, preservando 100% da reprodutibilidade da etapa **TabVibNet**.

---

## 1. Motivação e Problemas Estruturais Diagnosticados na Fase 1

A auditoria não destrutiva da árvore atual do repositório identificou os seguintes pontos que dificultam a manutenção e a extensão para modelos 1D no domínio do tempo:

1. **Duplicação de Código e *Boilerplate***:
   - `scripts/validate_dataset_tl.py` (1.018 linhas) e `src/data/validate_dataset_tl.py` (1.018 linhas) são cópias 99,9% idênticas.
   - `scripts/exp_tl_balance_strategies.py` (1.410 linhas) e `scripts/tl/exp_tl_balance_strategies.py` (1.405 linhas) são cópias quase idênticas, e compartilham ~90% do código de carregamento, cache de features e agregação com `scripts/tl/exp_tl_four_variants.py` (1.389 linhas) e `scripts/tl/exp_tl_eight_configs.py` (1.459 linhas).
   - `src/models/build_tabnet_resnet.py` e `src/models/build_tabnet_resnet_balance.py` duplicam `HybridDLModel` e funções de treino, além de conterem uma classe `ResidualBlock1D` não utilizada.
   - `src/preprocessing/dataset_normalization.py` é um módulo órfão (não importado por nenhum script), enquanto `src/preprocessing/normalize_dataset_tl.py` reimplementa a normalização por RMS saudável como CLI.
2. **Caminhos Frágeis (*Hardcoded*) e Divergências de Configuração**:
   - Caminhos absolutos `/home/vfrocha/...` em `src/data/audit_vibdata.py`, `src/data/make_dataset.py`, `src/data/make_dataset_tl.py`, `scripts/check_class_distribution.py`, `scripts/find_normal.py` e `scripts/rename_normal.sh`.
   - Caminhos relativos quebrados (`../..`) e imports quebrados (`from src.experiments.engine_tabular`) nos 13 scripts legados de `src/experiments/old_experiments/`.
   - Divergência de taxa de amostragem ($f_s$) para `Mechanical_Gear` (`5000` vs `10000` Hz) e `Electric_Motor` (`50000` vs `42000` Hz) entre `src/data/make_dataset.py` e `scripts/exp_tl_cross_dataset_mlp_tabnet.py`.
3. **Arquivos Placeholder Vazios (0 bytes)**:
   - `src/__init__.py`, `src/train_source.py`, `src/train_target.py`, `src/models/architectures.py`, `src/models/transfer.py`.

---

## 2. Arquitetura Modular Proposta

A estrutura alvo separa claramente os componentes comuns reutilizáveis (`data`, `preprocessing`, `evaluation`) das duas famílias de modelagem (`tabular` para o **TabVibNet** e `signal` para o futuro **VibNet-1D** no domínio do tempo):

```text
VibNet-1D/
├── configs/
│   ├── datasets.yaml                      # Fonte única de verdade: fs, unidade física, classe normal, papel (source/target)
│   ├── tabvibnet/
│   │   ├── baseline_loco.yaml
│   │   ├── tl_four_variants.yaml
│   │   ├── tl_aux_features.yaml
│   │   ├── tl_balance_strategies.yaml
│   │   ├── tl_contrastive_mlp.yaml
│   │   └── tl_cross_dataset_mlp_tabnet.yaml
│   └── time_domain/
│       └── template_1d_loco.yaml          # Configuração futura para modelos de sinal temporal
├── src/
│   └── vibnet/
│       ├── __init__.py
│       ├── data/                          # Componentes comuns de dados
│       │   ├── __init__.py
│       │   ├── registry.py                # Metadados canônicos dos 10 datasets (lê configs/datasets.yaml)
│       │   ├── ingestion.py               # Leitura de arquivos brutos e janelamento de 1s (ex-make_dataset_tl.py)
│       │   ├── loader.py                  # Carregamento de Parquet por variante (raw, detrended, raw_rms, detrended_rms)
│       │   └── validation.py              # Validação física e estatística (consolida validate_dataset_tl.py)
│       ├── preprocessing/                 # Pré-processamento compartilhado de sinais e atributos
│       │   ├── __init__.py
│       │   ├── signal_norm.py             # Detrend linear e normalização por RMS saudável de referência (RMS_ref)
│       │   └── scaling.py                 # StandardScaler isolado por domínio/fold (leak-free)
│       ├── evaluation/                    # Protocolos, balanceamento, métricas e reprodutibilidade comuns
│       │   ├── __init__.py
│       │   ├── protocols.py               # Geradores de splits LOCO, Severity LOCO e One-Class
│       │   ├── balancing.py               # SMOTE, hybrid e class_weights (isolados no treino do alvo + proteção BatchNorm1d)
│       │   ├── metrics.py                 # Cálculo padronizado de Macro-F1, Bal Acc, ACC, ROC-AUC com ROC_AUC_Valid
│       │   ├── aggregation.py             # Geração de results.csv, summary*.csv e pareamento estatístico
│       │   └── provenance.py              # Captura de git_commit, git_dirty, comando, seeds, pacotes e hashes SHA-256
│       ├── tabular/                       # Família 1: TabVibNet (Features Estatísticas/Espectrais/WPD/EMD)
│       │   ├── __init__.py
│       │   ├── features/
│       │   │   ├── __init__.py
│       │   │   ├── extractors_v2.py       # 11 temporais + 5 espectrais FFT
│       │   │   ├── signalai.py            # 125 atributos (temporais, PSD Welch, WPD db4, EMD IMFs)
│       │   │   └── cache.py               # Extração em paralelo + cache determinístico por variante/dataset
│       │   └── models/
│       │       ├── __init__.py
│       │       ├── sklearn_baselines.py   # RF, XGBoost, SVM, Isolation Forest, One-Class SVM
│       │       ├── mlp_multihead.py       # HybridDLModel (CE multi-cabeça + fine-tuning no alvo)
│       │       ├── mlp_contrastive.py     # SupConLoss por domínio fonte + encoders 256_128 / 128_64
│       │       └── tabnet_transfer.py     # TabNetPretrainer não supervisionado + TabNetClassifier (com drop_last seguro)
│       └── signal/                        # Família 2: VibNet-1D no Domínio do Tempo (Etapa Futura)
│           ├── __init__.py
│           ├── datasets.py                # Dataset PyTorch 1D com reamostragem/padding opcional para fs heterogêneos
│           ├── encoders_1d.py             # Futuros backbones 1D (CNN-1D, ResNet-1D, WDCNN, SincNet)
│           └── transfer_1d.py             # Adaptação multi-cabeça e SupCon 1D sobre encoders temporais
├── experiments/
│   ├── tabvibnet/                         # Pontos de entrada limpos da família TabVibNet
│   │   ├── run_baseline_loco.py
│   │   ├── run_tl_variants_and_aux.py
│   │   ├── run_tl_balance_strategies.py
│   │   ├── run_tl_contrastive_mlp.py
│   │   └── run_tl_cross_dataset_mlp_tabnet.py
│   └── time_domain/                       # Pontos de entrada futuros para modelos no domínio do tempo
│       └── README.md
├── scripts/                               # Utilitários operacionais e wrappers de retrocompatibilidade
│   ├── audit_experimental_results.py
│   ├── validate_dataset_tl.py             # Wrapper fino chamando src.vibnet.data.validation
│   └── verify_*.py
├── tests/                                 # Suíte de testes unitários e de regressão
│   ├── test_audit_experimental_results.py
│   ├── test_mlp_contrastive.py
│   ├── test_loco_isolation_and_metrics.py
│   └── test_dataset_registry.py
├── docs/                                  # Documentação científica e ADRs
└── results/                               # Resultados organizados por família (preservando todos os existentes)
    ├── tabvibnet/                         # Symlinks ou novos runs de TabVibNet
    └── time_domain/                       # Novos runs de modelos 1D no domínio do tempo
```

---

## 3. Mapeamento Detalhado de Arquivos Afetados e Dependências

| Arquivo Atual | Destino Planejado na Migração | Ação Planejada | Dependências Internas |
|---|---|---|---|
| `src/features/extractors_v2.py` | `src/vibnet/tabular/features/extractors_v2.py` | Mover + manter re-export em `src/features/` | `numpy`, `scipy`, `signalai` |
| `src/features/signalai.py` | `src/vibnet/tabular/features/signalai.py` | Mover + manter re-export em `src/features/` | `numpy`, `scipy`, `pywt`, `EMD` |
| `src/data/make_dataset_tl.py` | `src/vibnet/data/ingestion.py` | Extrair lógica parametrizável sem caminho `/home/vfrocha` fixo | `src/vibnet/data/registry.py` |
| `src/preprocessing/normalize_dataset_tl.py` + `dataset_normalization.py` | `src/vibnet/preprocessing/signal_norm.py` | Unificar módulo e CLI de normalização por `RMS_ref` | `src/vibnet/data/registry.py` |
| `scripts/validate_dataset_tl.py` + `src/data/validate_dataset_tl.py` | `src/vibnet/data/validation.py` | Eliminar duplicação de 1.018 linhas; manter `scripts/validate_dataset_tl.py` como entrypoint CLI | `src/vibnet/data/registry.py` |
| `src/models/build_tabnet_resnet.py` + `build_tabnet_resnet_balance.py` | `src/vibnet/tabular/models/mlp_multihead.py` e `tabnet_transfer.py` | Unificar modelos, remover `ResidualBlock1D` morto e extrair `_balance_target_training` para `evaluation/balancing.py` | `src/vibnet/evaluation/` |
| `src/models/build_mlp_contrastive.py` | `src/vibnet/tabular/models/mlp_contrastive.py` | Mover + manter wrapper de import em `src/models/` | `src/vibnet/evaluation/` |
| `src/models/build_sklearn.py` + `build_pytorch_baselines.py` | `src/vibnet/tabular/models/sklearn_baselines.py` | Corrigir assinatura de retorno de `train_and_evaluate_pure_mlp` (incluir `acc` para retornar 5-tupla consistente) | `src/vibnet/evaluation/metrics.py` |
| `scripts/tl/exp_tl_four_variants.py`, `exp_tl_eight_configs.py`, `scripts/exp_tl_balance_strategies.py` | `experiments/tabvibnet/` | Substituir as ~4.200 linhas duplicadas por runners enxutos baseados em `src/vibnet/evaluation/` e `src/vibnet/tabular/` | `src/vibnet/tabular/`, `src/vibnet/evaluation/` |
| `scripts/exp_tl_contrastive_mlp.py` e `scripts/exp_tl_cross_dataset_mlp_tabnet.py` | `experiments/tabvibnet/` (mantendo wrappers em `scripts/`) | Extrair `get_code_identification()` para `src/vibnet/evaluation/provenance.py` e corrigir $f_s$ de `Mechanical_Gear` (`5000`) e `Electric_Motor` (`50000`) via `registry.py` | `src/vibnet/data/registry.py`, `src/vibnet/evaluation/provenance.py` |
| `src/train_source.py`, `src/train_target.py`, `src/models/architectures.py`, `src/models/transfer.py` | Remoção (arquivos de 0 bytes) | Limpar placeholders vazios após aprovação | Nenhuma |

---

## 4. Riscos Metodológicos e Operacionais da Migração

1. **Risco de Quebra de Imports em Scripts de Verificação Existentes (`verify_*.py` e `tests/`)**:
   - *Mitigação*: Manter *shims* de re-exportação em `src/features/`, `src/models/` e `scripts/` durante a transição para que comandos históricos (`python scripts/exp_tl_cross_dataset_mlp_tabnet.py`, `python scripts/verify_mlp_contrastive.py`) continuem funcionando sem alterações.
2. **Risco de Alteração na Sequência Pseudoaleatória (Seeds) ao Refatorar Loops**:
   - *Mitigação*: Preservar a ordem exata das chamadas `np.random.seed(seed)`, `torch.manual_seed(seed)`, `SMOTE(random_state=seed)` e instanciação de camadas nos módulos migrados, validando com teste de igualdade numérica em 1 fold sintético antes e depois da refatoração.
3. **Risco de Mistura de Artefatos Antigos e Novos em `results/`**:
   - *Mitigação*: **Nunca mover nem renomear os diretórios de runs já concluídos em `results/`**, pois os caminhos estão registrados em `reports/reproducibility_audit/results_inventory.csv` e na documentação científica. Apenas novos runs passarão a gravar em `results/tabvibnet/` ou `results/time_domain/`.
4. **Risco de Resolução de Taxa de Amostragem ($f_s$)**:
   - *Mitigação*: Centralizar `DATASETS_CONFIG` em `configs/datasets.yaml` e `src/vibnet/data/registry.py`, validando automaticamente contra o comprimento das janelas nos arquivos Parquet (`len(window) == fs_hz`).

---

## 5. Plano de Execução em Etapas e Testes Necessários (Após Aprovação)

A migração deve ser executada em **4 etapas incrementais e testáveis** (apenas após aprovação explícita deste plano):

- **Etapa M1 — Fundação Comum (`configs/datasets.yaml`, `src/vibnet/data/registry.py`, `src/vibnet/evaluation/provenance.py` e `metrics.py`)**:
  - *Teste exigido*: `tests/test_dataset_registry.py` verificando consistência de $f_s$, classes saudáveis e `RMS_ref` contra `data/processed_tl/normalization_manifest.json`.
- **Etapa M2 — Consolidação de Dados e Pré-processamento (`src/vibnet/data/validation.py` e `src/vibnet/preprocessing/signal_norm.py`)**:
  - *Teste exigido*: Executar `python scripts/validate_dataset_tl.py` (via wrapper) e verificar que produz exatamente os mesmos 179 checks de `validation_summary.csv`.
- **Etapa M3 — Migração do `TabVibNet` (`src/vibnet/tabular/` + *shims* de compatibilidade em `src/features/` e `src/models/`)**:
  - *Teste exigido*: Executar `py scripts/verify_mlp_contrastive.py`, `py scripts/verify_cross_dataset_mlp_tabnet.py` e `py -m unittest discover -s tests`.
- **Etapa M4 — Esqueleto de `src/vibnet/signal/` e `experiments/time_domain/`**:
  - *Teste exigido*: Teste de contrato verificando que o `DataLoader` 1D de `src/vibnet/signal/` gera exatamente os mesmos folds LOCO (`Test Condition`, `N_Train`, `N_Test`, distribuição de classes) que `src/vibnet/tabular/`.
