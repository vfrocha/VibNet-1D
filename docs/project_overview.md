# Visão Geral do Projeto — VibNet-1D e Etapa TabVibNet

## 1. Contexto e Objetivo Científico

O **VibNet-1D** é um projeto de pesquisa voltado ao diagnóstico inteligente de falhas em máquinas rotativas (rolamentos, engrenagens e motores elétricos) a partir de sinais de vibração, combinando aprendizado de máquina clássico, aprendizado profundo tabular e **transferência de aprendizado (*Transfer Learning* — TL)** entre diferentes bancadas experimentais (*cross-dataset*) e condições operacionais variáveis.

O desenvolvimento do projeto está estruturado em duas grandes etapas:

1. **Etapa 1 — TabVibNet (Concluída / Auditada)**: Classificação e transferência de aprendizado baseadas em um vetor unificado de **141 atributos estatísticos, espectrais e tempo-frequência** extraídos de janelas de 1 segundo do sinal de vibração. Essa etapa permitiu estabelecer *baselines* rigorosos sob avaliação *Leave-One-Condition-Out* (LOCO), diagnosticar o impacto físico da remoção de tendência (*detrending*) e da normalização por RMS de referência saudável (`detrended_rms`), investigar estratégias de balanceamento no domínio alvo (SMOTE e híbrido), testar pré-treinamento contrastivo supervisionado (*SupCon*) e comparar sistematicamente arquiteturas tabulares (`MLP` vs. `TabNet`) sob múltiplas combinações de domínios fonte.
2. **Etapa 2 — VibNet-1D no Domínio do Tempo (Planejada)**: Classificação e transferência de aprendizado operando diretamente sobre as janelas temporais brutas/normalizadas (`1D-CNN`, `ResNet-1D`, arquiteturas híbridas tempo-frequência e *encoders* auto-supervisionados 1D), compartilhando os mesmos protocolos de particionamento LOCO, normalização física por referência e rastreabilidade estabelecidos na Etapa TabVibNet.

---

## 2. Pipeline de Atributos do TabVibNet (141 Features)

Na etapa **TabVibNet**, cada arquivo bruto de vibração é segmentado em janelas não sobrepostas de **1 segundo** ($N = f_s$ amostras por janela). De cada janela são extraídos **141 atributos** (`src/features/extractors_v2.py` + `src/features/signalai.py`):

| Grupo de Atributos | Módulo de Origem | Quantidade | Descrição Resumida |
|---|---|---:|---|
| **Temporais Clássicos (v2)** | `src/features/extractors_v2.py` (`extract_time_features`) | 11 | `Mean`, `Std`, `Skewness`, `Kurtosis`, `RMS`, `Peak`, `Peak2Peak`, `CrestFactor`, `ShapeFactor`, `ImpulseFactor`, `ClearanceFactor` |
| **Espectrais FFT (v2)** | `src/features/extractors_v2.py` (`extract_freq_features`) | 5 | `SpectralCentroid`, `SpectralVariance`, `SpectralSkewness`, `SpectralKurtosis`, `PeakFreq` |
| **Temporais Avançados (SignalAI)** | `src/features/signalai.py` | 15 | Energia, potência média, raiz da amplitude, cruzamentos por zero (*zero-crossing*), entropia de Shannon temporal e fatores de forma/impulso estendidos |
| **Frequência / PSD Welch (SignalAI)** | `src/features/signalai.py` | 26 | Momentos espectrais (`f1`–`f13`), energias relativas e picos em 4 bandas espectrais (`b1`–`b4`), razão harmônica e entropia espectral |
| **Wavelet Packet — WPD (SignalAI)** | `src/features/signalai.py` | 18 | Energias absolutas, relativas e entropia dos 8 nós da decomposição *Wavelet Packet* (`db4`, nível 3), além de razão de alta/baixa frequência |
| **Decomposição Modal Empírica — EMD (SignalAI)** | `src/features/signalai.py` | 66 | Estatísticas temporais e espectrais das 3 primeiras Funções de Modo Intrínseco (`IMF1`, `IMF2`, `IMF3`, 22 atributos por IMF) |
| **Total Base (`TabVibNet`)** | `extract_all_features()` | **141** | Vetor padrão utilizado nos experimentos de *baseline* e transferência de aprendizado |
| **Atributos Auxiliares de Energia (Opcional)** | `scripts/tl/exp_tl_eight_configs.py` | +3 (144) | `Aux_Window_RMS`, `Aux_Window_B1`, `Aux_Window_B2` (avaliados e descartados após ablação demonstrar degradação em `detrended_rms`) |

---

## 3. Evolução Histórica dos Experimentos no Repositório

A auditoria identificou três gerações de scripts e resultados no repositório:

### Geração 1 — Exploração Inicial e *Baselines* Por Dataset (`src/experiments/old_experiments/` e `results/old_results/`)
- **Período dos artefatos**: Maio a Agosto de 2026 (`20260517` a `20260824`).
- **Escopo**: Validação inicial dos extratores de atributos (`exp_01` a `exp_05` em `CWRU_12k`), testes com `TabNet`, `Random Forest`, `SVM`, `XGBoost` e `MLP` em `HUST`, `PU`, `UORED` e `UOEMD`, e primeiros protótipos de transferência de aprendizado (`tl_results_20260803`, `tl_loco_results_20260810`, `tl_multihead_results_20260819`).
- **Limitações históricas detectadas**:
  - Os scripts em `src/experiments/old_experiments/` usavam caminhos relativos `../..` que quebraram quando movidos para a subpasta `old_experiments/`.
  - Em `src/models/build_tabular.py` (usado nos primeiros experimentos), o `TabNetClassifier` recebia `eval_set=[(X_test_scaled, y_test)]` para *early stopping*, configurando vazamento de seleção de época pelo fold de teste. Esse comportamento foi corrigido nas gerações seguintes (`build_tabnet_resnet_balance.py` e `exp_tl_cross_dataset_mlp_tabnet.py`).

### Geração 2 — Consolidação LOCO, Severidade, Anomalia e Diagnóstico de Normalização (`src/experiments/` e `results/`)
- **Período dos artefatos**: Agosto a Outubro de 2026 (`20260812` a `20261027`).
- **Escopo**:
  - **Detecção de Anomalias One-Class** (`run_anomaly_experiments.py` $\rightarrow$ `anomaly_results_20260812_084311.csv`): Treinamento exclusivo na classe `Normal` com `Isolation Forest`, `One-Class SVM` e `Autoencoder` (3 limiares).
  - **Baselines Supervisionados LOCO** (`run_baseline_experiments.py` $\rightarrow$ `baseline/baseline_loco_results_20260829_084039.csv`, `baseline_loco_results_20260928_142515.csv`, `baseline_loco_results_20261003_052352.csv`): Comparação de `Random Forest`, `XGBoost`, `SVM`, `TabNet` e `MLP (Scratch)` sob protocolo LOCO estrito.
  - **Protocolos de Severidade e MERT em CWRU** (`run_cwru12k_severity.py`, `run_cwru48k_severity.py`, `run_cwru48k_mert_protocol.py`, `run_unified_severity_baselines.py`).
  - **Busca de Subconjuntos de Fontes e Transferência Negativa** (`run_tl_subset_search.py`, `run_baseline_scratch.py` $\rightarrow$ `results/negativetransfer/`).
  - **Diagnóstico de Normalização em Transferência de Aprendizado** (`exp_normalization_*.py` $\rightarrow$ `results/normalization/`, `norm_tl_diagnosis_*`): Identificou que normalizar cada janela pelo seu próprio RMS destrói a discriminação energética entre sinal normal e falha, motivando a criação do pipeline `data/processed_tl/` com normalização pelo RMS médio da condição saudável do dataset (`RMS_ref`).

### Geração 3 — Experimentos TabVibNet Auditáveis com Metadados (`scripts/` e `results/tl_*/`)
- **Período dos artefatos**: Outubro de 2026 (`20261005` a `20261028`).
- **Características**: Diretórios `run_<timestamp>/` contendo `config.json`, `results.csv` (por fold/seed), múltiplos `summary*.csv` e logs completos.
- **Famílias experimentais concluídas**:
  1. **`tl_four_variants`** (`scripts/tl/exp_tl_four_variants.py` $\rightarrow$ `run_20261005_163159`): Compara as 4 variantes de pré-processamento (`raw`, `detrended`, `raw_rms`, `detrended_rms`) com `MLP` e `TABNET` no alvo `CWRU_48k`.
  2. **`tl_aux_features`** (`scripts/tl/exp_tl_eight_configs.py` $\rightarrow$ `run_20261012_093659`): Avalia as 4 variantes $\times$ presença/ausência das 3 features auxiliares (`Aux_Window_RMS`, `B1`, `B2`) em `CWRU_48k`.
  3. **`tl_balance_strategies`** (`scripts/exp_tl_balance_strategies.py` $\rightarrow$ 4 runs em `20261012`): Compara as estratégias de balanceamento no treino do alvo (`none`, `smote`, `hybrid`, `class_weights`) sobre `detrended_rms` em `CWRU_48k`.
  4. **`tl_contrastive_mlp`** (`scripts/exp_tl_contrastive_mlp.py` $\rightarrow$ 7 runs em `20261026`–`20261027`): Avalia o pré-treinamento contrastivo supervisionado por domínio (*SupCon* nos modos `joint` e `pretrain` vs. `baseline`) em duas arquiteturas de encoder (`baseline` [`128->64`] e `expanded` [`256->128->64`]) com cabeças `asymmetric` e `symmetric`, no alvo `CWRU_48k` com 3 seeds (`42, 123, 2026`).
  5. **`tl_cross_dataset_mlp_tabnet`** (`scripts/exp_tl_cross_dataset_mlp_tabnet.py` $\rightarrow$ `run_20261028_105603` e `run_20261028_173310`): Experimento fatorial completo comparando `MLP` vs. `TabNet` em `detrended_rms` + `smote`, cobrindo 4 fontes individuais e todas as 15 combinações não vazias de domínios fonte (`UORED`, `HUST`, `Mechanical_Gear`, `Electric_Motor`) sobre 5 domínios alvo (`CWRU_12k`, `CWRU_48k`, `HUST_Gearbox`, `PU`, `UOEMD`) e 3 seeds (`42, 123, 2026`).

---

## 4. Mapa Atual do Repositório

```text
VibNet-1D/
├── AGENTS.md                        # Diretrizes científicas e operacionais do repositório
├── README.md                        # Visão geral inicial do projeto
├── README_contrastive_mlp.md        # Documentação específica do experimento SupCon MLP
├── requirements.txt                 # Dependências Python (numpy, scipy, scikit-learn, torch, pytorch-tabnet, etc.)
├── docs/                            # Documentação científica e arquitetural
│   ├── dataset_mapping.md           # Mapeamento bruto -> classes e condições dos datasets
│   ├── distribuicao_classes_datasets.csv
│   ├── project_overview.md          # (Este documento)
│   ├── datasets_and_protocols.md    # Catálogo de datasets, pré-processamento e protocolos LOCO
│   ├── reproducibility.md           # Guia de reprodutibilidade, metadados e auditoria
│   ├── results/
│   │   └── tabvibnet_summary.md     # Síntese quantitativa e metodológica dos experimentos TabVibNet
│   └── decisions/                   # Registros de Decisão Metodológica e Arquitetural (ADRs)
├── src/
│   ├── data/                        # Ingestão, criação de Parquet e validação de integridade (validate_dataset_tl.py)
│   ├── preprocessing/               # Normalização por RMS saudável de referência (normalize_dataset_tl.py)
│   ├── features/                    # Extratores TabVibNet: extractors_v2.py (16) + signalai.py (125) = 141 features
│   ├── models/                      # Arquiteturas sklearn, PyTorch (HybridDLModel, SupCon MLP) e TabNet
│   └── experiments/                 # Scripts de experimentos da Geração 2 e subpasta old_experiments/ (Geração 1)
├── scripts/                         # Pontos de entrada e runners principais da Geração 3
│   ├── audit_experimental_results.py
│   ├── validate_dataset_tl.py
│   ├── exp_tl_balance_strategies.py
│   ├── exp_tl_contrastive_mlp.py
│   ├── exp_tl_cross_dataset_mlp_tabnet.py
│   ├── verify_mlp_contrastive.py
│   ├── verify_cross_dataset_mlp_tabnet.py
│   └── tl/                          # Runners de variantes e features auxiliares
├── tests/                           # Testes unitários (test_mlp_contrastive.py, test_audit_experimental_results.py)
├── results/                         # 220 artefatos de resultados brutos e agregados preservados
└── reports/
    └── reproducibility_audit/       # Inventário automatizado, métricas extraídas, anomalias e relatório Markdown
```

---

## 5. Diretrizes de Transição para a Arquitetura Modular (`src/vibnet/`)

Para preparar a próxima etapa de classificação diretamente no domínio do tempo sem quebrar a reprodutibilidade do **TabVibNet**, a evolução do código seguirá a separação planejada em `docs/decisions/004-planned-modular-architecture-vibnet.md`:

- `src/vibnet/data/`: Carregamento de janelas brutas/Parquet, manifesto de normalização e validação de integridade.
- `src/vibnet/preprocessing/`: Remoção de tendência linear (*detrend*), normalização por RMS saudável de referência (`RMS_ref`) e escalonamento livre de vazamento (*leak-free scaling*).
- `src/vibnet/evaluation/`: Geradores de partições LOCO, balanceamento isolado no treino do alvo (SMOTE/híbrido com proteção para lotes unitários em `BatchNorm1d`), cálculo padronizado de métricas (`Macro-F1`, `Balanced Accuracy`, `Accuracy`, `ROC-AUC` com flag `ROC_AUC_Valid`) e registro automático de metadados de reprodutibilidade (`git_commit`, `git_dirty`, hashes SHA-256).
- `src/vibnet/tabular/`: Extratores das 141 features estatísticas (`TabVibNet`) e modelos tabulares (`MLP`, `TabNet`, `SupCon MLP`, baselines `sklearn`).
- `src/vibnet/signal/`: Futuros modelos 1D no domínio do tempo (`1D-CNN`, `ResNet-1D`, *encoders* temporais), consumindo exatamente as mesmas partições LOCO e variantes de `data/processed_tl/` para permitir comparação científica direta contra o `TabVibNet`.
