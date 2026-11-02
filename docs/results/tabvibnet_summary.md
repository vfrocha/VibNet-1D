# Síntese Científica dos Experimentos Concluídos — Etapa TabVibNet

Este documento consolida todos os experimentos concluídos na etapa **TabVibNet** (classificação e transferência de aprendizado baseadas nas **141 features estatísticas, espectrais, WPD e EMD**), seguindo rigorosamente os artefatos verificados em `results/` e inventariados por `scripts/audit_experimental_results.py`.

> **Nota Metodológica**: Nenhum resultado foi interpolado ou combinado entre protocolos incompatíveis. Informações não registradas nos artefatos históricos das Gerações 1 e 2 (como o `git_commit` SHA exato anterior à introdução do `config.json`) estão explicitamente marcadas como `[Não recuperado do artefato original]`.

---

## Visão Geral das Famílias Experimentais Concluídas

| # | Família Experimental | Script Gerador Principal | Protocolo | Alvos (`Targets`) | Fontes (`Sources`) | Modelos | Artefato Canônico Principal |
|---:|---|---|---|---|---|---|---|
| **1** | **Validação e Auditoria de Datasets (`dataset_validation_tl`)** | `scripts/validate_dataset_tl.py` | Auditoria física e estatística | 10 datasets (4 variantes) | — | — | `results/dataset_validation_tl/validation_summary.csv` |
| **2** | **Baselines Supervisionados LOCO (`baseline_loco`)** | `src/experiments/run_baseline_experiments.py` | `LOCO` | `CWRU_12k`, `CWRU_48k`, `HUST_Gearbox`, `UOEMD`, `PU` | Nenhuma (`Scratch`) | `RF`, `XGBoost`, `SVM`, `TabNet`, `MLP (Scratch)` | `results/baseline/baseline_loco_results_20260829_084039.csv`, `results/baseline_loco_results_20261003_052352.csv` |
| **3** | **Variantes de Pré-processamento em TL (`tl_four_variants`)** | `scripts/tl/exp_tl_four_variants.py` | `LOCO` (12 folds) | `CWRU_48k` | `CWRU_12k\|MFPT` | `MLP` (`HybridDLModel`), `TABNET` | `results/tl_four_variants/run_20261005_163159/results.csv` |
| **4** | **Ablação de Features Auxiliares de Energia (`tl_aux_features`)** | `scripts/tl/exp_tl_eight_configs.py` | `LOCO` (12 folds) | `CWRU_48k` | `CWRU_12k\|MFPT` | `MLP`, `TABNET` | `results/tl_aux_features/run_20261012_093659/results.csv` |
| **5** | **Estratégias de Balanceamento no Alvo (`tl_balance_strategies`)** | `scripts/exp_tl_balance_strategies.py` | `LOCO` (12 folds) | `CWRU_48k` | `CWRU_12k\|MFPT` | `MLP`, `TABNET` | `results/tl_balance_strategies/run_20261012_145854` a `150855` |
| **6** | **Diagnóstico Multi-Alvo e Ablação de Normalização (`norm_tl_diagnosis`)** | Scripts de diagnóstico de normalização | `LOCO` | `CWRU_48k`, `PU`, `UOEMD`, `HUST_Gearbox` | Variável | `MLP`, `TABNET` | `results/norm_tl_diagnosis_multitarget_results_20261001_160735.csv`, `results/norm_tl_diagnosis_ablation_20261027_093126_results.csv` |
| **7** | **Transferência Contrastiva Supervisionada (`tl_contrastive_mlp`)** | `scripts/exp_tl_contrastive_mlp.py` | `LOCO` (12 folds $\times$ 3 seeds: `42, 123, 2026`) | `CWRU_48k` | `CWRU_12k\|MFPT` | `MLP` (`baseline`, `joint`, `pretrain`; encoders `baseline`/`expanded`; cabeças `asymmetric`/`symmetric`) | `results/tl_contrastive_mlp/run_20261027_112749` a `113503` |
| **8** | **Fatorial Cross-Dataset `MLP` vs `TabNet` (`tl_cross_dataset_mlp_tabnet`)** | `scripts/exp_tl_cross_dataset_mlp_tabnet.py` | `LOCO` (66 folds totais $\times$ 3 seeds: `42, 123, 2026`) | `CWRU_12k`, `CWRU_48k`, `HUST_Gearbox`, `PU`, `UOEMD` | 15 combinações não vazias de `{UORED, HUST, Mechanical_Gear, Electric_Motor}` | `MLP`, `TabNet` | `results/tl_cross_dataset_mlp_tabnet/run_20261028_105603` e `run_20261028_173310` |
| **9** | **Experimentos Complementares (Anomalia, Severidade, MERT, Negative Transfer)** | `src/experiments/run_*.py` | `One-Class LOCO`, `Severity LOCO`, `MERT`, `LOCO Subset` | `CWRU_12k`, `CWRU_48k`, `HUST_Gearbox`, `UOEMD`, `PU` | Variável | `Isolation Forest`, `OC-SVM`, `AE`, `RF`, `XGB`, `SVM`, `MLP`, `TabNet` | `results/anomaly_results_*.csv`, `results/cwru*_severity_*.csv`, `results/negativetransfer/*.csv` |

---

## Experimento 1 — Validação Física e Estatística dos Datasets (`dataset_validation_tl`)

- **Objetivo**: Auditar a integridade das 4 variantes de pré-processamento (`raw`, `detrended`, `raw_rms`, `detrended_rms`) nos 10 datasets antes da modelagem de transferência de aprendizado.
- **Script**: `scripts/validate_dataset_tl.py` (cópia idêntica em `src/data/validate_dataset_tl.py`).
- **Artefatos de Saída**:
  - `results/dataset_validation_tl/validation_summary.csv` (179 verificações automatizadas)
  - `results/dataset_validation_tl/dataset_condition_class_summary.csv` (880 combinações `variante × dataset × condição × classe`)
  - `results/dataset_validation_tl/VALIDATION_REPORT.md` e `results/dataset_validation_tl/plots/*.png`
- **Resultados Observados**:
  - **174 `PASS`, 4 `WARN`, 1 `FAIL`** dentre as 179 verificações.
  - Nenhum valor `NaN`, `Inf` ou janela de variância zero nos 68.892 sinais por variante.
  - Em `raw_rms` e `detrended_rms`, todas as classes saudáveis próprias apresentaram razão $\overline{\text{RMS}} / \text{RMS}_{\text{ref}} = 1{,}000000$.
  - O único `FAIL` formal refere-se ao `CWRU_48k` não possuir `Class_0` própria, utilizando intencionalmente o $\text{RMS}_{\text{ref}} = 0{,}067780$ da `Class_0` de `CWRU_12k`.
  - Os 4 `WARN` correspondem ao desbalanceamento severo esperado em `Electric_Motor` (razão 6,5:1) e `MFPT` (razão 3,5:1) e à presença de 1 única condição (`Default`) nos domínios fonte puros.

---

## Experimento 2 — Baselines Supervisionados sob Protocolo LOCO (`baseline_loco`)

- **Objetivo**: Estabelecer o desempenho de referência sem transferência de aprendizado (*from scratch*) de 5 classificadores tabulares sobre as 141 features em cada domínio alvo.
- **Dados**: `data/processed/` (janelas de 1 s com `detrend` linear aplicado na ingestão por `src/data/make_dataset.py`).
  - Run `20260829_084039`: `CWRU_12k` (4 folds), `CWRU_48k` (12 folds), `HUST_Gearbox` (30 folds), `UOEMD` (16 folds).
  - Runs `20260928_142515` e `20261003_052352`: `CWRU_12k` (4 folds), `CWRU_48k` (12 folds), `HUST_Gearbox` (30 folds), `PU` (4 folds).
- **Protocolo**: `LOCO` estrito (`LeaveOneGroupOut` por `Condition`).
- **Modelos**: `Random Forest`, `XGBoost`, `SVM` (com `GridSearchCV` 3-fold interno ao treino), `TabNet` e `MLP (Scratch)`.
- **Pré-processamento e Balanceamento**: `StandardScaler` ajustado no treino de cada fold. No run `20261003_052352`, foi adicionado `SMOTE(random_state=42)` no treino de cada fold em `run_baseline_experiments.py` (linhas 67–74).
- **Seeds**: `42` (`git_commit`: `[Não recuperado do artefato original]`).
- **Resultados Quantitativos (`Macro-F1` Médio $\pm$ DP entre Folds LOCO)**:

| Dataset Alvo | Folds LOCO | Run / Configuração | `Random Forest` | `XGBoost` | `SVM` | `TabNet` | `MLP (Scratch)` |
|---|---:|---|---:|---:|---:|---:|---:|
| **`CWRU_12k`** | 4 | `20260829` (Sem SMOTE) | $0{,}9087 \pm 0{,}0668$ | $0{,}8989 \pm 0{,}0460$ | $0{,}8975 \pm 0{,}0689$ | $0{,}8897 \pm 0{,}0618$ | $0{,}9058 \pm 0{,}0784$ |
| **`CWRU_12k`** | 4 | `20261003` (Com SMOTE) | $0{,}9139 \pm 0{,}0607$ | $0{,}8862 \pm 0{,}0497$ | $0{,}8975 \pm 0{,}0689$ | **$0{,}9173 \pm 0{,}0634$** | *[Coluna deslocada]* ($0{,}9243^*$ recuperado de `Bal Acc`) |
| **`CWRU_48k`** | 12 | `20260829` (Sem SMOTE) | $0{,}8532 \pm 0{,}2804$ | $0{,}8093 \pm 0{,}2792$ | $0{,}8366 \pm 0{,}2979$ | **$0{,}9306 \pm 0{,}1464$** | $0{,}8858 \pm 0{,}1750$ |
| **`CWRU_48k`** | 12 | `20261003` (Com SMOTE) | $0{,}8532 \pm 0{,}2804$ | $0{,}8093 \pm 0{,}2792$ | $0{,}8366 \pm 0{,}2979$ | **$0{,}9289 \pm 0{,}1156$** | *[Coluna deslocada]* ($0{,}9054^*$ recuperado de `Bal Acc`) |
| **`HUST_Gearbox`** | 30 | `20260829` / `20261003` | $0{,}9586 \pm 0{,}1170$ | $0{,}9558 \pm 0{,}1157$ | $0{,}9616 \pm 0{,}1248$ | **$0{,}9854 \pm 0{,}0422$** (`0829`) / $0{,}9795$ (`1003`) | $0{,}9561 \pm 0{,}1437$ (`0829`) |
| **`UOEMD`** | 16 | `20260829` (Sem SMOTE) | **$0{,}8650 \pm 0{,}1456$** | $0{,}8355 \pm 0{,}1612$ | $0{,}7777 \pm 0{,}1239$ | $0{,}8300 \pm 0{,}0994$ | $0{,}8104 \pm 0{,}0877$ |
| **`PU`** | 4 | `20260928` (Sem SMOTE) | $0{,}6519 \pm 0{,}1669$ | $0{,}6280 \pm 0{,}2130$ | $0{,}5840 \pm 0{,}1880$ | $0{,}6000 \pm 0{,}1172$ | *[Coluna deslocada]* ($0{,}6163^*$ recuperado de `Bal Acc`) |
| **`PU`** | 4 | `20261003` (Com SMOTE) | $0{,}8570 \pm 0{,}2608$ | $0{,}9074 \pm 0{,}1154$ | $0{,}8051 \pm 0{,}2897$ | **$0{,}9078 \pm 0{,}1135$** | *[Coluna deslocada]* ($0{,}8144^*$ recuperado de `Bal Acc`) |

- **Limitações e Anomalias Detectadas**:
  1. **Bug de deslocamento de colunas (`SHIFTED_COLUMN_CORRUPTION`)**: Em `src/experiments/run_baseline_experiments.py` (linha 116), o script desempacota `mlp_acc, mlp_bacc, mlp_f1, mlp_auc, *_ = train_and_evaluate_pure_mlp(...)`, mas `build_pytorch_baselines.py` retorna apenas 4 elementos `(bal_acc, macro_f1, roc_auc, model)`. Como consequência, nos arquivos `baseline_loco_results_20260928_142515.csv` e `baseline_loco_results_20261003_052352.csv`, as 50 linhas de `MLP (Scratch)` têm `Bal Acc` gravado em `Accuracy`, `Macro F1` gravado em `Bal Acc`, `ROC-AUC` gravado em `Macro F1` e a string `"PureMLPBaseline(...)"` gravada em `ROC-AUC`. O arquivo anterior `results/baseline/baseline_loco_results_20260829_084039.csv` (7 colunas, sem coluna `Accuracy`) **não possui esse bug**.
  2. **Uso de `build_tabular.py` no `TabNet` de baseline**: `run_baseline_experiments.py` importa `build_tabular`, que passava `eval_set=[(X_test_scaled, y_test)]` para *early stopping* do `TabNet`. Nos experimentos da Geração 3 (`tl_*`), esse vazamento foi eliminado via `build_tabnet_resnet_balance.py` e `exp_tl_cross_dataset_mlp_tabnet.py`.

---

## Experimento 3 — Comparação das 4 Variantes de Pré-processamento em TL (`tl_four_variants`)

- **Objetivo**: Determinar qual combinação de remoção de tendência linear (*detrend*) e normalização pelo RMS saudável de referência (`RMS_ref`) maximiza a transferência de aprendizado para `CWRU_48k`.
- **Script e Artefatos**:
  - Script: `scripts/tl/exp_tl_four_variants.py` (usando `src/models/build_tabnet_resnet.py`).
  - Diretório: `results/tl_four_variants/run_20261005_163159/` (`config.json`, `results.csv` [96 linhas], `summary.csv`, `summary_by_variant_model.csv`, `run.log`).
- **Configuração (`config.json`)**:
  - **Alvo**: `CWRU_48k` (12 folds LOCO, 602 amostras de teste somadas nos folds válidos).
  - **Fontes**: `CWRU_12k|MFPT`.
  - **Variantes**: `raw`, `detrended`, `raw_rms`, `detrended_rms` (`141` features, `include_window_rms_feature: false`).
  - **Balanceamento**: Nenhum (`none`).
  - **Seed**: `42`.
- **Resultados Quantitativos (`results/tl_four_variants/run_20261005_163159/summary_by_variant_model.csv` e `summary.csv`)**:

| Variante | Modelo | Folds Válidos | `Macro-F1` (Mean $\pm$ DP) | `Macro-F1` Mediana | `ACC` Mean | `Bal Acc` Mean | `ROC-AUC` Mean* | `Macro_F1_Selection_Score` (Variante) |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **`detrended_rms`** | **`MLP`** | 12/12 | **$0{,}9202 \pm 0{,}1353$** | $1{,}0000$ | $0{,}9319$ | $0{,}9298$ | $0{,}8980$ | **$0{,}9055$** (1º) |
| **`detrended_rms`** | **`TABNET`** | 12/12 | **$0{,}8909 \pm 0{,}1774$** | $0{,}9500$ | $0{,}8995$ | $0{,}9014$ | $0{,}8945$ | **$0{,}9055$** (1º) |
| `raw_rms` | `MLP` | 12/12 | $0{,}8667 \pm 0{,}2123$ | $0{,}9704$ | $0{,}8912$ | $0{,}8917$ | $0{,}9050$ | $0{,}8715$ (2º) |
| `raw_rms` | `TABNET` | 12/12 | $0{,}8762 \pm 0{,}1980$ | $0{,}9418$ | $0{,}8914$ | $0{,}8863$ | $0{,}8942$ | $0{,}8715$ (2º) |
| `detrended` | `MLP` | 12/12 | $0{,}8977 \pm 0{,}1471$ | $0{,}9581$ | $0{,}9093$ | $0{,}9054$ | $0{,}9027$ | $0{,}8550$ (3º) |
| `detrended` | `TABNET` | 12/12 | $0{,}8124 \pm 0{,}1929$ | $0{,}8509$ | $0{,}8424$ | $0{,}8423$ | $0{,}8651$ | $0{,}8550$ (3º) |
| `raw` | `MLP` | 12/12 | $0{,}8305 \pm 0{,}2388$ | $0{,}8986$ | $0{,}8594$ | $0{,}8597$ | $0{,}8996$ | $0{,}8323$ (4º) |
| `raw` | `TABNET` | 12/12 | $0{,}8341 \pm 0{,}2163$ | $0{,}9146$ | $0{,}8549$ | $0{,}8514$ | $0{,}8885$ | $0{,}8323$ (4º) |

> **\*Nota sobre `ROC-AUC` Mean**: Inclui o fold `Load_0HP_Sev_0.021` com fallback `ROC-AUC = 0.0` devido à ausência da `Class_3` no teste; nos outros 11 folds, o `ROC-AUC` médio em `detrended_rms` supera $0{,}98$.
- **Conclusão Observada**: A combinação de remoção de tendência linear seguida de normalização pelo RMS saudável (`detrended_rms`) produziu o maior `Macro-F1` tanto para `MLP` ($0{,}9202$, ganho de $+8{,}97$ p.p. sobre `raw`) quanto para `TABNET` ($0{,}8909$, ganho de $+5{,}68$ p.p. sobre `raw`), vencendo também pelo `Macro_F1_Selection_Score` ($0{,}9055$).

---

## Experimento 4 — Ablação de Features Auxiliares de Energia (`tl_aux_features`)

- **Objetivo**: Avaliar se a reinjeção explícita da energia absoluta da janela (`Aux_Window_RMS`) e das energias de baixa/média frequência (`Aux_Window_B1`, `Aux_Window_B2`), totalizando 144 features (`rms_b1_b2`), melhora ou degrada a transferência de aprendizado em comparação às 141 features base (`base`).
- **Script e Artefatos**:
  - Script: `scripts/tl/exp_tl_eight_configs.py`.
  - Diretório: `results/tl_aux_features/run_20261012_093659/` (`config.json`, `results.csv` [192 linhas], `summary_by_experiment.csv`, `summary_by_experiment_model.csv`, `run.log`).
- **Configuração**: Alvo `CWRU_48k` (12 folds LOCO), fontes `CWRU_12k|MFPT`, 4 variantes $\times$ 2 conjuntos de features (`base` [141] vs. `rms_b1_b2` [144]), `seed = 42`.
- **Resultados Quantitativos (`results/tl_aux_features/run_20261012_093659/summary_by_experiment_model.csv` e `summary_by_experiment.csv`)**:

| Experimento (`Variante__Features`) | Nº Features | `MLP` Macro-F1 (Mean $\pm$ DP) | `TABNET` Macro-F1 (Mean $\pm$ DP) | `Macro_F1_Selection_Score` |
|---|---:|---:|---:|---:|
| **`detrended_rms__base`** | **141** | **$0{,}9202 \pm 0{,}1353$** | **$0{,}8909 \pm 0{,}1774$** | **$0{,}9055$** (1º) |
| `raw_rms__base` | 141 | $0{,}8667 \pm 0{,}2123$ | $0{,}8762 \pm 0{,}1980$ | $0{,}8715$ (2º) |
| `detrended__base` | 141 | $0{,}8977 \pm 0{,}1471$ | $0{,}8124 \pm 0{,}1929$ | $0{,}8550$ (3º) |
| `raw_rms__rms_b1_b2` | 144 | $0{,}8806 \pm 0{,}1887$ | $0{,}8227 \pm 0{,}1914$ | $0{,}8517$ (4º) |
| `raw__rms_b1_b2` | 144 | $0{,}8899 \pm 0{,}1876$ | $0{,}8074 \pm 0{,}2062$ | $0{,}8486$ (5º) |
| `detrended_rms__rms_b1_b2` | 144 | $0{,}8799 \pm 0{,}2046$ | $0{,}7966 \pm 0{,}1393$ | $0{,}8383$ (6º) |
| `detrended__rms_b1_b2` | 144 | $0{,}9002 \pm 0{,}1870$ | $0{,}7681 \pm 0{,}2287$ | $0{,}8342$ (7º) |
| `raw__base` | 141 | $0{,}8305 \pm 0{,}2388$ | $0{,}8341 \pm 0{,}2163$ | $0{,}8323$ (8º) |

- **Conclusão Observada**: Em `detrended_rms`, o vetor base de 141 features já incorpora o RMS normalizado pela referência saudável (uma vez que o atributo `RMS` de `extractors_v2.py` é linear na escala do sinal $\tilde{w} = w/\text{RMS}_{\text{ref}}$). Adicionar features auxiliares redundantes (`rms_b1_b2`) reduziu o `Macro-F1` de `detrended_rms` em $-4{,}03$ p.p. no `MLP` e $-9{,}42$ p.p. no `TABNET`, justificando a fixação definitiva em `141` features (`include_window_rms_feature = False`).

---

## Experimento 5 — Estratégias de Balanceamento no Domínio Alvo (`tl_balance_strategies`)

- **Objetivo**: Avaliar o efeito de 4 estratégias de balanceamento aplicadas exclusivamente ao conjunto de treino do domínio alvo (`none`, `smote`, `hybrid`, `class_weights`) sobre a variante `detrended_rms`.
- **Script e Artefatos**:
  - Script: `scripts/exp_tl_balance_strategies.py` (cópia em `scripts/tl/exp_tl_balance_strategies.py`), utilizando `src/models/build_tabnet_resnet_balance.py` (que corrige a fixação de `seed = 42` no `MLP`).
  - Diretórios (`CWRU_48k`, fontes `CWRU_12k|MFPT`, variante `detrended_rms`, 24 linhas por run):
    - `results/tl_balance_strategies/run_20261012_145854/` (`none`)
    - `results/tl_balance_strategies/run_20261012_150217/` (`class_weights`)
    - `results/tl_balance_strategies/run_20261012_150540/` (`smote`)
    - `results/tl_balance_strategies/run_20261012_150855/` (`hybrid`)
- **Resultados Quantitativos na Variante Vencedora `detrended_rms` (`CWRU_48k`, 12 Folds, Seed `42`)**:

| Estratégia de Balanceamento | Run Directory | `MLP` Macro-F1 (Mean $\pm$ DP) | `TABNET` Macro-F1 (Mean $\pm$ DP) | `ACC` Mean (`MLP` / `TABNET`) | `Bal Acc` Mean (`MLP` / `TABNET`) |
|---|---|---:|---:|---:|---:|
| **`smote`** | `run_20261012_150540` | **$0{,}9389 \pm 0{,}1289$** | $0{,}8851 \pm 0{,}1687$ | $0{,}9485$ / $0{,}8940$ | $0{,}9486$ / $0{,}8944$ |
| **`hybrid`** | `run_20261012_150855` | **$0{,}9389 \pm 0{,}1289$** | $0{,}8851 \pm 0{,}1687$ | $0{,}9485$ / $0{,}8940$ | $0{,}9486$ / $0{,}8944$ |
| `none` | `run_20261012_145854` | $0{,}9202 \pm 0{,}1353$ | **$0{,}8909 \pm 0{,}1774$** | $0{,}9319$ / $0{,}8995$ | $0{,}9298$ / $0{,}9014$ |
| `class_weights` | `run_20261012_150217` | $0{,}8525 \pm 0{,}1680$ | $0{,}8352 \pm 0{,}1495$ | $0{,}8735$ / $0{,}8500$ | $0{,}8677$ / $0{,}8429$ |

- **Explicação Metodológica — Por que `smote` e `hybrid` são idênticos bit a bit em `CWRU_48k`**:
  - Em `src/models/build_tabnet_resnet_balance.py` (linhas 60–95), a estratégia `hybrid` define a meta de amostras por classe como a **mediana inteira arredondada das contagens de treino**: `target_count = int(round(float(np.median(counts))))`.
  - No dataset `CWRU_48k` (que possui 3 classes), as contagens de treino em **todos os 12 folds LOCO** apresentam empate entre as duas classes majoritárias (por exemplo, `[184, 194, 194]` ou `[184, 200, 200]`).
  - A mediana de três números `[a, b, b]` com $a \le b$ é exatamente $b = \max(\text{counts})$. Assim, nenhuma classe possui contagem acima de `target_count` (zero subamostragem), e a classe minoritária sofre exatamente o mesmo *oversampling* via `SMOTE(random_state=seed)` até atingir $b$, gerando matrizes `(X_train_bal, y_train_bal)` idênticas às da estratégia `smote`.

---

## Experimento 6 — Diagnóstico Multi-Alvo e Ablação de Normalização (`norm_tl_diagnosis`)

- **Objetivo**: Avaliar sistematicamente estratégias de pré-treinamento, *fine-tuning* e normalização em múltiplos domínios alvo (`CWRU_48k`, `PU`, `UOEMD`, `HUST_Gearbox`).
- **Artefatos Encontrados**:
  - `results/norm_tl_diagnosis_multitarget_results_20261001_160735.csv` (1.736 linhas `fold_level`), acompanhado de `norm_tl_diagnosis_multitarget_summary_20261001_160735_by_strategy_model.csv` e `log_norm_tl_diagnosis_multitarget_20261001_160735.txt`.
  - `results/norm_tl_diagnosis_ablation_20261005_152759_results.csv` (2.976 linhas `fold_level`) e `results/norm_tl_diagnosis_ablation_20261027_093126_results.csv` (2.976 linhas `fold_level`, 24 estratégias $\times$ 2 modelos $\times$ 62 folds totais somando `CWRU_48k` [12], `PU` [4], `UOEMD` [16] e `HUST_Gearbox` [30]).
- **Limitação de Rastreabilidade**: Os scripts geradores originais dessas duas baterias intermediárias não estão versionados como arquivos separados na árvore atual (`[Script gerador original não versionado; substituído pelos runners da Geração 3]`).

---

## Experimento 7 — Transferência Contrastiva Supervisionada (`tl_contrastive_mlp`)

- **Objetivo**: Investigar o impacto da perda *Supervised Contrastive* (`SupConLoss`, $\tau = 0{,}07$) aplicada separadamente por domínio fonte em dois regimes (`joint`: perda conjunta CE + SupCon; e `pretrain`: pré-treinamento SupCon puro seguido de fine-tuning CE) frente ao pré-treinamento multi-cabeça puramente supervisionado (`baseline`), além da ablação do perfil do encoder (`baseline` [`128->64`] vs. `expanded` [`256->128->64`]) e do formato das cabeças (`asymmetric` vs. `symmetric`).
- **Script e Modelo**:
  - Runner: `scripts/exp_tl_contrastive_mlp.py`
  - Módulo de modelo: `src/models/build_mlp_contrastive.py`
  - Testes e verificação: `tests/test_mlp_contrastive.py` e `scripts/verify_mlp_contrastive.py`
- **Configuração Experimental (`config.json`)**:
  - **Alvo**: `CWRU_48k` (12 folds LOCO).
  - **Fontes**: `CWRU_12k|MFPT`.
  - **Variante**: `detrended_rms` (141 features), `balance_strategy: none`.
  - **Seeds**: `42` nos runs exploratórios (`20261026_*`) e **`3 seeds [42, 123, 2026]`** nos runs finais (`20261027_*`, 36 avaliações por configuração).
- **Resultados Quantitativos Consolidados (3 Seeds `[42, 123, 2026]`, Alvo `CWRU_48k` — `summary_across_seeds.csv`)**:

| Run Directory | Objetivo da Ablação | `Mode` | `Encoder Profile` | `Head Shape` | `Macro-F1` Mean $\pm$ Seed DP | `ACC` Mean $\pm$ Seed DP | `Bal Acc` Mean $\pm$ Seed DP | `ROC-AUC` OvR Mean $\pm$ Seed DP |
|---|---|---|---|---|---:|---:|---:|---:|
| **`run_20261027_112749`** | Modo de Treinamento (`baseline` vs `joint` vs `pretrain`) | **`pretrain`** | `baseline` (`128->64`) | `asymmetric` | **$0{,}9133 \pm 0{,}0116$** | **$0{,}9170 \pm 0{,}0127$** | **$0{,}9176 \pm 0{,}0114$** | **$0{,}9878 \pm 0{,}0009$** |
| `run_20261027_112749` | Modo de Treinamento | `joint` | `baseline` (`128->64`) | `asymmetric` | $0{,}9097 \pm 0{,}0104$ | $0{,}9164 \pm 0{,}0089$ | $0{,}9157 \pm 0{,}0076$ | $0{,}9782 \pm 0{,}0098$ |
| `run_20261027_112749` | Modo de Treinamento | `baseline` | `baseline` (`128->64`) | `asymmetric` | $0{,}9056 \pm 0{,}0101$ | $0{,}9144 \pm 0{,}0111$ | $0{,}9145 \pm 0{,}0119$ | $0{,}9744 \pm 0{,}0023$ |
| **`run_20261027_113203`** | Perfil do Encoder (`baseline` vs `expanded`) | **`joint`** | **`expanded` (`256->128->64`)** | `asymmetric` | **$0{,}9235 \pm 0{,}0131$** | **$0{,}9281 \pm 0{,}0107$** | **$0{,}9277 \pm 0{,}0111$** | **$0{,}9835 \pm 0{,}0046$** |
| `run_20261027_113203` | Perfil do Encoder | `joint` | `baseline` (`128->64`) | `asymmetric` | $0{,}9097 \pm 0{,}0104$ | $0{,}9164 \pm 0{,}0089$ | $0{,}9157 \pm 0{,}0076$ | $0{,}9782 \pm 0{,}0098$ |
| **`run_20261027_113503`** | Formato das Cabeças (`asymmetric` vs `symmetric`) | **`joint`** | `expanded` (`256->128->64`) | **`asymmetric`** | **$0{,}9235 \pm 0{,}0131$** | **$0{,}9281 \pm 0{,}0107$** | **$0{,}9277 \pm 0{,}0111$** | **$0{,}9835 \pm 0{,}0046$** |
| `run_20261027_113503` | Formato das Cabeças | `joint` | `expanded` (`256->128->64`) | `symmetric` | $0{,}8863 \pm 0{,}0150$ | $0{,}8908 \pm 0{,}0138$ | $0{,}8920 \pm 0{,}0118$ | $0{,}9722 \pm 0{,}0073$ |

- **Conclusão Observada**:
  1. Na arquitetura `baseline + asymmetric`, tanto `pretrain` ($0{,}9133$) quanto `joint` ($0{,}9097$) superaram consistentemente o modo puramente supervisionado `baseline` ($0{,}9056$) ao longo das 3 seeds (`42, 123, 2026`).
  2. A expansão da capacidade do encoder compartilhado para `expanded` (`256->128->64`) combinada com cabeças `asymmetric` (cabeças de fonte não lineares de 2 camadas e cabeça de alvo linear de 1 camada) elevou o `Macro-F1` médio em 3 seeds para **$0{,}9235 \pm 0{,}0131$** ($+3{,}72$ p.p. acima de cabeças `symmetric`).

---

## Experimento 8 — Estudo Fatorial Cross-Dataset `MLP` vs. `TabNet` (`tl_cross_dataset_mlp_tabnet`)

- **Objetivo**: Comparar de forma pareada `MLP` (`HybridDLModel` multi-cabeça) e `TabNet` (`TabNetPretrainer` não supervisionado nas fontes + `TabNetClassifier` no alvo) sob todas as 15 combinações não vazias de 4 domínios fonte heterogêneos sobre 5 domínios alvo.
- **Script e Verificação**:
  - Runner: `scripts/exp_tl_cross_dataset_mlp_tabnet.py` (inclui rastreamento `git_commit`, `git_dirty` e SHA-256 dos scripts em `config.json`).
  - Verificador: `scripts/verify_cross_dataset_mlp_tabnet.py`.
- **Configuração Experimental (`config.json`)**:
  - **Alvos (5)**: `CWRU_12k` (4 folds), `CWRU_48k` (12 folds), `HUST_Gearbox` (30 folds), `PU` (4 folds), `UOEMD` (16 folds) — total de **66 folds LOCO por seed**.
  - **Fontes (4)**: `UORED`, `HUST`, `Mechanical_Gear`, `Electric_Motor`.
  - **Variante e Balanceamento**: `detrended_rms` (141 features) + `smote` no treino do alvo.
  - **Seeds (3)**: `42, 123, 2026`.
- **Dois Runs Concluídos (`results/tl_cross_dataset_mlp_tabnet/`)**:
  1. **`run_20261028_105603`** (`git_commit: b07c7026...`, **1.980 avaliações** = 5 configs de fonte [`all_sources` + 4 `single_source`] $\times$ 66 folds $\times$ 2 modelos $\times$ 3 seeds):
     - Avaliações válidas: **1.965 / 1.980** (15 falhas de `TabNet` no fold `C4_1500rpm_0.7Nm_400N` de `PU`: 5 fontes $\times$ 3 seeds).
  2. **`run_20261028_173310`** (`git_commit: 69ca1860...`, **5.940 avaliações** = todas as **15 combinações não vazias** [`4 single_source`, `6 pair_sources`, `4 triple_sources`, `1 all_sources`] $\times$ 66 folds $\times$ 2 modelos $\times$ 3 seeds):
     - Avaliações válidas: **5.895 / 5.940** (45 falhas de `TabNet` no fold `C4_1500rpm_0.7Nm_400N` de `PU`: 15 combinações $\times$ 3 seeds).

### 8.1. Desempenho por Domínio Alvo: Melhor Fonte Única vs. Todas as 4 Fontes (`all_sources`) vs. Melhor Subconjunto (`run_20261028_173310`)

| Alvo (`Target`) | Modelo | Folds Válidos (3 Seeds) | Todas as 4 Fontes (`all_sources`) Macro-F1 | Melhor Fonte Individual (`single_source`) | Melhor Combinação Geral ( dentre as 15 em `run_173310`) | Placar Pareado `MLP` vs `TabNet` (`summary_paired_mlp_vs_tabnet.csv`) |
|---|---|---:|---:|---|---|---|
| **`CWRU_12k`** | **`MLP`** | 12/12 | $0{,}9058 \pm 0{,}0726$ | `HUST` ($0{,}9048 \pm 0{,}0688$) | **`UORED\|Electric_Motor`** (**$0{,}9110 \pm 0{,}0668$**) | **`MLP` vence 15/15** |
| **`CWRU_12k`** | **`TabNet`** | 12/12 | $0{,}8803 \pm 0{,}0592$ | `Mechanical_Gear` ($0{,}8778 \pm 0{,}0565$) | `UORED\|HUST\|Electric_Motor` ($0{,}8897 \pm 0{,}0568$) | (`TabNet` vence 0/15) |
| **`CWRU_48k`** | **`MLP`** | 36/36 | $0{,}8842 \pm 0{,}1889$ | `Mechanical_Gear` ($0{,}8984 \pm 0{,}1638$) | **`UORED\|Mechanical_Gear\|Electric_Motor`** (**$0{,}9232 \pm 0{,}1145$**) | **`MLP` vence 11/15** |
| **`CWRU_48k`** | **`TabNet`** | 36/36 | $0{,}9077 \pm 0{,}0975$ | `HUST` ($0{,}8858 \pm 0{,}1225$) | `UORED\|HUST\|Mechanical_Gear\|Electric_Motor` ($0{,}9077 \pm 0{,}0975$) | (`TabNet` vence 4/15) |
| **`HUST_Gearbox`** | **`MLP`** | 90/90 | $0{,}9594 \pm 0{,}1219$ | **`UORED`** (**$0{,}9630 \pm 0{,}1106$**) | **`UORED`** (**$0{,}9630 \pm 0{,}1106$**) | **`MLP` vence 13/15** |
| **`HUST_Gearbox`** | **`TabNet`** | 90/90 | $0{,}9465 \pm 0{,}1306$ | `Mechanical_Gear` ($0{,}9491 \pm 0{,}1376$) | `Mechanical_Gear\|Electric_Motor` ($0{,}9559 \pm 0{,}0980$) | (`TabNet` vence 2/15) |
| **`PU`** | **`MLP`** | 12/12 (9 comuns*) | **$0{,}8410 \pm 0{,}2491$** ($0{,}8082^*$ nos 9 comuns) | `UORED` ($0{,}8295 \pm 0{,}2550$) | **`UORED\|HUST\|Mechanical_Gear\|Electric_Motor`** (**$0{,}8410 \pm 0{,}2491$**) | **`MLP` vence 8/15** (nos 9 folds comuns) |
| **`PU`** | **`TabNet`** | 9/12* | $0{,}7952 \pm 0{,}3047^*$ | `Electric_Motor` ($0{,}8220 \pm 0{,}2651^*$) | `Mechanical_Gear\|Electric_Motor` ($0{,}8236 \pm 0{,}2718^*$) | (`TabNet` vence 7/15*) |
| **`UOEMD`** | **`MLP`** | 48/48 | $0{,}8358 \pm 0{,}1016$ | `HUST` ($0{,}8296 \pm 0{,}1141$) | **`UORED\|HUST\|Electric_Motor`** (**$0{,}8409 \pm 0{,}1071$**) | **`MLP` vence 15/15** |
| **`UOEMD`** | **`TabNet`** | 48/48 | $0{,}6362 \pm 0{,}1372$ | `UORED` ($0{,}6638 \pm 0{,}1081$) | `UORED\|Mechanical_Gear\|Electric_Motor` ($0{,}6844 \pm 0{,}1193$) | (`TabNet` vence 0/15) |

> **\*Nota Crítica sobre `PU` no `TabNet`**: As médias do `TabNet` em `PU` refletem apenas **3 dos 4 folds LOCO** (9 avaliações válidas em vez de 12), pois o fold `C4_1500rpm_0.7Nm_400N` falhou em 100% das execuções do `TabNet` (`Fold_Status = ERROR_MODEL_EXECUTION`). Na comparação pareada estrita sobre os mesmos folds válidos (`summary_paired_mlp_vs_tabnet.csv` em `run_20261028_173310`), o `MLP` superou o `TabNet` em **62 dos 75 pares `Alvo × Combinação de Fontes` (82,7%)**, incluindo 100% das combinações em `CWRU_12k` (15/15) e `UOEMD` (15/15, com vantagem média de $+15$ a $+21$ p.p. de `Macro-F1`).

### 8.2. Diagnóstico Técnico da Falha do `TabNet` no Fold `C4_1500rpm_0.7Nm_400N` de `PU`
- **Causa Raiz**: Quando `C4_1500rpm_0.7Nm_400N` é isolado para teste no dataset `PU` e o conjunto de treino das outras 3 condições é balanceado via SMOTE, o número resultante de amostras de treino pós-SMOTE é `N_Train_After = 8577`.
- Como `TabNetClassifier` é treinado com `batch_size = 64` e `drop_last = False` no `DataLoader` interno do `pytorch-tabnet`, o último mini-batch de cada época possui exatamente:
  $$8577 \bmod 64 = 1\text{ amostra}$$
- A camada `BatchNorm1d` da rede do `TabNet` em modo `train()` exige mais de 1 amostra por canal para estimar a variância do batch, lançando `ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 141])`.

---

## Experimento 9 — Experimentos Complementares da Geração 2

### 9.1. Detecção de Anomalias *One-Class* (`anomaly_results_20260812_084311.csv`)
- **Script**: `src/experiments/run_anomaly_experiments.py`.
- **Protocolo**: Treino exclusivo com amostras da classe `Normal` nos folds de treino LOCO; avaliação binária (`Normal` vs. `Anomalia`) no fold de teste em 4 datasets (`CWRU_12k`, `HUST_Gearbox`, `UOEMD`, `PU`, total de 50 folds $\times$ 5 modelos = 250 linhas).
- **Resultados (`F1 (Anomaly)` Médio $\pm$ DP)**:
  - `CWRU_12k` (4 folds): `Isolation Forest` atingiu **`F1 (Anomaly)` = $0{,}9723 \pm 0{,}0185$** (`Bal Acc` = $0{,}8641$).
  - `HUST_Gearbox` (30 folds): `AE (Percentil 99)` atingiu `F1 (Anomaly)` = $0{,}8491 \pm 0{,}0410$ (`Bal Acc` = $0{,}5267$).
  - `UOEMD` (16 folds): `AE (Percentil 99)` atingiu `F1 (Anomaly)` = $0{,}8882 \pm 0{,}0040$ (`Bal Acc` = $0{,}5000$, prevendo majoritariamente a classe anômala).
  - `PU` (4 folds): `AE (Percentil 99)` atingiu `F1 (Anomaly)` = $0{,}8955 \pm 0{,}0029$ (`Bal Acc` = $0{,}5009$).

### 9.2. Protocolos de Severidade e MERT em CWRU (`cwru*_severity_*`, `cwru48k_mert_*`)
- **Scripts**: `src/experiments/run_cwru12k_severity.py`, `run_cwru48k_severity.py`, `run_cwru48k_mert_protocol.py`, `run_unified_severity_baselines.py`.
- **Resultados em `cwru_unified_severity_loco_20260929_144600.csv`** (Severidade LOCO unificada com `RF`, `XGBoost`, `SVM`, `TabNet`, `MLP (Scratch)` e `Hybrid DL`):
  - Em `CWRU_12k` (4 severidades): `Random Forest` ($0{,}8730 \pm 0{,}1047$) e `TabNet` ($0{,}8608 \pm 0{,}1070$) lideraram.
  - Em `CWRU_48k` (3 severidades): `SVM` ($0{,}9218 \pm 0{,}0575$) e `MLP (Scratch)` ($0{,}9082 \pm 0{,}0622$) lideraram.
- **Limitação detectada**: Nos scripts isolados `run_cwru12k_severity.py` e `run_cwru48k_mert_protocol.py`, há um descompasso de desempacotamento (esperam 4 valores de `build_sklearn.train_and_evaluate`, que passou a retornar 5 valores), e diversas execuções de `GridSearchCV` falharam quando o número de condições era insuficiente para `cv=3`, registrando `0.0` (sinalizados como `ZERO_METRIC_FAILURE` no relatório de auditoria).

### 9.3. Busca Exaustiva de Subconjuntos de Fontes e Transferência Negativa (`results/negativetransfer/`)
- **Scripts**: `src/experiments/run_tl_subset_search.py` e `src/experiments/run_baseline_scratch.py`.
- **Artefatos**: `tl_subset_search_results_20260915_202128.csv` (1.134 linhas) e `tl_subset_search_results_20260917_102219.csv` (3.906 linhas, avaliando todas as 63 combinações não vazias de 6 fontes nos 62 folds LOCO dos 4 domínios alvo com `Hybrid DL`).
- **Achado Principal**: Demonstrou empiricamente a ocorrência de **transferência negativa** quando fontes com dinâmicas vibratórias díspares eram combinadas sem seleção criteriosa e antes da normalização por referência saudável (`detrended_rms`), motivando tanto o pipeline `data/processed_tl/` quanto o estudo fatorial controlado `tl_cross_dataset_mlp_tabnet`.

---

## Síntese das Conclusões Consolidadas do TabVibNet

1. **Pré-processamento Físico (`detrended_rms`) é Determinante**: Remover a tendência linear de cada janela e normalizar a amplitude pelo RMS médio da condição saudável de referência do dataset ($\text{RMS}_{\text{ref}}$) elevou o `Macro-F1` em `CWRU_48k` de $0{,}8305$ (`raw`) para **$0{,}9202$** (`MLP`) e de $0{,}8341$ para **$0{,}8909$** (`TABNET`).
2. **Vetor de 141 Features sem Redundância Auxiliar**: O vetor base de 141 atributos (`extractors_v2` + `signalai`) em `detrended_rms` já preserva a informação de severidade energética; a adição das 3 features auxiliares (`Aux_Window_RMS`, `B1`, `B2`) reduziu o `Macro_F1_Selection_Score` de $0{,}9055$ para $0{,}8383$.
3. **Balanceamento `SMOTE` e Arquitetura Contrastiva Assimétrica**: Em `CWRU_48k`, o balanceamento `smote` no treino do alvo elevou o `MLP` para **$0{,}9389 \pm 0{,}1289$**, enquanto a perda contrastiva conjunta (`joint`) com encoder expandido (`256->128->64`) e cabeças assimétricas alcançou **$0{,}9235 \pm 0{,}0131$** ao longo de 3 seeds (`42, 123, 2026`) mesmo sem SMOTE (`balance_strategy: none`).
4. **Superioridade do `MLP` Multi-Cabeça sobre o `TabNet` e Seleção de Subconjuntos de Fontes**: No experimento fatorial de 7.920 avaliações totais (`tl_cross_dataset_mlp_tabnet`), o `MLP` venceu o `TabNet` em **82,7% dos pares (62/75)**, e subconjuntos seletivos de 2 ou 3 fontes (`UORED|Mechanical_Gear|Electric_Motor` em `CWRU_48k` com $0{,}9232$; `UORED|Electric_Motor` em `CWRU_12k` com $0{,}9110$; `UORED|HUST|Electric_Motor` em `UOEMD` com $0{,}8409$; `UORED` em `HUST_Gearbox` com $0{,}9630$) superaram o uso indiscriminado de todas as 4 fontes (`all_sources`) em 4 dos 5 domínios alvo.
