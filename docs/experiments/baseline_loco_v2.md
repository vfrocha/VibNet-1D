# Benchmark Supervisionado Pareado e Livre de Vazamento (`baseline_loco_v2`)

- **Identificador do Experimento (`experiment_id`)**: `baseline_loco_v2`
- **Tarefa (`task`)**: `diagnosis` (diagnóstico multiclasse no vocabulário nativo de cada dataset alvo)
- **Script Principal**: [`src/experiments/run_baseline_loco_v2.py`](../../src/experiments/run_baseline_loco_v2.py)
- **Suíte de Testes Unitários**: [`tests/test_baseline_loco_v2.py`](../../tests/test_baseline_loco_v2.py)
- **Diretório de Saída Padrão**: `results/baseline_loco_v2/run_YYYYMMDD_HHMMSS/`

---

## 1. Objetivo do `baseline_loco_v2`

O experimento **`baseline_loco_v2`** estabelece uma referência supervisionada rigorosa, reprodutível, livre de vazamento (*leakage-free*) e **diretamente pareada** para avaliar os ganhos reais de Transferência de Aprendizado (*Transfer Learning* — TL) nos experimentos atuais (`scripts/exp_tl_cross_dataset_mlp_tabnet.py`) e futuros do repositório **VibNet-1D**.

Todos os modelos do `baseline_loco_v2` são treinados **exclusivamente nos dados das $K - 1$ condições operacionais de treinamento do próprio dataset alvo**, sem qualquer amostra de domínios fonte e sem pesos pré-treinados.

---

## 2. Por Que o Baseline Legado (`run_baseline_experiments.py`) Não Pode Ser Usado Diretamente na Comparação com TL

A inspeção do runner histórico [`src/experiments/run_baseline_experiments.py`](../../src/experiments/run_baseline_experiments.py) e de seus módulos auxiliares ([`src/models/build_tabular.py`](../../src/models/build_tabular.py), [`src/models/build_sklearn.py`](../../src/models/build_sklearn.py), [`src/models/build_pytorch_baselines.py`](../../src/models/build_pytorch_baselines.py)) identificou divergências metodológicas que impedem a comparação pareada direta com os experimentos de TL da Geração 3 (`tl_cross_dataset_mlp_tabnet`):

1. **Divergência de Pré-processamento de Sinal (`data/processed/` vs. `data/processed_tl/detrended_rms/`)**:
   - O runner legado carregava janelas de `data/processed/`, enquanto o benchmark de TL opera sobre `data/processed_tl/detrended_rms/` (remoção de tendência linear + normalização pelo $\text{RMS}_{\text{ref}}$ da classe saudável de referência do dataset, conforme [`ADR-001`](../decisions/001-rms-reference-normalization-and-detrending.md)).
2. **Ausência do Dataset Alvo `UOEMD`**:
   - O script legado avaliava apenas 4 datasets de diagnóstico (`CWRU_12k`, `CWRU_48k`, `PU`, `HUST_Gearbox`) e omitia `UOEMD` (16 folds LOCO, 8 classes), que faz parte dos 5 domínios alvo do benchmark `tl_cross_dataset_mlp_tabnet`.
3. **Vazamento do Fold de Teste no *Early Stopping* do `TabNet` ([`ADR-002`](../decisions/002-loco-isolation-and-target-balancing.md))**:
   - Em `src/models/build_tabular.py` (`train_and_evaluate_tabnet`), o conjunto de teste `eval_set=[(X_train_scaled, y_train), (X_test_scaled, y_test)]` era passado diretamente ao `TabNetClassifier.fit(..., patience=20)`, utilizando o fold de teste retido para interromper o treinamento e selecionar o melhor *checkpoint*.
4. **Validação Cruzada Interna Não Agrupada por Condição Operacional**:
   - Em `src/models/build_sklearn.py`, o `GridSearchCV(cv=3)` dividia janelas aleatoriamente sem agrupar por condição operacional, misturando janelas correlacionadas da mesma condição entre treino e validação interna.
5. **Ausência de `SMOTE` e Execução em Seed Única**:
   - O runner legado não avaliava o pareamento `none` vs. `smote` com auditoria de contagens por classe nem executava as 3 seeds canônicas (`[42, 123, 2026]`).
6. **Corrupção de Tupla no `MLP (Scratch)` e Substituição Silenciosa de Falhas por `0.0`**:
   - `train_and_evaluate_pure_mlp` retornava uma 4-tupla `(bal_acc, macro_f1, roc_auc, model)` que era desempacotada em 5 variáveis em `run_baseline_experiments.py`, deslocando a representação textual do módulo PyTorch para a coluna `ROC-AUC` (`SHIFTED_COLUMN_CORRUPTION`), além de substituir métricas de execuções falhas ou `ROC-AUC` indefinido por `0.0`.

> **Preservação Histórica**: Conforme as diretrizes do projeto (`AGENTS.md`), `src/experiments/run_baseline_experiments.py` e os arquivos `results/baseline_loco_results_*.csv` foram integralmente preservados como artefatos históricos.

---

## 3. Datasets Alvo e Contagem Canônica de Folds LOCO

O `baseline_loco_v2` avalia exatamente os 5 datasets alvo do benchmark *cross-dataset* de TL (`TARGET_DATASETS`), validados antes da execução contra o registro canônico [`src/vibnet/data/registry.py`](../../src/vibnet/data/registry.py):

| Dataset Alvo | Taxa de Amostragem ($f_s$) | Componente | Nº de Folds LOCO | Condições Operacionais (`Test_Condition`) |
|---|---:|---|---:|---|
| **`CWRU_12k`** | $12.000\text{ Hz}$ | Rolamento (*Drive End*) | **4** | `Load_0HP`, `Load_1HP`, `Load_2HP`, `Load_3HP` |
| **`CWRU_48k`** | $48.000\text{ Hz}$ | Rolamento (*Drive End*) | **12** | `Load_{0,1,2,3}HP_Sev_{0.007,0.014,0.021}` |
| **`HUST_Gearbox`** | $25.600\text{ Hz}$ | Caixa de Engrenagens | **30** | `Cond_{20,25,30,35,40}_{0..4}` e `Cond_L{0..4}_VS_0_40_0` |
| **`UOEMD`** | $42.000\text{ Hz}$ | Motor + Rolamento | **16** | 16 perfis de carga e velocidade (`Load_Loaded_*`, `Load_No_Load_*`) |
| **`PU`** | $64.000\text{ Hz}$ | Rolamento | **4** | `C1_1500rpm_0.7Nm_1000N` .. `C4_1500rpm_0.7Nm_400N` |
| **Total** | — | — | **66** | **66 folds LOCO por seed / modelo / estratégia de balanceamento** |

Para o benchmark completo:
$$\text{Total de avaliações} = 66\text{ folds} \times 3\text{ seeds} \times 2\text{ estratégias de balanceamento} \times 5\text{ modelos} = 1.980\text{ avaliações}.$$

---

## 4. Pré-processamento e Representação de Atributos (`141` Features)

Para garantir paridade estrita com `scripts/exp_tl_cross_dataset_mlp_tabnet.py`:

- **Variante de Pré-processamento**: `detrended_rms`, lida exclusivamente de `data/processed_tl/detrended_rms/<Dataset>/<Condition>/<Class>/*.npy`.
- **Bloqueio de Substituição Silenciosa**: `validate_data_root_is_detrended_rms()` rejeita explicitamente qualquer caminho que aponte para `data/processed/` ou para outra variante diferente de `detrended_rms`.
- **Extração Canônica de 141 Atributos**:
  - Utiliza `extract_fusion_features(X_windows, fs_hz, extract_advanced_features)` ([`src/features/signalai_wrapper.py`](../../src/features/signalai_wrapper.py) + [`src/features/extractors_v2.py`](../../src/features/extractors_v2.py)), seguida de `np.nan_to_num(..., nan=0.0, posinf=0.0, neginf=0.0)`.
  - Nenhum atributo auxiliar experimental é adicionado (`Feature_Dimension == 141`).

---

## 5. Isolamento LOCO Estrito e Prevenção de Vazamento

Em cada dataset $d$ e fold $k \in \{0, \dots, K-1\}$ correspondente à condição de teste $c_k$:

1. **Particionamento por Condição Operacional**:
   - **Conjunto de Teste (`test_mask`)**: 100% das janelas com `Condition == c_k`.
   - **Conjunto de Treinamento Alvo (`train_mask`)**: 100% das janelas das demais $K - 1$ condições (`Condition != c_k`).
   - Verificado em tempo de execução por `assert_no_condition_leakage()`, garantindo interseção vazia tanto de índices de amostras quanto de identificadores de condição operacional.
2. **Assinatura Criptográfica da Partição (`Sample_Split_Signature`)**:
   - Os identificadores determinísticos `<Dataset>/<Condition>/<Class>/<filename>.npy` de treino e teste geram um hash SHA-256 truncado (`Sample_Split_Signature`) registrado em `results.csv`, comprovando que todos os 5 modelos recebem exatamente as mesmas amostras de treino e teste em cada fold.
3. **Padronização (`StandardScaler`) Isolada no Treino**:
   - Ajustada (`fit_transform`) exclusivamente sobre `X_train_raw` ($K - 1$ condições).
   - O conjunto de teste recebe apenas `scaler.transform(X_test_raw)`.
4. **Balanceamento (`SMOTE`) Isolado no Treino Escalado**:
   - Aplicado exclusivamente sobre `(X_train_scaled, y_train_raw)` quando `balance_strategy == "smote"`.
   - O conjunto de teste jamais passa por `SMOTE`.

---

## 6. Estratégia de Validação Interna (Sem Vazamento da Condição de Teste)

Para algoritmos que utilizam validação interna para seleção de época (*early stopping* no `TabNet Scratch`):

1. **Particionamento Interno Agrupado por Condição (`create_internal_validation_split`)**:
   - Opera **exclusivamente dentro das $K - 1$ condições de treinamento** (`X_train_raw`, `y_train_raw`, `conds_train_raw`); a condição de teste retida $c_k$ jamais está presente.
   - Prioriza `GroupShuffleSplit` sobre os grupos de condição operacional de treinamento (`Validation_Split_Mode = "grouped_condition"`), retendo $\approx 20\%$ das condições de treinamento como `inner_val`, desde que todas as classes presentes em `y_train_raw` permaneçam representadas em `inner_train`.
   - Caso reter uma condição inteira remova alguma classe rara de `inner_train`, utiliza `StratifiedShuffleSplit` estritamente dentro de `X_train_raw` (`Validation_Split_Mode = "stratified_within_train"`).
2. **Protocolo de Duas Etapas no `TabNet Scratch`**:
   - **Etapa 1 (Seleção de Época Livre de Vazamento)**: Ajusta um `StandardScaler` interno exclusivamente em `X_inner_train`, aplica `balance_strategy` apenas em `X_inner_train_scaled`, e treina `TabNetClassifier` monitorando `eval_set=[(X_inner_val_scaled, y_inner_val)]` com `patience=15` para selecionar `selected_epochs = max(1, best_epoch + 1)`. O conjunto de teste `X_test_scaled` **nunca** é passado para `eval_set`.
   - **Etapa 2 (Refit no Treinamento Completo Permitido)**: Re-inicializa um `TabNetClassifier` do zero com a mesma `seed` e treina sobre a totalidade da partição de treinamento balanceada permitida `(X_train_bal, y_train_bal)` por exatamente `selected_epochs` épocas (`eval_set=[]`, `patience=0`), realizando então a predição única em `X_test_scaled`.

---

## 7. Modelos Avaliados e Hiperparâmetros

Todos os 5 modelos são treinados **do zero (*from scratch*)** sem dados de domínios fonte e sem pesos pré-treinados. Quando um fold de treinamento contém um subconjunto das classes do vocabulário global do dataset, `project_predictions_to_full_vocabulary()` mapeia rótulos contíguos locais de volta para o `LabelEncoder` canônico do dataset e projeta a matriz de probabilidades para `(n_test, n_total_classes)`.

| Modelo (`Model`) | Implementação | Hiperparâmetros Documentados (`config.json`) | Proteções Específicas |
|---|---|---|---|
| **`Random Forest`** | `sklearn.ensemble.RandomForestClassifier` | `n_estimators=200`, `max_depth=None`, `min_samples_split=2`, `min_samples_leaf=1`, `max_features="sqrt"`, `random_state=seed` | Projeção para vocabulário completo do dataset |
| **`SVM (RBF)`** | `sklearn.svm.SVC` | `kernel="rbf"`, `C=10.0`, `gamma="scale"`, `probability=True`, `random_state=seed` | Projeção para vocabulário completo do dataset |
| **`XGBoost`** | `xgboost.XGBClassifier` | `n_estimators=200`, `max_depth=6`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=0.8`, `eval_metric="mlogloss"`, `random_state=seed` | Rótulos locais contíguos $0 \dots C_{\text{present}}-1$ + projeção de probabilidades |
| **`MLP Scratch`** | `src.models.build_tabnet_resnet_balance.HybridDLModel` (apenas cabeça do target) | `MLPEncoder(141->256->128->64)` + `Bottleneck(64->128->64)` + `Linear(64->C_target)`, `AdamW(lr=1e-3, weight_decay=1e-4)`, `StepLR(10, 0.5)`, `epochs=30`, `batch_size=64` | Mesma capacidade arquitetural do MLP de TL, inicializado do zero; proteção `trim_single_remainder_sample` contra lotes finais de tamanho 1 no `BatchNorm1d` |
| **`TabNet Scratch`** | `pytorch_tabnet.tab_model.TabNetClassifier` | `n_d=16, n_a=16, n_steps=4, gamma=1.3, n_independent=2, n_shared=2, mask_type="sparsemax"`, `Adam(lr=2e-2, weight_decay=1e-5)`, `StepLR(10, 0.9)`, `max_epochs=30`, `patience=15`, `batch_size=64`, `virtual_batch_size=32`, `from_unsupervised=None` | *Early stopping* restrito à validação interna das condições de treino + *refit* no treino completo; proteção `trim_single_remainder_sample` contra falha de resto 1 no `BatchNorm1d` (ex.: `PU` fold `C4`) |

---

## 8. Estratégias de Balanceamento de Classes (`BALANCE_STRATEGIES`)

O experimento avalia duas configurações em cada fold e seed:

1. **`smote`** (*configuração primária de comparabilidade com TL*):
   - Aplica `SMOTE` exclusivamente sobre `X_train_scaled` com `random_state=seed` e `k_neighbors = min(5, minority_class_count - 1)`.
   - Caso alguma classe no conjunto de treinamento possua `< 2` amostras, **não** forja nem substitui silenciosamente o `SMOTE`: registra `Status = "FAILED_SMOTE_MIN_SAMPLES"`, detalha a classe em `Unbalanced_Classes` e `Error_Message`, e preenche as métricas com `NaN`.
2. **`none`** (*ablação de controle sem oversampling*):
   - Mantém a distribuição natural das classes nas $K - 1$ condições de treinamento escaladas.

---

## 9. Seeds, Métricas e Esquema dos Arquivos de Saída

- **Seeds Canônicas (`SEEDS`)**: `[42, 123, 2026]`, aplicadas a `random`, `numpy`, `torch` (CPU/CUDA), `SMOTE`, `Random Forest`, `SVM`, `XGBoost`, `DataLoader` do `MLP Scratch`, `TabNet Scratch` e divisões de validação interna.
- **Regras de Métricas**:
  - `Macro_F1`: Métrica primária do protocolo, calculada sobre as classes presentes/relevantes no fold (`f1_score(..., labels=relevant_classes, average="macro", zero_division=0)`), idêntica ao cálculo de `scripts/exp_tl_cross_dataset_mlp_tabnet.py`.
  - `Accuracy` e `Balanced_Accuracy`: Métricas secundárias.
  - `ROC_AUC` e `ROC_AUC_Valid`: Calculada via *One-vs-Rest* macro sobre as classes presentes no fold de teste quando `Test_Class_Count >= 2` (incluindo o caso de subconjunto binário `Test_Class_Count == 2`, como o fold `Load_0HP_Sev_0.021` de `CWRU_48k`). Quando `Test_Class_Count < 2` ou as probabilidades são inválidas, armazena `ROC_AUC = NaN` e `ROC_AUC_Valid = False` — **jamais substituindo por `0.0`**.

### Arquivos Gerados em `results/baseline_loco_v2/run_YYYYMMDD_HHMMSS/`

1. **`config.json`**: Configuração completa do experimento, hiperparâmetros de cada modelo e contrato de prevenção de vazamento.
2. **`results.csv`**: Tabela detalhada salva incrementalmente a cada avaliação, contendo as 21 colunas obrigatórias:
   - `Dataset`, `Task`, `Test_Condition`, `Fold`, `Seed`, `Model`, `Balance_Strategy`, `Preprocessing`, `Feature_Dimension`, `Accuracy`, `Balanced_Accuracy`, `Macro_F1`, `ROC_AUC`, `ROC_AUC_Valid`, `Test_Sample_Count`, `Test_Class_Count`, `Train_Sample_Count`, `Train_Class_Count`, `Status`, `Error_Message`, `Runtime_Seconds`
   - Além das colunas de auditoria: `Train_Sample_Count_Before_Balance`, `Class_Counts_Before`, `Class_Counts_After`, `SMOTE_Params`, `Unbalanced_Classes`, `ROC_AUC_Note`, `Validation_Split_Mode`, `Selected_Epochs`, `Sample_Split_Signature`.
3. **`summary.csv`**: Média e desvio-padrão entre os folds LOCO agrupados por `(Dataset, Model, Balance_Strategy, Seed)`.
4. **`summary_by_dataset_model.csv`**: Média e desvio-padrão agregados por `(Dataset, Model, Balance_Strategy)`.
5. **`summary_across_seeds.csv`**: Média e desvio-padrão entre as 3 seeds das médias LOCO de cada `(Dataset, Model, Balance_Strategy)`.
6. **`run_metadata.json`**: Proveniência completa da execução (`git_commit`, `git_branch`, `git_dirty`, versões de pacotes, hashes SHA-256 dos scripts e contabilidade de folds válidos/falhos).
7. **`run.log`**: Log cronológico detalhado da execução.
8. **`predictions.csv`** (quando `--save-predictions` é habilitado): Predições e vetores de probabilidade por amostra (`Sample_ID`) para testes estatísticos pareados com resultados de TL.

### Chave de Junção Pareada com Resultados de Transfer Learning (`tl_cross_dataset_mlp_tabnet`)

Para comparar diretamente `baseline_loco_v2` com `results/tl_cross_dataset_mlp_tabnet/run_*/results.csv`, realize o *join* pelas colunas equivalentes:

| `baseline_loco_v2` (`results.csv`) | `tl_cross_dataset_mlp_tabnet` (`results.csv`) |
|---|---|
| `Dataset` | `Target` |
| `Test_Condition` | `Fold` |
| `Seed` | `Seed` |
| `Preprocessing` (`detrended_rms`) | `Variant` (`detrended_rms`) |
| `Balance_Strategy` (`smote`) | `Balance` (`smote`) |

---

## 10. Comandos de Execução

### 10.1. Executar a Suíte de Testes Unitários e Anti-Vazamento

```bash
python -m unittest tests/test_baseline_loco_v2.py -v
```

### 10.2. Executar Smoke Test com Dados Reais (1 Fold de `CWRU_12k` e `UOEMD`, 1 Seed, `none` + `smote`, 5 Modelos)

```bash
python -m src.experiments.run_baseline_loco_v2 \
    --targets CWRU_12k UOEMD \
    --seeds 42 \
    --balance-strategies none smote \
    --max-folds 1 \
    --save-predictions
```

### 10.3. Executar o Benchmark Completo (`1.980` Avaliações: 5 Datasets $\times$ 66 Folds $\times$ 3 Seeds $\times$ 2 Balanceamentos $\times$ 5 Modelos)

```bash
python -m src.experiments.run_baseline_loco_v2 \
    --targets CWRU_12k CWRU_48k HUST_Gearbox UOEMD PU \
    --models "Random Forest" "SVM (RBF)" "XGBoost" "MLP Scratch" "TabNet Scratch" \
    --balance-strategies none smote \
    --seeds 42 123 2026 \
    --save-predictions
```

