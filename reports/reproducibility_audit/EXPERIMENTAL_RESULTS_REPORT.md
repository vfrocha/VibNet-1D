# VibNet-1D — Relatório Automatizado de Auditoria de Resultados Experimentais

## 1. Visão Geral do Inventário
- **Diretório auditado**: `results/`
- **Total de artefatos inventariados**: 220
  - `.csv`: 108, `.json`: 15, `.log`: 43, `.png`: 4, `.txt`: 50
- **Arquivos CSV analisados**: 108 (`dataset_audit`: 2, `fold_level`: 66, `summary`: 40)
- **Linhas de métricas extraídas (`metrics_long.csv`)**: 89445
- **Grupos descritivos gerados (`metrics_summary.csv`)**: 9984
- **Anomalias / avaliações inválidas detectadas (`anomalies_and_warnings.csv`)**: 1025
- **Erros de leitura de arquivo**: 0

## 2. Arquivos Derivados Gerados
- `results_inventory.csv`: inventário completo de todos os 220 artefatos com tipo, granularidade (`fold_level`, `summary`, `dataset_audit`), colunas de contexto (`dataset`, `model`, `seed`, `fold`), métricas reconhecidas e colunas não reconhecidas.
- `metrics_long.csv`: métricas extraídas linha a linha mantendo referência exata ao arquivo de origem (`relative_path`), granularidade, `dataset`, `model`, `sources`, `strategy`, `seed`, `fold`, `status`, flag de validade (`is_valid`) e `anomaly_flag`.
- `metrics_summary.csv`: estatísticas descritivas (`count`, `invalid_count`, `n_seeds`, `n_folds`, `mean`, `std`, `min`, `max`) calculadas **estritamente dentro de cada arquivo e protocolo**, filtrando avaliações corrompidas (`is_valid=False`).
- `anomalies_and_warnings.csv`: registro auditável de linhas com falha de execução (`NON_OK_STATUS`), deslocamento de colunas (`SHIFTED_COLUMN_CORRUPTION`), falhas de ajuste (`ZERO_METRIC_FAILURE`) e fallbacks de classe ausente em ROC-AUC (`ROC_AUC_ZERO_FALLBACK`, `ROC_AUC_NAN_MISSING_CLASS`).

## 3. Resumo de Anomalias e Avaliações Inválidas Detectadas
- **`NON_OK_STATUS`**: 60 ocorrências
- **`ROC_AUC_NAN_MISSING_CLASS`**: 8 ocorrências
- **`ROC_AUC_ZERO_FALLBACK`**: 700 ocorrências
- **`SHIFTED_COLUMN_CORRUPTION`**: 100 ocorrências
- **`ZERO_METRIC_FAILURE`**: 157 ocorrências

### Detalhamento das principais anomalias:
1. **`SHIFTED_COLUMN_CORRUPTION` (100 linhas)**: Nos arquivos `baseline_loco_results_20260928_142515.csv` (50 linhas) e `baseline_loco_results_20261003_052352.csv` (50 linhas), o modelo `MLP (Scratch)` teve 4 valores desempacotados em 5 colunas (`run_baseline_experiments.py` L116 vs `build_pytorch_baselines.py` L90). Nesses registros, `Accuracy` contém `Bal Acc`, `Bal Acc` contém `Macro F1`, `Macro F1` contém `ROC-AUC` e `ROC-AUC` contém a string da arquitetura `PureMLPBaseline(...)`. Essas linhas foram marcadas com `is_valid=False` para não contaminar `metrics_summary.csv`.
2. **`NON_OK_STATUS` (`ERROR_MODEL_EXECUTION`, 60 linhas)**: Nos runs `tl_cross_dataset_mlp_tabnet/run_20261028_105603` (15 linhas) e `run_20261028_173310` (45 linhas), o modelo `TabNet` falha no fold `C4_1500rpm_0.7Nm_400N` do dataset `PU` com `ValueError: Expected more than 1 value per channel when training, got input size torch.Size([1, 141])` no `BatchNorm1d`, pois após SMOTE o conjunto de treino tem `8577` amostras (`8577 % 64 = 1`, gerando último mini-batch unitário).
3. **`ZERO_METRIC_FAILURE` (157 linhas)**: Folds em que `Macro F1 == 0.0` e `Bal Acc == 0.0` (ex.: falhas de `GridSearchCV` nos scripts de severidade `cwru12k_severity_loco_results_*` e `cwru48k_mert_comparison_*`, além do fold `Load_0HP_Sev_0.021` em `CWRU_48k` para `Random Forest` e `SVM`).
4. **`ROC_AUC_ZERO_FALLBACK` (700 linhas) e `ROC_AUC_NAN_MISSING_CLASS` (8 linhas)**: No dataset `CWRU_48k`, o fold `Load_0HP_Sev_0.021` possui apenas 2 das 3 classes de falha presentes no teste (`N_Test_Classes_Present = 2`). Scripts legados registravam `ROC-AUC = 0.0` ao capturar `ValueError` no cálculo multiclasse OvR; o runner `exp_tl_cross_dataset_mlp_tabnet.py` passa a registrar `NaN` com `ROC_AUC_Valid = False`.

## 4. Colunas Não Reconhecidas por Arquivo
- `tl_cross_dataset_mlp_tabnet/run_20261028_105603/summary_paired_mlp_vs_tabnet.csv` (`summary`): `Macro F1_MLP_mean;Macro F1_MLP_std;Macro F1_TabNet_mean;Macro F1_TabNet_std;Macro F1_Diff_MLP_minus_TabNet;ACC_MLP_mean;ACC_MLP_std;ACC_TabNet_mean;ACC_TabNet_std;ACC_Diff_MLP_minus_TabNet;Bal Acc_MLP_mean;Bal Acc_MLP_std;Bal Acc_TabNet_mean;Bal Acc_TabNet_std;Bal Acc_Diff_MLP_minus_TabNet;ROC-AUC_MLP_mean;ROC-AUC_MLP_std;ROC-AUC_TabNet_mean;ROC-AUC_TabNet_std;ROC-AUC_Diff_MLP_minus_TabNet`
- `tl_cross_dataset_mlp_tabnet/run_20261028_173310/summary_paired_mlp_vs_tabnet.csv` (`summary`): `Macro F1_MLP_mean;Macro F1_MLP_std;Macro F1_TabNet_mean;Macro F1_TabNet_std;Macro F1_Diff_MLP_minus_TabNet;ACC_MLP_mean;ACC_MLP_std;ACC_TabNet_mean;ACC_TabNet_std;ACC_Diff_MLP_minus_TabNet;Bal Acc_MLP_mean;Bal Acc_MLP_std;Bal Acc_TabNet_mean;Bal Acc_TabNet_std;Bal Acc_Diff_MLP_minus_TabNet;ROC-AUC_MLP_mean;ROC-AUC_MLP_std;ROC-AUC_TabNet_mean;ROC-AUC_TabNet_std;ROC-AUC_Diff_MLP_minus_TabNet`

## 5. Destaques Quantitativos por Família Experimental (Apenas Avaliações Válidas `fold_level`, Métrica Primária: `macro_f1`)

| Família | Arquivo (`relative_path`) | Dataset | Modelo | Estratégia | Folds/Avaliações (`count`) | Seeds | Folds | Macro-F1 Mean | Macro-F1 Std |
|---|---|---|---|---|---:|---:|---:|---:|---:|
| `baseline` | `baseline/baseline_loco_results_20260829_084039.csv` | `CWRU_12k` | `MLP (Scratch)` | `` | 4 | - | 4 | **0.9058** | 0.0784 |
| `baseline` | `baseline/baseline_loco_results_20260829_084039.csv` | `CWRU_12k` | `Random Forest` | `` | 4 | - | 4 | **0.9087** | 0.0668 |
| `baseline` | `baseline/baseline_loco_results_20260829_084039.csv` | `CWRU_12k` | `SVM` | `` | 4 | - | 4 | **0.8975** | 0.0689 |
| `baseline` | `baseline/baseline_loco_results_20260829_084039.csv` | `CWRU_12k` | `TabNet` | `` | 4 | - | 4 | **0.8897** | 0.0618 |
| `baseline` | `baseline/baseline_loco_results_20260829_084039.csv` | `CWRU_12k` | `XGBoost` | `` | 4 | - | 4 | **0.8989** | 0.0460 |
| `baseline` | `baseline/baseline_loco_results_20260829_084039.csv` | `CWRU_48k` | `MLP (Scratch)` | `` | 12 | - | 12 | **0.8858** | 0.1750 |
| `baseline` | `baseline/baseline_loco_results_20260829_084039.csv` | `CWRU_48k` | `Random Forest` | `` | 12 | - | 12 | **0.8532** | 0.2804 |
| `baseline` | `baseline/baseline_loco_results_20260829_084039.csv` | `CWRU_48k` | `SVM` | `` | 12 | - | 12 | **0.8366** | 0.2979 |
| `baseline` | `baseline/baseline_loco_results_20260829_084039.csv` | `CWRU_48k` | `TabNet` | `` | 12 | - | 12 | **0.9306** | 0.1464 |
| `baseline` | `baseline/baseline_loco_results_20260829_084039.csv` | `CWRU_48k` | `XGBoost` | `` | 12 | - | 12 | **0.8093** | 0.2792 |
| `baseline` | `baseline/baseline_loco_results_20260829_084039.csv` | `HUST_Gearbox` | `MLP (Scratch)` | `` | 30 | - | 30 | **0.9561** | 0.1437 |
| `baseline` | `baseline/baseline_loco_results_20260829_084039.csv` | `HUST_Gearbox` | `Random Forest` | `` | 30 | - | 30 | **0.9586** | 0.1170 |
| `baseline` | `baseline/baseline_loco_results_20260829_084039.csv` | `HUST_Gearbox` | `SVM` | `` | 30 | - | 30 | **0.9616** | 0.1248 |
| `baseline` | `baseline/baseline_loco_results_20260829_084039.csv` | `HUST_Gearbox` | `TabNet` | `` | 30 | - | 30 | **0.9854** | 0.0422 |
| `baseline` | `baseline/baseline_loco_results_20260829_084039.csv` | `HUST_Gearbox` | `XGBoost` | `` | 30 | - | 30 | **0.9558** | 0.1157 |
| `baseline` | `baseline/baseline_loco_results_20260829_084039.csv` | `UOEMD` | `MLP (Scratch)` | `` | 16 | - | 16 | **0.8104** | 0.0877 |
| `baseline` | `baseline/baseline_loco_results_20260829_084039.csv` | `UOEMD` | `Random Forest` | `` | 16 | - | 16 | **0.8650** | 0.1456 |
| `baseline` | `baseline/baseline_loco_results_20260829_084039.csv` | `UOEMD` | `SVM` | `` | 16 | - | 16 | **0.7777** | 0.1239 |
| `baseline` | `baseline/baseline_loco_results_20260829_084039.csv` | `UOEMD` | `TabNet` | `` | 16 | - | 16 | **0.8300** | 0.0994 |
| `baseline` | `baseline/baseline_loco_results_20260829_084039.csv` | `UOEMD` | `XGBoost` | `` | 16 | - | 16 | **0.8355** | 0.1612 |
| `baseline_root` | `baseline_loco_results_20260928_142515.csv` | `CWRU_12k` | `Random Forest` | `` | 4 | - | 4 | **0.9139** | 0.0607 |
| `baseline_root` | `baseline_loco_results_20260928_142515.csv` | `CWRU_12k` | `SVM` | `` | 4 | - | 4 | **0.8975** | 0.0689 |
| `baseline_root` | `baseline_loco_results_20260928_142515.csv` | `CWRU_12k` | `TabNet` | `` | 4 | - | 4 | **0.9173** | 0.0634 |
| `baseline_root` | `baseline_loco_results_20260928_142515.csv` | `CWRU_12k` | `XGBoost` | `` | 4 | - | 4 | **0.8862** | 0.0497 |
| `baseline_root` | `baseline_loco_results_20260928_142515.csv` | `CWRU_48k` | `Random Forest` | `` | 12 | - | 12 | **0.8532** | 0.2804 |
| `baseline_root` | `baseline_loco_results_20260928_142515.csv` | `CWRU_48k` | `SVM` | `` | 12 | - | 12 | **0.8366** | 0.2979 |
| `baseline_root` | `baseline_loco_results_20260928_142515.csv` | `CWRU_48k` | `TabNet` | `` | 12 | - | 12 | **0.9289** | 0.1156 |
| `baseline_root` | `baseline_loco_results_20260928_142515.csv` | `CWRU_48k` | `XGBoost` | `` | 12 | - | 12 | **0.8093** | 0.2792 |
| `baseline_root` | `baseline_loco_results_20260928_142515.csv` | `HUST_Gearbox` | `Random Forest` | `` | 30 | - | 30 | **0.9586** | 0.1170 |
| `baseline_root` | `baseline_loco_results_20260928_142515.csv` | `HUST_Gearbox` | `SVM` | `` | 30 | - | 30 | **0.9616** | 0.1248 |
| `baseline_root` | `baseline_loco_results_20260928_142515.csv` | `HUST_Gearbox` | `TabNet` | `` | 30 | - | 30 | **0.9795** | 0.0827 |
| `baseline_root` | `baseline_loco_results_20260928_142515.csv` | `HUST_Gearbox` | `XGBoost` | `` | 30 | - | 30 | **0.9558** | 0.1157 |
| `baseline_root` | `baseline_loco_results_20260928_142515.csv` | `PU` | `Random Forest` | `` | 4 | - | 4 | **0.6519** | 0.1669 |
| `baseline_root` | `baseline_loco_results_20260928_142515.csv` | `PU` | `SVM` | `` | 4 | - | 4 | **0.5840** | 0.1880 |
| `baseline_root` | `baseline_loco_results_20260928_142515.csv` | `PU` | `TabNet` | `` | 4 | - | 4 | **0.6000** | 0.1172 |
| `baseline_root` | `baseline_loco_results_20260928_142515.csv` | `PU` | `XGBoost` | `` | 4 | - | 4 | **0.6280** | 0.2130 |
| `baseline_root` | `baseline_loco_results_20261003_052352.csv` | `CWRU_12k` | `Random Forest` | `` | 4 | - | 4 | **0.9139** | 0.0607 |
| `baseline_root` | `baseline_loco_results_20261003_052352.csv` | `CWRU_12k` | `SVM` | `` | 4 | - | 4 | **0.8975** | 0.0689 |
| `baseline_root` | `baseline_loco_results_20261003_052352.csv` | `CWRU_12k` | `TabNet` | `` | 4 | - | 4 | **0.9173** | 0.0634 |
| `baseline_root` | `baseline_loco_results_20261003_052352.csv` | `CWRU_12k` | `XGBoost` | `` | 4 | - | 4 | **0.8862** | 0.0497 |
| `baseline_root` | `baseline_loco_results_20261003_052352.csv` | `CWRU_48k` | `Random Forest` | `` | 12 | - | 12 | **0.8532** | 0.2804 |
| `baseline_root` | `baseline_loco_results_20261003_052352.csv` | `CWRU_48k` | `SVM` | `` | 12 | - | 12 | **0.8366** | 0.2979 |
| `baseline_root` | `baseline_loco_results_20261003_052352.csv` | `CWRU_48k` | `TabNet` | `` | 12 | - | 12 | **0.9289** | 0.1156 |
| `baseline_root` | `baseline_loco_results_20261003_052352.csv` | `CWRU_48k` | `XGBoost` | `` | 12 | - | 12 | **0.8093** | 0.2792 |
| `baseline_root` | `baseline_loco_results_20261003_052352.csv` | `HUST_Gearbox` | `Random Forest` | `` | 30 | - | 30 | **0.9586** | 0.1170 |
| `baseline_root` | `baseline_loco_results_20261003_052352.csv` | `HUST_Gearbox` | `SVM` | `` | 30 | - | 30 | **0.9616** | 0.1248 |
| `baseline_root` | `baseline_loco_results_20261003_052352.csv` | `HUST_Gearbox` | `TabNet` | `` | 30 | - | 30 | **0.9795** | 0.0827 |
| `baseline_root` | `baseline_loco_results_20261003_052352.csv` | `HUST_Gearbox` | `XGBoost` | `` | 30 | - | 30 | **0.9558** | 0.1157 |
| `baseline_root` | `baseline_loco_results_20261003_052352.csv` | `PU` | `Random Forest` | `` | 4 | - | 4 | **0.8570** | 0.2608 |
| `baseline_root` | `baseline_loco_results_20261003_052352.csv` | `PU` | `SVM` | `` | 4 | - | 4 | **0.8051** | 0.2897 |
| `baseline_root` | `baseline_loco_results_20261003_052352.csv` | `PU` | `TabNet` | `` | 4 | - | 4 | **0.9078** | 0.1135 |
| `baseline_root` | `baseline_loco_results_20261003_052352.csv` | `PU` | `XGBoost` | `` | 4 | - | 4 | **0.9074** | 0.1154 |
| `tl_aux_features` | `tl_aux_features/run_20261012_093659/results.csv` | `CWRU_48k` | `MLP` | `Variant=detrended | Experiment=detrended__base` | 12 | 1 | 12 | **0.8977** | 0.1471 |
| `tl_aux_features` | `tl_aux_features/run_20261012_093659/results.csv` | `CWRU_48k` | `MLP` | `Variant=detrended | Experiment=detrended__rms_b1_b2` | 12 | 1 | 12 | **0.9002** | 0.1870 |
| `tl_aux_features` | `tl_aux_features/run_20261012_093659/results.csv` | `CWRU_48k` | `MLP` | `Variant=detrended_rms | Experiment=detrended_rms__base` | 12 | 1 | 12 | **0.9202** | 0.1353 |
| `tl_aux_features` | `tl_aux_features/run_20261012_093659/results.csv` | `CWRU_48k` | `MLP` | `Variant=detrended_rms | Experiment=detrended_rms__rms_b1_b2` | 12 | 1 | 12 | **0.8799** | 0.2046 |
| `tl_aux_features` | `tl_aux_features/run_20261012_093659/results.csv` | `CWRU_48k` | `MLP` | `Variant=raw | Experiment=raw__base` | 12 | 1 | 12 | **0.8305** | 0.2388 |
| `tl_aux_features` | `tl_aux_features/run_20261012_093659/results.csv` | `CWRU_48k` | `MLP` | `Variant=raw | Experiment=raw__rms_b1_b2` | 12 | 1 | 12 | **0.8899** | 0.1876 |
| `tl_aux_features` | `tl_aux_features/run_20261012_093659/results.csv` | `CWRU_48k` | `MLP` | `Variant=raw_rms | Experiment=raw_rms__base` | 12 | 1 | 12 | **0.8667** | 0.2123 |
| `tl_aux_features` | `tl_aux_features/run_20261012_093659/results.csv` | `CWRU_48k` | `MLP` | `Variant=raw_rms | Experiment=raw_rms__rms_b1_b2` | 12 | 1 | 12 | **0.8806** | 0.1887 |
| `tl_aux_features` | `tl_aux_features/run_20261012_093659/results.csv` | `CWRU_48k` | `TABNET` | `Variant=detrended | Experiment=detrended__base` | 12 | 1 | 12 | **0.8124** | 0.1929 |
| `tl_aux_features` | `tl_aux_features/run_20261012_093659/results.csv` | `CWRU_48k` | `TABNET` | `Variant=detrended | Experiment=detrended__rms_b1_b2` | 12 | 1 | 12 | **0.7681** | 0.2287 |
| `tl_aux_features` | `tl_aux_features/run_20261012_093659/results.csv` | `CWRU_48k` | `TABNET` | `Variant=detrended_rms | Experiment=detrended_rms__base` | 12 | 1 | 12 | **0.8909** | 0.1774 |
| `tl_aux_features` | `tl_aux_features/run_20261012_093659/results.csv` | `CWRU_48k` | `TABNET` | `Variant=detrended_rms | Experiment=detrended_rms__rms_b1_b2` | 12 | 1 | 12 | **0.7966** | 0.1393 |
| `tl_aux_features` | `tl_aux_features/run_20261012_093659/results.csv` | `CWRU_48k` | `TABNET` | `Variant=raw | Experiment=raw__base` | 12 | 1 | 12 | **0.8341** | 0.2163 |
| `tl_aux_features` | `tl_aux_features/run_20261012_093659/results.csv` | `CWRU_48k` | `TABNET` | `Variant=raw | Experiment=raw__rms_b1_b2` | 12 | 1 | 12 | **0.8074** | 0.2062 |
| `tl_aux_features` | `tl_aux_features/run_20261012_093659/results.csv` | `CWRU_48k` | `TABNET` | `Variant=raw_rms | Experiment=raw_rms__base` | 12 | 1 | 12 | **0.8762** | 0.1980 |
| `tl_aux_features` | `tl_aux_features/run_20261012_093659/results.csv` | `CWRU_48k` | `TABNET` | `Variant=raw_rms | Experiment=raw_rms__rms_b1_b2` | 12 | 1 | 12 | **0.8227** | 0.1914 |
| `tl_balance_strategies` | `tl_balance_strategies/run_20261012_145854/results.csv` | `CWRU_48k` | `MLP` | `Balance Strategy=none | Variant=detrended_rms` | 12 | 1 | 12 | **0.9202** | 0.1353 |
| `tl_balance_strategies` | `tl_balance_strategies/run_20261012_145854/results.csv` | `CWRU_48k` | `TABNET` | `Balance Strategy=none | Variant=detrended_rms` | 12 | 1 | 12 | **0.8909** | 0.1774 |
| `tl_balance_strategies` | `tl_balance_strategies/run_20261012_150217/results.csv` | `CWRU_48k` | `MLP` | `Balance Strategy=undersample | Variant=detrended_rms` | 12 | 1 | 12 | **0.8525** | 0.1680 |
| `tl_balance_strategies` | `tl_balance_strategies/run_20261012_150217/results.csv` | `CWRU_48k` | `TABNET` | `Balance Strategy=undersample | Variant=detrended_rms` | 12 | 1 | 12 | **0.8352** | 0.1495 |
| `tl_balance_strategies` | `tl_balance_strategies/run_20261012_150540/results.csv` | `CWRU_48k` | `MLP` | `Balance Strategy=smote | Variant=detrended_rms` | 12 | 1 | 12 | **0.9389** | 0.1289 |
| `tl_balance_strategies` | `tl_balance_strategies/run_20261012_150540/results.csv` | `CWRU_48k` | `TABNET` | `Balance Strategy=smote | Variant=detrended_rms` | 12 | 1 | 12 | **0.8851** | 0.1687 |
| `tl_balance_strategies` | `tl_balance_strategies/run_20261012_150855/results.csv` | `CWRU_48k` | `MLP` | `Balance Strategy=hybrid | Variant=detrended_rms` | 12 | 1 | 12 | **0.9389** | 0.1289 |
| `tl_balance_strategies` | `tl_balance_strategies/run_20261012_150855/results.csv` | `CWRU_48k` | `TABNET` | `Balance Strategy=hybrid | Variant=detrended_rms` | 12 | 1 | 12 | **0.8851** | 0.1687 |
| `tl_contrastive_mlp` | `tl_contrastive_mlp/run_20261026_212435/results.csv` | `CWRU_48k` | `` | `Balance Strategy=none | Variant=detrended_rms | Mode=baseline | Encoder Profile=baseline | Head Shape=asymmetric` | 12 | 1 | 12 | **0.6432** | 0.2179 |
| `tl_contrastive_mlp` | `tl_contrastive_mlp/run_20261026_212739/results.csv` | `CWRU_48k` | `` | `Balance Strategy=none | Variant=detrended_rms | Mode=baseline | Encoder Profile=baseline | Head Shape=asymmetric` | 12 | 1 | 12 | **0.9373** | 0.0691 |
| `tl_contrastive_mlp` | `tl_contrastive_mlp/run_20261026_212739/results.csv` | `CWRU_48k` | `` | `Balance Strategy=none | Variant=detrended_rms | Mode=joint | Encoder Profile=baseline | Head Shape=asymmetric` | 12 | 1 | 12 | **0.9324** | 0.1229 |
| `tl_contrastive_mlp` | `tl_contrastive_mlp/run_20261026_212739/results.csv` | `CWRU_48k` | `` | `Balance Strategy=none | Variant=detrended_rms | Mode=pretrain | Encoder Profile=baseline | Head Shape=asymmetric` | 12 | 1 | 12 | **0.9030** | 0.1350 |
| `tl_contrastive_mlp` | `tl_contrastive_mlp/run_20261026_212945/results.csv` | `CWRU_48k` | `` | `Balance Strategy=none | Variant=detrended_rms | Mode=joint | Encoder Profile=baseline | Head Shape=asymmetric` | 12 | 1 | 12 | **0.9324** | 0.1229 |
| `tl_contrastive_mlp` | `tl_contrastive_mlp/run_20261026_212945/results.csv` | `CWRU_48k` | `` | `Balance Strategy=none | Variant=detrended_rms | Mode=joint | Encoder Profile=expanded | Head Shape=asymmetric` | 12 | 1 | 12 | **0.8896** | 0.1340 |
| `tl_contrastive_mlp` | `tl_contrastive_mlp/run_20261026_213141/results.csv` | `CWRU_48k` | `` | `Balance Strategy=none | Variant=detrended_rms | Mode=joint | Encoder Profile=expanded | Head Shape=asymmetric` | 12 | 1 | 12 | **0.8896** | 0.1340 |
| `tl_contrastive_mlp` | `tl_contrastive_mlp/run_20261026_213141/results.csv` | `CWRU_48k` | `` | `Balance Strategy=none | Variant=detrended_rms | Mode=joint | Encoder Profile=expanded | Head Shape=symmetric` | 12 | 1 | 12 | **0.9169** | 0.1067 |
| `tl_contrastive_mlp` | `tl_contrastive_mlp/run_20261027_112749/results.csv` | `CWRU_48k` | `` | `Balance Strategy=none | Variant=detrended_rms | Mode=baseline | Encoder Profile=baseline | Head Shape=asymmetric` | 36 | 3 | 12 | **0.9056** | 0.1386 |
| `tl_contrastive_mlp` | `tl_contrastive_mlp/run_20261027_112749/results.csv` | `CWRU_48k` | `` | `Balance Strategy=none | Variant=detrended_rms | Mode=joint | Encoder Profile=baseline | Head Shape=asymmetric` | 36 | 3 | 12 | **0.9097** | 0.1227 |
| `tl_contrastive_mlp` | `tl_contrastive_mlp/run_20261027_112749/results.csv` | `CWRU_48k` | `` | `Balance Strategy=none | Variant=detrended_rms | Mode=pretrain | Encoder Profile=baseline | Head Shape=asymmetric` | 36 | 3 | 12 | **0.9133** | 0.0902 |
| `tl_contrastive_mlp` | `tl_contrastive_mlp/run_20261027_113203/results.csv` | `CWRU_48k` | `` | `Balance Strategy=none | Variant=detrended_rms | Mode=joint | Encoder Profile=baseline | Head Shape=asymmetric` | 36 | 3 | 12 | **0.9097** | 0.1227 |
| `tl_contrastive_mlp` | `tl_contrastive_mlp/run_20261027_113203/results.csv` | `CWRU_48k` | `` | `Balance Strategy=none | Variant=detrended_rms | Mode=joint | Encoder Profile=expanded | Head Shape=asymmetric` | 36 | 3 | 12 | **0.9235** | 0.0844 |
| `tl_contrastive_mlp` | `tl_contrastive_mlp/run_20261027_113503/results.csv` | `CWRU_48k` | `` | `Balance Strategy=none | Variant=detrended_rms | Mode=joint | Encoder Profile=expanded | Head Shape=asymmetric` | 36 | 3 | 12 | **0.9235** | 0.0844 |
| `tl_contrastive_mlp` | `tl_contrastive_mlp/run_20261027_113503/results.csv` | `CWRU_48k` | `` | `Balance Strategy=none | Variant=detrended_rms | Mode=joint | Encoder Profile=expanded | Head Shape=symmetric` | 36 | 3 | 12 | **0.8863** | 0.1206 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `CWRU_12k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 12 | 3 | 4 | **0.9045** | 0.0653 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `CWRU_12k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 12 | 3 | 4 | **0.9048** | 0.0688 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `CWRU_12k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 12 | 3 | 4 | **0.9006** | 0.0694 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `CWRU_12k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 12 | 3 | 4 | **0.9031** | 0.0680 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `CWRU_12k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=all_sources` | 12 | 3 | 4 | **0.9058** | 0.0726 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `CWRU_12k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 12 | 3 | 4 | **0.8624** | 0.0718 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `CWRU_12k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 12 | 3 | 4 | **0.8765** | 0.0751 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `CWRU_12k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 12 | 3 | 4 | **0.8778** | 0.0565 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `CWRU_12k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 12 | 3 | 4 | **0.8715** | 0.0629 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `CWRU_12k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=all_sources` | 12 | 3 | 4 | **0.8803** | 0.0592 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `CWRU_48k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 36 | 3 | 12 | **0.8888** | 0.1865 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `CWRU_48k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 36 | 3 | 12 | **0.8984** | 0.1629 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `CWRU_48k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 36 | 3 | 12 | **0.8984** | 0.1638 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `CWRU_48k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 36 | 3 | 12 | **0.8964** | 0.1626 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `CWRU_48k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=all_sources` | 36 | 3 | 12 | **0.8842** | 0.1889 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `CWRU_48k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 36 | 3 | 12 | **0.8330** | 0.1385 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `CWRU_48k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 36 | 3 | 12 | **0.8858** | 0.1225 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `CWRU_48k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 36 | 3 | 12 | **0.8213** | 0.1506 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `CWRU_48k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 36 | 3 | 12 | **0.8636** | 0.1297 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `CWRU_48k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=all_sources` | 36 | 3 | 12 | **0.9077** | 0.0975 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `HUST_Gearbox` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 90 | 3 | 30 | **0.9591** | 0.1301 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `HUST_Gearbox` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 90 | 3 | 30 | **0.9506** | 0.1333 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `HUST_Gearbox` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 90 | 3 | 30 | **0.9458** | 0.1481 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `HUST_Gearbox` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 90 | 3 | 30 | **0.9630** | 0.1106 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `HUST_Gearbox` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=all_sources` | 90 | 3 | 30 | **0.9594** | 0.1219 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `HUST_Gearbox` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 90 | 3 | 30 | **0.9371** | 0.1403 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `HUST_Gearbox` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 90 | 3 | 30 | **0.9429** | 0.1391 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `HUST_Gearbox` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 90 | 3 | 30 | **0.9491** | 0.1376 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `HUST_Gearbox` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 90 | 3 | 30 | **0.9302** | 0.1412 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `HUST_Gearbox` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=all_sources` | 90 | 3 | 30 | **0.9465** | 0.1306 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `PU` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 12 | 3 | 4 | **0.8115** | 0.2847 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `PU` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 12 | 3 | 4 | **0.8184** | 0.2557 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `PU` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 12 | 3 | 4 | **0.8141** | 0.2628 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `PU` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 12 | 3 | 4 | **0.8295** | 0.2550 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `PU` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=all_sources` | 12 | 3 | 4 | **0.8410** | 0.2491 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `PU` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 9 | 3 | 3 | **0.8220** | 0.2651 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `PU` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 9 | 3 | 3 | **0.8079** | 0.2788 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `PU` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 9 | 3 | 3 | **0.7960** | 0.3024 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `PU` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 9 | 3 | 3 | **0.8006** | 0.2944 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `PU` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=all_sources` | 9 | 3 | 3 | **0.7952** | 0.3047 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `UOEMD` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 48 | 3 | 16 | **0.8270** | 0.1065 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `UOEMD` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 48 | 3 | 16 | **0.8296** | 0.1141 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `UOEMD` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 48 | 3 | 16 | **0.8284** | 0.1088 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `UOEMD` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 48 | 3 | 16 | **0.8251** | 0.1228 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `UOEMD` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=all_sources` | 48 | 3 | 16 | **0.8358** | 0.1016 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `UOEMD` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 48 | 3 | 16 | **0.6301** | 0.1270 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `UOEMD` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 48 | 3 | 16 | **0.6470** | 0.1210 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `UOEMD` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 48 | 3 | 16 | **0.6313** | 0.1225 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `UOEMD` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 48 | 3 | 16 | **0.6638** | 0.1081 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_105603/results.csv` | `UOEMD` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=all_sources` | 48 | 3 | 16 | **0.6362** | 0.1372 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 12 | 3 | 4 | **0.9045** | 0.0653 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 12 | 3 | 4 | **0.9048** | 0.0688 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 12 | 3 | 4 | **0.9077** | 0.0706 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 12 | 3 | 4 | **0.9047** | 0.0710 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 12 | 3 | 4 | **0.9037** | 0.0688 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 12 | 3 | 4 | **0.9006** | 0.0694 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 12 | 3 | 4 | **0.9059** | 0.0666 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 12 | 3 | 4 | **0.9031** | 0.0680 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 12 | 3 | 4 | **0.9110** | 0.0668 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 12 | 3 | 4 | **0.9014** | 0.0700 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 12 | 3 | 4 | **0.9037** | 0.0705 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 12 | 3 | 4 | **0.9071** | 0.0704 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=all_sources` | 12 | 3 | 4 | **0.9058** | 0.0726 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 12 | 3 | 4 | **0.9071** | 0.0690 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 12 | 3 | 4 | **0.9056** | 0.0642 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 12 | 3 | 4 | **0.8624** | 0.0718 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 12 | 3 | 4 | **0.8765** | 0.0751 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 12 | 3 | 4 | **0.8815** | 0.0610 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 12 | 3 | 4 | **0.8525** | 0.0724 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 12 | 3 | 4 | **0.8851** | 0.0536 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 12 | 3 | 4 | **0.8778** | 0.0565 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 12 | 3 | 4 | **0.8672** | 0.0535 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 12 | 3 | 4 | **0.8715** | 0.0629 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 12 | 3 | 4 | **0.8731** | 0.0627 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 12 | 3 | 4 | **0.8665** | 0.0683 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 12 | 3 | 4 | **0.8897** | 0.0568 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 12 | 3 | 4 | **0.8827** | 0.0606 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=all_sources` | 12 | 3 | 4 | **0.8803** | 0.0592 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 12 | 3 | 4 | **0.8698** | 0.0656 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_12k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 12 | 3 | 4 | **0.8826** | 0.0552 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 36 | 3 | 12 | **0.8888** | 0.1865 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 36 | 3 | 12 | **0.8984** | 0.1629 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 36 | 3 | 12 | **0.9052** | 0.1663 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 36 | 3 | 12 | **0.8578** | 0.2240 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 36 | 3 | 12 | **0.8824** | 0.1953 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 36 | 3 | 12 | **0.8984** | 0.1638 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 36 | 3 | 12 | **0.9175** | 0.1479 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 36 | 3 | 12 | **0.8964** | 0.1626 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 36 | 3 | 12 | **0.9214** | 0.1390 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 36 | 3 | 12 | **0.9115** | 0.1490 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 36 | 3 | 12 | **0.8885** | 0.1797 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 36 | 3 | 12 | **0.8933** | 0.1753 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=all_sources` | 36 | 3 | 12 | **0.8842** | 0.1889 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 36 | 3 | 12 | **0.8972** | 0.1879 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 36 | 3 | 12 | **0.9232** | 0.1145 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 36 | 3 | 12 | **0.8330** | 0.1385 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 36 | 3 | 12 | **0.8858** | 0.1225 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 36 | 3 | 12 | **0.8306** | 0.1582 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 36 | 3 | 12 | **0.8679** | 0.1212 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 36 | 3 | 12 | **0.8551** | 0.1335 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 36 | 3 | 12 | **0.8213** | 0.1506 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 36 | 3 | 12 | **0.8424** | 0.1268 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 36 | 3 | 12 | **0.8636** | 0.1297 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 36 | 3 | 12 | **0.8286** | 0.1556 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 36 | 3 | 12 | **0.8587** | 0.1471 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 36 | 3 | 12 | **0.9001** | 0.1072 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 36 | 3 | 12 | **0.8794** | 0.1221 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=all_sources` | 36 | 3 | 12 | **0.9077** | 0.0975 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 36 | 3 | 12 | **0.8980** | 0.1280 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `CWRU_48k` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 36 | 3 | 12 | **0.8530** | 0.1733 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 90 | 3 | 30 | **0.9591** | 0.1301 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 90 | 3 | 30 | **0.9506** | 0.1333 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 90 | 3 | 30 | **0.9521** | 0.1352 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 90 | 3 | 30 | **0.9566** | 0.1331 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 90 | 3 | 30 | **0.9535** | 0.1328 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 90 | 3 | 30 | **0.9458** | 0.1481 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 90 | 3 | 30 | **0.9545** | 0.1358 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 90 | 3 | 30 | **0.9630** | 0.1106 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 90 | 3 | 30 | **0.9571** | 0.1317 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 90 | 3 | 30 | **0.9597** | 0.1260 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 90 | 3 | 30 | **0.9556** | 0.1252 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 90 | 3 | 30 | **0.9609** | 0.1125 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=all_sources` | 90 | 3 | 30 | **0.9594** | 0.1219 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 90 | 3 | 30 | **0.9511** | 0.1391 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 90 | 3 | 30 | **0.9569** | 0.1319 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 90 | 3 | 30 | **0.9371** | 0.1403 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 90 | 3 | 30 | **0.9429** | 0.1391 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 90 | 3 | 30 | **0.9418** | 0.1467 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 90 | 3 | 30 | **0.9393** | 0.1455 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 90 | 3 | 30 | **0.9427** | 0.1385 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 90 | 3 | 30 | **0.9491** | 0.1376 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 90 | 3 | 30 | **0.9559** | 0.0980 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 90 | 3 | 30 | **0.9302** | 0.1412 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 90 | 3 | 30 | **0.9474** | 0.1264 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 90 | 3 | 30 | **0.9460** | 0.1489 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 90 | 3 | 30 | **0.9481** | 0.1458 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 90 | 3 | 30 | **0.9479** | 0.1392 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=all_sources` | 90 | 3 | 30 | **0.9465** | 0.1306 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 90 | 3 | 30 | **0.9472** | 0.1314 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `HUST_Gearbox` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 90 | 3 | 30 | **0.9552** | 0.1156 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 12 | 3 | 4 | **0.8115** | 0.2847 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 12 | 3 | 4 | **0.8184** | 0.2557 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 12 | 3 | 4 | **0.8165** | 0.2548 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 12 | 3 | 4 | **0.8161** | 0.2617 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 12 | 3 | 4 | **0.8364** | 0.2379 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 12 | 3 | 4 | **0.8141** | 0.2628 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 12 | 3 | 4 | **0.8195** | 0.2503 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 12 | 3 | 4 | **0.8295** | 0.2550 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 12 | 3 | 4 | **0.8308** | 0.2605 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 12 | 3 | 4 | **0.8288** | 0.2572 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 12 | 3 | 4 | **0.8272** | 0.2435 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 12 | 3 | 4 | **0.8269** | 0.2719 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=all_sources` | 12 | 3 | 4 | **0.8410** | 0.2491 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 12 | 3 | 4 | **0.8167** | 0.2783 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 12 | 3 | 4 | **0.8317** | 0.2620 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 9 | 3 | 3 | **0.8220** | 0.2651 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 9 | 3 | 3 | **0.8079** | 0.2788 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 9 | 3 | 3 | **0.7912** | 0.3068 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 9 | 3 | 3 | **0.8022** | 0.2842 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 9 | 3 | 3 | **0.7855** | 0.3061 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 9 | 3 | 3 | **0.7960** | 0.3024 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 9 | 3 | 3 | **0.8236** | 0.2718 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 9 | 3 | 3 | **0.8006** | 0.2944 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 9 | 3 | 3 | **0.7763** | 0.3162 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 9 | 3 | 3 | **0.7591** | 0.3490 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 9 | 3 | 3 | **0.8001** | 0.2884 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 9 | 3 | 3 | **0.7943** | 0.3057 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=all_sources` | 9 | 3 | 3 | **0.7952** | 0.3047 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 9 | 3 | 3 | **0.7696** | 0.3383 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `PU` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 9 | 3 | 3 | **0.7820** | 0.3208 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 48 | 3 | 16 | **0.8270** | 0.1065 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 48 | 3 | 16 | **0.8296** | 0.1141 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 48 | 3 | 16 | **0.8268** | 0.1083 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 48 | 3 | 16 | **0.8232** | 0.1179 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 48 | 3 | 16 | **0.8284** | 0.1023 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 48 | 3 | 16 | **0.8284** | 0.1088 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 48 | 3 | 16 | **0.8225** | 0.1064 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 48 | 3 | 16 | **0.8251** | 0.1228 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 48 | 3 | 16 | **0.8292** | 0.1136 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 48 | 3 | 16 | **0.8406** | 0.1112 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 48 | 3 | 16 | **0.8409** | 0.1071 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 48 | 3 | 16 | **0.8300** | 0.1026 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=all_sources` | 48 | 3 | 16 | **0.8358** | 0.1016 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 48 | 3 | 16 | **0.8377** | 0.1016 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `MLP` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 48 | 3 | 16 | **0.8345** | 0.1021 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 48 | 3 | 16 | **0.6301** | 0.1270 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 48 | 3 | 16 | **0.6470** | 0.1210 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 48 | 3 | 16 | **0.6555** | 0.1160 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 48 | 3 | 16 | **0.6484** | 0.1256 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 48 | 3 | 16 | **0.6607** | 0.1377 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 48 | 3 | 16 | **0.6313** | 0.1225 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 48 | 3 | 16 | **0.6425** | 0.1191 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=single_source` | 48 | 3 | 16 | **0.6638** | 0.1081 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 48 | 3 | 16 | **0.6553** | 0.1484 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 48 | 3 | 16 | **0.6616** | 0.1241 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 48 | 3 | 16 | **0.6588** | 0.1305 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 48 | 3 | 16 | **0.6205** | 0.1296 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=all_sources` | 48 | 3 | 16 | **0.6362** | 0.1372 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=pair_sources` | 48 | 3 | 16 | **0.6715** | 0.1109 |
| `tl_cross_dataset_mlp_tabnet` | `tl_cross_dataset_mlp_tabnet/run_20261028_173310/results.csv` | `UOEMD` | `TabNet` | `Balance=smote | Variant=detrended_rms | Source_Mode=triple_sources` | 48 | 3 | 16 | **0.6844** | 0.1193 |
| `tl_four_variants` | `tl_four_variants/run_20261005_163159/results.csv` | `CWRU_48k` | `MLP` | `detrended` | 12 | 1 | 12 | **0.8977** | 0.1471 |
| `tl_four_variants` | `tl_four_variants/run_20261005_163159/results.csv` | `CWRU_48k` | `MLP` | `detrended_rms` | 12 | 1 | 12 | **0.9202** | 0.1353 |
| `tl_four_variants` | `tl_four_variants/run_20261005_163159/results.csv` | `CWRU_48k` | `MLP` | `raw` | 12 | 1 | 12 | **0.8305** | 0.2388 |
| `tl_four_variants` | `tl_four_variants/run_20261005_163159/results.csv` | `CWRU_48k` | `MLP` | `raw_rms` | 12 | 1 | 12 | **0.8667** | 0.2123 |
| `tl_four_variants` | `tl_four_variants/run_20261005_163159/results.csv` | `CWRU_48k` | `TABNET` | `detrended` | 12 | 1 | 12 | **0.8124** | 0.1929 |
| `tl_four_variants` | `tl_four_variants/run_20261005_163159/results.csv` | `CWRU_48k` | `TABNET` | `detrended_rms` | 12 | 1 | 12 | **0.8909** | 0.1774 |
| `tl_four_variants` | `tl_four_variants/run_20261005_163159/results.csv` | `CWRU_48k` | `TABNET` | `raw` | 12 | 1 | 12 | **0.8341** | 0.2163 |
| `tl_four_variants` | `tl_four_variants/run_20261005_163159/results.csv` | `CWRU_48k` | `TABNET` | `raw_rms` | 12 | 1 | 12 | **0.8762** | 0.1980 |

## 6. Cautelas Metodológicas e Regras de Comparação
- **Proibição de fusão entre protocolos incompatíveis**: Resultados de `LOCO`, `Severity LOCO`, `MERT`, `5-Fold CV` e `One-Class Anomaly` avaliam hipóteses e partições distintas e jamais são combinados entre arquivos.
- **Separação estrita de granularidade**: Arquivos `fold_level` e `summary` coexistem nos diretórios de saída (`results/tl_*`); utilize o campo `granularity == 'fold_level'` para reanálises estatísticas a partir das avaliações primárias.
- **Preservação dos dados brutos**: Nenhum arquivo em `results/` foi modificado, movido ou sobrescrito por esta auditoria.

## 7. Erros de Leitura
- Nenhum erro de leitura nos 220 arquivos inventariados.
