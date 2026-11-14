# Relatório Técnico — Baselines Clássicos de Machine Learning (`scratch_vs_tl_classical_baselines`)

> **AVISO DE ESCOPO (EXECUÇÃO PILOTO / SINTÉTICA DE VALIDAÇÃO)**: Este diretório corresponde a uma execução de validação em subconjunto ou dados sintéticos (`is_synthetic_smoke=True`). Conforme `AGENTS.md`, resultados de teste rápido validam a integridade funcional e anti-vazamento do pipeline.


- **Commit Git**: `15f72c25c8444915168cc1e6fd29b0f93efc3cad` (`git_dirty=True`)
- **Manifesto de Referência Reutilizado**: `/home/vfrocha/VibNet-1D/results/scratch_vs_tl/run_20261105_141743/fold_manifest.csv`
- **Total de Avaliações Executadas**: `12` (`OK=12`, `Falhas=0`)
- **Datasets Alvo**: `['CWRU_12k']`
- **Modelos (`architecture`)**: `['RandomForest', 'SVM', 'XGBoost']`
- **Método (`method`)**: `ClassicalBaseline`
- **Estratégias de Balanceamento**: `['none', 'smote']`
- **Seeds**: `[42]`

---

## 1. Configuração Experimental e Otimização Interna de Hiperparâmetros por Fold

Os três classificadores clássicos (`RandomForest`, `SVM`, `XGBoost`) foram ajustados e treinados exclusivamente nos dados de treinamento de cada fold LOCO do dataset de destino:
1. **Seleção de Hiperparâmetros Estritamente no Conjunto de Treino (`Inner StratifiedKFold`)**: em cada bloco `(target_dataset, fold_id, seed, balance_strategy)`, o conjunto de treinamento original (`X_train_raw`, `y_train_raw`) é particionado em subfolds estratificados (`StratifiedKFold(n_splits=3, shuffle=True, random_state=seed)`). Em cada subfold interno, `StandardScaler` e (quando `balance_strategy == "smote"`) `SMOTE` são aplicados **exclusivamente sobre a partição interna de treino**, avaliando a partição interna de validação sem `SMOTE` pela métrica primária `Macro-F1`.
2. **Refit Canônico no Fold Externo**: a combinação vencedora de hiperparâmetros (`selected_hyperparameters`) é treinada sobre o conjunto de treino canônico do fold externo (`X_train_bal`, `y_train_bal`), reutilizando exatamente os mesmos splits (`train_sample_ids` e `test_sample_ids`), `StandardScaler` (`split_signature`) e balanceamento `SMOTE` (`preprocessing_signature`) registrados em `results/scratch_vs_tl/run_20261105_141743/fold_manifest.csv`, e avaliada uma única vez na condição de teste isolada (`X_test_scaled`).

```json
{
  "hyperparameter_tuning": {
    "enabled": true,
    "strategy": "grid_search_stratified_kfold",
    "cv_folds": 3,
    "scoring": "macro_f1",
    "inner_scaling_and_smote_strict": true,
    "param_grids": {
      "RandomForest": {
        "n_estimators": [
          200,
          500
        ],
        "max_depth": [
          null,
          15,
          30
        ],
        "min_samples_leaf": [
          1,
          2
        ],
        "max_features": [
          "sqrt"
        ]
      },
      "SVM": {
        "kernel": [
          "rbf"
        ],
        "C": [
          0.1,
          1.0,
          10.0
        ],
        "gamma": [
          "scale",
          "auto"
        ]
      },
      "XGBoost": {
        "n_estimators": [
          200,
          400
        ],
        "max_depth": [
          4,
          6
        ],
        "learning_rate": [
          0.03,
          0.1
        ],
        "subsample": [
          0.8
        ],
        "colsample_bytree": [
          0.8
        ]
      }
    }
  },
  "base_hyperparameters": {
    "RandomForest": {
      "n_estimators": 500,
      "max_depth": null,
      "min_samples_split": 2,
      "min_samples_leaf": 1,
      "max_features": "sqrt",
      "class_weight": null,
      "n_jobs": -1
    },
    "SVM": {
      "kernel": "rbf",
      "C": 1.0,
      "gamma": "scale",
      "probability": true,
      "class_weight": null
    },
    "XGBoost": {
      "objective": "multi:softprob",
      "n_estimators": 400,
      "max_depth": 6,
      "learning_rate": 0.05,
      "subsample": 0.8,
      "colsample_bytree": 0.8,
      "reg_lambda": 1.0,
      "min_child_weight": 1,
      "eval_metric": "mlogloss",
      "tree_method": "hist",
      "n_jobs": -1,
      "verbosity": 0
    }
  }
}
```

---

## 2. Verificação de Integridade e Reprodutibilidade

- **Assinaturas de Split e Pré-processamento Verificadas**: `True`
- **Duplicatas em `results.csv`**: `0`
- **Duplicatas em `predictions.csv`**: `0`
- **Predições Individuais Validadas**: `7440`

---

## 3. Resultados por Dataset, Modelo e Estratégia de Balanceamento (Média e Desvio Padrão)

```text
target_dataset architecture            method balance_strategy  n_folds  n_seeds  macro_f1_mean  macro_f1_std  balanced_accuracy_mean  balanced_accuracy_std  accuracy_mean  accuracy_std  roc_auc_mean  roc_auc_std
      CWRU_12k RandomForest ClassicalBaseline             none        2        1       0.897524      0.058065                0.897405               0.058254       0.875677      0.076382      0.974990     0.023019
      CWRU_12k          SVM ClassicalBaseline             none        2        1       0.860094      0.099156                0.863743               0.094384       0.833490      0.122786      0.933196     0.050533
      CWRU_12k      XGBoost ClassicalBaseline             none        2        1       0.896372      0.065638                0.895651               0.066420       0.873021      0.086768      0.974565     0.027414
      CWRU_12k RandomForest ClassicalBaseline            smote        2        1       0.894722      0.052289                0.894883               0.052413       0.872604      0.069679      0.976539     0.020897
      CWRU_12k          SVM ClassicalBaseline            smote        2        1       0.858193      0.096467                0.861806               0.091645       0.831146      0.119472      0.933482     0.050523
      CWRU_12k      XGBoost ClassicalBaseline            smote        2        1       0.889541      0.058046                0.889401               0.059546       0.865938      0.079108      0.972634     0.028061
```

---

## 4. Resumo Global por Modelo (`Architecture-Level Summary`)

```text
architecture            method balance_strategy  n_datasets  total_evaluations  valid_evaluations  failed_evaluations  macro_f1_mean  macro_f1_std  macro_f1_median  accuracy_mean  accuracy_std  accuracy_median  balanced_accuracy_mean  balanced_accuracy_std  balanced_accuracy_median  roc_auc_mean  roc_auc_std  roc_auc_median
RandomForest ClassicalBaseline             none           1                  2                  2                   0       0.897524      0.058065         0.897524       0.875677      0.076382         0.875677                0.897405               0.058254                  0.897405      0.974990     0.023019        0.974990
         SVM ClassicalBaseline             none           1                  2                  2                   0       0.860094      0.099156         0.860094       0.833490      0.122786         0.833490                0.863743               0.094384                  0.863743      0.933196     0.050533        0.933196
     XGBoost ClassicalBaseline             none           1                  2                  2                   0       0.896372      0.065638         0.896372       0.873021      0.086768         0.873021                0.895651               0.066420                  0.895651      0.974565     0.027414        0.974565
RandomForest ClassicalBaseline            smote           1                  2                  2                   0       0.894722      0.052289         0.894722       0.872604      0.069679         0.872604                0.894883               0.052413                  0.894883      0.976539     0.020897        0.976539
         SVM ClassicalBaseline            smote           1                  2                  2                   0       0.858193      0.096467         0.858193       0.831146      0.119472         0.831146                0.861806               0.091645                  0.861806      0.933482     0.050523        0.933482
     XGBoost ClassicalBaseline            smote           1                  2                  2                   0       0.889541      0.058046         0.889541       0.865938      0.079108         0.865938                0.889401               0.059546                  0.889401      0.972634     0.028061        0.972634
```

---

## 5. Comparações Estatísticas Pareadas entre Baselines Clássicos (Wilcoxon + Holm)

```text
target_dataset balance_strategy               comparison      model_a      model_b            metric        statistical_unit  n_paired_folds  model_a_mean  model_b_mean  mean_diff_a_minus_b  median_diff_a_minus_b  std_diff  ci95_low  ci95_high  n_folds_a_better  n_folds_b_better  n_folds_tied  wilcoxon_statistic  p_value_raw  p_value_holm  significant_alpha_0_05                                                                         methodological_note
      CWRU_12k             none XGBoost vs. RandomForest      XGBoost RandomForest          macro_f1 loco_fold_seed_averaged               2      0.896372      0.897524            -0.001153              -0.001153  0.007573 -0.069195   0.066889                 1                 1             0                 1.0          1.0           1.0                   False Low sample size (N=2 LOCO folds < 6 minimum for alpha=0.05 two-sided Wilcoxon significance)
      CWRU_12k             none XGBoost vs. RandomForest      XGBoost RandomForest          accuracy loco_fold_seed_averaged               2      0.873021      0.875677            -0.002656              -0.002656  0.010386 -0.095967   0.090655                 1                 1             0                 1.0          1.0           1.0                   False Low sample size (N=2 LOCO folds < 6 minimum for alpha=0.05 two-sided Wilcoxon significance)
      CWRU_12k             none XGBoost vs. RandomForest      XGBoost RandomForest balanced_accuracy loco_fold_seed_averaged               2      0.895651      0.897405            -0.001754              -0.001754  0.008167 -0.075131   0.071622                 1                 1             0                 1.0          1.0           1.0                   False Low sample size (N=2 LOCO folds < 6 minimum for alpha=0.05 two-sided Wilcoxon significance)
      CWRU_12k             none XGBoost vs. RandomForest      XGBoost RandomForest           roc_auc loco_fold_seed_averaged               2      0.974565      0.974990            -0.000425              -0.000425  0.004396 -0.039918   0.039068                 1                 1             0                 1.0          1.0           1.0                   False Low sample size (N=2 LOCO folds < 6 minimum for alpha=0.05 two-sided Wilcoxon significance)
      CWRU_12k             none     RandomForest vs. SVM RandomForest          SVM          macro_f1 loco_fold_seed_averaged               2      0.897524      0.860094             0.037430               0.037430  0.041091 -0.331759   0.406619                 2                 0             0                 0.0          0.5           1.0                   False Low sample size (N=2 LOCO folds < 6 minimum for alpha=0.05 two-sided Wilcoxon significance)
      CWRU_12k             none     RandomForest vs. SVM RandomForest          SVM          accuracy loco_fold_seed_averaged               2      0.875677      0.833490             0.042187               0.042187  0.046404 -0.374735   0.459110                 2                 0             0                 0.0          0.5           1.0                   False Low sample size (N=2 LOCO folds < 6 minimum for alpha=0.05 two-sided Wilcoxon significance)
      CWRU_12k             none     RandomForest vs. SVM RandomForest          SVM balanced_accuracy loco_fold_seed_averaged               2      0.897405      0.863743             0.033662               0.033662  0.036131 -0.290959   0.358284                 2                 0             0                 0.0          0.5           1.0                   False Low sample size (N=2 LOCO folds < 6 minimum for alpha=0.05 two-sided Wilcoxon significance)
      CWRU_12k             none     RandomForest vs. SVM RandomForest          SVM           roc_auc loco_fold_seed_averaged               2      0.974990      0.933196             0.041794               0.041794  0.027514 -0.205410   0.288999                 2                 0             0                 0.0          0.5           1.0                   False Low sample size (N=2 LOCO folds < 6 minimum for alpha=0.05 two-sided Wilcoxon significance)
      CWRU_12k             none          XGBoost vs. SVM      XGBoost          SVM          macro_f1 loco_fold_seed_averaged               2      0.896372      0.860094             0.036277               0.036277  0.033518 -0.264870   0.337424                 2                 0             0                 0.0          0.5           1.0                   False Low sample size (N=2 LOCO folds < 6 minimum for alpha=0.05 two-sided Wilcoxon significance)
      CWRU_12k             none          XGBoost vs. SVM      XGBoost          SVM          accuracy loco_fold_seed_averaged               2      0.873021      0.833490             0.039531               0.039531  0.036018 -0.284080   0.363142                 2                 0             0                 0.0          0.5           1.0                   False Low sample size (N=2 LOCO folds < 6 minimum for alpha=0.05 two-sided Wilcoxon significance)
      CWRU_12k             none          XGBoost vs. SVM      XGBoost          SVM balanced_accuracy loco_fold_seed_averaged               2      0.895651      0.863743             0.031908               0.031908  0.027964 -0.219337   0.283153                 2                 0             0                 0.0          0.5           1.0                   False Low sample size (N=2 LOCO folds < 6 minimum for alpha=0.05 two-sided Wilcoxon significance)
      CWRU_12k             none          XGBoost vs. SVM      XGBoost          SVM           roc_auc loco_fold_seed_averaged               2      0.974565      0.933196             0.041369               0.041369  0.023119 -0.166343   0.249081                 2                 0             0                 0.0          0.5           1.0                   False Low sample size (N=2 LOCO folds < 6 minimum for alpha=0.05 two-sided Wilcoxon significance)
      CWRU_12k            smote XGBoost vs. RandomForest      XGBoost RandomForest          macro_f1 loco_fold_seed_averaged               2      0.889541      0.894722            -0.005182              -0.005182  0.005757 -0.056909   0.046546                 0                 2             0                 0.0          0.5           1.0                   False Low sample size (N=2 LOCO folds < 6 minimum for alpha=0.05 two-sided Wilcoxon significance)
      CWRU_12k            smote XGBoost vs. RandomForest      XGBoost RandomForest          accuracy loco_fold_seed_averaged               2      0.865938      0.872604            -0.006667              -0.006667  0.009428 -0.091375   0.078041                 0                 1             1                 0.5          1.0           1.0                   False Low sample size (N=2 LOCO folds < 6 minimum for alpha=0.05 two-sided Wilcoxon significance)
      CWRU_12k            smote XGBoost vs. RandomForest      XGBoost RandomForest balanced_accuracy loco_fold_seed_averaged               2      0.889401      0.894883            -0.005482              -0.005482  0.007133 -0.069571   0.058606                 0                 2             0                 0.0          0.5           1.0                   False Low sample size (N=2 LOCO folds < 6 minimum for alpha=0.05 two-sided Wilcoxon significance)
      CWRU_12k            smote XGBoost vs. RandomForest      XGBoost RandomForest           roc_auc loco_fold_seed_averaged               2      0.972634      0.976539            -0.003904              -0.003904  0.007165 -0.068276   0.060467                 1                 1             0                 1.0          1.0           1.0                   False Low sample size (N=2 LOCO folds < 6 minimum for alpha=0.05 two-sided Wilcoxon significance)
      CWRU_12k            smote     RandomForest vs. SVM RandomForest          SVM          macro_f1 loco_fold_seed_averaged               2      0.894722      0.858193             0.036530               0.036530  0.044178 -0.360396   0.433455                 2                 0             0                 0.0          0.5           1.0                   False Low sample size (N=2 LOCO folds < 6 minimum for alpha=0.05 two-sided Wilcoxon significance)
      CWRU_12k            smote     RandomForest vs. SVM RandomForest          SVM          accuracy loco_fold_seed_averaged               2      0.872604      0.831146             0.041458               0.041458  0.049792 -0.405906   0.488823                 2                 0             0                 0.0          0.5           1.0                   False Low sample size (N=2 LOCO folds < 6 minimum for alpha=0.05 two-sided Wilcoxon significance)
      CWRU_12k            smote     RandomForest vs. SVM RandomForest          SVM balanced_accuracy loco_fold_seed_averaged               2      0.894883      0.861806             0.033077               0.033077  0.039232 -0.319408   0.385563                 2                 0             0                 0.0          0.5           1.0                   False Low sample size (N=2 LOCO folds < 6 minimum for alpha=0.05 two-sided Wilcoxon significance)
      CWRU_12k            smote     RandomForest vs. SVM RandomForest          SVM           roc_auc loco_fold_seed_averaged               2      0.976539      0.933482             0.043057               0.043057  0.029626 -0.223126   0.309239                 2                 0             0                 0.0          0.5           1.0                   False Low sample size (N=2 LOCO folds < 6 minimum for alpha=0.05 two-sided Wilcoxon significance)
      CWRU_12k            smote          XGBoost vs. SVM      XGBoost          SVM          macro_f1 loco_fold_seed_averaged               2      0.889541      0.858193             0.031348               0.031348  0.038421 -0.313850   0.376546                 2                 0             0                 0.0          0.5           1.0                   False Low sample size (N=2 LOCO folds < 6 minimum for alpha=0.05 two-sided Wilcoxon significance)
      CWRU_12k            smote          XGBoost vs. SVM      XGBoost          SVM          accuracy loco_fold_seed_averaged               2      0.865938      0.831146             0.034792               0.034792  0.040364 -0.327865   0.397448                 2                 0             0                 0.0          0.5           1.0                   False Low sample size (N=2 LOCO folds < 6 minimum for alpha=0.05 two-sided Wilcoxon significance)
      CWRU_12k            smote          XGBoost vs. SVM      XGBoost          SVM balanced_accuracy loco_fold_seed_averaged               2      0.889401      0.861806             0.027595               0.027595  0.032099 -0.260802   0.315992                 2                 0             0                 0.0          0.5           1.0                   False Low sample size (N=2 LOCO folds < 6 minimum for alpha=0.05 two-sided Wilcoxon significance)
      CWRU_12k            smote          XGBoost vs. SVM      XGBoost          SVM           roc_auc loco_fold_seed_averaged               2      0.972634      0.933482             0.039152               0.039152  0.022462 -0.162659   0.240964                 2                 0             0                 0.0          0.5           1.0                   False Low sample size (N=2 LOCO folds < 6 minimum for alpha=0.05 two-sided Wilcoxon significance)
```

---

## 6. Rankings por Dataset e Média de Ranks (`Macro-F1`)

```text
                    scope target_dataset architecture            method balance_strategy        metric  metric_value  rank
 dataset_classical_models       CWRU_12k RandomForest ClassicalBaseline             none macro_f1_mean      0.897524   1.0
 dataset_classical_models       CWRU_12k          SVM ClassicalBaseline             none macro_f1_mean      0.860094   3.0
 dataset_classical_models       CWRU_12k      XGBoost ClassicalBaseline             none macro_f1_mean      0.896372   2.0
 dataset_classical_models       CWRU_12k RandomForest ClassicalBaseline            smote macro_f1_mean      0.894722   1.0
 dataset_classical_models       CWRU_12k          SVM ClassicalBaseline            smote macro_f1_mean      0.858193   3.0
 dataset_classical_models       CWRU_12k      XGBoost ClassicalBaseline            smote macro_f1_mean      0.889541   2.0
mean_rank_across_datasets    ALL_TARGETS RandomForest ClassicalBaseline             none macro_f1_mean      0.897524   1.0
mean_rank_across_datasets    ALL_TARGETS          SVM ClassicalBaseline             none macro_f1_mean      0.860094   3.0
mean_rank_across_datasets    ALL_TARGETS      XGBoost ClassicalBaseline             none macro_f1_mean      0.896372   2.0
mean_rank_across_datasets    ALL_TARGETS RandomForest ClassicalBaseline            smote macro_f1_mean      0.894722   1.0
mean_rank_across_datasets    ALL_TARGETS          SVM ClassicalBaseline            smote macro_f1_mean      0.858193   3.0
mean_rank_across_datasets    ALL_TARGETS      XGBoost ClassicalBaseline            smote macro_f1_mean      0.889541   2.0
```

---

## 7. Desvios em Relação ao Protocolo Original

- **Nenhum desvio metodológico**: todos os folds, amostras de treino/teste, transformações `StandardScaler`, reamostragens `SMOTE`, vocabulários de classes e definições de métricas (`Accuracy`, `Balanced_Accuracy`, `Macro_F1`, `ROC_AUC`) são idênticos aos do benchmark original `run_20261105_141743`.
