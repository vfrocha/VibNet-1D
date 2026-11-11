# Relatório Técnico — Benchmark Scratch vs. Transfer Learning (`scratch_vs_tl`)

> **ESCOPO DE DADOS REAIS**: Execução realizada sobre as janelas reais de `data/processed_tl/detrended_rms/`.


- **Commit Git**: `a234ba75d41ab63e24a531deca747ccafd143ee3` (`git_dirty=True`)
- **Total de Avaliações Executadas**: `2376` (`OK=2376`, `Falhas=0`)
- **Datasets Alvo**: `['CWRU_12k', 'CWRU_48k', 'HUST_Gearbox', 'UOEMD', 'PU']`
- **Arquiteturas**: `['MLP', 'TabNet']`
- **Métodos**: `['Scratch', 'Fine-Tuning', 'Linear Probing']`
- **Estratégias de Balanceamento**: `['none', 'smote']`
- **Seeds**: `[42, 123, 2026]`

---

## 1. Configuração Experimental (`Experimental Setup`)

O benchmark avalia de forma pareada três regimes de treinamento para `MLP` e `TabNet`:
1. **`Scratch`**: Inicialização aleatória; treinamento exclusivo nas $K - 1$ condições operacionais do domínio alvo.
2. **`Fine-Tuning`**: Pré-treinamento do encoder exclusivamente no pool de domínios fonte disjunto (`UORED`, `HUST`, `Mechanical_Gear`, `Electric_Motor`), seguido de adaptação supervisionada completa (encoder + cabeça linear do alvo) nas $K - 1$ condições do alvo.
3. **`Linear Probing`**: Inicialização a partir do mesmo encoder pré-treinado nas fontes, com congelamento estrito de todos os parâmetros e buffers do encoder (`requires_grad=False`, `.eval()`, verificado programaticamente com `max_abs_diff == 0.0`), treinando apenas a cabeça linear do domínio alvo.

---

## 2. Auditoria de Compatibilidade e Reprodutibilidade

- A auditoria completa encontra-se em `compatibility_audit.csv` e `results/scratch_vs_tl/audit/audit_report.md`.
- Todos os métodos pareados em cada `(target_dataset, fold_id, seed, balance_strategy)` compartilham exatamente o mesmo `split_signature` e `preprocessing_signature` registrados em `fold_manifest.csv` e os mesmos identificadores `sample_id` em `predictions.csv`.

---

## 3. Resultados por Dataset Alvo (`Dataset-Level Results`)

```text
target_dataset architecture         method balance_strategy  n_folds  n_seeds  macro_f1_mean  macro_f1_std  balanced_accuracy_mean  roc_auc_mean
      CWRU_12k          MLP        Scratch             none        4        3       0.903959      0.068991                0.905306      0.965246
      CWRU_12k          MLP    Fine-Tuning             none        4        3       0.907849      0.064439                0.908845      0.966858
      CWRU_12k          MLP Linear Probing             none        4        3       0.729942      0.034565                0.732097      0.883279
      CWRU_12k       TabNet        Scratch             none        4        3       0.775949      0.081637                0.780443      0.921680
      CWRU_12k       TabNet    Fine-Tuning             none        4        3       0.832681      0.048407                0.834494      0.944999
      CWRU_12k       TabNet Linear Probing             none        4        3       0.537724      0.074980                0.556651      0.791299
      CWRU_12k          MLP        Scratch            smote        4        3       0.903441      0.065819                0.904624      0.964306
      CWRU_12k          MLP    Fine-Tuning            smote        4        3       0.909035      0.065218                0.910039      0.963859
      CWRU_12k          MLP Linear Probing            smote        4        3       0.728068      0.035311                0.731817      0.883534
      CWRU_12k       TabNet        Scratch            smote        4        3       0.806339      0.064375                0.808851      0.929452
      CWRU_12k       TabNet    Fine-Tuning            smote        4        3       0.797990      0.058046                0.800341      0.933244
      CWRU_12k       TabNet Linear Probing            smote        4        3       0.532536      0.052772                0.556312      0.789439
      CWRU_48k          MLP        Scratch             none       12        3       0.898587      0.171206                0.910317      0.970449
      CWRU_48k          MLP    Fine-Tuning             none       12        3       0.894656      0.165711                0.904431      0.957958
      CWRU_48k          MLP Linear Probing             none       12        3       0.441535      0.188033                0.470040      0.691617
      CWRU_48k       TabNet        Scratch             none       12        3       0.691573      0.252914                0.710251      0.844190
      CWRU_48k       TabNet    Fine-Tuning             none       12        3       0.747191      0.200623                0.759854      0.904267
      CWRU_48k       TabNet Linear Probing             none       12        3       0.424252      0.167078                0.454034      0.619429
      CWRU_48k          MLP        Scratch            smote       12        3       0.893166      0.183441                0.907011      0.966053
      CWRU_48k          MLP    Fine-Tuning            smote       12        3       0.891686      0.182451                0.903638      0.960135
      CWRU_48k          MLP Linear Probing            smote       12        3       0.450285      0.191729                0.478175      0.690687
      CWRU_48k       TabNet        Scratch            smote       12        3       0.689752      0.221954                0.714616      0.885319
      CWRU_48k       TabNet    Fine-Tuning            smote       12        3       0.706308      0.200270                0.728241      0.885170
      CWRU_48k       TabNet Linear Probing            smote       12        3       0.422596      0.163720                0.455688      0.623790
  HUST_Gearbox          MLP        Scratch             none       30        3       0.954843      0.137522                0.958519      0.983981
  HUST_Gearbox          MLP    Fine-Tuning             none       30        3       0.957182      0.136716                0.961111      0.986074
  HUST_Gearbox          MLP Linear Probing             none       30        3       0.856696      0.172621                0.871111      0.959741
  HUST_Gearbox       TabNet        Scratch             none       30        3       0.921914      0.147727                0.930370      0.977111
  HUST_Gearbox       TabNet    Fine-Tuning             none       30        3       0.914004      0.153605                0.921481      0.977148
  HUST_Gearbox       TabNet Linear Probing             none       30        3       0.763202      0.197746                0.795556      0.923074
  HUST_Gearbox          MLP        Scratch            smote       30        3       0.954843      0.137522                0.958519      0.983981
  HUST_Gearbox          MLP    Fine-Tuning            smote       30        3       0.957182      0.136716                0.961111      0.986074
  HUST_Gearbox          MLP Linear Probing            smote       30        3       0.856696      0.172621                0.871111      0.959741
  HUST_Gearbox       TabNet        Scratch            smote       30        3       0.921914      0.147727                0.930370      0.977111
  HUST_Gearbox       TabNet    Fine-Tuning            smote       30        3       0.914004      0.153605                0.921481      0.977148
  HUST_Gearbox       TabNet Linear Probing            smote       30        3       0.763202      0.197746                0.795556      0.923074
         UOEMD          MLP        Scratch             none       16        3       0.815213      0.103811                0.831250      0.973627
         UOEMD          MLP    Fine-Tuning             none       16        3       0.795750      0.100457                0.814323      0.975007
         UOEMD          MLP Linear Probing             none       16        3       0.366209      0.101260                0.413802      0.812891
         UOEMD       TabNet        Scratch             none       16        3       0.526399      0.106286                0.563542      0.886228
         UOEMD       TabNet    Fine-Tuning             none       16        3       0.530970      0.100526                0.572917      0.888683
         UOEMD       TabNet Linear Probing             none       16        3       0.274433      0.096144                0.328385      0.734874
         UOEMD          MLP        Scratch            smote       16        3       0.815213      0.103811                0.831250      0.973627
         UOEMD          MLP    Fine-Tuning            smote       16        3       0.795750      0.100457                0.814323      0.975007
         UOEMD          MLP Linear Probing            smote       16        3       0.366209      0.101260                0.413802      0.812891
         UOEMD       TabNet        Scratch            smote       16        3       0.526399      0.106286                0.563542      0.886228
         UOEMD       TabNet    Fine-Tuning            smote       16        3       0.530970      0.100526                0.572917      0.888683
         UOEMD       TabNet Linear Probing            smote       16        3       0.274433      0.096144                0.328385      0.734874
            PU          MLP        Scratch             none        4        3       0.818559      0.270436                0.844680      0.959326
            PU          MLP    Fine-Tuning             none        4        3       0.838740      0.247677                0.859812      0.962448
            PU          MLP Linear Probing             none        4        3       0.746930      0.182040                0.721570      0.888228
            PU       TabNet        Scratch             none        4        3       0.802860      0.282383                0.808904      0.902760
            PU       TabNet    Fine-Tuning             none        4        3       0.804083      0.277998                0.823442      0.923839
            PU       TabNet Linear Probing             none        4        3       0.642426      0.203882                0.647121      0.845405
            PU          MLP        Scratch            smote        4        3       0.824791      0.267474                0.850069      0.946508
            PU          MLP    Fine-Tuning            smote        4        3       0.834128      0.259476                0.857031      0.956992
            PU          MLP Linear Probing            smote        4        3       0.758870      0.186388                0.772433      0.896640
            PU       TabNet        Scratch            smote        4        3       0.764375      0.252378                0.798685      0.902073
            PU       TabNet    Fine-Tuning            smote        4        3       0.804544      0.286755                0.821801      0.929817
            PU       TabNet Linear Probing            smote        4        3       0.607386      0.174647                0.668130      0.843456
```

---

## 4. Resultados por Arquitetura (`Architecture-Level Results`)

```text
architecture         method balance_strategy  n_datasets  total_evaluations  valid_evaluations  failed_evaluations  macro_f1_mean  macro_f1_std  macro_f1_median  accuracy_mean  accuracy_std  accuracy_median  balanced_accuracy_mean  balanced_accuracy_std  balanced_accuracy_median  roc_auc_mean  roc_auc_std  roc_auc_median
         MLP        Scratch             none           5                198                198                   0       0.899422      0.155573         0.966583       0.908487      0.134144         0.966667                0.908777               0.136415                  0.966667      0.976381     0.057504        1.000000
         MLP    Fine-Tuning             none           5                198                198                   0       0.896510      0.154127         0.982283       0.905954      0.135002         0.980729                0.905914               0.138158                  0.982456      0.975683     0.064614        1.000000
         MLP Linear Probing             none           5                198                198                   0       0.647972      0.268603         0.695387       0.672662      0.247919         0.666667                0.669838               0.248044                  0.682420      0.866423     0.153742        0.886931
      TabNet        Scratch             none           5                198                198                   0       0.768090      0.234379         0.832112       0.783028      0.217537         0.841667                0.784972               0.216311                  0.841667      0.923046     0.135235        0.979167
      TabNet    Fine-Tuning             none           5                198                198                   0       0.779227      0.219959         0.838848       0.797538      0.198940         0.833333                0.796380               0.201322                  0.839912      0.937272     0.091692        0.979167
      TabNet Linear Probing             none           5                198                198                   0       0.562100      0.264588         0.555556       0.598739      0.247388         0.567708                0.596732               0.246199                  0.611115      0.809548     0.162848        0.811446
         MLP        Scratch            smote           5                198                198                   0       0.898782      0.157507         0.966583       0.908067      0.135651         0.966667                0.908461               0.138331                  0.966667      0.974748     0.060483        1.000000
         MLP    Fine-Tuning            smote           5                198                198                   0       0.895763      0.158647         0.973181       0.905437      0.138216         0.970833                0.905673               0.141218                  0.973684      0.975566     0.063913        1.000000
         MLP Linear Probing            smote           5                198                198                   0       0.650173      0.268276         0.688825       0.674725      0.247182         0.666667                0.674383               0.249164                  0.685709      0.866779     0.154825        0.887479
      TabNet        Scratch            smote           5                198                198                   0       0.767268      0.226602         0.834680       0.786319      0.205283         0.833333                0.786868               0.207972                  0.841192      0.930953     0.108179        0.966458
      TabNet    Fine-Tuning            smote           5                198                198                   0       0.769719      0.221884         0.822222       0.788066      0.200709         0.833333                0.788463               0.201801                  0.833333      0.933449     0.092337        0.973656
      TabNet Linear Probing            smote           5                198                198                   0       0.559361      0.262409         0.548453       0.595959      0.245332         0.566667                0.598286               0.245619                  0.581488      0.810110     0.161965        0.811334
```

---

## 5. Comparação por Método: Scratch vs. Fine-Tuning vs. Linear Probing

```text
        method balance_strategy  n_architectures  n_datasets  total_evaluations  valid_evaluations  failed_evaluations  macro_f1_mean  macro_f1_std  macro_f1_median  accuracy_mean  accuracy_std  accuracy_median  balanced_accuracy_mean  balanced_accuracy_std  balanced_accuracy_median  roc_auc_mean  roc_auc_std  roc_auc_median
       Scratch             none                2           5                396                396                   0       0.833756      0.209263         0.932258       0.845758      0.191104         0.925463                0.846875               0.190942                  0.925000      0.949713     0.107160        0.998897
   Fine-Tuning             none                2           5                396                396                   0       0.837869      0.198557         0.920759       0.851746      0.178253         0.914583                0.851147               0.180944                  0.923794      0.956477     0.081517        0.998475
Linear Probing             none                2           5                396                396                   0       0.605036      0.269713         0.556124       0.635700      0.250094         0.633333                0.633285               0.249509                  0.633333      0.837985     0.160702        0.862368
       Scratch            smote                2           5                396                396                   0       0.833025      0.205711         0.915954       0.847193      0.184146         0.914062                0.847665               0.186602                  0.925292      0.952851     0.090231        0.998571
   Fine-Tuning            smote                2           5                396                396                   0       0.832741      0.202702         0.920909       0.846752      0.181856         0.925000                0.847068               0.183574                  0.929167      0.954508     0.082062        0.998284
Linear Probing            smote                2           5                396                396                   0       0.604767      0.268894         0.555556       0.635342      0.249088         0.633333                0.636334               0.250004                  0.633333      0.838445     0.160758        0.864554
```

---

## 6. Impacto do Balanceamento com SMOTE (`Impact of SMOTE`)

- Em cada fold LOCO, `none` preserva a distribuição natural das $K - 1$ condições de treino, enquanto `smote` realiza interpolação k-NN (`k_neighbors = min(5, minority_count - 1)`) exclusivamente no conjunto de treinamento escalado (`X_train_scaled`).
- Os resultados discriminados por `balance_strategy` (`none` vs. `smote`) constam nas tabelas das Seções 3, 4, 5 e 7.

---

## 7. Análise Estatística Pareada e Rankings (`Statistical Analysis`)

- **Unidade Estatística Primária**: Fold LOCO ($N = K_T$ folds por dataset alvo), agregando primeiramente a média das seeds repetidas dentro de cada fold para evitar pseudorreplicação entre seeds correlacionadas.
- **Correção para Múltiplas Comparações**: Ajuste *step-down* de Holm-Bonferroni (`p_value_holm`).

### Testes Pareados (Wilcoxon Signed-Rank + Holm)

```text
target_dataset architecture balance_strategy                 comparison            metric  n_paired_folds  scratch_mean  tl_mean  mean_diff_tl_minus_scratch  p_value_raw  p_value_holm  significant_alpha_0_05
      CWRU_12k          MLP             none    Fine-Tuning vs. Scratch          macro_f1               4      0.903959 0.907849                    0.003890     0.250000      1.000000                   False
      CWRU_12k          MLP             none    Fine-Tuning vs. Scratch balanced_accuracy               4      0.905306 0.908845                    0.003539     0.125000      0.750000                   False
      CWRU_12k          MLP             none    Fine-Tuning vs. Scratch          accuracy               4      0.885972 0.890365                    0.004392     0.250000      1.000000                   False
      CWRU_12k          MLP             none    Fine-Tuning vs. Scratch           roc_auc               4      0.965246 0.966858                    0.001612     0.250000      1.000000                   False
      CWRU_12k          MLP             none Linear Probing vs. Scratch          macro_f1               4      0.903959 0.729942                   -0.174017     0.125000      0.750000                   False
      CWRU_12k          MLP             none Linear Probing vs. Scratch balanced_accuracy               4      0.905306 0.732097                   -0.173209     0.125000      0.750000                   False
      CWRU_12k          MLP             none Linear Probing vs. Scratch          accuracy               4      0.885972 0.682283                   -0.203689     0.125000      0.750000                   False
      CWRU_12k          MLP             none Linear Probing vs. Scratch           roc_auc               4      0.965246 0.883279                   -0.081967     0.125000      0.875000                   False
      CWRU_12k          MLP            smote    Fine-Tuning vs. Scratch          macro_f1               4      0.903441 0.909035                    0.005593     0.250000      1.000000                   False
      CWRU_12k          MLP            smote    Fine-Tuning vs. Scratch balanced_accuracy               4      0.904624 0.910039                    0.005415     0.250000      1.000000                   False
      CWRU_12k          MLP            smote    Fine-Tuning vs. Scratch          accuracy               4      0.885191 0.891649                    0.006458     0.250000      1.000000                   False
      CWRU_12k          MLP            smote    Fine-Tuning vs. Scratch           roc_auc               4      0.964306 0.963859                   -0.000447     0.625000      1.000000                   False
      CWRU_12k          MLP            smote Linear Probing vs. Scratch          macro_f1               4      0.903441 0.728068                   -0.175373     0.125000      0.750000                   False
      CWRU_12k          MLP            smote Linear Probing vs. Scratch balanced_accuracy               4      0.904624 0.731817                   -0.172807     0.125000      0.750000                   False
      CWRU_12k          MLP            smote Linear Probing vs. Scratch          accuracy               4      0.885191 0.681545                   -0.203646     0.125000      0.750000                   False
      CWRU_12k          MLP            smote Linear Probing vs. Scratch           roc_auc               4      0.964306 0.883534                   -0.080772     0.125000      0.750000                   False
      CWRU_12k       TabNet             none    Fine-Tuning vs. Scratch          macro_f1               4      0.775949 0.832681                    0.056732     0.125000      0.875000                   False
      CWRU_12k       TabNet             none    Fine-Tuning vs. Scratch balanced_accuracy               4      0.780443 0.834494                    0.054051     0.125000      0.875000                   False
      CWRU_12k       TabNet             none    Fine-Tuning vs. Scratch          accuracy               4      0.745920 0.804497                    0.058576     0.125000      0.875000                   False
      CWRU_12k       TabNet             none    Fine-Tuning vs. Scratch           roc_auc               4      0.921680 0.944999                    0.023319     0.125000      0.769043                   False
      CWRU_12k       TabNet             none Linear Probing vs. Scratch          macro_f1               4      0.775949 0.537724                   -0.238225     0.125000      0.875000                   False
      CWRU_12k       TabNet             none Linear Probing vs. Scratch balanced_accuracy               4      0.780443 0.556651                   -0.223792     0.125000      0.875000                   False
      CWRU_12k       TabNet             none Linear Probing vs. Scratch          accuracy               4      0.745920 0.537170                   -0.208750     0.125000      0.875000                   False
      CWRU_12k       TabNet             none Linear Probing vs. Scratch           roc_auc               4      0.921680 0.791299                   -0.130380     0.125000      0.769043                   False
      CWRU_12k       TabNet            smote    Fine-Tuning vs. Scratch          macro_f1               4      0.806339 0.797990                   -0.008349     0.875000      1.000000                   False
      CWRU_12k       TabNet            smote    Fine-Tuning vs. Scratch balanced_accuracy               4      0.808851 0.800341                   -0.008510     0.875000      1.000000                   False
      CWRU_12k       TabNet            smote    Fine-Tuning vs. Scratch          accuracy               4      0.772943 0.763212                   -0.009731     0.875000      1.000000                   False
      CWRU_12k       TabNet            smote    Fine-Tuning vs. Scratch           roc_auc               4      0.929452 0.933244                    0.003792     0.875000      1.000000                   False
      CWRU_12k       TabNet            smote Linear Probing vs. Scratch          macro_f1               4      0.806339 0.532536                   -0.273803     0.125000      0.875000                   False
      CWRU_12k       TabNet            smote Linear Probing vs. Scratch balanced_accuracy               4      0.808851 0.556312                   -0.252539     0.125000      0.875000                   False
      CWRU_12k       TabNet            smote Linear Probing vs. Scratch          accuracy               4      0.772943 0.527005                   -0.245938     0.125000      0.875000                   False
      CWRU_12k       TabNet            smote Linear Probing vs. Scratch           roc_auc               4      0.929452 0.789439                   -0.140013     0.125000      0.875000                   False
      CWRU_48k          MLP             none    Fine-Tuning vs. Scratch          macro_f1              12      0.898587 0.894656                   -0.003931     0.675781      1.000000                   False
      CWRU_48k          MLP             none    Fine-Tuning vs. Scratch balanced_accuracy              12      0.910317 0.904431                   -0.005886     0.617188      1.000000                   False
      CWRU_48k          MLP             none    Fine-Tuning vs. Scratch          accuracy              12      0.911626 0.903395                   -0.008230     0.417969      1.000000                   False
      CWRU_48k          MLP             none    Fine-Tuning vs. Scratch           roc_auc              12      0.970449 0.957958                   -0.012491     0.367188      1.000000                   False
      CWRU_48k          MLP             none Linear Probing vs. Scratch          macro_f1              12      0.898587 0.441535                   -0.457052     0.000488      0.004395                    True
      CWRU_48k          MLP             none Linear Probing vs. Scratch balanced_accuracy              12      0.910317 0.470040                   -0.440278     0.000488      0.004395                    True
      CWRU_48k          MLP             none Linear Probing vs. Scratch          accuracy              12      0.911626 0.474331                   -0.437294     0.000488      0.004395                    True
      CWRU_48k          MLP             none Linear Probing vs. Scratch           roc_auc              12      0.970449 0.691617                   -0.278832     0.000488      0.004395                    True
      CWRU_48k          MLP            smote    Fine-Tuning vs. Scratch          macro_f1              12      0.893166 0.891686                   -0.001480     0.733398      1.000000                   False
      CWRU_48k          MLP            smote    Fine-Tuning vs. Scratch balanced_accuracy              12      0.907011 0.903638                   -0.003373     0.650391      1.000000                   False
      CWRU_48k          MLP            smote    Fine-Tuning vs. Scratch          accuracy              12      0.907253 0.901852                   -0.005401     0.580078      1.000000                   False
      CWRU_48k          MLP            smote    Fine-Tuning vs. Scratch           roc_auc              12      0.966053 0.960135                   -0.005918     0.070312      0.492188                   False
      CWRU_48k          MLP            smote Linear Probing vs. Scratch          macro_f1              12      0.893166 0.450285                   -0.442881     0.000488      0.004395                    True
      CWRU_48k          MLP            smote Linear Probing vs. Scratch balanced_accuracy              12      0.907011 0.478175                   -0.428836     0.000488      0.004395                    True
      CWRU_48k          MLP            smote Linear Probing vs. Scratch          accuracy              12      0.907253 0.486265                   -0.420988     0.000488      0.004395                    True
      CWRU_48k          MLP            smote Linear Probing vs. Scratch           roc_auc              12      0.966053 0.690687                   -0.275366     0.000488      0.004395                    True
      CWRU_48k       TabNet             none    Fine-Tuning vs. Scratch          macro_f1              12      0.691573 0.747191                    0.055618     0.203613      0.875000                   False
      CWRU_48k       TabNet             none    Fine-Tuning vs. Scratch balanced_accuracy              12      0.710251 0.759854                    0.049603     0.301270      1.000000                   False
      CWRU_48k       TabNet             none    Fine-Tuning vs. Scratch          accuracy              12      0.703858 0.769547                    0.065689     0.151367      0.875000                   False
      CWRU_48k       TabNet             none    Fine-Tuning vs. Scratch           roc_auc              12      0.844190 0.904267                    0.060077     0.109863      0.769043                   False
      CWRU_48k       TabNet             none Linear Probing vs. Scratch          macro_f1              12      0.691573 0.424252                   -0.267321     0.000488      0.003906                    True
      CWRU_48k       TabNet             none Linear Probing vs. Scratch balanced_accuracy              12      0.710251 0.454034                   -0.256217     0.000977      0.007812                    True
      CWRU_48k       TabNet             none Linear Probing vs. Scratch          accuracy              12      0.703858 0.452932                   -0.250926     0.000977      0.007812                    True
      CWRU_48k       TabNet             none Linear Probing vs. Scratch           roc_auc              12      0.844190 0.619429                   -0.224761     0.000488      0.004395                    True
      CWRU_48k       TabNet            smote    Fine-Tuning vs. Scratch          macro_f1              12      0.689752 0.706308                    0.016556     0.791016      1.000000                   False
      CWRU_48k       TabNet            smote    Fine-Tuning vs. Scratch balanced_accuracy              12      0.714616 0.728241                    0.013624     0.834961      1.000000                   False
      CWRU_48k       TabNet            smote    Fine-Tuning vs. Scratch          accuracy              12      0.720782 0.732253                    0.011471     0.834961      1.000000                   False
      CWRU_48k       TabNet            smote    Fine-Tuning vs. Scratch           roc_auc              12      0.885319 0.885170                   -0.000149     0.969727      1.000000                   False
      CWRU_48k       TabNet            smote Linear Probing vs. Scratch          macro_f1              12      0.689752 0.422596                   -0.267156     0.000488      0.003906                    True
      CWRU_48k       TabNet            smote Linear Probing vs. Scratch balanced_accuracy              12      0.714616 0.455688                   -0.258929     0.000488      0.003906                    True
      CWRU_48k       TabNet            smote Linear Probing vs. Scratch          accuracy              12      0.720782 0.457099                   -0.263683     0.000488      0.003923                    True
      CWRU_48k       TabNet            smote Linear Probing vs. Scratch           roc_auc              12      0.885319 0.623790                   -0.261528     0.000488      0.004395                    True
  HUST_Gearbox          MLP             none    Fine-Tuning vs. Scratch          macro_f1              30      0.954843 0.957182                    0.002338     0.940476      1.000000                   False
  HUST_Gearbox          MLP             none    Fine-Tuning vs. Scratch balanced_accuracy              30      0.958519 0.961111                    0.002593     0.889685      1.000000                   False
  HUST_Gearbox          MLP             none    Fine-Tuning vs. Scratch          accuracy              30      0.958519 0.961111                    0.002593     0.723178      1.000000                   False
  HUST_Gearbox          MLP             none    Fine-Tuning vs. Scratch           roc_auc              30      0.983981 0.986074                    0.002093     0.499337      1.000000                   False
  HUST_Gearbox          MLP             none Linear Probing vs. Scratch          macro_f1              30      0.954843 0.856696                   -0.098148     0.000702      0.005614                    True
  HUST_Gearbox          MLP             none Linear Probing vs. Scratch balanced_accuracy              30      0.958519 0.871111                   -0.087407     0.000580      0.004643                    True
  HUST_Gearbox          MLP             none Linear Probing vs. Scratch          accuracy              30      0.958519 0.871111                   -0.087407     0.000558      0.004461                    True
  HUST_Gearbox          MLP             none Linear Probing vs. Scratch           roc_auc              30      0.983981 0.959741                   -0.024241     0.008301      0.066411                   False
  HUST_Gearbox          MLP            smote    Fine-Tuning vs. Scratch          macro_f1              30      0.954843 0.957182                    0.002338     0.940476      1.000000                   False
  HUST_Gearbox          MLP            smote    Fine-Tuning vs. Scratch balanced_accuracy              30      0.958519 0.961111                    0.002593     0.889685      1.000000                   False
  HUST_Gearbox          MLP            smote    Fine-Tuning vs. Scratch          accuracy              30      0.958519 0.961111                    0.002593     0.723178      1.000000                   False
  HUST_Gearbox          MLP            smote    Fine-Tuning vs. Scratch           roc_auc              30      0.983981 0.986074                    0.002093     0.499337      1.000000                   False
  HUST_Gearbox          MLP            smote Linear Probing vs. Scratch          macro_f1              30      0.954843 0.856696                   -0.098148     0.000702      0.005614                    True
  HUST_Gearbox          MLP            smote Linear Probing vs. Scratch balanced_accuracy              30      0.958519 0.871111                   -0.087407     0.000580      0.004643                    True
  HUST_Gearbox          MLP            smote Linear Probing vs. Scratch          accuracy              30      0.958519 0.871111                   -0.087407     0.000558      0.004461                    True
  HUST_Gearbox          MLP            smote Linear Probing vs. Scratch           roc_auc              30      0.983981 0.959741                   -0.024241     0.008301      0.066411                   False
  HUST_Gearbox       TabNet             none    Fine-Tuning vs. Scratch          macro_f1              30      0.921914 0.914004                   -0.007910     0.926124      1.000000                   False
  HUST_Gearbox       TabNet             none    Fine-Tuning vs. Scratch balanced_accuracy              30      0.930370 0.921481                   -0.008889     0.917937      1.000000                   False
  HUST_Gearbox       TabNet             none    Fine-Tuning vs. Scratch          accuracy              30      0.930370 0.921481                   -0.008889     0.868991      1.000000                   False
  HUST_Gearbox       TabNet             none    Fine-Tuning vs. Scratch           roc_auc              30      0.977111 0.977148                    0.000037     0.958487      1.000000                   False
  HUST_Gearbox       TabNet             none Linear Probing vs. Scratch          macro_f1              30      0.921914 0.763202                   -0.158712     0.000078      0.000706                    True
  HUST_Gearbox       TabNet             none Linear Probing vs. Scratch balanced_accuracy              30      0.930370 0.795556                   -0.134815     0.000072      0.000719                    True
  HUST_Gearbox       TabNet             none Linear Probing vs. Scratch          accuracy              30      0.930370 0.795556                   -0.134815     0.000072      0.000718                    True
  HUST_Gearbox       TabNet             none Linear Probing vs. Scratch           roc_auc              30      0.977111 0.923074                   -0.054037     0.000528      0.004395                    True
  HUST_Gearbox       TabNet            smote    Fine-Tuning vs. Scratch          macro_f1              30      0.921914 0.914004                   -0.007910     0.926124      1.000000                   False
  HUST_Gearbox       TabNet            smote    Fine-Tuning vs. Scratch balanced_accuracy              30      0.930370 0.921481                   -0.008889     0.917937      1.000000                   False
  HUST_Gearbox       TabNet            smote    Fine-Tuning vs. Scratch          accuracy              30      0.930370 0.921481                   -0.008889     0.868991      1.000000                   False
  HUST_Gearbox       TabNet            smote    Fine-Tuning vs. Scratch           roc_auc              30      0.977111 0.977148                    0.000037     0.958487      1.000000                   False
  HUST_Gearbox       TabNet            smote Linear Probing vs. Scratch          macro_f1              30      0.921914 0.763202                   -0.158712     0.000078      0.000706                    True
  HUST_Gearbox       TabNet            smote Linear Probing vs. Scratch balanced_accuracy              30      0.930370 0.795556                   -0.134815     0.000072      0.000719                    True
  HUST_Gearbox       TabNet            smote Linear Probing vs. Scratch          accuracy              30      0.930370 0.795556                   -0.134815     0.000072      0.000718                    True
  HUST_Gearbox       TabNet            smote Linear Probing vs. Scratch           roc_auc              30      0.977111 0.923074                   -0.054037     0.000528      0.004395                    True
            PU          MLP             none    Fine-Tuning vs. Scratch          macro_f1               4      0.818559 0.838740                    0.020181     0.375000      1.000000                   False
            PU          MLP             none    Fine-Tuning vs. Scratch balanced_accuracy               4      0.844680 0.859812                    0.015132     0.125000      0.750000                   False
            PU          MLP             none    Fine-Tuning vs. Scratch          accuracy               4      0.855292 0.882059                    0.026767     0.375000      1.000000                   False
            PU          MLP             none    Fine-Tuning vs. Scratch           roc_auc               4      0.959326 0.962448                    0.003122     0.500000      1.000000                   False
            PU          MLP             none Linear Probing vs. Scratch          macro_f1               4      0.818559 0.746930                   -0.071629     0.250000      1.000000                   False
            PU          MLP             none Linear Probing vs. Scratch balanced_accuracy               4      0.844680 0.721570                   -0.123110     0.125000      0.750000                   False
            PU          MLP             none Linear Probing vs. Scratch          accuracy               4      0.855292 0.805104                   -0.050188     0.375000      1.000000                   False
            PU          MLP             none Linear Probing vs. Scratch           roc_auc               4      0.959326 0.888228                   -0.071098     0.125000      0.875000                   False
            PU          MLP            smote    Fine-Tuning vs. Scratch          macro_f1               4      0.824791 0.834128                    0.009337     0.625000      1.000000                   False
            PU          MLP            smote    Fine-Tuning vs. Scratch balanced_accuracy               4      0.850069 0.857031                    0.006961     0.625000      1.000000                   False
            PU          MLP            smote    Fine-Tuning vs. Scratch          accuracy               4      0.862265 0.876888                    0.014622     0.625000      1.000000                   False
            PU          MLP            smote    Fine-Tuning vs. Scratch           roc_auc               4      0.946508 0.956992                    0.010484     0.500000      1.000000                   False
            PU          MLP            smote Linear Probing vs. Scratch          macro_f1               4      0.824791 0.758870                   -0.065921     0.250000      1.000000                   False
            PU          MLP            smote Linear Probing vs. Scratch balanced_accuracy               4      0.850069 0.772433                   -0.077636     0.250000      1.000000                   False
            PU          MLP            smote Linear Probing vs. Scratch          accuracy               4      0.862265 0.804086                   -0.058180     0.250000      1.000000                   False
            PU          MLP            smote Linear Probing vs. Scratch           roc_auc               4      0.946508 0.896640                   -0.049868     0.125000      0.750000                   False
            PU       TabNet             none    Fine-Tuning vs. Scratch          macro_f1               4      0.802860 0.804083                    0.001223     0.875000      1.000000                   False
            PU       TabNet             none    Fine-Tuning vs. Scratch balanced_accuracy               4      0.808904 0.823442                    0.014537     0.625000      1.000000                   False
            PU       TabNet             none    Fine-Tuning vs. Scratch          accuracy               4      0.830531 0.843462                    0.012931     0.500000      1.000000                   False
            PU       TabNet             none    Fine-Tuning vs. Scratch           roc_auc               4      0.902760 0.923839                    0.021080     0.375000      1.000000                   False
            PU       TabNet             none Linear Probing vs. Scratch          macro_f1               4      0.802860 0.642426                   -0.160434     0.125000      0.875000                   False
            PU       TabNet             none Linear Probing vs. Scratch balanced_accuracy               4      0.808904 0.647121                   -0.161783     0.125000      0.875000                   False
            PU       TabNet             none Linear Probing vs. Scratch          accuracy               4      0.830531 0.703015                   -0.127516     0.250000      1.000000                   False
            PU       TabNet             none Linear Probing vs. Scratch           roc_auc               4      0.902760 0.845405                   -0.057355     0.375000      1.000000                   False
            PU       TabNet            smote    Fine-Tuning vs. Scratch          macro_f1               4      0.764375 0.804544                    0.040169     0.250000      1.000000                   False
            PU       TabNet            smote    Fine-Tuning vs. Scratch balanced_accuracy               4      0.798685 0.821801                    0.023116     0.375000      1.000000                   False
            PU       TabNet            smote    Fine-Tuning vs. Scratch          accuracy               4      0.807024 0.840347                    0.033324     0.375000      1.000000                   False
            PU       TabNet            smote    Fine-Tuning vs. Scratch           roc_auc               4      0.902073 0.929817                    0.027744     0.125000      0.875000                   False
            PU       TabNet            smote Linear Probing vs. Scratch          macro_f1               4      0.764375 0.607386                   -0.156989     0.125000      0.875000                   False
            PU       TabNet            smote Linear Probing vs. Scratch balanced_accuracy               4      0.798685 0.668130                   -0.130555     0.125000      0.875000                   False
            PU       TabNet            smote Linear Probing vs. Scratch          accuracy               4      0.807024 0.654807                   -0.152217     0.125000      0.875000                   False
            PU       TabNet            smote Linear Probing vs. Scratch           roc_auc               4      0.902073 0.843456                   -0.058617     0.375000      1.000000                   False
         UOEMD          MLP             none    Fine-Tuning vs. Scratch          macro_f1              16      0.815213 0.795750                   -0.019463     0.021393      0.149750                   False
         UOEMD          MLP             none    Fine-Tuning vs. Scratch balanced_accuracy              16      0.831250 0.814323                   -0.016927     0.043627      0.305391                   False
         UOEMD          MLP             none    Fine-Tuning vs. Scratch          accuracy              16      0.831250 0.814323                   -0.016927     0.043627      0.305391                   False
         UOEMD          MLP             none    Fine-Tuning vs. Scratch           roc_auc              16      0.973627 0.975007                    0.001380     0.899933      1.000000                   False
         UOEMD          MLP             none Linear Probing vs. Scratch          macro_f1              16      0.815213 0.366209                   -0.449004     0.000031      0.000305                    True
         UOEMD          MLP             none Linear Probing vs. Scratch balanced_accuracy              16      0.831250 0.413802                   -0.417448     0.000031      0.000305                    True
         UOEMD          MLP             none Linear Probing vs. Scratch          accuracy              16      0.831250 0.413802                   -0.417448     0.000031      0.000305                    True
         UOEMD          MLP             none Linear Probing vs. Scratch           roc_auc              16      0.973627 0.812891                   -0.160737     0.000031      0.000305                    True
         UOEMD          MLP            smote    Fine-Tuning vs. Scratch          macro_f1              16      0.815213 0.795750                   -0.019463     0.021393      0.149750                   False
         UOEMD          MLP            smote    Fine-Tuning vs. Scratch balanced_accuracy              16      0.831250 0.814323                   -0.016927     0.043627      0.305391                   False
         UOEMD          MLP            smote    Fine-Tuning vs. Scratch          accuracy              16      0.831250 0.814323                   -0.016927     0.043627      0.305391                   False
         UOEMD          MLP            smote    Fine-Tuning vs. Scratch           roc_auc              16      0.973627 0.975007                    0.001380     0.899933      1.000000                   False
         UOEMD          MLP            smote Linear Probing vs. Scratch          macro_f1              16      0.815213 0.366209                   -0.449004     0.000031      0.000305                    True
         UOEMD          MLP            smote Linear Probing vs. Scratch balanced_accuracy              16      0.831250 0.413802                   -0.417448     0.000031      0.000305                    True
         UOEMD          MLP            smote Linear Probing vs. Scratch          accuracy              16      0.831250 0.413802                   -0.417448     0.000031      0.000305                    True
         UOEMD          MLP            smote Linear Probing vs. Scratch           roc_auc              16      0.973627 0.812891                   -0.160737     0.000031      0.000305                    True
         UOEMD       TabNet             none    Fine-Tuning vs. Scratch          macro_f1              16      0.526399 0.530970                    0.004571     0.860260      1.000000                   False
         UOEMD       TabNet             none    Fine-Tuning vs. Scratch balanced_accuracy              16      0.563542 0.572917                    0.009375     0.958754      1.000000                   False
         UOEMD       TabNet             none    Fine-Tuning vs. Scratch          accuracy              16      0.563542 0.572917                    0.009375     0.979370      1.000000                   False
         UOEMD       TabNet             none    Fine-Tuning vs. Scratch           roc_auc              16      0.886228 0.888683                    0.002455     0.668549      1.000000                   False
         UOEMD       TabNet             none Linear Probing vs. Scratch          macro_f1              16      0.526399 0.274433                   -0.251966     0.000031      0.000305                    True
         UOEMD       TabNet             none Linear Probing vs. Scratch balanced_accuracy              16      0.563542 0.328385                   -0.235156     0.000433      0.003897                    True
         UOEMD       TabNet             none Linear Probing vs. Scratch          accuracy              16      0.563542 0.328385                   -0.235156     0.000436      0.003923                    True
         UOEMD       TabNet             none Linear Probing vs. Scratch           roc_auc              16      0.886228 0.734874                   -0.151354     0.000031      0.000305                    True
         UOEMD       TabNet            smote    Fine-Tuning vs. Scratch          macro_f1              16      0.526399 0.530970                    0.004571     0.860260      1.000000                   False
         UOEMD       TabNet            smote    Fine-Tuning vs. Scratch balanced_accuracy              16      0.563542 0.572917                    0.009375     0.958754      1.000000                   False
         UOEMD       TabNet            smote    Fine-Tuning vs. Scratch          accuracy              16      0.563542 0.572917                    0.009375     0.979370      1.000000                   False
         UOEMD       TabNet            smote    Fine-Tuning vs. Scratch           roc_auc              16      0.886228 0.888683                    0.002455     0.668549      1.000000                   False
         UOEMD       TabNet            smote Linear Probing vs. Scratch          macro_f1              16      0.526399 0.274433                   -0.251966     0.000031      0.000305                    True
         UOEMD       TabNet            smote Linear Probing vs. Scratch balanced_accuracy              16      0.563542 0.328385                   -0.235156     0.000433      0.003897                    True
         UOEMD       TabNet            smote Linear Probing vs. Scratch          accuracy              16      0.563542 0.328385                   -0.235156     0.000436      0.003923                    True
         UOEMD       TabNet            smote Linear Probing vs. Scratch           roc_auc              16      0.886228 0.734874                   -0.151354     0.000031      0.000305                    True
```

### Rankings dos Métodos (`Macro-F1`)

```text
                      scope target_dataset architecture         method balance_strategy        metric  metric_value  rank
dataset_within_architecture       CWRU_12k          MLP        Scratch             none macro_f1_mean      0.903959   2.0
dataset_within_architecture       CWRU_12k          MLP    Fine-Tuning             none macro_f1_mean      0.907849   1.0
dataset_within_architecture       CWRU_12k          MLP Linear Probing             none macro_f1_mean      0.729942   3.0
dataset_within_architecture       CWRU_12k       TabNet        Scratch             none macro_f1_mean      0.775949   2.0
dataset_within_architecture       CWRU_12k       TabNet    Fine-Tuning             none macro_f1_mean      0.832681   1.0
dataset_within_architecture       CWRU_12k       TabNet Linear Probing             none macro_f1_mean      0.537724   3.0
dataset_within_architecture       CWRU_12k          MLP        Scratch            smote macro_f1_mean      0.903441   2.0
dataset_within_architecture       CWRU_12k          MLP    Fine-Tuning            smote macro_f1_mean      0.909035   1.0
dataset_within_architecture       CWRU_12k          MLP Linear Probing            smote macro_f1_mean      0.728068   3.0
dataset_within_architecture       CWRU_12k       TabNet        Scratch            smote macro_f1_mean      0.806339   1.0
dataset_within_architecture       CWRU_12k       TabNet    Fine-Tuning            smote macro_f1_mean      0.797990   2.0
dataset_within_architecture       CWRU_12k       TabNet Linear Probing            smote macro_f1_mean      0.532536   3.0
dataset_within_architecture       CWRU_48k          MLP        Scratch             none macro_f1_mean      0.898587   1.0
dataset_within_architecture       CWRU_48k          MLP    Fine-Tuning             none macro_f1_mean      0.894656   2.0
dataset_within_architecture       CWRU_48k          MLP Linear Probing             none macro_f1_mean      0.441535   3.0
dataset_within_architecture       CWRU_48k       TabNet        Scratch             none macro_f1_mean      0.691573   2.0
dataset_within_architecture       CWRU_48k       TabNet    Fine-Tuning             none macro_f1_mean      0.747191   1.0
dataset_within_architecture       CWRU_48k       TabNet Linear Probing             none macro_f1_mean      0.424252   3.0
dataset_within_architecture       CWRU_48k          MLP        Scratch            smote macro_f1_mean      0.893166   1.0
dataset_within_architecture       CWRU_48k          MLP    Fine-Tuning            smote macro_f1_mean      0.891686   2.0
dataset_within_architecture       CWRU_48k          MLP Linear Probing            smote macro_f1_mean      0.450285   3.0
dataset_within_architecture       CWRU_48k       TabNet        Scratch            smote macro_f1_mean      0.689752   2.0
dataset_within_architecture       CWRU_48k       TabNet    Fine-Tuning            smote macro_f1_mean      0.706308   1.0
dataset_within_architecture       CWRU_48k       TabNet Linear Probing            smote macro_f1_mean      0.422596   3.0
dataset_within_architecture   HUST_Gearbox          MLP        Scratch             none macro_f1_mean      0.954843   2.0
dataset_within_architecture   HUST_Gearbox          MLP    Fine-Tuning             none macro_f1_mean      0.957182   1.0
dataset_within_architecture   HUST_Gearbox          MLP Linear Probing             none macro_f1_mean      0.856696   3.0
dataset_within_architecture   HUST_Gearbox       TabNet        Scratch             none macro_f1_mean      0.921914   1.0
dataset_within_architecture   HUST_Gearbox       TabNet    Fine-Tuning             none macro_f1_mean      0.914004   2.0
dataset_within_architecture   HUST_Gearbox       TabNet Linear Probing             none macro_f1_mean      0.763202   3.0
dataset_within_architecture   HUST_Gearbox          MLP        Scratch            smote macro_f1_mean      0.954843   2.0
dataset_within_architecture   HUST_Gearbox          MLP    Fine-Tuning            smote macro_f1_mean      0.957182   1.0
dataset_within_architecture   HUST_Gearbox          MLP Linear Probing            smote macro_f1_mean      0.856696   3.0
dataset_within_architecture   HUST_Gearbox       TabNet        Scratch            smote macro_f1_mean      0.921914   1.0
dataset_within_architecture   HUST_Gearbox       TabNet    Fine-Tuning            smote macro_f1_mean      0.914004   2.0
dataset_within_architecture   HUST_Gearbox       TabNet Linear Probing            smote macro_f1_mean      0.763202   3.0
dataset_within_architecture          UOEMD          MLP        Scratch             none macro_f1_mean      0.815213   1.0
dataset_within_architecture          UOEMD          MLP    Fine-Tuning             none macro_f1_mean      0.795750   2.0
dataset_within_architecture          UOEMD          MLP Linear Probing             none macro_f1_mean      0.366209   3.0
dataset_within_architecture          UOEMD       TabNet        Scratch             none macro_f1_mean      0.526399   2.0
dataset_within_architecture          UOEMD       TabNet    Fine-Tuning             none macro_f1_mean      0.530970   1.0
dataset_within_architecture          UOEMD       TabNet Linear Probing             none macro_f1_mean      0.274433   3.0
dataset_within_architecture          UOEMD          MLP        Scratch            smote macro_f1_mean      0.815213   1.0
dataset_within_architecture          UOEMD          MLP    Fine-Tuning            smote macro_f1_mean      0.795750   2.0
dataset_within_architecture          UOEMD          MLP Linear Probing            smote macro_f1_mean      0.366209   3.0
dataset_within_architecture          UOEMD       TabNet        Scratch            smote macro_f1_mean      0.526399   2.0
dataset_within_architecture          UOEMD       TabNet    Fine-Tuning            smote macro_f1_mean      0.530970   1.0
dataset_within_architecture          UOEMD       TabNet Linear Probing            smote macro_f1_mean      0.274433   3.0
dataset_within_architecture             PU          MLP        Scratch             none macro_f1_mean      0.818559   2.0
dataset_within_architecture             PU          MLP    Fine-Tuning             none macro_f1_mean      0.838740   1.0
dataset_within_architecture             PU          MLP Linear Probing             none macro_f1_mean      0.746930   3.0
dataset_within_architecture             PU       TabNet        Scratch             none macro_f1_mean      0.802860   2.0
dataset_within_architecture             PU       TabNet    Fine-Tuning             none macro_f1_mean      0.804083   1.0
dataset_within_architecture             PU       TabNet Linear Probing             none macro_f1_mean      0.642426   3.0
dataset_within_architecture             PU          MLP        Scratch            smote macro_f1_mean      0.824791   2.0
dataset_within_architecture             PU          MLP    Fine-Tuning            smote macro_f1_mean      0.834128   1.0
dataset_within_architecture             PU          MLP Linear Probing            smote macro_f1_mean      0.758870   3.0
dataset_within_architecture             PU       TabNet        Scratch            smote macro_f1_mean      0.764375   2.0
dataset_within_architecture             PU       TabNet    Fine-Tuning            smote macro_f1_mean      0.804544   1.0
dataset_within_architecture             PU       TabNet Linear Probing            smote macro_f1_mean      0.607386   3.0
  mean_rank_across_datasets    ALL_TARGETS          MLP        Scratch             none macro_f1_mean      0.878232   1.6
  mean_rank_across_datasets    ALL_TARGETS          MLP    Fine-Tuning             none macro_f1_mean      0.878835   1.4
  mean_rank_across_datasets    ALL_TARGETS          MLP Linear Probing             none macro_f1_mean      0.628262   3.0
  mean_rank_across_datasets    ALL_TARGETS       TabNet        Scratch             none macro_f1_mean      0.743739   1.8
  mean_rank_across_datasets    ALL_TARGETS       TabNet    Fine-Tuning             none macro_f1_mean      0.765786   1.2
  mean_rank_across_datasets    ALL_TARGETS       TabNet Linear Probing             none macro_f1_mean      0.528407   3.0
  mean_rank_across_datasets    ALL_TARGETS          MLP        Scratch            smote macro_f1_mean      0.878291   1.6
  mean_rank_across_datasets    ALL_TARGETS          MLP    Fine-Tuning            smote macro_f1_mean      0.877556   1.4
  mean_rank_across_datasets    ALL_TARGETS          MLP Linear Probing            smote macro_f1_mean      0.632026   3.0
  mean_rank_across_datasets    ALL_TARGETS       TabNet        Scratch            smote macro_f1_mean      0.741756   1.6
  mean_rank_across_datasets    ALL_TARGETS       TabNet    Fine-Tuning            smote macro_f1_mean      0.750763   1.4
  mean_rank_across_datasets    ALL_TARGETS       TabNet Linear Probing            smote macro_f1_mean      0.520031   3.0
```

---

## 8. Limitações Metodológicas (`Limitations`)

1. **Tamanho Amostral em Datasets com 4 Condições Operacionais (`CWRU_12k` e `PU`)**:
   - Com $K_T = 4$ folds LOCO, o menor $p$-valor bicaudal possível no teste exato de Wilcoxon signed-rank é $2 / 2^4 = 0.125 > 0.05$. Assim, em `CWRU_12k` e `PU`, diferenças consistentes em todos os 4 folds devem ser interpretadas descritivamente (intervalo de confiança e proporção de folds vencidos `n_folds_tl_better = 4/4`), enquanto `CWRU_48k` ($K_T = 12$), `UOEMD` ($K_T = 16$) e `HUST_Gearbox` ($K_T = 30$) possuem potência estatística para inferência formal em $\alpha = 0.05$.
2. **Correlação entre Folds LOCO**:
   - Dois folds LOCO de um mesmo dataset compartilham $(K_T - 2) / (K_T - 1)$ das condições de treinamento, uma propriedade intrínseca de qualquer esquema *leave-one-group-out*.

---

## 9. Conclusões Principais (`Main Conclusions`)

- O pipeline independente `scratch_vs_tl` garante paridade estrita de amostras, pré-processamento `detrended_rms` (141 atributos), isolamento LOCO e verificação programática de congelamento no `Linear Probing` (`backbone_param_max_abs_diff == 0.0`).
- A distinção entre melhorias descritivas de média, consistência entre folds e significância estatística corrigida por Holm está documentada em `statistical_tests.csv` e `rankings.csv`.
