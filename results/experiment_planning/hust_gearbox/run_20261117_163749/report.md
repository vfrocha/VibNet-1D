# Estudo exploratório HUST_Gearbox — Macro-LOCO

- Protocolo: `load`; folds: Load_0, Load_1, Load_2, Load_3, Load_4.
- Seeds: 42, 100, 2024, 7, 999; modelos: Random Forest e SVM RBF.
- Janelas: 1780; atributos: 141; classes: Class_Broken, Class_Missing, Class_Normal.
- Métrica primária: Macro-F1. AUC reportada como OvR macro.
- Busca de hiperparâmetros: não realizada; configurações fixas registradas em `config.json`.
- IDs de gravação disjuntos entre treino e teste verificados para todos os folds.

## Resultados por fold e modelo

| Fold | Modelo | Accuracy | Balanced Accuracy | Macro-F1 | Macro ROC-AUC OvR |
|---|---|---:|---:|---:|---:|
| Load_0 | RandomForest | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Load_0 | SVM_RBF | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Load_1 | RandomForest | 0.9789 ± 0.0091 | 0.9789 ± 0.0091 | 0.9789 ± 0.0092 | 1.0000 ± 0.0000 |
| Load_1 | SVM_RBF | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Load_2 | RandomForest | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Load_2 | SVM_RBF | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| Load_3 | RandomForest | 0.9878 ± 0.0164 | 0.9878 ± 0.0164 | 0.9877 ± 0.0164 | 1.0000 ± 0.0000 |
| Load_3 | SVM_RBF | 0.9722 ± 0.0000 | 0.9722 ± 0.0000 | 0.9722 ± 0.0000 | 0.9995 ± 0.0001 |
| Load_4 | RandomForest | 0.8674 ± 0.0063 | 0.8711 ± 0.0061 | 0.8678 ± 0.0065 | 0.9784 ± 0.0055 |
| Load_4 | SVM_RBF | 0.7600 ± 0.0000 | 0.7667 ± 0.0000 | 0.7583 ± 0.0000 | 0.9521 ± 0.0009 |

## Limitações

Estudo exploratório em uma máquina e representação pré-processada disponível; não constitui experimento de TL, DA ou DG.