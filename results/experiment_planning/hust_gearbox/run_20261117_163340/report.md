# Estudo exploratório HUST_Gearbox — Macro-LOCO

- Protocolo: `speed`; folds: 20Hz, 25Hz, 30Hz, 35Hz, 40Hz, VS_0_40Hz.
- Seeds: 42, 100, 2024, 7, 999; modelos: Random Forest e SVM RBF.
- Janelas: 1780; atributos: 141; classes: Class_Broken, Class_Missing, Class_Normal.
- Métrica primária: Macro-F1. AUC reportada como OvR macro.
- Busca de hiperparâmetros: não realizada; configurações fixas registradas em `config.json`.
- IDs de gravação disjuntos entre treino e teste verificados para todos os folds.

## Resultados por fold e modelo

| Fold | Modelo | Accuracy | Balanced Accuracy | Macro-F1 | Macro ROC-AUC OvR |
|---|---|---:|---:|---:|---:|
| 20Hz | RandomForest | 0.7269 ± 0.0232 | 0.7360 ± 0.0224 | 0.7321 ± 0.0240 | 0.9115 ± 0.0141 |
| 20Hz | SVM_RBF | 0.7172 ± 0.0000 | 0.7267 ± 0.0000 | 0.7233 ± 0.0000 | 0.8606 ± 0.0015 |
| 25Hz | RandomForest | 0.8627 ± 0.0146 | 0.8627 ± 0.0146 | 0.8564 ± 0.0167 | 0.9923 ± 0.0070 |
| 25Hz | SVM_RBF | 0.9333 ± 0.0000 | 0.9333 ± 0.0000 | 0.9327 ± 0.0000 | 0.9961 ± 0.0010 |
| 30Hz | RandomForest | 0.8373 ± 0.0234 | 0.8373 ± 0.0234 | 0.8264 ± 0.0281 | 0.9324 ± 0.0127 |
| 30Hz | SVM_RBF | 0.8067 ± 0.0000 | 0.8067 ± 0.0000 | 0.7889 ± 0.0000 | 1.0000 ± 0.0000 |
| 35Hz | RandomForest | 0.9267 ± 0.0254 | 0.9267 ± 0.0254 | 0.9242 ± 0.0268 | 0.9948 ± 0.0017 |
| 35Hz | SVM_RBF | 0.9600 ± 0.0000 | 0.9600 ± 0.0000 | 0.9599 ± 0.0000 | 0.9992 ± 0.0003 |
| 40Hz | RandomForest | 0.8924 ± 0.0326 | 0.8960 ± 0.0315 | 0.8928 ± 0.0345 | 1.0000 ± 0.0000 |
| 40Hz | SVM_RBF | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 | 1.0000 ± 0.0000 |
| VS_0_40Hz | RandomForest | 0.6133 ± 0.0429 | 0.6133 ± 0.0429 | 0.5028 ± 0.0263 | 0.7269 ± 0.0114 |
| VS_0_40Hz | SVM_RBF | 0.5867 ± 0.0000 | 0.5867 ± 0.0000 | 0.4745 ± 0.0000 | 0.8352 ± 0.0008 |

## Limitações

Estudo exploratório em uma máquina e representação pré-processada disponível; não constitui experimento de TL, DA ou DG.