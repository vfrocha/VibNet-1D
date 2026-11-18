# Estudo exploratório HUST_Gearbox — Macro-LOCO

- Protocolo: `speed`; folds: 20Hz.
- Seeds: 42; modelos: Random Forest e SVM RBF.
- Janelas: 1780; atributos: 141; classes: Class_Broken, Class_Missing, Class_Normal.
- Métrica primária: Macro-F1. AUC reportada como OvR macro.
- Busca de hiperparâmetros: não realizada; configurações fixas registradas em `config.json`.
- IDs de gravação disjuntos entre treino e teste verificados para todos os folds.

## Resultados por fold e modelo

| Fold | Modelo | Accuracy | Balanced Accuracy | Macro-F1 | Macro ROC-AUC OvR |
|---|---|---:|---:|---:|---:|
| 20Hz | RandomForest | 0.7517 ± 0.0000 | 0.7600 ± 0.0000 | 0.7591 ± 0.0000 | 0.9270 ± 0.0000 |
| 20Hz | SVM_RBF | 0.7172 ± 0.0000 | 0.7267 ± 0.0000 | 0.7233 ± 0.0000 | 0.8611 ± 0.0000 |

## Limitações

Estudo exploratório em uma máquina e representação pré-processada disponível; não constitui experimento de TL, DA ou DG.