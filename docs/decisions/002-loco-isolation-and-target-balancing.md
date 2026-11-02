# ADR 002: Isolamento Estrito sob Protocolo LOCO, Balanceamento no Alvo e Casos de Borda Numéricos

- **Status**: Aceito e Auditado
- **Contexto**: Avaliação de generalização entre condições operacionais (`LOCO`) nos experimentos TabVibNet
- **Artefatos Relacionados**:
  - `src/models/build_tabnet_resnet_balance.py`
  - `src/models/build_mlp_contrastive.py`
  - `scripts/exp_tl_balance_strategies.py`
  - `scripts/exp_tl_cross_dataset_mlp_tabnet.py`
  - `results/tl_balance_strategies/`
  - `results/tl_cross_dataset_mlp_tabnet/`

## 1. Problema e Riscos de Vazamento Metodológico

Em sinais de vibração de máquinas rotativas, janelas extraídas de um mesmo arquivo ou de uma mesma condição operacional (mesma rotação, torque e força radial) possuem forte correlação espectral de fundo. Particionamentos aleatórios (*k-fold* estratificado convencional) vazam essa assinatura operacional entre treino e teste, inflando artificialmente as métricas para $\sim 0{,}99$–$1{,}00$. Além disso, a auditoria identificou quatro riscos técnicos em avaliações LOCO:
1. **Vazamento por *Early Stopping* no Teste**: Em `src/models/build_tabular.py` (linha 54), o `TabNetClassifier.fit()` recebia `eval_set=[(X_test_scaled, y_test)]` com `patience=20`, selecionando a época de parada com base no fold de teste.
2. **Escalonamento e Balanceamento Antes do Split**: Ajustar `StandardScaler` ou `SMOTE` antes de separar a condição de teste contamina as estatísticas de média/desvio e interpola vizinhos sintéticos com o teste.
3. **Empate de Mediana na Estratégia `hybrid` em Datasets de 3 Classes (`CWRU_48k`)**: Comportamento idêntico entre `smote` e `hybrid` em `CWRU_48k`.
4. **Falha de `BatchNorm1d` com Último Mini-Batch de Tamanho 1 (`N_train % batch_size == 1`) e Classe Ausente no Teste (`Load_0HP_Sev_0.021`)**.

## 2. Decisões Metodológicas Adotadas

1. **Isolamento Completo do Fold de Teste LOCO**:
   - Em cada fold $k$, a condição operacional de teste $c_k$ é separada antes de qualquer estatística.
   - O `StandardScaler` do alvo é ajustado (`fit`) exclusivamente nas $K-1$ condições de treino (`X_train_target`), aplicando apenas `transform` em `X_test_target`.
   - O balanceamento (`SMOTE` com `k_neighbors = min(5, min_class_count - 1)`) é aplicado **somente sobre `(X_train_scaled, y_train)`** do domínio alvo.
   - O *early stopping* supervisionado pelo conjunto de teste foi **removido** dos pipelines de transferência (`build_tabnet_resnet_balance.py`, `build_mlp_contrastive.py` e `exp_tl_cross_dataset_mlp_tabnet.py`).
2. **Adoção de `SMOTE` como Estratégia Padrão de Balanceamento no Treino do Alvo**:
   - No experimento `tl_balance_strategies` em `CWRU_48k` (`detrended_rms`), `smote` ($0{,}9389 \pm 0{,}1289$ no `MLP`) superou `none` ($0{,}9202$) e `class_weights` ($0{,}8525$).
   - Documentou-se que `hybrid` e `smote` produzem resultados idênticos em `CWRU_48k` porque, em todos os 12 folds de 3 classes, as duas maiores classes possuem contagens iguais (ex.: `[184, 194, 194]`), fazendo com que $\text{mediana}([184, 194, 194]) = 194 = \max$, anulando a etapa de subamostragem.
3. **Tratamento Transparente de `ROC-AUC` em Folds com Classe Ausente no Teste**:
   - No fold `Load_0HP_Sev_0.021` de `CWRU_48k` (`N_Test_Classes_Present = 2`, sem amostras de `Class_3`), `exp_tl_cross_dataset_mlp_tabnet.py` registra explicitamente `ROC-AUC = NaN`, `ROC_AUC_Valid = False` e `ROC_AUC_Note = "missing_test_classes:[2]"`, evitando distorcer a média de `ROC-AUC` com zeros artificiais (`0.0`).
   - O cálculo de `Macro-F1` é restrito às classes efetivamente presentes em `y_test` (`labels=present_classes`).

## 3. Recomendação Técnica para Correção do Caso de Borda de `BatchNorm1d` (`PU` Fold `C4`)

Nos runs `tl_cross_dataset_mlp_tabnet/run_20261028_*`, o `TabNetClassifier` falhou no fold `C4_1500rpm_0.7Nm_400N` do dataset `PU` porque `N_Train_After = 8577` deixa resto $8577 \bmod 64 = 1$, gerando um último mini-batch de 1 amostra na camada `BatchNorm1d`. Na futura biblioteca modular (`src/vibnet/evaluation/` e `src/vibnet/tabular/`), todo loop de treinamento com `BatchNorm1d` deve:
- Utilizar `drop_last=True` no `DataLoader` de treinamento quando `len(dataset) > batch_size`, **ou**
- Ajustar dinamicamente o `batch_size` (ou descartar 1 amostra sintética excedente do SMOTE quando `N_train % batch_size == 1`) antes de invocar `TabNetClassifier.fit()`.
