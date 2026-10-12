# Experimento de balanceamento — VibNet-1D

## Arquivos

- `exp_tl_balance_strategies.py`: runner LOCO baseado no experimento existente, configurado por padrão para `detrended_rms` e para o target `CWRU_48k`.
- `build_tabnet_resnet_balance.py`: versão do módulo de treinamento com balanceamento opcional no treino supervisionado do target.

## Estratégias

- `none`: sem balanceamento (baseline).
- `undersample`: reduz cada classe do treino target à contagem da classe menos frequente.
- `smote`: aumenta as classes minoritárias até a contagem da classe mais frequente.
- `hybrid`: define como alvo comum a mediana das contagens originais; aplica SMOTE nas classes abaixo do alvo e undersampling nas classes acima.

O balanceamento ocorre **depois do `StandardScaler.fit_transform()` ajustado exclusivamente nos dados originais de treino do target**. As sources (`CWRU_12k` e `MFPT`) e o conjunto de teste LOCO não são balanceados nem usados para ajustar o scaler do target. A avaliação e a seleção de folds seguem a regra de validade do script original.

Se uma classe tiver apenas uma amostra no treino de um fold, SMOTE/híbrido falharão explicitamente nesse fold/estratégia, pois não é possível gerar vizinhos sintéticos com uma única amostra. O erro é registrado no CSV; o fold não deve ser silenciosamente removido.

## Instalação no repositório

Copie os arquivos para:

- `scripts/tl/exp_tl_balance_strategies.py`
- `src/models/build_tabnet_resnet_balance.py`

O runner importa o novo módulo de treinamento, preservando `src/models/build_tabnet_resnet.py` original sem alterações.

## Execução

Na raiz do repositório, execute cada estratégia com a mesma seed. Por padrão, o experimento usa `detrended_rms`, MLP e TabNet:

```bash
python scripts/tl/exp_tl_balance_strategies.py --balance-strategy none --seed 42
python scripts/tl/exp_tl_balance_strategies.py --balance-strategy undersample --seed 42
python scripts/tl/exp_tl_balance_strategies.py --balance-strategy smote --seed 42
python scripts/tl/exp_tl_balance_strategies.py --balance-strategy hybrid --seed 42
```

Para um teste rápido antes da execução completa:

```bash
python scripts/tl/exp_tl_balance_strategies.py --balance-strategy hybrid --models mlp --seed 42
```

Para repetir com mais seeds, use o mesmo conjunto de seeds em todas as estratégias (por exemplo, 42, 123 e 2026), sem alterar outras configurações.

## Saídas

Cada execução cria uma pasta em `results/tl_balance_strategies/run_<timestamp>/`, contendo `results.csv`, `config.json`, `experiment.log` e os resumos gerados pelo runner. O CSV inclui a coluna `Balance Strategy` para permitir a consolidação posterior.

## Verificações antes de executar

1. Confirme que o ambiente possui `imbalanced-learn`, `scikit-learn`, `torch` e `pytorch-tabnet` instalados.
2. Rode primeiro o teste rápido com MLP.
3. Confira no log as linhas `[BALANCE]`, com contagens antes/depois, e valide que `none`, `undersample`, `smote` e `hybrid` produziram as contagens esperadas.
4. Confirme que as fontes e o teste permanecem inalterados. A coluna `N Train Target` representa o tamanho antes do balanceamento; as contagens após balanceamento aparecem no log.
