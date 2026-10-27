# Experimentos MLP e supervised contrastive loss

## Protocolo e arquitetura

O runner `scripts/exp_tl_contrastive_mlp.py` reutiliza o carregamento e a extração de features de `scripts/exp_tl_balance_strategies.py`. Usa `CWRU_12k` e `MFPT` como fontes, `CWRU_48k` como destino, a variante já processada `detrended_rms` e avaliação LOCO por condição. `CWRU_48k` não contém amostras normais; o experimento é diagnóstico apenas entre as classes de falha presentes.

O módulo `src/models/build_mlp_contrastive.py` oferece encoder baseline (`256 -> 128 -> 64`) e expandido (`512 -> 256 -> 128 -> 64`), com cabeças pós-latente assimétrica (`128 -> 64 -> classes`), simétrica (larguras do encoder em ordem inversa) ou linear. São cabeças classificadoras, sem decoder. Cada domínio possui vocabulário e cabeça próprios. SupCon compara exemplos apenas dentro do minibatch de um domínio.

Modos: `baseline` supervisionado sem SupCon; `joint` combina cross-entropy e SupCon ponderada; `pretrain` faz SupCon supervisionada nas fontes e depois treino supervisionado multi-domínio e fine-tuning no target. Seeds, temperatura, peso contrastivo, épocas, encoder, cabeça e balanceamento são configuráveis.

## Execução

Execute da raiz `VibNet-1D/`. Comece com uma configuração e uma seed; o runner exige os dados locais e extrai features reais.
O padrão esperado é `data/processed_tl/detrended_rms/<dataset>/<condição>/<classe>/*.npy`; a raiz do projeto é localizada a partir do script, inclusive quando o runner compartilhado está em `scripts/` ou `scripts/tl/`.

Testes sintéticos do módulo (não carregam os datasets):

```bash
python -m unittest discover -s tests -v
```

Uma execução curta de smoke test no pipeline ainda percorre as condições LOCO dos datasets disponíveis; reduza épocas e use uma configuração só:

```bash
python scripts/exp_tl_contrastive_mlp.py --modes baseline --epochs 1 --finetune-epochs 0 --batch-size 32
```

Comparação A, mantendo encoder e cabeça fixos:

```bash
python scripts/exp_tl_contrastive_mlp.py --modes baseline joint pretrain --encoder-profiles baseline --head-shapes asymmetric --seeds 42
```

Comparação B, exemplo com `joint` escolhido em validação independente:

```bash
python scripts/exp_tl_contrastive_mlp.py --modes joint --encoder-profiles baseline expanded --head-shapes asymmetric --seeds 42
```

Comparação C, exemplo com `joint` e encoder `expanded` escolhidos em validação independente:

```bash
python scripts/exp_tl_contrastive_mlp.py --modes joint --encoder-profiles expanded --head-shapes asymmetric symmetric --seeds 42
```

Parâmetros SupCon e balanceamento são configuráveis:

```bash
python scripts/exp_tl_contrastive_mlp.py --modes joint --temperature 0.07 --contrastive-weight 0.1 --balance-strategy hybrid --seeds 42 123 2026
```

Cada execução cria `results/tl_contrastive_mlp/run_<timestamp>/` com `config.json`, `experiment.log`, `results.csv` (estados `OK`, `SKIPPED` e `ERROR`) e `summary.csv`. Registra ACC, balanced accuracy, Macro-F1 e ROC-AUC OvR macro; AUC fica ausente quando indefinida. O custo cresce aproximadamente com combinações × seeds × condições LOCO × épocas, mais fine-tuning; faça primeiro uma execução pequena.

## Interpretação científica

O scaler do target é ajustado somente nas condições de treino; a condição LOCO não participa do treino, balanceamento ou seleção de checkpoint. Não use métricas dos folds de teste para escolher hiperparâmetros/estratégias e depois reportar os mesmos folds como confirmação imparcial: escolha em validação independente/nested e confirme com seeds previamente definidas. Folds onde alguma classe de teste não aparece no treino do target ficam registrados como `SKIPPED`.

SMOTE aumenta classes até a contagem máxima; híbrido usa o alvo mediano, com SMOTE nas menores e undersampling nas maiores. As contagens diferem em distribuições assimétricas, mas podem coincidir quando os alvos calculados coincidem; consulte as contagens antes/depois por fold. SMOTE requer pelo menos duas amostras por classe. Falhas ficam registradas e não devem ser silenciosamente removidas.

Nos resultados anteriores em `results/tl_balance_strategies/run_20261012_150540/` (SMOTE) e `run_20261012_150855/` (híbrido), as métricas coincidem em 12/12 linhas por modelo. Os logs mostram, por exemplo, contagens `[184, 194, 194]` e `[184, 200, 200]`: em ambos os casos a mediana é igual à maior contagem porque duas classes empatam no máximo. Assim, os dois métodos pediram o mesmo aumento sem undersampling; isso explica a igualdade observada nessa execução, mas não torna as implementações equivalentes em distribuições mais assimétricas.
