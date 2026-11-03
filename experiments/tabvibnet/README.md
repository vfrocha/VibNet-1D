# Experimentos `TabVibNet` (`experiments/tabvibnet/`)

Este diretório reúne os pontos de entrada modulares da família experimental **`TabVibNet`** (classificação de falhas em máquinas rotativas a partir do vetor de 141 features estatísticas, espectrais, WPD e EMD extraídas de janelas de `2.048` amostras).

Cada runner carrega sua configuração declarativa em [`configs/tabvibnet/`](../../configs/tabvibnet/), registra a proveniência completa da execução (`run_metadata.json` via [`src.vibnet.evaluation.provenance`](../../src/vibnet/evaluation/provenance.py)) e delega a execução científica aos módulos validados do repositório, preservando 100% da compatibilidade com [`scripts/exp_tl_*.py`](../../scripts/).

---

## Runners disponíveis

| Runner | Configuração YAML | Script legado correspondente | Descrição |
| :--- | :--- | :--- | :--- |
| [`run_variants_and_balance.py`](run_variants_and_balance.py) | [`configs/tabvibnet/tl_variants_and_balance.yaml`](../../configs/tabvibnet/tl_variants_and_balance.yaml) | [`scripts/exp_tl_four_variants.py`](../../scripts/exp_tl_four_variants.py), [`scripts/exp_tl_aux_features.py`](../../scripts/exp_tl_aux_features.py), [`scripts/exp_tl_balance_strategies.py`](../../scripts/exp_tl_balance_strategies.py) | Ablação das 4 variantes de pré-processamento (`raw`, `detrended`, `raw_rms`, `detrended_rms`), features auxiliares (`141` vs `144`) e estratégias de balanceamento no treino target. |
| [`run_contrastive_tl.py`](run_contrastive_tl.py) | [`configs/tabvibnet/tl_contrastive_mlp.yaml`](../../configs/tabvibnet/tl_contrastive_mlp.yaml) | [`scripts/exp_tl_contrastive_mlp.py`](../../scripts/exp_tl_contrastive_mlp.py) | Ablação fatorial (`mode`, `encoder`, `head`) de aprendizado contrastivo supervisionado (`SupConLoss`) com `MultiHeadMLP` (`CWRU_12k|MFPT -> CWRU_48k`, 3 seeds). |
| [`run_cross_dataset_tl.py`](run_cross_dataset_tl.py) | [`configs/tabvibnet/tl_cross_dataset_mlp_tabnet.yaml`](../../configs/tabvibnet/tl_cross_dataset_mlp_tabnet.yaml) | [`scripts/exp_tl_cross_dataset_mlp_tabnet.py`](../../scripts/exp_tl_cross_dataset_mlp_tabnet.py) | Benchmark completo de Transfer Learning multi-source (`15` combinações de `4` fontes $\times$ `5` targets = `66` folds LOCO $\times$ `3` seeds $\times$ `MLP` vs `TabNet`). |

---

## Como inspecionar configurações e proveniência (Dry-Run)

Qualquer runner pode ser executado com `--dry-run` (inclusive em ambientes Python padrão sem bibliotecas de aprendizado profundo instaladas) para validar a configuração YAML, verificar a combinatória de fontes/targets e gerar um manifesto `run_metadata.json` de inspeção:

```powershell
python experiments/tabvibnet/run_cross_dataset_tl.py --dry-run
python experiments/tabvibnet/run_contrastive_tl.py --dry-run
python experiments/tabvibnet/run_variants_and_balance.py --dry-run
```
