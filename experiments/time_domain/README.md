# Experimentos no Domínio do Tempo (`experiments/time_domain/`)

Este diretório está reservado para a próxima etapa científica do projeto **`VibNet-1D`**: classificação e transferência de aprendizado **diretamente a partir do sinal de vibração 1D no domínio do tempo**, sem extração manual das 141 features estatísticas/espectrais/WPD/EMD do `TabVibNet`.

> **Nota de Escopo Atual (`docs/reproducibility-audit`):**  
> Conforme diretriz explícita da auditoria e arquitetura modular ([`ADR-004`](../../docs/decisions/004-planned-modular-architecture-vibnet.md)), **nenhum novo modelo 1D foi implementado prematuramente nesta etapa**. Apenas os contratos de dados, alinhamento de folds LOCO e configuração declarativa foram estabelecidos e testados.

---

## Contratos Científicos de Comparação Justa (`TabVibNet` vs `VibNet-1D`)

Para que qualquer comparação futura entre um modelo temporal 1D (ex.: `1D-CNN`, `1D-ResNet`, `WDCNN`, `1D-Transformer`) e os resultados consolidados do `TabVibNet` ([`docs/results/tabvibnet_summary.md`](../../docs/results/tabvibnet_summary.md)) seja cientificamente válida, todo experimento adicionado a `experiments/time_domain/` deve obedecer aos seguintes contratos:

1. **Mesmas Janelas Pré-Processadas (`detrended_rms`, `L = 2.048`, `C = 1`):**
   - Carregar as janelas diretamente de `data/processed_tl/detrended_rms/` utilizando [`TimeDomainDatasetSpec`](../../src/vibnet/signal/contracts.py) e [`TimeDomainWindowBatch`](../../src/vibnet/signal/contracts.py).
2. **Alinhamento 1-para-1 de Folds LOCO:**
   - Utilizar [`TimeDomainLOCOAdapter`](../../src/vibnet/signal/contracts.py) e validar o particionamento com [`verify_tabular_signal_loco_alignment`](../../src/vibnet/signal/contracts.py), garantindo exatamente os mesmos `66` folds LOCO nos `5` datasets targets (`CWRU_12k` [4], `CWRU_48k` [12], `HUST_Gearbox` [30], `PU` [4], `UOEMD` [16]).
3. **Mesmas Combinações de Fontes e Vocabulários Separados (`ADR-003`):**
   - Utilizar [`build_source_combinations`](../../src/vibnet/evaluation/protocols.py) para gerar os `15` subconjuntos não-vazios de `['UORED', 'HUST', 'Mechanical_Gear', 'Electric_Motor']`, mantendo cabeças de classificação específicas por domínio (`H_s` e `H_T`).
4. **Mesmas Seeds e Proteção de `BatchNorm1d` (`ADR-002`):**
   - Executar com as seeds `[42, 123, 2026]` e aplicar [`trim_single_remainder_sample`](../../src/vibnet/evaluation/balancing.py) (ou `drop_last=True` no `DataLoader` de treino) para evitar falhas quando `N_train % batch_size == 1` (como no fold `C4` do dataset `PU`).
5. **Mesmas Métricas e Registro de Proveniência:**
   - Reportar `Macro_F1` como métrica primária via [`compute_fold_metrics`](../../src/vibnet/evaluation/metrics.py) (com flag `ROC_AUC_Valid`) e gravar `run_metadata.json` via [`collect_run_provenance`](../../src/vibnet/evaluation/provenance.py).
   - Usar o template declarativo [`configs/time_domain/template_1d_loco.yaml`](../../configs/time_domain/template_1d_loco.yaml).
