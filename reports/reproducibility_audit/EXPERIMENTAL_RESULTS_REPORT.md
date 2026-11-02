# VibNet-1D — Auditoria de resultados

- Diretório: `/home/vfrocha/VibNet-1D/results`
- Arquivos inventariados: 173
- CSVs: 108
- Linhas de métricas extraídas: 79454
- Erros de leitura: 0

## Arquivos gerados
- `results_inventory.csv`: inventário e status de leitura.
- `metrics_long.csv`: métricas reconhecidas com caminho de origem.
- `metrics_summary.csv`: estatísticas descritivas por arquivo/dataset/modelo/estratégia.

## Limitações e cautelas
- Não compare resultados de arquivos diferentes sem verificar folds, seeds, dataset, pré-processamento, balanceamento e protocolo.
- Linhas por fold e linhas agregadas podem coexistir; verifique a granularidade antes de resumir.
- Métricas/colunas com nomes desconhecidos podem não ser extraídas automaticamente.
- Resultados originais não são alterados; relatórios são derivados.

## Erros de leitura
Nenhum.
