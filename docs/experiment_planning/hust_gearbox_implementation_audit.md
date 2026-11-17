# Implementação do estudo exploratório HUST_Gearbox

## Auditoria do ambiente

Foram consultados `audit_summary.md`, `hust_gearbox_preliminary.md`,
`hust_gearbox_split_analysis.csv` e `experimental_protocol_proposal.md`.

- O projeto já possui a preparação de janelas em `src/data/make_dataset.py` e
  `src/data/make_dataset_tl.py`, normalização em
  `src/preprocessing/normalize_dataset_tl.py`, extração canônica de 141 atributos
  em `src/experiments/run_baseline_loco_v2.py` e baselines/validação em
  `src/experiments/run_scratch_vs_tl_classical_baselines.py`.
- O formato preparado esperado é
  `data/processed_tl/detrended_rms/HUST_Gearbox/<condição>/<classe>/sNNNNN_wNN.npy`.
  `wNN` identifica a janela. A execução do pesquisador mostrou que `sNNNNN`
  pode se repetir em condições diferentes; por isso, o runner identifica a
  gravação pelo trio condição/classe/prefixo e também registra o prefixo de
  origem separadamente. Todas as janelas com a mesma identidade composta ficam
  juntas no fold; o manifesto preserva também os caminhos de cada janela.
- A auditoria específica documenta 30 condições, 900 janelas, 90 gravações,
  três classes (`Class_Normal`, `Class_Broken`, `Class_Missing`), 10 janelas por
  gravação e 25,6 kHz. Os nomes de condição são `Cond_<20|25|30|35|40>_<0..4>`
  e `Cond_L<0..4>_VS_0_40_0`.
- **Os dados não estão disponíveis nesta máquina.** Não há diretório `data/`,
  nem `vibdata/raw/HUST_gearbox/`. Manifestos e resultados antigos estão no
  repositório, mas não contêm as janelas necessárias para repetir o estudo.
- Há inconsistência entre a auditoria específica (3 classes/900 janelas) e
  `src/vibnet/data/registry.py` (vocabulário/contagem diferentes). A implementação
  obedece à auditoria HUST Gearbox solicitada e falha explicitamente se os dados
  carregados não apresentarem exatamente as três classes canônicas auditadas e
  as 30 condições esperadas. O alias saudável descrito abaixo é a única
  harmonização permitida.
- O dataset pode conter `Class_Healthy` e `Class_Normal` em pastas distintas. O
  repositório já documenta essa equivalência em `scripts/rename_normal.sh` e em
  `normalize_dataset_tl.py`. O runner mapeia `Class_Healthy` para `Class_Normal`
  somente durante a leitura para representar o mesmo rótulo saudável; não muda
  os diretórios. Os caminhos originais ficam registrados nos IDs de amostra.

Para executar, é necessário disponibilizar as janelas processadas no caminho
acima, preservando os nomes `sNNNNN_wNN.npy` e as pastas por condição e classe.
Caso só os sinais brutos estejam disponíveis, obtenha o dataset HUST_Gearbox
pelos meios aprovados pelo pesquisador. Se optar explicitamente pelo mecanismo
de aquisição do projeto, a sequência documentada é:

```bash
python scripts/setup/download_datasets.py --datasets HUST_Gearbox
python scripts/setup/prepare_datasets.py --datasets HUST_Gearbox \
  --variants processed_1d detrended detrended_rms
```

Esses comandos são instruções para uma execução futura, não foram executados
nesta implementação. A preparação existente pode ser consultada em
`scripts/setup/README.md`, `src/data/make_dataset.py` e
`src/preprocessing/normalize_dataset_tl.py`. A validação pode ser feita com
`python scripts/validate_dataset_tl.py --datasets HUST_Gearbox --variants detrended_rms`.

## Implementação e protocolo

Novo runner: `src/experiments/run_hust_gearbox_macro_loco.py`.

- `speed`: seis folds `20Hz`, `25Hz`, `30Hz`, `35Hz`, `40Hz`, `VS_0_40Hz`.
- `load`: cinco folds `Load_0` a `Load_4`.
- Cada fold testa todas as condições pertencentes ao grupo e verifica IDs de
  gravação disjuntos entre treino e teste.
- Random Forest: 100 árvores, `max_features=sqrt`, pesos balanceados, um worker.
- SVM: RBF, `C=1`, pesos balanceados, probabilidades; `StandardScaler` ajustado
  dentro do pipeline de treino do fold.
- Não há busca de hiperparâmetros. Cada modelo é ajustado no conjunto completo
  permitido de treino; seeds explícitas controlam RF e SVM.
- Métricas: Accuracy, Balanced Accuracy, Macro-F1, ROC-AUC OvR macro e suporte
  por classe.
- Cada execução cria uma pasta `run_<timestamp>` exclusiva em
  `results/experiment_planning/hust_gearbox/`, com `results.csv`,
  `fold_manifest.csv`, `config.json`, `run.log` e `report.md`.

## Verificações e execução

O teste mínimo de dados reais não pôde ser executado porque as janelas estão
ausentes. Após disponibilizá-las, execute:

```powershell
# Um fold e uma seed (modo reduzido)
python -m src.experiments.run_hust_gearbox_macro_loco --protocol speed --fold 20Hz --seed 42

# Macro-LOCO completo por velocidade (seeds padrão: 42, 100, 2024, 7, 999)
python -m src.experiments.run_hust_gearbox_macro_loco --protocol speed

# Macro-LOCO completo por carga
python -m src.experiments.run_hust_gearbox_macro_loco --protocol load

# Consultar a execução mais recente
$run = Get-ChildItem results/experiment_planning/hust_gearbox -Directory |
  Sort-Object Name -Descending | Select-Object -First 1
Import-Csv (Join-Path $run.FullName 'results.csv') | Format-Table
Get-Content (Join-Path $run.FullName 'report.md')
```

Os comandos do experimento completo executam os protocolos separadamente e
criam diretórios de saída distintos, sem sobrescrever execuções anteriores.
