# Auditoria Geral de Datasets — VibNet-1D (TL, DA e DG)

## 1. Objetivo e Escopo da Auditoria

Este documento consolida a auditoria técnica dos datasets do projeto **VibNet-1D** visando a definição de um protocolo experimental rigoroso e livre de vazamento de dados para **Transfer Learning (TL)**, **Domain Adaptation (DA)** e **Domain Generalization (DG)** com **TabNet**, **Random Forest (RF)** e **SVM** sobre o espaço de **141 características** (`TabVibNet` / `extract_fusion_features`).

A auditoria contrastou quatro fontes de informação presentes no repositório e no ambiente de execução:
1. **Metadados e adaptadores brutos da biblioteca `vibdata`** (`labels.csv` global e `<DATASET>.csv` de cada base);
2. **Scripts reais de preparação e normalização** ([`src/data/make_dataset.py`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/src/data/make_dataset.py), [`src/data/make_dataset_tl.py`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/src/data/make_dataset_tl.py), [`src/preprocessing/normalize_dataset_tl.py`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/src/preprocessing/normalize_dataset_tl.py));
3. **Manifestos e inventários reais de janelas processadas em disco** ([`results/scratch_vs_tl/run_20261105_141743/fold_manifest.csv`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/results/scratch_vs_tl/run_20261105_141743/fold_manifest.csv) e [`docs/distribuicao_classes_datasets.csv`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/docs/distribuicao_classes_datasets.csv));
4. **Arquivos de configuração e registro** ([`configs/datasets.yaml`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/configs/datasets.yaml) e [`src/vibnet/data/registry.py`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/src/vibnet/data/registry.py)).

Os dados tabulares consolidados desta auditoria encontram-se em [`docs/experiment_planning/dataset_inventory_summary.csv`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/docs/experiment_planning/dataset_inventory_summary.csv).

---

## 2. Classificação de Escopo dos Datasets

Conforme as decisões metodológicas estabelecidas:

| Categoria no Protocolo | Datasets | Papel Experimental |
| :--- | :--- | :--- |
| **Datasets de Origem (Source)** | `UORED`, `HUST` (HUST Bearing), `MFPT` *(candidato auditado)* | Pré-treinamento não supervisionado do `TabNetPretrainer` (TL, DA, DG) e treinamento supervisionado de origem (DA e DG). |
| **Datasets-Alvo para TL** | `CWRU_12k`, `CWRU_48k`, `PU`, `UOEMD`, `HUST_Gearbox` | Avaliação supervisionada no domínio-alvo com cabeça específica do dataset-alvo (LOCO / extrapolação de regime). |
| **Datasets-Alvo para DA e DG** | `CWRU_12k`, `CWRU_48k`, `PU` | Avaliação entre domínios de rolamentos usando a ontologia global de 5 classes (`Normal`, `Ball`, `Inner`, `Outer`, `Inner & Outer`). |
| **Fora do Escopo Inicial** | `UOC` | Excluído desta etapa experimental. |
| **Candidatos Futuros (Extensões)** | `IMS`, `Mechanical_Gear`, `Electric_Motor` | Documentados no inventário, sem processamento ou inclusão na rodada experimental imediata. |

---

## 3. Quadro Resumo Verificado dos Datasets Principais

A tabela abaixo apresenta os parâmetros reais verificados para cada dataset (janelas sem sobreposição de $1{,}0\text{ s}$, onde $L = f_s$):

| Dataset | Papel | Componente | $f_s$ (Hz) | $L$ (pts / 1s) | Overlap | Gravações Brutas (`vibdata`) | Gravações Processadas | Janelas em Disco (`detrended_rms`) | Nº Condições | Status da Auditoria |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **`UORED`** | Origem | Rolamento | 42.000 | 42.000 | 0% | 60 | 37 | **370** *(600 brutas)* | 18 | **Alerta: extração parcial em disco** (faltam `Inner` e `Outer` no `.npy` atual) |
| **`HUST`** | Origem | Rolamento | 51.200 | 51.200 | 0% | 99 | 99 | **951** | 3 | **OK — Completo** (cobre todas as 5 classes globais + 2 compostas com esfera) |
| **`MFPT`** | Origem (Cand.) | Rolamento | 97.656 / 48.828 | 48.828 | 0% | 20 | 20 | **78** *(harmonizado)* / **126** *(histórico)* | 8 | **Apto com reamostragem** ($97.656 \rightarrow 48.828\text{ Hz}$ em 6 arquivos) |
| **`CWRU_12k`** | Alvo (TL/DA/DG) | Rolamento | 12.000 | 12.000 | 0% | 252 | 252 | **2.520** | 4 | **OK — Completo** (`Normal`, `Inner`, `Outer`, `Ball`) |
| **`CWRU_48k`** | Alvo (TL/DA/DG) | Rolamento | 48.000 | 48.000 | 0% | 66 | 66 | **602** | 12 | **Alerta: sem classe `Normal` nativa** e célula `Load_0HP_Sev_0.014/Class_1` vazia |
| **`PU`** | Alvo (TL/DA/DG) | Rolamento | 64.000 | 64.000 | 0% | 2.559 | 1.621 | **6.476** *(10.236 brutas)* | 4 | **Alerta crítico: classe `Inner` (`Class_28`, 880 arq.) ausente em disco** |
| **`UOEMD`** | Alvo (TL) | Motor + Rol. | 42.000 | 42.000 | 0% | 156 | 156 | **1.560** | 6 | **OK — Completo** (8 classes perfeitamente balanceadas, 195 janelas/classe) |
| **`HUST_Gearbox`**| Alvo (TL) | Engrenagem | 25.600 | 25.600 | 0% | 90 | 90 | **900** | 30 | **OK — Completo** (3 classes $\times$ 30 condições = 10 janelas/classe/condição) |
| **`UOC`** | Excluído | Rolamento | 200.000 | 200.000 | 0% | 36 | — | — | — | Fora do escopo inicial |
| **`IMS`** | Futuro | Rolamento | 20.000 | 20.480 | 0% | 4.848 | — | — | 3 | Candidato futuro (*run-to-failure*) |
| **`Mechanical_Gear`**| Futuro | Engrenagem | 25.600 | 25.600 | 0% | — | — | *(225 em doc)* | 3 | Candidato futuro |
| **`Electric_Motor`** | Futuro | Motor | 42.000 | 42.000 | 0% | — | — | *(510 em doc)* | 3 | Candidato futuro |

---

## 4. Detalhamento por Dataset Auditado

### 4.1. `UORED` (University of Ottawa Rolling-Element Dataset — Constant Speed)
- **Taxa de amostragem e janelamento:** $f_s = 42.000\text{ Hz}$, gravações de $10\text{ s}$ ($420.000$ amostras), fatiadas via `SimpleSplit(42000)` sem sobreposição em $10$ janelas de $1{,}0\text{ s}$ por arquivo.
- **Inventário bruto (`vibdata/raw/UORED/UORED.csv`):** 60 gravações `.mat` distribuídas em 5 rótulos:
  - `label=47` (`Normal`): 20 arquivos (`H_1_1..H_10_2`) $\rightarrow$ 200 janelas esperadas;
  - `label=48` (`Inner Race`): 10 arquivos (`I_1_1..I_5_2`) $\rightarrow$ 100 janelas esperadas;
  - `label=49` (`Outer Race`): 10 arquivos (`O_6_1..O_10_2`) $\rightarrow$ 100 janelas esperadas;
  - `label=50` (`Ball`): 10 arquivos (`B_11_1..B_15_2`) $\rightarrow$ 100 janelas esperadas;
  - `label=51` (`Cage`): 10 arquivos (`C_16_1..C_20_2`) $\rightarrow$ 100 janelas esperadas.
- **Inventário efetivo em disco (`data/processed_tl/detrended_rms/UORED`):** **370 janelas** de 37 gravações, distribuídas em 18 pastas de condição (`Bearing_1`, `Bearing_11` a `Bearing_20`) e **apenas 3 classes**:
  - `Class_Normal`: 180 janelas (18 gravações);
  - `Class_Ball`: 100 janelas (10 gravações);
  - `Class_Cage`: 90 janelas (9 gravações).
- **Causa da discrepância identificada no código:** No adaptador `UORED_raw.__getitem__` da biblioteca `vibdata`, a chave interna do dicionário MATLAB é buscada assumindo `tag_name = os.path.basename(f_name).replace(".mat", "")`. Em 23 dos 60 arquivos `.mat` (incluindo todos os 10 arquivos `I_*` de `Inner` e todos os 10 arquivos `O_*` de `Outer`), a chave interna do `.mat` difere ligeiramente do nome do arquivo, lançando `KeyError`. Como [`src/data/make_dataset.py`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/src/data/make_dataset.py) envolve a iteração em `try ... except Exception: continue`, esses 23 arquivos foram ignorados silenciosamente.
- **Impacto metodológico:** No estado atual em disco, o `UORED` fornece amostras para pré-treinamento não supervisionado (370 janelas) e para as classes `Normal` e `Ball` em DA/DG, mas **não possui `Inner` nem `Outer` processados**. Caso deseje usar `UORED` isoladamente como domínio de origem supervisionado, é necessário corrigir a leitura da chave `.mat` no carregamento do `UORED`. Combinado com `HUST` e `MFPT`, todas as 5 classes globais já ficam cobertas.

### 4.2. `HUST` (HUST Bearing Dataset)
- **Taxa de amostragem e janelamento:** $f_s = 51.200\text{ Hz}$, janelas de $1{,}0\text{ s}$ ($51.200$ amostras), sem sobreposição.
- **Inventário bruto e em disco (`99` gravações $\rightarrow$ `951` janelas):** 100% das gravações de `HUST.csv` foram processadas com sucesso em 3 condições operacionais de carga elétrica (`Load_0W`: 320 janelas, `Load_200W`: 313 janelas, `Load_400W`: 318 janelas) e **7 classes**:
  - `Class_Normal` (`label=40`, `Normal`): **150 janelas** (50 por carga, 15 gravações);
  - `Class_41` (`label=41`, `Inner Race`): **150 janelas** (50 por carga, 15 gravações);
  - `Class_42` (`label=42`, `Outer Race`): **150 janelas** (50 por carga, 15 gravações);
  - `Class_43` (`label=43`, `Ball`): **100 janelas** (30 em `0W`, 40 em `200W`, 30 em `400W`, 12 gravações);
  - `Class_45` (`label=45`, `Inner Race and Outer Race`): **150 janelas** (50 por carga, 15 gravações) $\rightarrow$ mapeia diretamente para a classe composta independente **`Inner & Outer`**;
  - `Class_44` (`label=44`, `Inner Race and Ball`): **120 janelas** (40 por carga, 12 gravações) $\rightarrow$ fora da ontologia de 5 classes;
  - `Class_46` (`label=46`, `Outer Race and Ball`): **131 janelas** (50 em `0W`, 33 em `200W`, 48 em `400W`, 15 gravações) $\rightarrow$ fora da ontologia de 5 classes.
- **Papel chave no protocolo:** `HUST` é o **único dataset de origem que contém todas as 5 classes conceituais globais** (`Normal`, `Ball`, `Inner`, `Outer`, `Inner & Outer`), somando **700 janelas** nas 5 classes globais e **951 janelas** totais para o pré-treinamento não supervisionado do TabNet.

### 4.3. `MFPT` (Machinery Failure Prevention Technology)
- Auditado em detalhe no documento dedicado [`docs/experiment_planning/mfpt_audit.md`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/docs/experiment_planning/mfpt_audit.md). Possui **20 arquivos `.mat`** em 3 classes (`Normal`, `Inner`, `Outer`), exigindo decimação por fator 2 (`97.656 Hz` $\rightarrow$ `48.828 Hz`) nos 6 arquivos gravados a $97.656\text{ Hz}$ (`baseline_1..3.mat` e `OuterRaceFault_1..3.mat`).

### 4.4. `CWRU_12k` (Case Western Reserve University — 12 kHz)
- **Taxa de amostragem e janelamento:** $f_s = 12.000\text{ Hz}$, janelas de $1{,}0\text{ s}$ ($12.000$ amostras), sem sobreposição.
- **Inventário em disco:** **2.520 janelas** provenientes de 252 gravações em 4 condições de carga (`Load_0HP`: 600, `Load_1HP`: 640, `Load_2HP`: 640, `Load_3HP`: 640) e **4 classes**:
  - `Class_0` (`Normal`, `id=0` em `vibdata/labels.csv`): **280 janelas** (40 em `Load_0HP`, 80 em `Load_1..3HP`);
  - `Class_1` (`Inner Race`, `id=1`): **760 janelas** (190 por condição);
  - `Class_2` (`Outer Race`, `id=2`): **720 janelas** (180 por condição);
  - `Class_3` (`Ball`, `id=3`): **760 janelas** (190 por condição).
- **Verificação crítica de mapeamento:** A inspeção direta de `vibdata/raw/labels.csv` e `CWRU.csv` comprovou que `Class_2` é **`Outer Race`** (96 registros brutos) e `Class_3` é **`Ball`** (100 registros brutos). O comentário existente em [`scripts/exp_tl_cross_dataset_mlp_tabnet.py`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/scripts/exp_tl_cross_dataset_mlp_tabnet.py) que invertia `Class_2` e `Class_3` está incorreto e **não deve ser utilizado**.

### 4.5. `CWRU_48k` (Case Western Reserve University — 48 kHz)
- **Taxa de amostragem e janelamento:** $f_s = 48.000\text{ Hz}$, janelas de $1{,}0\text{ s}$ ($48.000$ amostras), sem sobreposição.
- **Inventário em disco:** **602 janelas** de 66 gravações distribuídas em **12 condições compostas** (`Load_{0..3}HP_Sev_{0.007, 0.014, 0.021}`) e **3 classes de falha**:
  - `Class_1` (`Inner`): **194 janelas**;
  - `Class_2` (`Outer`): **204 janelas**;
  - `Class_3` (`Ball`): **204 janelas**.
- **Limitações estruturais verificadas:**
  1. **Ausência de classe `Normal` nativa em 48 kHz:** Não há pasta `Class_0` em `CWRU_48k` (apenas falhas de rolamento). Por isso, `CWRU_48k` é avaliado exclusivamente em **diagnóstico multiclasse** e utiliza o $\text{RMS}_{\text{ref}}$ das janelas normais de `CWRU_12k` no protocolo TL.
  2. **Célula vazia em `Load_0HP_Sev_0.014`:** Nessa condição específica, `Class_1` (`Inner`) possui **0 janelas** (apenas `Class_2`: 10 e `Class_3`: 10). Em uma avaliação LOCO de 12 folds, o fold `Load_0HP_Sev_0.014` não possui amostras de teste da classe `Inner`. Alternativamente, agrupar os folds de `CWRU_48k` pelas **4 cargas de motor (`Load_0HP`..`Load_3HP`)** garante que todas as 3 classes (`Inner`, `Outer`, `Ball`) estejam presentes em todos os folds de teste!

### 4.6. `PU` (Paderborn University Bearing Dataset)
- **Taxa de amostragem e janelamento:** $f_s = 64.000\text{ Hz}$, gravações de $4{,}0\text{ s}$ ($256.000$ amostras), fatiadas em 4 janelas de $1{,}0\text{ s}$ ($64.000$ amostras) sem sobreposição.
- **Inventário bruto (`vibdata/raw/PU/PU.csv`):** 2.559 arquivos `.mat` nas 4 condições operacionais (`C1_1500rpm_0.7Nm_1000N`, `C2_900rpm_0.7Nm_1000N`, `C3_1500rpm_0.1Nm_1000N`, `C4_1500rpm_0.7Nm_400N`) e 4 rótulos:
  - `label=26` (`Normal`, rolamentos `K001..K006`): 480 arquivos;
  - `label=27` (`Outer Ring Fault`, rolamentos `KA01..KA30`): 959 arquivos;
  - `label=28` (`Inner Ring Fault`, rolamentos `KI01..KI21`): 880 arquivos;
  - `label=29` (`Outer And Inner Ring Fault`, rolamentos `KB23, KB24, KB27`): 240 arquivos.
- **Inventário efetivo em disco (`data/processed_tl/detrended_rms/PU`):** **6.476 janelas** (de 1.621 gravações) em **3 classes**:
  - `Class_Normal` (`Normal`): **1.918 janelas** (~480 por condição);
  - `Class_27` (`Outer`): **3.811 janelas** (~952 por condição);
  - `Class_29` (`Inner & Outer`): **747 janelas** (~187 por condição);
  - **`Class_28` (`Inner`): 0 janelas em disco!** Todos os 880 arquivos `KI*` falharam silenciosamente durante a execução original de `make_dataset.py` e não foram gravados em `data/processed/PU`.
- **Implicação para TL, DA e DG:** Nos dados atualmente em disco, `PU` testa `Normal`, `Outer` e a classe composta `Inner & Outer`. Se desejar avaliar também `Inner` (`Class_28`) no `PU`, será necessário reprocessar os 880 arquivos `KI*` brutos antes da rodada definitiva.

### 4.7. `UOEMD` (University of Ottawa Electric Motor Dataset — Variable Speed)
- **Taxa de amostragem e janelamento:** $f_s = 42.000\text{ Hz}$, gravações de $10\text{ s}$ fatiadas em 10 janelas de $1{,}0\text{ s}$ ($42.000$ amostras), sem sobreposição.
- **Inventário em disco:** **1.560 janelas** de 156 gravações, perfeitamente distribuídas em **6 condições operacionais** (`Load_0_Const_Speed`, `Load_0_Linear_Speed`, `Load_0_Step_Speed`, `Load_1_Const_Speed`, `Load_1_Linear_Speed`, `Load_1_Step_Speed`, 260 janelas por condição) e **8 classes de falha elétrica/mecânica** (195 janelas por classe no total: `Class_Healthy`, `Class_Rotor_Unbalance`, `Class_Rotor_Misalignment`, `Class_Stator_Winding`, `Class_Voltage_Unbalance`, `Class_Bowed_Rotor`, `Class_Broken_Bars`, `Class_Faulty_Bearing`).
- **Adequação:** Excelente alvo para **Transfer Learning (TL)** (avalia a transferência de representações pré-treinadas em rolamentos para um domínio misto motor/rolamento sob velocidade constante, rampa linear e degrau). Excluído de DA/DG por possuir taxonomia de falhas elétricas/estruturais de motor incompatível com a ontologia de 5 classes de rolamentos.

### 4.8. `HUST_Gearbox`
- Auditado em profundidade no documento dedicado [`docs/experiment_planning/hust_gearbox_preliminary.md`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/docs/experiment_planning/hust_gearbox_preliminary.md). Contém **900 janelas** ($f_s = 25.600\text{ Hz}$) perfeitamente balanceadas entre **3 classes de engrenagem** (`Class_Normal`: 300, `Class_Broken`: 300, `Class_Missing`: 300) e **30 condições operacionais** (25 de velocidade constante e 5 de velocidade variável `0–40–0 Hz`).

---

## 5. Inconsistências Encontradas entre Código, Configs e Documentação

1. **`src/vibnet/data/registry.py` vs. Pipeline Real (`make_dataset.py` e `configs/datasets.yaml`):**
   - [`src/vibnet/data/registry.py`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/src/vibnet/data/registry.py) declara `window_length_samples=2048` e `overlap_ratio=0.5` para todos os 10 datasets. Na realidade, [`src/data/make_dataset.py`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/src/data/make_dataset.py) e [`src/data/make_dataset_tl.py`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/src/data/make_dataset_tl.py) utilizam **janelas de $1{,}0\text{ s}$ ($L = f_s$) com `overlap = 0`**.
   - Os valores de `total_windows_audit` em `registry.py` (ex.: `PU = 7680`, `UORED = 600`) refletem estimativas teóricas pré-processamento e divergem dos arquivos efetivamente presentes em `data/processed_tl/detrended_rms/` (`PU = 6476`, `UORED = 370`).
2. **Comentário invertido de classes do CWRU em `scripts/exp_tl_cross_dataset_mlp_tabnet.py`:**
   - As linhas 94–95 de `scripts/exp_tl_cross_dataset_mlp_tabnet.py` comentam `Class_2 -> Ball` e `Class_3 -> Outer Race`. Na tabela oficial `vibdata/raw/labels.csv` usada na criação das pastas `.npy`, **`Class_2` é `Outer Race` (`id=2`) e `Class_3` é `Ball` (`id=3`)**.
3. **Captura silenciosa de exceções (`except Exception: continue`) em `src/data/make_dataset.py`:**
   - Ocultou a perda de 23 arquivos `.mat` do `UORED` (classes `Inner` e `Outer`) e de 880 arquivos `.mat` do `PU` (`Class_28` / `Inner Ring Fault`).

