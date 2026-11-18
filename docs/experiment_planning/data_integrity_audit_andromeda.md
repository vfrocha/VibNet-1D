# Auditoria de Integridade dos Dados — VibNet-1D (Andromeda)

- **Data da Auditoria:** 18/11/2026
- **Ambiente Auditado:** Servidor `andromeda` (`Linux 6.8.0-65-generic #68~22.04.1-Ubuntu`, CPU `Intel(R) Core(TM) i9-10900 CPU @ 2.80GHz`, 20 núcleos lógicos)
- **Diretório Raiz Auditado:** `/home/vfrocha/VibNet-1D/data`
- **Modo de Operação:** Estritamente de leitura (nenhum arquivo de dados, manifesto ou script foi modificado, movido ou excluído; nenhum script de processamento foi reexecutado).

---

## 1. Resumo Executivo

A auditoria completa por varredura de sistema de arquivos, inspeção de cabeçalhos NumPy (`shape`, `dtype`, finitude, RMS) e cálculo exaustivo de hashes criptográficos **SHA-256** sobre todos os **51.744 arquivos `.npy`** (~13,48 GB) presentes em `/home/vfrocha/VibNet-1D/data/` na máquina **Andromeda**, cruzada com os manifests locais e com o relatório de instalação [`results/setup/installation_report.json`](file:///home/vfrocha/VibNet-1D/results/setup/installation_report.json), **confirmou integralmente a suspeita de duplicação de arquivos, resíduos de execuções anteriores e contaminação entre cópia de diretórios e reprocessamento**.

### Principais Conclusões

1. **Integridade Física Individual dos Arrays (`.npy`):**
   - Dos **51.744 arquivos `.npy`** em disco (`38.249` em `data/processed/` e `13.495` em `data/processed_tl/detrended_rms/`), **0 arquivos estão corrompidos, vazios (0 bytes), com `NaN`/`Inf` ou com RMS nulo**. Todos possuem `dtype = float64` e comprimentos compatíveis com as janelas configuradas nos scripts que os geraram.

2. **Duplicação Residual Crítica no Dataset `HUST` na Máquina Andromeda (+150 arquivos / +100% na classe Normal):**
   - Tanto em `data/processed/HUST/` quanto em `data/processed_tl/detrended_rms/HUST/`, existem **1.101 arquivos `.npy`** em vez dos **951 arquivos legítimos**.
   - A análise de timestamps (`mtime`) e SHA-256 comprovou que `src/data/make_dataset.py` e `scripts/rename_normal.sh` foram executados duas vezes (em `23/06/2026` e em `06/07/2026`). Na segunda execução de `rename_normal.sh` (`mv "$d/Class_40" "$d/Class_Normal"`), como o diretório `Class_Normal/` já existia, o comando `mv` moveu `Class_40` para **dentro** de `Class_Normal`, criando a subpasta anômala de nível 4 **`Load_{0,200,400}W/Class_Normal/Class_40/sXXXXX_wYY.npy`** (150 arquivos).
   - Os 150 arquivos em `Class_Normal/Class_40/` possuem **exatamente o mesmo conteúdo (100% de colisão SHA-256)** dos 150 arquivos em `Class_Normal/`, duplicando artificialmente a classe saudável de 150 para 300 janelas em qualquer `rglob("*.npy")` e corrompendo os manifests de validação (`rms_statistics.csv` e `file_comparison.csv`).

3. **Duplicação Total entre `CWRU_12k` e `CWRU_12k_Severity` em `data/processed/` (2.520 arquivos idênticos):**
   - Em `data/processed/`, os diretórios `CWRU_12k` (2.520 arquivos, 4 condições) e `CWRU_12k_Severity` (2.520 arquivos, 20 condições) possuem **100% de colisão SHA-256 (2.520 pares idênticos byte a byte)**, pois `src/data/make_dataset.py` (linhas 220–245) grava simultaneamente as mesmas janelas nas duas árvores de diretórios.

4. **Duplicações Internas Legítimas / Oriundas da Fonte Bruta em `PU` e `HUST_Gearbox`:**
   - Em **`PU`** (`data/processed/PU` e `data/processed_tl/detrended_rms/PU`), existem **12 grupos de SHA-256 idênticos (16 arquivos redundantes)** dentro da classe `Class_27` (`C2` e `C4`), correspondentes a gravações brutas repetidas no catálogo do dataset Paderborn (`s00194 == s00198` em `C2`; `s00180 == s00204 == s00224` e `s00196 == s00220` em `C4`).
   - Em **`HUST_Gearbox`** (`data/processed/HUST_Gearbox` e `data/processed_tl/detrended_rms/HUST_Gearbox`), existem **10 pares de SHA-256 idênticos (10 janelas de 1 gravação)** entre `Cond_30_0/Class_Normal/s00082_w00..w09.npy` e `Cond_30_1/Class_Normal/s00031_w00..w09.npy`, indicando que o arquivo bruto saudável em 30 Hz foi compartilhado ou duplicado entre as cargas 0 e 1 na base original.

5. **Comprovação Forense do Problema de Cópia entre Máquinas + Execução do Instalador (`results/setup/installation_report.json`):**
   - O arquivo [`results/setup/installation_report.json`](file:///home/vfrocha/VibNet-1D/results/setup/installation_report.json) (trazido para a Andromeda via `git pull` no commit `b5e8e5d` em `18/11/2026 08:13`) foi gerado em `17/11/2026 13:32 UTC` em **outra máquina Linux** (`Ubuntu 24.04.5 LTS`, kernel `7.0.0-42-generic`, CPU `Intel Core i9-14900K`, GPU `NVIDIA RTX A6000`).
   - Nessa segunda máquina, o diretório `data/` foi copiado da Andromeda (contendo `data/processed/` e `data/processed_tl/detrended_rms/`, mas sem `data/processed_tl/{raw,detrended,raw_rms}`) e, em seguida, `scripts/setup/install_linux.sh` / `prepare_datasets.py` foi executado.
   - Como `prepare_datasets.py` e `normalize_dataset_tl.py` **não limpam os diretórios de saída antes de gravar**, a geração de `detrended` forçou a reexecução de `detrended_rms` por cima dos arquivos copiados da Andromeda, causando **duplicação massiva em `detrended_rms`** naquela máquina:
     - **`PU`**: subiu de `10.181` (`detrended`) para **`12.099`** em `detrended_rms` ($+1.918$ arquivos residuais de `Class_Normal` somados aos novos `Class_26`);
     - **`UOEMD`**: dobrou de `1.280` (`detrended`) para **`2.560`** em `detrended_rms` ($+1.280$ arquivos com prefixos `sXXXXX` distintos devido à ordem de `os.listdir`/`os.walk` nos arquivos brutos);
     - **`HUST_Gearbox`**: subiu de `900` (`detrended`) para **`1.780`** em `detrended_rms` ($+300$ de `Class_Normal` vs `Class_Healthy` e $+580$ de prefixos `sXXXXX` distintos);
     - **`Mechanical_Gear`**: dobrou de `180` (`detrended`) para **`360`** em `detrended_rms` ($+180$ arquivos com prefixos `sXXXXX` distintos);
     - **`Electric_Motor`**: subiu de `66` (`detrended`) para **`130`** em `detrended_rms` ($+64$ arquivos com prefixos `sXXXXX` distintos);
     - **`UORED`**: somou `370` arquivos antigos copiados da Andromeda com `230` novos arquivos gerados na outra máquina, totalizando **`600`** arquivos em `detrended_rms`.

---

## 2. Inventário Completo de `data/` na Máquina Andromeda

### 2.1. Estrutura Geral de Diretórios e Arquivos de Metadados

Na máquina **Andromeda** (`/home/vfrocha/VibNet-1D/data/`), a árvore de diretórios física atual é:

```text
data/
├── .dataloader.py.swp                              # Arquivo residual de swap do Vim (20.480 bytes, 17/05/2026)
├── external/                                       # Vazio (0 arquivos)
├── raw/                                            # Vazio (0 arquivos; sem download_manifest.json)
├── processed/                                      # 12 diretórios de datasets (38.249 arquivos .npy, 8.103,78 MB)
│   ├── CWRU_12k/                                   # 2.520 arquivos .npy (4 condições x 4 classes)
│   ├── CWRU_12k_Severity/                          # 2.520 arquivos .npy (20 condições x 4 classes)
│   ├── CWRU_48k/                                   #   602 arquivos .npy (12 condições x 3 classes)
│   ├── Electric_Motor/                             #    66 arquivos .npy (30 condições x 4 classes)
│   ├── HUST/                                       # 1.101 arquivos .npy (3 condições; inclui subpasta anômala Class_Normal/Class_40)
│   ├── HUST_Gearbox/                               #   900 arquivos .npy (30 condições x 3 classes)
│   ├── IMS/                                        # 21.184 arquivos .npy (6 condições x 7 classes)
│   ├── MFPT/                                       #   114 arquivos .npy (9 condições x 3 classes — versão não harmonizada)
│   ├── Mechanical_Gear/                            #   180 arquivos .npy (6 condições x 6 classes)
│   ├── PU/                                         # 6.476 arquivos .npy (4 condições x 3 classes — sem Class_28/Inner)
│   ├── UOC/                                        #   936 arquivos .npy (1 condição x 5 classes)
│   ├── UOEMD/                                      # 1.280 arquivos .npy (16 condições x 8 classes)
│   └── UORED/                                      #   370 arquivos .npy (18 condições x 3 classes — sem Inner e Outer)
└── processed_tl/                                   # Sem subpastas raw/, detrended/ ou raw_rms/; sem processing_manifest.json
    ├── detrended_rms/                              # 9 diretórios de datasets (13.495 arquivos .npy, 4.781,55 MB)
    │   ├── normalization_manifest.csv              # Manifesto de normalização RMS (9 linhas, 28/10/2026)
    │   ├── CWRU_12k/                               # 2.520 arquivos .npy
    │   ├── CWRU_48k/                               #   602 arquivos .npy
    │   ├── Electric_Motor/                         #    66 arquivos .npy
    │   ├── HUST/                                   # 1.101 arquivos .npy (inclui subpasta anômala Class_Normal/Class_40)
    │   ├── HUST_Gearbox/                           #   900 arquivos .npy
    │   ├── Mechanical_Gear/                        #   180 arquivos .npy
    │   ├── PU/                                     # 6.476 arquivos .npy
    │   ├── UOEMD/                                  # 1.280 arquivos .npy
    │   └── UORED/                                  #   370 arquivos .npy
    └── validation/                                 # Relatórios CSV de validação (28/10/2026 09:54)
        ├── dataset_validation.csv
        ├── file_comparison.csv
        ├── reference_validation.csv
        └── rms_statistics.csv
```

> [!IMPORTANT]
> Na máquina **Andromeda**, os diretórios `data/raw/`, `data/processed_tl/raw/`, `data/processed_tl/detrended/` e `data/processed_tl/raw_rms/` **não existem ou estão vazios**, e o ambiente virtual `.venv/` também não está presente localmente. Todos os arquivos em `data/processed_tl/detrended_rms/` na Andromeda foram gerados em `28/10/2026` usando a flag `--use-processed-fallback` de [`src/preprocessing/normalize_dataset_tl.py`](file:///home/vfrocha/VibNet-1D/src/preprocessing/normalize_dataset_tl.py), que lê diretamente de `data/processed/<dataset>` (onde os sinais já passaram por `Detrend() + SimpleSplit()`).

### 2.2. Contagem, Tamanho, Dimensões e Integridade por Dataset (Andromeda)

A tabela a seguir consolida a verificação física de todos os diretórios em `data/processed/` e `data/processed_tl/detrended_rms/` na máquina Andromeda:

| Variante (Diretório) | Dataset | Qtd. `.npy` | Tamanho Total (MB) | Shape Verificado | Dtype | Arq. Vazios / Inválidos | `NaN` / `Inf` | RMS Zero | Nº Condições | Nº Classes | Intervalo de Modificação (`mtime` UTC) |
| :--- | :--- | ---: | ---: | :---: | :---: | ---: | ---: | ---: | ---: | ---: | :---: |
| `data/processed` | `CWRU_12k` | 2.520 | 231,02 | `(12000,)` | `float64` | 0 | 0 | 0 | 4 | 4 | `2026-09-28 11:10:17` – `11:10:48` |
| `data/processed` | `CWRU_12k_Severity` | 2.520 | 231,02 | `(12000,)` | `float64` | 0 | 0 | 0 | 20 | 4 | `2026-09-28 11:10:17` – `11:10:48` |
| `data/processed` | `CWRU_48k` | 602 | 220,53 | `(48000,)` | `float64` | 0 | 0 | 0 | 12 | 3 | `2026-09-28 11:10:43` – `11:10:47` |
| `data/processed` | `Electric_Motor` | 66 | 25,19 | `(50000,)` | `float64` | 0 | 0 | 0 | 30 | 4 | `2026-07-09 15:42:56` – `15:42:57` |
| `data/processed` | `HUST` | **1.101** | 430,21 | `(51200,)` | `float64` | 0 | 0 | 0 | 3 | **7 + 1*** | `2026-06-23 12:20:28` – `2026-07-06 14:00:46` |
| `data/processed` | `HUST_Gearbox` | 900 | 175,89 | `(25600,)` | `float64` | 0 | 0 | 0 | 30 | 3 | `2026-07-28 20:02:24` – `20:02:39` |
| `data/processed` | `IMS` | 21.184 | 3.235,01 | `(20000,)` | `float64` | 0 | 0 | 0 | 6 | 7 | `2026-07-09 17:38:23` – `17:54:41` |
| `data/processed` | `MFPT` | **114** | 42,48 | `(48828,)` | `float64` | 0 | 0 | 0 | 9 | 3 | `2026-07-09 17:21:49` – `17:21:51` |
| `data/processed` | `Mechanical_Gear` | 180 | 6,89 | `(5000,)` | `float64` | 0 | 0 | 0 | 6 | 6 | `2026-07-09 15:43:13` – `15:43:14` |
| `data/processed` | `PU` | **6.476** | 3.162,90 | `(64000,)` | `float64` | 0 | 0 | 0 | 4 | 3 | `2026-07-06 13:56:16` – `14:00:27` |
| `data/processed` | `UOC` | 936 | 14,74 | `(2048,)` | `float64` | 0 | 0 | 0 | 1 | 5 | `2026-07-09 19:22:00` – `19:23:22` |
| `data/processed` | `UOEMD` | 1.280 | 410,31 | `(42000,)` | `float64` | 0 | 0 | 0 | 16 | 8 | `2026-07-28 19:31:40` – `19:32:11` |
| `data/processed` | `UORED` | **370** | 118,61 | `(42000,)` | `float64` | 0 | 0 | 0 | 18 | 3 | `2026-07-09 17:04:09` – `17:04:15` |
| **Subtotal `processed`** | **13 diretórios** | **38.249** | **8.304,80** | — | `float64` | **0** | **0** | **0** | — | — | — |
| `detrended_rms` | `CWRU_12k` | 2.520 | 231,02 | `(12000,)` | `float64` | 0 | 0 | 0 | 4 | 4 | `2026-10-28 12:50:11` – `12:50:34` |
| `detrended_rms` | `CWRU_48k` | 602 | 220,53 | `(48000,)` | `float64` | 0 | 0 | 0 | 12 | 3 | `2026-10-28 12:50:36` – `12:50:46` |
| `detrended_rms` | `Electric_Motor` | 66 | 25,19 | `(50000,)` | `float64` | 0 | 0 | 0 | 30 | 4 | `2026-10-28 12:53:46` – `12:53:46` |
| `detrended_rms` | `HUST` | **1.101** | 430,21 | `(51200,)` | `float64` | 0 | 0 | 0 | 3 | **7 + 1*** | `2026-10-28 12:53:39` – `12:53:46` |
| `detrended_rms` | `HUST_Gearbox` | 900 | 175,89 | `(25600,)` | `float64` | 0 | 0 | 0 | 30 | 3 | `2026-10-28 12:52:55` – `12:53:20` |
| `detrended_rms` | `Mechanical_Gear` | 180 | 6,89 | `(5000,)` | `float64` | 0 | 0 | 0 | 6 | 6 | `2026-10-28 12:53:46` – `12:53:46` |
| `detrended_rms` | `PU` | **6.476** | 3.162,90 | `(64000,)` | `float64` | 0 | 0 | 0 | 4 | 3 | `2026-10-28 12:51:20` – `12:52:50` |
| `detrended_rms` | `UOEMD` | 1.280 | 410,31 | `(42000,)` | `float64` | 0 | 0 | 0 | 16 | 8 | `2026-10-28 12:53:24` – `12:53:38` |
| `detrended_rms` | `UORED` | **370** | 118,61 | `(42000,)` | `float64` | 0 | 0 | 0 | 18 | 3 | `2026-10-28 12:53:38` – `12:53:38` |
| **Subtotal `detrended_rms`**| **9 diretórios** | **13.495** | **4.781,55** | — | `float64` | **0** | **0** | **0** | — | — | — |
| **TOTAL GERAL EM `data/`** | **22 diretórios** | **51.744** | **13.086,35** | — | `float64` | **0** | **0** | **0** | — | — | — |

*\*Nota: Em `HUST`, além das 7 classes em nível 3 (`Class_Normal`, `Class_41`..`Class_46`), há 1 subdiretório anômalo em nível 4 (`Class_Normal/Class_40`) com 150 arquivos.*

### 2.3. Distribuição Detalhada por Dataset, Condição e Classe (Andromeda)

1. **`CWRU_12k` (`2.520` janelas em `processed` e `detrended_rms`):**
   - **Condições (4):** `Load_0HP` (600), `Load_1HP` (640), `Load_2HP` (640), `Load_3HP` (640).
   - **Classes (4):** `Class_0` (`Normal`: 280 — sendo 40 em `0HP` e 80 em `1..3HP`), `Class_1` (`Inner`: 760 — 190/cond.), `Class_2` (`Outer`: 720 — 180/cond.), `Class_3` (`Ball`: 760 — 190/cond.).
2. **`CWRU_12k_Severity` (`2.520` janelas apenas em `processed`):**
   - **Condições (20):** `Load_{0..3}HP_Sev_0.000` (40 em `0HP`, 80 em `1..3HP`, contendo apenas `Class_0`), `Load_{0..3}HP_Sev_{0.007, 0.014, 0.021}` (180 cada: 60 `Class_1`, 60 `Class_2`, 60 `Class_3`), `Load_{0..3}HP_Sev_0.028` (20 cada: 10 `Class_1`, 10 `Class_3`, sem `Class_2`).
   - **Classes (4):** `Class_0` (280), `Class_1` (760), `Class_2` (720), `Class_3` (760).
3. **`CWRU_48k` (`602` janelas em `processed` e `detrended_rms`):**
   - **Condições (12):** `Load_0HP_Sev_0.007` (30), `Load_0HP_Sev_0.014` (20 — **0 janelas em `Class_1`**), `Load_0HP_Sev_0.021` (18: 10 `Class_1`, 4 `Class_2`, 4 `Class_3`), `Load_1HP_Sev_0.007` (60), `Load_1HP_Sev_0.014` (54: 14 `Class_1`, 20 `Class_2`, 20 `Class_3`), `Load_1HP_Sev_0.021` (60), `Load_{2,3}HP_Sev_{0.007,0.014,0.021}` (60 cada: 20 por classe).
   - **Classes (3):** `Class_1` (194), `Class_2` (204), `Class_3` (204). Não possui classe `Normal`.
4. **`Electric_Motor` (`66` janelas em `processed` e `detrended_rms`):**
   - **Condições (30):** `Cond_01` a `Cond_30` (24 condições com 2 janelas e 6 condições [`Cond_05, 11, 14, 16, 17, 19`] com 3 janelas; cada condição contém apenas 1 classe).
   - **Classes (4):** `Class_Healthy` (4 — apenas em `Cond_01` e `Cond_02`), `Class_Electrical_Fault` (10), `Class_Mechanical_Fault` (20), `Class_Combined_Fault` (32).
5. **`HUST` (`1.101` janelas em `processed` e `detrended_rms`, sendo `951` válidas + `150` duplicadas):**
   - **Condições (3):**
     - `Load_0W`: 325 janelas em nível 3 (`Class_Normal`: 50, `Class_41`: 50, `Class_42`: 50, `Class_43`: 36, `Class_44`: 40, `Class_45`: 50, `Class_46`: 49) + **50 janelas duplicadas** em `Load_0W/Class_Normal/Class_40/` = 375 arquivos.
     - `Load_200W`: 315 janelas em nível 3 (`Class_Normal`: 50, `Class_41`: 50, `Class_42`: 50, `Class_43`: 33, `Class_44`: 40, `Class_45`: 50, `Class_46`: 42) + **50 janelas duplicadas** em `Load_200W/Class_Normal/Class_40/` = 365 arquivos.
     - `Load_400W`: 311 janelas em nível 3 (`Class_Normal`: 50, `Class_41`: 50, `Class_42`: 50, `Class_43`: 31, `Class_44`: 40, `Class_45`: 50, `Class_46`: 40) + **50 janelas duplicadas** em `Load_400W/Class_Normal/Class_40/` = 361 arquivos.
6. **`HUST_Gearbox` (`900` janelas em `processed` e `detrended_rms`):**
   - **Condições (30):** 25 estacionárias (`Cond_{20,25,30,35,40}_{0..4}`) e 5 variáveis (`Cond_L{0..4}_VS_0_40_0`), todas com exatas 30 janelas (10 por classe).
   - **Classes (3):** `Class_Normal` (300), `Class_Broken` (300), `Class_Missing` (300).
7. **`IMS` (`21.184` janelas apenas em `processed`):**
   - **Condições (6):** `Test_1_Bearing_{1..4}` (4.312 cada), `Test_2_Bearing_{1,2}` (1.968 cada).
   - **Classes (7):** `Class_6` (18.392), `Class_7` (500), `Class_8` (68), `Class_9` (600), `Class_10` (112), `Class_11` (900), `Class_12` (612).
8. **`MFPT` (`114` janelas apenas em `processed`):**
   - **Condições (9):** `Load_0` (3 `Class_24`), `Load_25` (3 `Class_25`), `Load_{50,100,150,200,250,300}` (6 cada: 3 `Class_24`, 3 `Class_25`), `Load_270` (**72 janelas não decimadas de 0,5 s**: 36 `Class_23` e 36 `Class_25`).
   - **Classes (3):** `Class_23` (`Normal`: 36), `Class_24` (`Inner`: 21), `Class_25` (`Outer`: 57).
9. **`Mechanical_Gear` (`180` janelas em `processed` e `detrended_rms`):**
   - **Condições (6):** `Cond_{8.33203125Hz, 25.0Hz, 40.0Hz}_{0.0Nm, 80.0Nm}` (30 janelas por condição, 5 por classe).
   - **Classes (6):** `Class_Healthy` (30), `Class_Eccentricity` (30), `Class_Missing Tooth` (30), `Class_Root Crack` (30), `Class_Surface Fault` (30), `Class_Tooth Chipped Fault` (30).
10. **`PU` (`6.476` janelas em `processed` e `detrended_rms`):**
    - **Condições (4):**
      - `C1_1500rpm_0.7Nm_1000N` (1.629: `Class_Normal`=480, `Class_27`=954, `Class_29`=195);
      - `C2_900rpm_0.7Nm_1000N` (1.612: `Class_Normal`=479, `Class_27`=955, `Class_29`=178);
      - `C3_1500rpm_0.1Nm_1000N` (1.621: `Class_Normal`=480, `Class_27`=950, `Class_29`=191);
      - `C4_1500rpm_0.7Nm_400N` (1.614: `Class_Normal`=479, `Class_27`=952, `Class_29`=183).
    - **Classes (3):** `Class_Normal` (1.918), `Class_27` (`Outer`: 3.811), `Class_29` (`Inner & Outer`: 747). **`Class_28` (`Inner`) está completamente ausente (0 janelas)**.
11. **`UOC` (`936` janelas apenas em `processed`):**
    - **Condição (1):** `Cond_Unknown` (936).
    - **Classes (5):** `Class_35` (104), `Class_36` (104), `Class_37` (104), `Class_38` (104), `Class_39` (520).
12. **`UOEMD` (`1.280` janelas em `processed` e `detrended_rms`):**
    - **Condições (16):** `Load_{Loaded,No_Load}_Speed_{15Hz,30Hz,45Hz,60Hz,Dec_45_to_15Hz,Dec_60_to_30Hz,Inc_15_to_45Hz,Inc_30_to_60Hz}` (80 janelas por condição, 10 por classe).
    - **Classes (8):** `Class_Normal`, `Class_Bowed Rotor`, `Class_Broken Rotor Bar`, `Class_Faulty Bearing`, `Class_Misalignment`, `Class_Stator Winding`, `Class_Unbalance`, `Class_Voltage Imbalance` (160 janelas cada).
13. **`UORED` (`370` janelas em `processed` e `detrended_rms`):**
    - **Condições (18):** `Bearing_{1..4, 6..9}` (10 janelas `Class_Normal` cada), `Bearing_{11..15}` (30 cada: 10 `Class_Normal`, 20 `Class_Ball`), `Bearing_{16..19}` (30 cada: 10 `Class_Normal`, 20 `Class_Cage`), `Bearing_20` (20: 10 `Class_Normal`, 10 `Class_Cage`). (`Bearing_5` e `Bearing_10` ausentes).
    - **Classes (3):** `Class_Normal` (180), `Class_Ball` (100), `Class_Cage` (90). (`Class_Inner Race` e `Class_Outer Race` ausentes).

---

## 3. Tabela de Contagens por Variante (Andromeda vs. Manifests e Máquina Secundária)

A tabela abaixo contrasta as contagens reais encontradas em disco na **Andromeda** com as contagens registradas no relatório [`results/setup/installation_report.json`](file:///home/vfrocha/VibNet-1D/results/setup/installation_report.json) (gerado na máquina `i9-14900K / RTX A6000` após cópia e reprocessamento) e na documentação de planejamento ([`docs/experiment_planning/dataset_inventory_summary.csv`](file:///home/vfrocha/VibNet-1D/docs/experiment_planning/dataset_inventory_summary.csv)):

| Dataset | Andromeda: `data/processed` (`processed_1d`) | Andromeda: `processed_tl/raw` | Andromeda: `processed_tl/detrended` | Andromeda: `processed_tl/raw_rms` | Andromeda: `processed_tl/detrended_rms` | Relatório `installation_report.json`: `detrended` | Relatório `installation_report.json`: `detrended_rms` | Contagem Canônica Limpa Esperada (1,0 s) | Status na Andromeda |
| :--- | ---: | :---: | :---: | :---: | ---: | ---: | ---: | ---: | :--- |
| **`CWRU_12k`** | 2.520 | Ausente | Ausente | Ausente | 2.520 | 2.520 | 2.520 | **2.520** | Íntegro (usa `Class_0`) |
| **`CWRU_12k_Severity`** | **2.520** | Ausente | Ausente | Ausente | Ausente | N/A | N/A | *(Espelho de `CWRU_12k`)* | Duplicata integral de `CWRU_12k` |
| **`CWRU_48k`** | 602 | Ausente | Ausente | Ausente | 602 | 602 | 602 | **602** | Íntegro (sem classe `Normal`) |
| **`MFPT`** | **114** | Ausente | Ausente | Ausente | **Ausente** | 78 | 78 | **78** *(harmonizado)* | Desatualizado em `processed` (sem decimação 97,6 kHz) e ausente em `detrended_rms` |
| **`PU`** | **6.476** | Ausente | Ausente | Ausente | **6.476** | 10.181 | **12.099** *(+1.918 dup)* | **10.181** *(c/ `Class_28`)* ou **6.476** *(s/ `Class_28`)* | Parcial (sem `Class_28`); na outra máquina sofreu fusão de `Class_Normal` + `Class_26` |
| **`UORED`** | **370** | Ausente | Ausente | Ausente | **370** | 230 | **600** *(370+230)* | **600** *(60 arq. brutos)* | Parcial (faltam 23 arq. `Inner`/`Outer`); na outra máquina misturou 2 extrações |
| **`UOEMD`** | 1.280 | Ausente | Ausente | Ausente | 1.280 | 1.280 | **2.560** *(2x dup)* | **1.280** *(16 cond.)* | Íntegro na Andromeda; duplicado ($2\times$) na outra máquina |
| **`HUST`** | **1.101** | Ausente | Ausente | Ausente | **1.101** | Ausente | **1.101** *(+150 dup)* | **951** | **Contaminado (+150 duplicatas em `Class_Normal/Class_40`)** |
| **`HUST_Gearbox`** | 900 | Ausente | Ausente | Ausente | 900 | 900 | **1.780** *(+880 dup)* | **900** | Íntegro na Andromeda (10 pares idênticos da fonte); duplicado na outra máquina |
| **`Mechanical_Gear`** | 180 | Ausente | Ausente | Ausente | 180 | 180 | **360** *(2x dup)* | **180** | Íntegro na Andromeda; duplicado ($2\times$) na outra máquina |
| **`Electric_Motor`** | 66 | Ausente | Ausente | Ausente | 66 | 66 | **130** *(+64 dup)* | **66** | Íntegro na Andromeda; duplicado na outra máquina |
| **`IMS`** | 21.184 | Ausente | Ausente | Ausente | Ausente | N/A | N/A | — *(Fora do escopo TL)* | Presente apenas em `data/processed` |
| **`UOC`** | 936 | Ausente | Ausente | Ausente | Ausente | N/A | N/A | — *(Excluído)* | Presente apenas em `data/processed` |

---

## 4. Duplicações Confirmadas por SHA-256

O cálculo do hash SHA-256 sobre os **51.744 arquivos `.npy`** na máquina Andromeda identificou exatamente **2.864 grupos de arquivos com conteúdo binário 100% idêntico** (`2.868` arquivos extras além de um representante único por grupo).

A análise abaixo classifica rigorosamente cada um dos **2.864 grupos** conforme sua natureza (erro residual de processamento vs. espelhamento arquitetural vs. repetição na fonte bruta):

### Resumo Global dos Grupos Duplicados por SHA-256 (Andromeda)

| Categoria de Duplicação | Variante / Escopo | Datasets Envolvidos | Grupos SHA-256 | Arquivos Redundantes | Classificação de Auditoria |
| :--- | :--- | :--- | ---: | ---: | :--- |
| **1. Pasta residual aninhada (`Class_Normal/Class_40`)** | Intra-dataset (`processed`) | `HUST` | **150** | **150** | **ERRO CRÍTICO (Resíduo de `rename_normal.sh`)** |
| **2. Pasta residual aninhada (`Class_Normal/Class_40`)** | Intra-dataset (`detrended_rms`) | `HUST` | **150** | **150** | **ERRO CRÍTICO (Propagado de `processed/HUST`)** |
| **3. Espelhamento `CWRU_12k` vs `CWRU_12k_Severity`** | Inter-datasets (`processed`) | `CWRU_12k` $\leftrightarrow$ `CWRU_12k_Severity` | **2.520** | **2.520** | **REDUNDÂNCIA DE PROJETO (`make_dataset.py`)** |
| **4. Gravação idêntica entre `Cond_30_0` e `Cond_30_1`** | Intra-dataset (`processed`) | `HUST_Gearbox` | **10** | **10** | **DUPLICAÇÃO NA FONTE BRUTA (Vazamento LOCO entre `30_0` e `30_1`)** |
| **5. Gravação idêntica entre `Cond_30_0` e `Cond_30_1`** | Intra-dataset (`detrended_rms`) | `HUST_Gearbox` | **10** | **10** | **DUPLICAÇÃO NA FONTE BRUTA (Propagada para `detrended_rms`)** |
| **6. Gravações repetidas em `Class_27` (`C2` e `C4`)** | Intra-dataset (`processed`) | `PU` | **12** | **16** | **DUPLICAÇÃO NA FONTE BRUTA (Pares/trincas de `.mat` idênticos)** |
| **7. Gravações repetidas em `Class_27` (`C2` e `C4`)** | Intra-dataset (`detrended_rms`) | `PU` | **12** | **16** | **DUPLICAÇÃO NA FONTE BRUTA (Propagada para `detrended_rms`)** |
| **8. Entre variantes (`processed` vs `detrended_rms`)** | Inter-variantes | Todos os 9 datasets | **0** | **0** | **OK (`RMS_ref != 1.0` altera todos os arrays)** |
| **TOTAL CONFIRMADO POR SHA-256** | — | — | **2.864** | **2.872** | — |

### 4.1. Detalhamento: Duplicação Crítica Residual em `HUST` (150 grupos em `processed` + 150 grupos em `detrended_rms`)

- **Caminhos afetados:**
  - `data/processed/HUST/Load_{0W,200W,400W}/Class_Normal/sXXXXX_wYY.npy` (150 arquivos, `mtime = 2026-06-23 12:20:28..40 UTC`)
  - `data/processed/HUST/Load_{0W,200W,400W}/Class_Normal/Class_40/sXXXXX_wYY.npy` (150 arquivos, `mtime = 2026-07-06 14:00:29..46 UTC`)
  - E seus correspondentes normalizados em `data/processed_tl/detrended_rms/HUST/`.
- **Evidência SHA-256 (exemplos verificados):**
  - `Load_0W/Class_Normal/s00008_w00.npy` e `Load_0W/Class_Normal/Class_40/s00008_w00.npy`:
    - Em `processed`: `sha256 = b78bc5a66ef27949b401f941320700c19878af6ee122250aad578d71822601b3`
    - Em `detrended_rms`: `sha256 = d3b470c2aa917e8a411a12250837373007c2cbc3663af65394a9824683335c69`
  - `Load_0W/Class_Normal/s00008_w01.npy` e `Load_0W/Class_Normal/Class_40/s00008_w01.npy`:
    - Em `processed`: `sha256 = 6d1b00ea28f6d83f0423eec9d1dc039f98a6d5be972ae98d6475f317571e87c9`
    - Em `detrended_rms`: `sha256 = 8b881085b7eec049e7b9e8a24e66e74f03517f8271271fd7e159e1116586c4d2`
- **Impacto:** Qualquer carregador que use busca recursiva (`rglob("*.npy")`) ou que leia subpastas de `HUST` carrega **duas vezes** todas as 150 janelas normais (totalizando 300 janelas normais e 1.101 no dataset) ou interpreta `Class_Normal` como nome de condição e `Class_40` como classe distinta.

### 4.2. Detalhamento: Duplicação entre Condições em `HUST_Gearbox` (10 grupos em `processed` + 10 grupos em `detrended_rms`)

As 10 janelas da gravação `s00082` em `Cond_30_0/Class_Normal/` são **idênticas byte a byte** às 10 janelas da gravação `s00031` em `Cond_30_1/Class_Normal/` (com nomes de arquivo diferentes!):

| Janela | Caminho em `Cond_30_0` | Caminho em `Cond_30_1` | SHA-256 em `data/processed/HUST_Gearbox` | SHA-256 em `data/processed_tl/detrended_rms/HUST_Gearbox` |
| :---: | :--- | :--- | :--- | :--- |
| `w00` | `Cond_30_0/Class_Normal/s00082_w00.npy` | `Cond_30_1/Class_Normal/s00031_w00.npy` | `fa463b48651c5332...` | `e39561ab9b0b5e73...` |
| `w01` | `Cond_30_0/Class_Normal/s00082_w01.npy` | `Cond_30_1/Class_Normal/s00031_w01.npy` | `dab25b56b6ed7768...` | `a8e3dccf4ca4d6df...` |
| `w02` | `Cond_30_0/Class_Normal/s00082_w02.npy` | `Cond_30_1/Class_Normal/s00031_w02.npy` | `cc73323151adb1ff...` | `7d23013aaeb018ab...` |
| `w03` | `Cond_30_0/Class_Normal/s00082_w03.npy` | `Cond_30_1/Class_Normal/s00031_w03.npy` | `0ada1dfa041605ab...` | `9f7a32698a37ebfc...` |
| `w04` | `Cond_30_0/Class_Normal/s00082_w04.npy` | `Cond_30_1/Class_Normal/s00031_w04.npy` | `e84ea435dafbb512...` | `1247ae91c1c33901...` |
| `w05` | `Cond_30_0/Class_Normal/s00082_w05.npy` | `Cond_30_1/Class_Normal/s00031_w05.npy` | `c58fc260f5615648...` | `66a9917f729ca952...` |
| `w06` | `Cond_30_0/Class_Normal/s00082_w06.npy` | `Cond_30_1/Class_Normal/s00031_w06.npy` | `1ab2242e9bb391b5...` | `5e483ebe5c9db574...` |
| `w07` | `Cond_30_0/Class_Normal/s00082_w07.npy` | `Cond_30_1/Class_Normal/s00031_w07.npy` | `aecbcd565eef68f2...` | `d91ec4cc7f90b95d...` |
| `w08` | `Cond_30_0/Class_Normal/s00082_w08.npy` | `Cond_30_1/Class_Normal/s00031_w08.npy` | `d2844c64a5bd47fc...` | `5baf07efb8198cac...` |
| `w09` | `Cond_30_0/Class_Normal/s00082_w09.npy` | `Cond_30_1/Class_Normal/s00031_w09.npy` | `8dfd423919334026...` | `c08d264252b2b00e...` |

> [!WARNING]
> **Implicação Metodológica em `HUST_Gearbox`:** No protocolo LOCO de 30 condições individuais, quando `Cond_30_0` é teste, `Cond_30_1` está no treino (e vice-versa), causando vazamento exato das 10 janelas de `Class_Normal`. Já no protocolo **Macro-LOCO por Regime de Velocidade (`30Hz`)**, ambas `Cond_30_0` e `Cond_30_1` caem juntas no mesmo fold (`30Hz`), eliminando o vazamento entre treino e teste (mas no protocolo Macro-LOCO por Carga `Load_0` vs `Load_1`, elas ficam em folds separados).

### 4.3. Detalhamento: Duplicações Internas em `PU` (12 grupos / 16 arquivos extras em `processed` e `detrended_rms`)

Dentro da mesma condição e classe (`Class_27` = `Outer Ring Fault`), 3 gravações brutas de 4 janelas possuem cópias idênticas com índices `sXXXXX` distintos:
1. Em `C2_900rpm_0.7Nm_1000N/Class_27/`:
   - `s00194_w00..w03.npy` == `s00198_w00..w03.npy` (4 grupos de pares idênticos; SHA-256 em `processed`: `d40ba7fd...`, `91ddbf9f...`, `f44ccf61...`, `23daa555...`).
2. Em `C4_1500rpm_0.7Nm_400N/Class_27/`:
   - `s00180_w00..w03.npy` == `s00204_w00..w03.npy` == `s00224_w00..w03.npy` (4 grupos de **trincas idênticas**, gerando 8 arquivos redundantes; SHA-256 em `processed`: `6cde79f1...`, `2613420e...`, `b85ec2a2...`, `a439dc9b...`).
   - `s00196_w00..w03.npy` == `s00220_w00..w03.npy` (4 grupos de pares idênticos; SHA-256 em `processed`: `d49de18c...`, `d2f983ec...`, `e4bf4629...`, `37f5587e...`).

> [!NOTE]
> Como esses arquivos duplicados em `PU` pertencem à **mesma condição operacional** (`C2` ou `C4`), em avaliações **LOCO** (onde toda a condição `C2` ou `C4` vai integralmente para o treino ou para o teste) não há vazamento entre treino e teste. Porém, no protocolo de **Domain Adaptation (DA — split 20% adaptação / 80% teste por gravação `sXXXXX`)**, se `s00180` cair nos 20% de adaptação e `s00204` ou `s00224` cair nos 80% de teste, haverá **vazamento de 12 janelas idênticas entre adaptação e teste** caso não sejam deduplicadas por SHA-256!

---

## 5. Comparação entre `detrended` (`data/processed/`) e `detrended_rms` (`data/processed_tl/detrended_rms/`)

### 5.1. Relação Direta na Máquina Andromeda

1. **Inexistência de `data/processed_tl/detrended/`:** Na Andromeda, a pasta `data/processed_tl/detrended/` nunca foi criada. Em seu lugar, o diretório `data/processed/` atua como a variante `detrended` (pois `src/data/make_dataset.py` aplica `Sequential([Detrend(), SimpleSplit(...)])`).
2. **Correspondência 1-para-1 nos 9 Datasets de `cross_dataset`:**
   - Para os 9 datasets presentes em `data/processed_tl/detrended_rms/` (`CWRU_12k`, `CWRU_48k`, `Electric_Motor`, `HUST`, `HUST_Gearbox`, `Mechanical_Gear`, `PU`, `UOEMD`, `UORED`), a correspondência de caminhos relativos com `data/processed/<dataset>` é **100% exata** (`13.495` arquivos em ambos, `0` ausentes, `0` extras).
   - Inclusive o defeito estrutural de `data/processed/HUST/Load_*W/Class_Normal/Class_40/` foi copiado de forma idêntica para `data/processed_tl/detrended_rms/HUST/Load_*W/Class_Normal/Class_40/`.
   - **0 arquivos** possuem o mesmo SHA-256 entre `data/processed/<ds>` e `data/processed_tl/detrended_rms/<ds>`, pois todos os sinais foram divididos por $\text{RMS}_{\text{ref}} \neq 1{,}0$ e passaram no teste matemático $\text{RMS}(x_{\text{norm}}) = \text{RMS}(x) / \text{RMS}_{\text{ref}}$ com erro relativo $< 10^{-15}$.
3. **Datasets Presentes em `data/processed/` mas Ausentes em `data/processed_tl/detrended_rms/` (Andromeda):**
   - `MFPT` (`114` arquivos em `data/processed/MFPT/`, `0` em `detrended_rms/`): ficou de fora porque `normalize_dataset_tl.py` foi executado em `28/10/2026` com `--datasets cross_dataset` (lista de 9 datasets que não inclui `MFPT`).
   - `CWRU_12k_Severity` (`2.520` arquivos), `IMS` (`21.184` arquivos) e `UOC` (`936` arquivos): presentes apenas em `data/processed/`.

### 5.2. Relação entre `detrended` e `detrended_rms` no Relatório `results/setup/installation_report.json` (Máquina Secundária)

No relatório [`results/setup/installation_report.json`](file:///home/vfrocha/VibNet-1D/results/setup/installation_report.json) (gerado após copiar `data/` da Andromeda para a máquina `i9-14900K / RTX A6000` e rodar `install_linux.sh`), a variante `data/processed_tl/detrended/` foi gerada do zero a partir de `data/raw/`, enquanto `data/processed_tl/detrended_rms/` **já continha os arquivos copiados da Andromeda** e recebeu por cima os novos arquivos normalizados:

| Dataset | `detrended` (Gerado Limpo na Máquina 2) | `detrended_rms` (Contaminado na Máquina 2) | Excesso em `detrended_rms` | Causa Exata da Divergência entre `detrended` e `detrended_rms` |
| :--- | ---: | ---: | ---: | :--- |
| **`PU`** | 10.181 | **12.099** | **+1.918** | `detrended` gerou `Class_26` (1.918), `Class_27` (3.811), `Class_28` (3.705) e `Class_29` (747). Em `detrended_rms`, esses 10.181 arquivos foram gravados ao lado da pasta antiga **`Class_Normal` (1.918 arquivos)** copiada da Andromeda ($10.181 + 1.918 = 12.099$). |
| **`UORED`** | 230 | **600** | **+370** | `detrended` extraiu 23 arquivos `.mat` (230 janelas: `Normal`, `Inner Race`, `Outer Race`, `Cage`). Em `detrended_rms`, essas 230 janelas somaram-se às **370 janelas antigas** copiadas da Andromeda ($370 + 230 = 600$). |
| **`UOEMD`** | 1.280 | **2.560** | **+1.280** | Como `s{i:05d}` depende da ordem de `os.walk`/`os.listdir` sobre os arquivos brutos, as 128 gravações receberam índices `sXXXXX` diferentes na Máquina 2, não sobrescrevendo os 1.280 arquivos copiados da Andromeda ($1.280 + 1.280 = 2.560$). |
| **`HUST_Gearbox`** | 900 | **1.780** | **+880** | As 300 janelas saudáveis foram gravadas em `Class_Healthy`, mantendo as 300 antigas em `Class_Normal` ($+300$). Nas classes `Broken` e `Missing` (600 janelas), 580 janelas receberam prefixos `sXXXXX` distintos e apenas 20 colidiram ($900 + 300 + 580 = 1.780$). |
| **`Mechanical_Gear`** | 180 | **360** | **+180** | As 36 gravações (5 janelas cada) receberam prefixos `sXXXXX` distintos na Máquina 2 em todas as 36 células condição/classe ($180 + 180 = 360$). |
| **`Electric_Motor`** | 66 | **130** | **+64** | De 30 gravações (66 janelas), 29 receberam prefixos `sXXXXX` distintos e 1 gravação (2 janelas) colidiu e foi sobrescrita ($66 + 66 - 2 = 130$). |
| **`HUST`** | 0 *(Ausente)* | **1.101** | **+1.101** *(vs `detrended`)* | `HUST.zip` falhou no download na Máquina 2 (`CORRUPTED_INVALID_ZIP`), então `detrended` não foi criado e `detrended_rms` manteve os 1.101 arquivos (com `Class_Normal/Class_40`) copiados da Andromeda. |

---

## 6. Arquivos Residuais ou Suspeitos Identificados

1. **[`data/.dataloader.py.swp`](file:///home/vfrocha/VibNet-1D/data/.dataloader.py.swp)** (`20.480` bytes, `mtime = 17/05/2026`):
   - Arquivo oculto de recuperação de sessão do editor Vim (`swap file`) deixado na raiz de `data/`.
2. **`data/processed/HUST/Load_{0W,200W,400W}/Class_Normal/Class_40/*.npy` (150 arquivos) e `data/processed_tl/detrended_rms/HUST/Load_{0W,200W,400W}/Class_Normal/Class_40/*.npy` (150 arquivos):**
   - Diretórios residuais criados pela dupla execução de `make_dataset.py` + `scripts/rename_normal.sh`. Devem ser removidos na reconstrução.
3. **`data/processed/MFPT/` (114 arquivos de `09/07/2026`):**
   - Gerados antes da introdução de `harmonize_mfpt_sample_rate()`. Os 72 arquivos em `Load_270/Class_23` (36) e `Load_270/Class_25` (36) têm taxa real de $97.656\text{ Hz}$ ($0{,}5\text{ s}$ por janela de 48.828 pontos), causando erro de fator $2\times$ na escala de frequência da FFT.
4. **`data/processed/CWRU_12k_Severity/` (2.520 arquivos):**
   - Cópia integral com outra organização de pastas de `CWRU_12k` (231,02 MB). Não faz parte dos 10 datasets canônicos de `src/vibnet/data/registry.py`.
5. **`data/processed/IMS/` (21.184 arquivos, 3,24 GB) e `data/processed/UOC/` (936 arquivos, 14,74 MB):**
   - Datasets de experimentos antigos (`09/07/2026`) que ocupam ~3,25 GB em `data/processed/` e estão fora do escopo atual de Transfer Learning.

---

## 7. Auditoria dos Manifests e Inconsistências Encontradas

Foram auditados todos os manifests e relatórios existentes no repositório:

1. **[`data/processed_tl/detrended_rms/normalization_manifest.csv`](file:///home/vfrocha/VibNet-1D/data/processed_tl/detrended_rms/normalization_manifest.csv) (`28/10/2026 09:53`):**
   - Registra 9 datasets (`CWRU_12k`, `CWRU_48k`, `PU`, `HUST_Gearbox`, `UOEMD`, `UORED`, `HUST`, `Mechanical_Gear`, `Electric_Motor`).
   - **Inconsistência 1:** Para `HUST`, registra `reference_windows = 150` (pois filtrou `path.parent.name == "Class_Normal"`), mas `input_windows = 1101` (pois normalizou também as 150 duplicatas aninhadas em `Class_Normal/Class_40`).
   - **Inconsistência 2:** `MFPT` está ausente do manifesto na Andromeda.
2. **[`data/processed_tl/validation/rms_statistics.csv`](file:///home/vfrocha/VibNet-1D/data/processed_tl/validation/rms_statistics.csv) e [`data/processed_tl/validation/file_comparison.csv`](file:///home/vfrocha/VibNet-1D/data/processed_tl/validation/file_comparison.csv) (`28/10/2026 09:54`):**
   - **Inconsistência Crítica:** Em `rms_statistics.csv` (linhas 332–334 e 668–670), aparecem **6 linhas de datasets inexistentes chamados `Load_0W`, `Load_200W` e `Load_400W`**, com `condition = Class_Normal` e `class = Class_40`. Isso ocorre porque `grouped_summary()` em [`src/data/validate_dataset_tl.py`](file:///home/vfrocha/VibNet-1D/src/data/validate_dataset_tl.py#L283-L293) assume que caminhos relativos de 4 componentes (`len(parts) >= 4`) têm `parts[0]` como nome do dataset. Quando encontrou `Load_0W/Class_Normal/Class_40/s00008_w00.npy` dentro de `HUST`, deslocou todas as colunas!
   - Em `file_comparison.csv`, os mesmos 150 arquivos foram registrados com `dataset = HUST`, `condition = Class_Normal` e `class = Class_40`.
3. **[`docs/experiment_planning/preprocessing_audit.md`](file:///home/vfrocha/VibNet-1D/docs/experiment_planning/preprocessing_audit.md#L41-L47) vs. [`normalization_manifest.csv`](file:///home/vfrocha/VibNet-1D/data/processed_tl/detrended_rms/normalization_manifest.csv):**
   - Os valores de $\text{RMS}_{\text{ref}}$ citados no texto de `preprocessing_audit.md` (`CWRU_12k = 0,0735`, `UORED = 0,1047`, `HUST = 2,1743`, `PU = 0,2828`, `UOEMD = 0,1066`, `HUST_Gearbox = 0,5887`) **divergem completamente** dos valores reais gravados em `data/processed_tl/detrended_rms/normalization_manifest.csv` e validados em `reference_validation.csv`:
     - `CWRU_12k` / `CWRU_48k`: **`0.06803570025020977`** (280 janelas `Class_0`)
     - `PU`: **`0.27637237210403554`** (1.918 janelas `Class_Normal`)
     - `HUST_Gearbox`: **`0.09268339993814448`** (300 janelas `Class_Normal`)
     - `UOEMD`: **`4.26087249166203`** (160 janelas `Class_Normal`)
     - `UORED`: **`5.850288142457682`** (180 janelas `Class_Normal`)
     - `HUST`: **`0.18114055825864453`** (150 janelas `Class_Normal`)
     - `Mechanical_Gear`: **`0.0052909948776746225`** (30 janelas `Class_Healthy`)
     - `Electric_Motor`: **`1760.7113218645923`** (4 janelas `Class_Healthy`)
4. **[`docs/experiment_planning/dataset_inventory_summary.csv`](file:///home/vfrocha/VibNet-1D/docs/experiment_planning/dataset_inventory_summary.csv) e [`docs/experiment_planning/dataset_audit.md`](file:///home/vfrocha/VibNet-1D/docs/experiment_planning/dataset_audit.md) vs. Disco da Andromeda:**
   - Para `UOEMD`, `dataset_audit.md` descreve `1.560` janelas em `6` condições (`Load_0/1_Const/Linear/Step`). Em disco na Andromeda (tanto em `processed` quanto em `detrended_rms`), `UOEMD` possui **`1.280` janelas em `16` condições** (`Load_{Loaded,No_Load}_Speed_*`).
   - Para `HUST`, `dataset_audit.md` reporta `951` janelas em disco (que é a contagem apenas de nível 3, ignorando as `150` duplicatas aninhadas em nível 4 que elevam o total físico em disco para `1.101`).
5. **[`docs/distribuicao_classes_datasets.csv`](file:///home/vfrocha/VibNet-1D/docs/distribuicao_classes_datasets.csv):**
   - Registra `HUST,Normal,300` (contabilizando as 150 janelas normais + 150 duplicadas em `Class_Normal/Class_40`) e `PU,Normal,3836` (evidenciando que em algum momento anterior `PU` também teve `Class_26` [1.918] + `Class_Normal` [1.918] duplicados simultaneamente).
6. **[`results/setup/installation_report.json`](file:///home/vfrocha/VibNet-1D/results/setup/installation_report.json) e [`results/setup/installation_report.md`](file:///home/vfrocha/VibNet-1D/results/setup/installation_report.md):**
   - Embora estejam versionados no repositório na Andromeda, **não correspondem à máquina Andromeda** (foram gerados na máquina `i9-14900K / RTX A6000` em `17/11/2026`). Além disso, na Andromeda os arquivos `data/raw/download_manifest.json` e `data/processed_tl/processing_manifest.json` não existem.

---

## 8. Verificação dos Scripts de Processamento e Causas Raiz no Código

A inspeção estática dos scripts de preparação e normalização identificou **quatro vulnerabilidades estruturais** que explicam todas as anomalias observadas:

1. **Ausência Total de Limpeza dos Diretórios de Destino (`rmtree` ausente):**
   - [`src/data/make_dataset.py`](file:///home/vfrocha/VibNet-1D/src/data/make_dataset.py#L230-L245), [`src/data/make_dataset_tl.py`](file:///home/vfrocha/VibNet-1D/src/data/make_dataset_tl.py#L253-L265), [`src/preprocessing/normalize_dataset_tl.py`](file:///home/vfrocha/VibNet-1D/src/preprocessing/normalize_dataset_tl.py#L299-L319) e [`scripts/setup/prepare_datasets.py`](file:///home/vfrocha/VibNet-1D/scripts/setup/prepare_datasets.py#L295-L300) utilizam exclusivamente `os.makedirs(final_dir, exist_ok=True)` ou `out_dir.mkdir(parents=True, exist_ok=True)` antes de salvar `np.save(...)`.
   - **Consequência:** Nenhum script limpa arquivos `.npy` ou subpastas preexistentes do dataset/variante antes de reprocessá-lo. Qualquer mudança de nome de classe, mudança de índice `s{i:05d}` ou mudança no número de janelas acumula arquivos novos junto aos antigos.

2. **Armadilha de `mv` em [`scripts/rename_normal.sh`](file:///home/vfrocha/VibNet-1D/scripts/rename_normal.sh#L9-L22):**
   - O script executa `[ -d "$d/Class_40" ] && mv "$d/Class_40" "$d/Class_Normal"`.
   - Em POSIX/bash, quando o diretório de destino `"$d/Class_Normal"` **não existe**, `mv` renomeia `Class_40` para `Class_Normal`. Porém, quando `"$d/Class_Normal"` **já existe** (de uma execução anterior), `mv` move a pasta `Class_40` para **dentro** de `Class_Normal`, criando `"$d/Class_Normal/Class_40"`! Foi exatamente isso que contaminou `HUST` em `06/07/2026`.

3. **Divergência de Nomes de Classes entre `make_dataset.py` / `prepare_datasets.py` e `rename_normal.sh`:**
   - `make_dataset.py` e `prepare_datasets.py` geram os nomes brutos do `vibdata`: `Class_0` (em `CWRU`), `Class_26` (em `PU`), `Class_40` (em `HUST`) e `Class_Healthy` (em `HUST_Gearbox`).
   - Como na Andromeda `rename_normal.sh` havia sido executado manualmente no passado sobre `data/processed/` (renomeando `PU` para `Class_Normal`, `HUST` para `Class_Normal` e `HUST_Gearbox` para `Class_Normal`), quando `prepare_datasets.py` rodou na outra máquina sobre uma cópia de `detrended_rms/`, gravou `Class_26` e `Class_Healthy` ao lado dos `Class_Normal` antigos.

4. **Não-Determinismo do Prefixo `s{i:05d}` entre Máquinas (`os.walk` / `os.listdir` no `vibdata`) + Reuso Superficial em `prepare_datasets.py`:**
   - O nome de cada janela é `f"s{i:05d}_w{w_idx:02d}.npy"`, onde `i` é o índice de iteração do objeto bruto `ds[i]`. Para datasets cujos adaptadores no `vibdata` varrem diretórios sem ordenação lexicográfica explícita (`UOEMD`, `HUST_Gearbox`, `Mechanical_Gear`, `Electric_Motor`), a ordem `i` depende da ordem física de inodes do sistema de arquivos (`ext4`).
   - Quando os arquivos brutos são extraídos em outra máquina, o mesmo arquivo `.mat` recebe outro índice `i`, gerando outro nome `sYYYYY_wYY.npy`. Sem limpeza prévia do diretório de destino, reprocessar o dataset duplica quase todas as janelas.
   - Adicionalmente, `inspect_existing_processed_variant()` em [`scripts/setup/prepare_datasets.py`](file:///home/vfrocha/VibNet-1D/scripts/setup/prepare_datasets.py#L139-L172) inspeciona apenas 3 arquivos (`índice 0, meio e fim`) quanto a `size == expected_window_len`. Por isso, não detectou que `data/processed/MFPT` tinha 114 janelas não harmonizadas (pois cada janela tinha 48.828 amostras, embora 72 delas correspondessem a 0,5 s a 97.656 Hz) nem que `HUST` tinha 150 arquivos aninhados em `Class_Normal/Class_40`.

---

## 9. Recomendações para Reconstrução e Saneamento Seguro dos Dados

Para garantir reprodutibilidade estrita, ausência de duplicatas residuais e consistência entre `data/processed/`, `data/processed_tl/` e os manifests, recomenda-se o seguinte plano de saneamento (a ser executado somente após aprovação explícita do pesquisador):

### 9.1. Saneamento Imediato e Cirúrgico na Máquina Andromeda (Caso Não Reprocesse Tudo do Bruto)

Se o objetivo na Andromeda for manter os datasets atuais sem precisar baixar novamente todos os `.zip` brutos em `data/raw/`:
1. **Remover as 150 duplicatas aninhadas de `HUST` em ambas as variantes:**
   - Remover exclusivamente as 3 subpastas anômalas de nível 4 em `data/processed/HUST/Load_{0W,200W,400W}/Class_Normal/Class_40/` (150 arquivos) e em `data/processed_tl/detrended_rms/HUST/Load_{0W,200W,400W}/Class_Normal/Class_40/` (150 arquivos), restando exatamente as **951 janelas legítimas** (sendo 150 em `Class_Normal`).
   - Nota: Como `normalize_dataset_tl.py` já havia calculado `RMS_ref = 0.18114055825864453` usando apenas as 150 janelas de `Class_Normal` (ignorando `Class_40` no cálculo da referência), os valores numéricos das 951 janelas restantes em `detrended_rms/HUST` **já estão matematicamente corretos**; basta atualizar `input_windows` de `1101` para `951` no manifesto e reexecutar `validate_dataset_tl.py`.
2. **Remover o arquivo de swap residual:**
   - Remover `data/.dataloader.py.swp`.
3. **Harmonizar e incluir o `MFPT` (78 janelas):**
   - Baixar/disponibilizar `MFPT_raw` em `data/raw/`, limpar `data/processed/MFPT/` (removendo as 114 janelas antigas não decimadas) e processar `MFPT` com `harmonize_mfpt_sample_rate()` + `normalize_dataset_tl.py`, gerando as **78 janelas exatas de 1,0 s** em `data/processed/MFPT` e `data/processed_tl/detrended_rms/MFPT`.

### 9.2. Correções Preventivas Obrigatórias nos Scripts Antes de Qualquer Reprocessamento (`install_linux.sh` / `prepare_datasets.py`)

Tanto na Andromeda quanto na outra máquina (`i9-14900K / RTX A6000`), **antes** de rodar `prepare_datasets.py` ou `normalize_dataset_tl.py`:
1. **Limpeza atômica por `(dataset, variant)` antes de gravar:**
   - Em `prepare_datasets.py` (`generate_base_windows_for_raw_source`) e em `normalize_dataset_tl.py` (`normalize_dataset`), quando um dataset/variante for (re)gerado, remover previamente o diretório de saída daquele dataset (`shutil.rmtree(dataset_output_dir, ignore_errors=True)`) antes de gravar as novas janelas. Isso impede 100% do acúmulo de arquivos residuais (`12.099` em `PU`, `2.560` em `UOEMD`, `1.780` em `HUST_Gearbox`, `600` em `UORED`, `360` em `Mechanical_Gear`, `130` em `Electric_Motor`).
2. **Descontinuar `scripts/rename_normal.sh` e padronizar nomes de classe diretamente no pipeline Python:**
   - Nunca executar `mv "$d/Class_X" "$d/Class_Normal"` via shell sobre diretórios existentes. A resolução de aliases saudáveis já é suportada nativamente por `discover_normal_class()` em `normalize_dataset_tl.py`.
3. **Na máquina secundária (`i9-14900K / RTX A6000`):**
   - Como `data/processed_tl/detrended_rms/` naquela máquina está contaminado pela fusão dos arquivos copiados da Andromeda com os arquivos recém-gerados, deve-se **apagar `data/processed_tl/detrended_rms/`** naquela máquina e reexecutar `python scripts/preprocessing/normalize_dataset_tl.py` (ou `prepare_datasets.py --variants detrended_rms --force`) a partir de `data/processed_tl/detrended/` limpo, seguido de `python scripts/setup/validate_installation.py` para atualizar `results/setup/installation_report.{json,md}`.

