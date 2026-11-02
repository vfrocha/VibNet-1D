# Datasets, Pré-processamento e Protocolos de Avaliação — VibNet-1D

Este documento cataloga os 10 conjuntos de dados de vibração utilizados no projeto **VibNet-1D**, as transformações de pré-processamento no domínio do tempo (`data/processed_tl/`) e os protocolos de avaliação adotados nos experimentos **TabVibNet**.

---

## 1. Catálogo dos Datasets

Todos os sinais são segmentados em **janelas não sobrepostas de 1 segundo** ($L = f_s$ amostras por janela). Cada dataset preserva seu **vocabulário de classes nativo** e suas condições operacionais originais (`Condition`), evitando mapeamentos artificiais entre falhas de naturezas mecânicas distintas.

| Dataset | Componente Principal | Papel Principal nos Experimentos TL | Taxa de Amostragem ($f_s$) | Unidade Física | Total de Janelas | Nº de Classes | Nº de Condições (`LOCO` Folds) | Classe Saudável de Referência (`RMS_ref`) |
|---|---|---|---:|---|---:|---:|---:|---|
| **`CWRU_12k`** | Rolamento | Fonte (`Source`) / Baseline LOCO | 12.000 Hz | Aceleração ($g$) | 1.981 | 4 | 4 | `Class_0` (Normal, 141 janelas, $\text{RMS}_{\text{ref}} = 0{,}067780$) |
| **`CWRU_48k`** | Rolamento | Alvo (`Target`) / Baseline LOCO | 48.000 Hz | Aceleração ($g$) | 626 | 3 | 12 | **Ausente no próprio dataset**; utiliza `Class_0` do `CWRU_12k` ($\text{RMS}_{\text{ref}} = 0{,}067780$) |
| **`PU`** | Rolamento | Alvo (`Target`) / Baseline LOCO | 64.000 Hz | Aceleração ($m/s^2$) | 12.785 | 6 | 4 | `K00` (Normal, 2.397 janelas, $\text{RMS}_{\text{ref}} = 0{,}272094$) |
| **`UOEMD`** | Rolamento + Motor | Alvo (`Target`) / Baseline LOCO | 42.000 Hz | Aceleração ($g$ / V) | 1.600 | 5 | 16 | `H` (Healthy, 320 janelas, $\text{RMS}_{\text{ref}} = 0{,}019937$) |
| **`HUST_Gearbox`** | Caixa de Engrenagens | Alvo (`Target`) / Baseline LOCO | 25.600 Hz | Aceleração ($g$) | 720 | 4 | 30 | `Normal` (180 janelas, $\text{RMS}_{\text{ref}} = 0{,}635542$) |
| **`HUST`** | Rolamento | Fonte (`Source`) | 51.200 Hz | Aceleração ($m/s^2$) | 1.012 | 7 | 4 (ou 1*) | `N` (Normal, 144 janelas, $\text{RMS}_{\text{ref}} = 2{,}173862$) |
| **`MFPT`** | Rolamento | Fonte (`Source`) | 48.828 Hz | Aceleração ($g$) | 126 | 3 | 1 | `Normal` (18 janelas, $\text{RMS}_{\text{ref}} = 1{,}012798$) |
| **`UORED`** | Rolamento | Fonte (`Source`) | 42.000 Hz | Aceleração ($m/s^2$) | 420 | 5 | 1 | `H` (Healthy, 100 janelas, $\text{RMS}_{\text{ref}} = 0{,}104741$) |
| **`Mechanical_Gear`** | Engrenagem | Fonte Auxiliar (Geração 2) | 5.000 Hz (ou 10.000 Hz**) | Aceleração | 16.017 | 4 | 1 | `normal` (5.052 janelas, $\text{RMS}_{\text{ref}} = 0{,}873096$) |
| **`Electric_Motor`** | Motor + Rolamento | Fonte Auxiliar (Geração 2) | 50.000 Hz (ou 42.000 Hz**) | Aceleração | 33.600 | 3 | 1 | `normal` (3.200 janelas, $\text{RMS}_{\text{ref}} = 2{,}096993$) |

> **\*Nota sobre condições em `HUST`**: No script original `src/experiments/old_experiments/run_hust.py`, as 4 cargas (`20W`, `40W`, `60W`, `80W`) eram extraídas do nome do arquivo para avaliação LOCO. Já na ingestão unificada (`src/data/make_dataset_tl.py`), `HUST`, `MFPT`, `UORED`, `Mechanical_Gear` e `Electric_Motor` têm `Condition = "Default"` (1 condição única) porque atuam exclusivamente como **domínios fonte** (cujos dados são usados integralmente no pré-treinamento).
>
> **\*\*Divergência documentada de taxa de amostragem ($f_s$)**: Em `src/data/make_dataset.py`, `src/data/make_dataset_tl.py`, `docs/dataset_mapping.md` e `src/experiments/*.py`, os datasets `Mechanical_Gear` e `Electric_Motor` são processados com $f_s = 5.000\text{ Hz}$ e $f_s = 50.000\text{ Hz}$, respectivamente (confirmado pelo comprimento das janelas auditadas em `results/dataset_validation_tl/dataset_condition_class_summary.csv`: `5.000` e `50.000` amostras). Contudo, no dicionário `DATASETS_CONFIG` de `scripts/exp_tl_cross_dataset_mlp_tabnet.py` (linhas 79–80), constam os valores `10000` para `Mechanical_Gear` e `42000` para `Electric_Motor`. Como `Mechanical_Gear` e `Electric_Motor` foram utilizados como domínios fonte nos runs `run_20261028_105603` e `run_20261028_173310`, essa divergência reescalona linearmente as 5 features espectrais FFT (`extract_freq_features` em `extractors_v2.py`) dessas duas fontes durante a extração de atributos daquele script.

---

## 2. Vocabulário de Classes e Condições Operacionais por Dataset

Os dados auditados em `docs/distribuicao_classes_datasets.csv` e `results/dataset_validation_tl/dataset_condition_class_summary.csv` estabelecem a distribuição exata por dataset:

1. **`CWRU_12k`** (1.981 janelas, 4 classes, 4 condições de carga):
   - **Classes**: `Class_0` (Normal: 141), `Class_1` (Inner Race: 604), `Class_2` (Ball: 608), `Class_3` (Outer Race: 628).
   - **Condições (`Condition`)**: `Load_0HP` (395), `Load_1HP` (528), `Load_2HP` (529), `Load_3HP` (529).
2. **`CWRU_48k`** (626 janelas, 3 classes, 12 condições Carga $\times$ Severidade):
   - **Classes**: `Class_1` (Inner Race: 211), `Class_2` (Ball: 213), `Class_3` (Outer Race: 202). **Não possui `Class_0` (Normal)** nos arquivos de 48 kHz processados.
   - **Condições (`Condition`)**: 12 combinações `Load_{0,1,2,3}HP_Sev_{0.007,0.014,0.021}`.
   - **Particularidade crítica no fold `Load_0HP_Sev_0.021`**: Possui 32 janelas contendo apenas **2 das 3 classes** (`Class_1`: 11, `Class_2`: 21; **`Class_3` possui 0 amostras**). Isso torna o cálculo padrão de `ROC-AUC` multiclasse *One-vs-Rest* indefinido nesse fold (`N_Test_Classes_Present = 2`).
3. **`PU` — Paderborn University** (12.785 janelas, 6 classes, 4 condições operacionais):
   - **Classes**: `K00` (Saudável: 2.397), `KA0` (Dano externo artificial: 3.197), `KI0` (Dano interno artificial: 2.796), `KA1`–`KA3` (Dano externo real: 1.198), `KI1`–`KI2` (Dano interno real: 2.000), `KB2` (Dano múltiplo: 1.197).
   - **Condições (`Condition`)**:
     - `C1_1500rpm_0.7Nm_1000N` (3.200 janelas)
     - `C2_900rpm_0.7Nm_1000N` (3.194 janelas)
     - `C3_1500rpm_0.1Nm_1000N` (3.194 janelas)
     - `C4_1500rpm_0.7Nm_400N` (3.197 janelas; quando `C4` é o fold de teste, o treino possui $12.785 - 3.197 = 9.588$ janelas distribuídas com classe majoritária de 2.397 amostras em `KA0` $\rightarrow$ após SMOTE de 6 classes, $6 \times 1.429{,}5 \rightarrow 8.577$ amostras nos subconjuntos ou contagens que deixam resto 1 na divisão por `batch_size=64`).
4. **`UOEMD` — University of Ottawa Electric Motor & Bearing** (1.600 janelas, 5 classes, 16 condições):
   - **Classes** ( perfeitamente balanceadas: 320 janelas por classe, 20 por condição): `H` (Healthy), `I` (Inner Race), `O` (Outer Race), `B` (Ball), `C` (Cage).
   - **Condições (`Condition`)**: 16 combinações de perfil de velocidade e carga (`Condition_01` a `Condition_16` / perfis de rotação constantes e variáveis com carga).
5. **`HUST_Gearbox`** (720 janelas, 4 classes, 30 condições):
   - **Classes** (180 janelas por classe, 6 por condição): `Normal`, `Broken` (dente quebrado), `Miss` (dente faltante), `Root` (trinca na raiz).
   - **Condições (`Condition`)**: 30 condições operacionais variando frequência de rotação e carga (`C01` a `C30`).
6. **Domínios Fonte Dedicados (`Condition = "Default"`)**:
   - **`HUST`** (1.012 janelas, 7 classes): `N` (144), `I` (145), `O` (145), `B` (144), `IO` (145), `IB` (144), `OB` (145).
   - **`MFPT`** (126 janelas, 3 classes): `Normal` (18), `Inner_Race` (63), `Outer_Race` (45).
   - **`UORED`** (420 janelas, 5 classes): `H` (100), `I` (80), `O` (80), `B` (80), `C` (80).
   - **`Mechanical_Gear`** (16.017 janelas, 4 classes): `normal` (5.052), `eccentricity` (4.060), `missing_tooth` (3.249), `root_crack` (3.656).
   - **`Electric_Motor`** (33.600 janelas, 3 classes): `normal` (3.200), `stator_short` (9.600), `bearing_fault` (20.800).

---

## 3. Variantes de Pré-processamento de Sinal (`data/processed_tl/`)

Para investigar de forma isolada o efeito da remoção de tendência linear (*detrending*) e da normalização de amplitude por condição saudável de referência, o pipeline `src/data/make_dataset_tl.py` + `src/preprocessing/normalize_dataset_tl.py` gera quatro variantes paralelas dos 10 datasets em formato Parquet:

| Variante | Diretório | Remoção de Tendência (`scipy.signal.detrend(w, type='linear')`) | Normalização por RMS Saudável de Referência ($w / \text{RMS}_{\text{ref}}$) |
|---|---|:---:|:---:|
| **`raw`** | `data/processed_tl/raw/` | Não | Não |
| **`detrended`** | `data/processed_tl/detrended/` | **Sim** | Não |
| **`raw_rms`** | `data/processed_tl/raw_rms/` | Não | **Sim** ($\text{RMS}_{\text{ref}}$ calculado sobre `raw`) |
| **`detrended_rms`** | `data/processed_tl/detrended_rms/` | **Sim** | **Sim** ($\text{RMS}_{\text{ref}}$ calculado sobre `detrended`) |

### Definição Matemática da Normalização por Referência Saudável (`RMS_ref`)

Seja $\mathcal{W}_0^{(d)}$ o conjunto de todas as janelas pertencentes à classe saudável (*Normal/Healthy*) do dataset $d$. O RMS de cada janela $w = (w_1, \dots, w_L)$ é definido como:

$$\text{RMS}(w) = \sqrt{\frac{1}{L}\sum_{t=1}^{L} w_t^2}$$

O fator de referência do dataset $d$ é a média dos valores de RMS das janelas saudáveis:

$$\text{RMS}_{\text{ref}}^{(d)} = \frac{1}{|\mathcal{W}_0^{(d)}|} \sum_{w \in \mathcal{W}_0^{(d)}} \text{RMS}(w)$$

Todas as janelas $w$ (saudáveis e com falha) do dataset $d$ são então divididas pelo escalar constante $\text{RMS}_{\text{ref}}^{(d)}$:

$$\tilde{w} = \frac{w}{\text{RMS}_{\text{ref}}^{(d)}}$$

- **Propriedade física fundamental**: Como $\text{RMS}_{\text{ref}}^{(d)}$ é constante para todo o dataset $d$, as janelas saudáveis passam a ter $\mathbb{E}[\text{RMS}(\tilde{w})] = 1{,}0$, enquanto as janelas com falha preservam sua razão energética relativa ($\text{RMS}(\tilde{w}) > 1{,}0$ quando a falha amplifica a vibração), ao mesmo tempo em que as escalas absolutas de diferentes sensores/bancadas ($g$, $m/s^2$, V) são alinhadas em torno da unidade saudável.
- **Exceção documentada (`CWRU_48k`)**: Como `CWRU_48k` não possui amostras da classe `Class_0` nos dados processados, ele utiliza $\text{RMS}_{\text{ref}}^{(\text{CWRU\_12k})}$ (mesma bancada física CWRU, acelerômetro no *Drive End*).

---

## 4. Protocolos de Avaliação

> **Regra Científica Fundamental (`AGENTS.md`)**: Resultados obtidos sob protocolos distintos avaliam hipóteses diferentes e **nunca devem ser combinados ou comparados diretamente sem explicitar suas diferenças**.

### 4.1. Protocolo Primário: `LOCO` (*Leave-One-Condition-Out*)
- **Aplicação**: Experimentos principais de *baseline* (`baseline_loco_results_*`) e todos os experimentos de Transferência de Aprendizado da Geração 3 (`tl_four_variants`, `tl_aux_features`, `tl_balance_strategies`, `tl_contrastive_mlp`, `tl_cross_dataset_mlp_tabnet`).
- **Particionamento**: Para um dataset alvo com $K$ condições operacionais distintas ($K=12$ em `CWRU_48k`, $K=4$ em `PU`, $K=16$ em `UOEMD`, $K=30$ em `HUST_Gearbox`, $K=4$ em `CWRU_12k`), realizam-se $K$ folds. No fold $k$:
  - **Teste**: 100% das janelas da condição $c_k$.
  - **Treino do Alvo**: Todas as janelas das demais $K-1$ condições operacionais do dataset alvo.
- **Regras de Isolamento (Prevenção de Vazamento)**:
  1. Os dados dos domínios fonte (`Sources`) são utilizados exclusivamente na etapa de pré-treinamento do *encoder* compartilhado (com seu próprio `StandardScaler` ajustado por domínio fonte ou no conjunto de pré-treinamento).
  2. Na etapa de *fine-tuning* no domínio alvo, o `StandardScaler` é ajustado (`fit_transform`) **exclusivamente sobre as $K-1$ condições de treino do alvo**; o fold de teste $c_k$ recebe apenas `transform`.
  3. Qualquer estratégia de balanceamento (`SMOTE`, `hybrid`, `class_weights`) é aplicada **apenas sobre as amostras de treino do alvo** após o isolamento do fold de teste $c_k$.
  4. Nenhum *checkpoint* ou hiperparâmetro é selecionado monitorando o fold de teste $c_k$.

### 4.2. Protocolo `Severity LOCO` (Generalização entre Severidades de Dano)
- **Aplicação**: `run_cwru12k_severity.py`, `run_cwru48k_severity.py`, `run_unified_severity_baselines.py` (`cwru12k_severity_loco_results_*`, `cwru48k_severity_loco_results_*`, `cwru_unified_severity_loco_*`).
- **Particionamento**: Mantém a condição de severidade de defeito (`0.007"`, `0.014"`, `0.021"` ou `0.028"`) isolada no teste, treinando nas demais severidades (e repetindo amostras saudáveis quando aplicável em `CWRU_12k`). Avalia se o classificador reconhece o tipo de falha (`Inner`, `Ball`, `Outer`) em um tamanho de defeito inédito.

### 4.3. Protocolo `MERT` em `CWRU_48k`
- **Aplicação**: `run_cwru48k_mert_protocol.py` (`cwru48k_mert_comparison_20261012_062814.csv`).
- **Particionamento**: Agrupa as 12 condições de `CWRU_48k` pelas 4 cargas mecânicas (`Load_0HP`, `Load_1HP`, `Load_2HP`, `Load_3HP`), misturando as severidades dentro de cada carga (4 folds de carga em vez de 12 folds Carga $\times$ Severidade). Não é diretamente comparável ao LOCO de 12 folds de `CWRU_48k`.

### 4.4. Protocolo `One-Class Anomaly Detection`
- **Aplicação**: `run_anomaly_experiments.py` (`anomaly_results_20260812_084311.csv`).
- **Particionamento**: Em cada fold LOCO, o modelo (`Isolation Forest`, `One-Class SVM`, `Autoencoder`) é treinado **exclusivamente com amostras da classe saudável (`Normal`)** das condições de treino, e avaliado na tarefa binária (`Normal = 0` vs. `Anomalia = 1`) na condição de teste.

### 4.5. Protocolo `5-Fold Cross-Validation` Estratificado (Geração 1)
- **Aplicação**: Primeiros experimentos exploratórios em `src/experiments/old_experiments/exp_01` a `exp_05` (`resultados_exp01_*` a `exp05_*`).
- **Limitação**: Como janelas adjacentes da mesma condição operacional caem simultaneamente no treino e no teste, atinge métricas próximas de `1.0000` por correlação intra-condição; foi substituído pelo protocolo `LOCO` a partir da Geração 2.

---

## 5. Métricas de Avaliação

1. **Métrica Primária — `Macro-F1`**:
   $$\text{Macro-F1} = \frac{1}{|\mathcal{C}_{\text{present}}|} \sum_{c \in \mathcal{C}_{\text{present}}} \frac{2 P_c R_c}{P_c + R_c}$$
   Calculada com `f1_score(y_true, y_pred, average='macro', zero_division=0)`. Nos runners mais recentes (`build_mlp_contrastive.py`, `exp_tl_cross_dataset_mlp_tabnet.py`), é restrita explicitamente a `labels=present_classes` do fold de teste para evitar penalização espúria quando um fold de teste não contém todas as classes do dataset (como `Load_0HP_Sev_0.021` em `CWRU_48k`).
2. **Métricas Secundárias**:
   - **`Balanced Accuracy` (`Bal Acc`)**: Média dos *recalls* das classes presentes no fold de teste (`balanced_accuracy_score`).
   - **`Accuracy` (`ACC`)**: Fração global de acertos no fold de teste (`accuracy_score`).
   - **`ROC-AUC` (`roc_auc_ovr_macro`)**: Área sob a curva ROC multiclasse *One-vs-Rest* com média macro (`roc_auc_score(y_true, y_prob, multi_class='ovr', average='macro')`). Requer que todas as classes do modelo estejam presentes em `y_true`; caso contrário, é reportada como `NaN` com `ROC_AUC_Valid = False` (ou `0.0` nos runners legados, devidamente sinalizado pela auditoria como `ROC_AUC_ZERO_FALLBACK`).
   - **Critério composto de seleção multi-modelo** (em `summary.csv` de `tl_four_variants`, `tl_aux_features` e `tl_balance_strategies`):
     $$\text{Macro\_F1\_Selection\_Score} = \overline{\text{Macro-F1}}_{\text{modelos}} - 0{,}5 \cdot s_{\text{modelos}}(\text{Macro-F1})$$
     que recompensa configurações com alto `Macro-F1` médio e baixa divergência entre `MLP` e `TABNET`.
