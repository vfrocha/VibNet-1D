# Revisão Metodológica e Protocolo Experimental da Matriz DA, DG e TL — VibNet-1D

- **Data:** 18/11/2026
- **Ambiente Auditado:** Máquina `andromeda` (`/home/vfrocha/VibNet-1D`, branch `main`)
- **Escopo:** Atualização da especificação metodológica, implementação de cabeças de classificação com espaço de classes variável (`S1`: 2, `S2`: 5, `S3`: 2, `S4`: 5) e expansão por nome canônico no DA, auditoria de checkpoints e pré-treinamento TabNet, auditoria de unidades físicas e normalização geométrica em $\text{m/s}^2$ (`1 g = 9.80665 m/s²`), identificação matemática das 126 features invariantes a escala e 15 sensíveis, preparação e decimação isolada do `MFPT` (`97.656 Hz -> 48.828 Hz`, 78 janelas de $1{,}0\text{ s}$), diagnóstico de calibração externa `CWRU_12k -> CWRU_48k`, geração dos manifestos de partições por gravação, baselines clássicos (`RF` e `SVM`) e validação por 17 testes obrigatórios.

---

## 1. Configurações de Origem, Destino e Espaço de Classes Variável das Cabeças (Seções 2 e 3)

### 1.1. Fontes e Destinos
- **Fontes (`UOC` explicitamente excluído):** `UORED`, `HUST` (*HUST Bearing*, distinto de `HUST_Gearbox`), `MFPT`.
- **Configurações de Origem:**
  - **`S1`**: `UORED`
  - **`S2`**: `HUST` (*HUST Bearing*)
  - **`S3`**: `MFPT`
  - **`S4`**: `UORED + HUST + MFPT`
- **Destinos de DA e DG (3 destinos de rolamentos):** `CWRU_12k`, `CWRU_48k`, `PU`.
- **Destinos de TL (5 destinos):** `CWRU_12k`, `CWRU_48k`, `PU`, `HUST_Gearbox`, `UOEMD`.

### 1.2. Correção Fundamental do Espaço de Classes das Cabeças (`VariableClassHeadCheckpoint`)
O vocabulário canônico harmonizado de falhas de rolamentos compreende 5 classes conceituais (agrupando severidades de uma mesma falha e mantendo `Inner & Outer` como classe composta independente):
1. `Normal` (`id=0`)
2. `Ball` (`id=1`)
3. `Inner` (`id=2`)
4. `Outer` (`id=3`)
5. `Inner & Outer` (`id=4`)

**Não são impostas 5 saídas a todas as cabeças iniciais.** Cada cabeça inicial possui exatamente a dimensão correspondente às classes disponíveis na sua configuração de origem:

| Configuração | Fontes | Classes Supervisionadas Disponíveis na Cabeça Inicial | Dimensão da Cabeça Inicial (`out_features`) | Mapeamento Local `índice -> classe` |
| :---: | :--- | :--- | :---: | :--- |
| **`S1`** | `UORED` | `Normal`, `Ball` | **2** | `{0: "Normal", 1: "Ball"}` |
| **`S2`** | `HUST` *(HUST Bearing)* | `Normal`, `Ball`, `Inner`, `Outer`, `Inner & Outer` | **5** | `{0: "Normal", 1: "Ball", 2: "Inner", 3: "Outer", 4: "Inner & Outer"}` |
| **`S3`** | `MFPT` | `Inner`, `Outer` *(`Class_23 / Normal` usado para $\text{RMS}_{\text{ref}}$)* | **2** | `{0: "Inner", 1: "Outer"}` |
| **`S4`** | `UORED + HUST + MFPT` | `Normal`, `Ball`, `Inner`, `Outer`, `Inner & Outer` | **5** | `{0: "Normal", 1: "Ball", 2: "Inner", 3: "Outer", 4: "Inner & Outer"}` |

### 1.3. Comportamento das Cabeças em DG vs. DA

#### A. Domain Generalization (DG — Seção 3.1)
- O encoder pré-treinado (`TabNetPretrainer.network.embedder` + `encoder`) permanece **100% congelado** (`requires_grad=False`, `.eval()`).
- A cabeça de classificação é treinada exclusivamente nas fontes (`S1`: 2 saídas; `S2`: 5 saídas; `S3`: 2 saídas; `S4`: 5 saídas) e **não sofre expansão nem adaptação com dados do destino**.
- Na avaliação nos destinos (`100%` de `CWRU_12k`, `CWRU_48k` e `PU`), **nenhuma amostra de classe desconhecida pela origem é removida do teste**. Se a classe verdadeira do destino não existir na cabeça da origem (ex.: `Inner` e `Outer` ao avaliar `S1` em `CWRU_12k`), a predição é necessariamente incorreta ($\text{TP}_c = 0 \implies \text{Recall}_c = 0{,}0$ e $\text{F1}_c = 0{,}0$), penalizando rigorosamente `Macro-F1`, `Balanced Accuracy`, `Accuracy` e a matriz de confusão.

#### B. Domain Adaptation (DA — Seção 3.2)
Implementado em [`expand_head_for_da_target()`](file:///home/vfrocha/VibNet-1D/src/vibnet/evaluation/da_dg_tl_protocol.py) e [`train_variable_class_head_with_frozen_tabnet_encoder()`](file:///home/vfrocha/VibNet-1D/src/vibnet/evaluation/da_dg_tl_protocol.py):
1. Treina-se a cabeça inicial nas classes disponíveis na configuração de origem (`S1`: 2, `S2`: 5, `S3`: 2, `S4`: 5).
2. Carregam-se o encoder pré-treinado (mantido **100% congelado**) e a cabeça inicial da fonte.
3. Para cada destino, `expand_head_for_da_target()` constrói uma cabeça específica cujo espaço de saída é a união canônica $\mathcal{C}_{\text{exp}} = \mathcal{C}_{\text{source}} \cup \mathcal{C}_{\text{target\_adapt}}$:
   - **Correspondência estrita por nome canônico** (nunca por posição presumida): por exemplo, ao expandir `S3` (`{0: "Inner", 1: "Outer"}`) para `CWRU_12k` (`Normal, Ball, Inner, Outer`), a nova cabeça de 4 saídas tem ordem canônica `{0: "Normal", 1: "Ball", 2: "Inner", 3: "Outer"}`; os pesos e *biases* de `"Inner"` (que estavam na linha `0` de `S3`) são copiados bit a bit para a linha `2`, e os de `"Outer"` (linha `1` de `S3`) são copiados bit a bit para a linha `3`.
   - **Inicialização reproduzível das novas saídas:** Cada nova classe $c \in \mathcal{C}_{\text{target\_adapt}} \setminus \mathcal{C}_{\text{source}}$ recebe pesos e *bias* inicializados por um `torch.Generator` semeado deterministicamente com `SHA-256(f"DA_EXPAND::{seed}::{class_name}")`.
4. Adapta-se **somente a cabeça expandida** utilizando os 20% rotulados do destino (`adapt`) pelo número de épocas selecionado na validação das fontes (`selected_epoch`), avaliando nos 80% restantes (`test`).
5. O checkpoint registra separadamente:
   - `transferred_from_source_classes`: classes transferidas da origem;
   - `introduced_by_expansion_classes`: classes novas introduzidas pela expansão da cabeça;
   - `learned_during_adaptation_classes`: classes efetivamente ajustadas com exemplos na partição de adaptação do destino.

| Origem | Destino DA | Classes na Cabeça Inicial | Classes no Destino | Cabeça Expandida (`dim`) | Classes Transferidas da Origem | Classes Introduzidas na Expansão |
| :---: | :--- | :--- | :--- | :--- | :--- | :--- |
| **`S1`** | `CWRU_12k` | `Normal, Ball` (2) | `Normal, Ball, Inner, Outer` (4) | `Normal, Ball, Inner, Outer` (**4**) | `Normal, Ball` | `Inner, Outer` |
| **`S1`** | `CWRU_48k` | `Normal, Ball` (2) | `Ball, Inner, Outer` (3) | `Normal, Ball, Inner, Outer` (**4**) | `Normal, Ball` | `Inner, Outer` |
| **`S1`** | `PU` | `Normal, Ball` (2) | `Normal, Outer, Inner & Outer` (3) | `Normal, Ball, Outer, Inner & Outer` (**4**) | `Normal, Ball` | `Outer, Inner & Outer` |
| **`S3`** | `CWRU_12k` | `Inner, Outer` (2) | `Normal, Ball, Inner, Outer` (4) | `Normal, Ball, Inner, Outer` (**4**) | `Inner, Outer` | `Normal, Ball` |
| **`S3`** | `CWRU_48k` | `Inner, Outer` (2) | `Ball, Inner, Outer` (3) | `Ball, Inner, Outer` (**3**) | `Inner, Outer` | `Ball` |
| **`S3`** | `PU` | `Inner, Outer` (2) | `Normal, Outer, Inner & Outer` (3) | `Normal, Inner, Outer, Inner & Outer` (**4**) | `Inner, Outer` | `Normal, Inner & Outer` |
| **`S2` / `S4`** | Todos (`CWRU_12k/48k`, `PU`) | Todas as 5 classes (5) | Subconjunto de 3 ou 4 classes | Todas as 5 classes (**5**) | Todas as 5 classes | Nenhuma (`0`) |

---

## 2. Auditoria do Encoder TabNet e do Pré-Treinamento (Seção 4)

1. **Checkpoints em Disco:** A varredura completa do repositório confirmou que **não existem arquivos binários `.pt`, `.pth` ou `.zip` de modelos TabNet pré-treinados salvos anteriormente em disco**. Nos benchmarks anteriores ([`src/experiments/run_scratch_vs_tl_benchmark.py`](file:///home/vfrocha/VibNet-1D/src/experiments/run_scratch_vs_tl_benchmark.py)), o `TabNetPretrainer` era treinado em memória RAM no início de cada execução e mantido em `_PRETRAINER_CACHE`.
2. **Natureza e Procedimento do Pré-Treinamento:**
   - **Auto-supervisionado (*self-supervised masked feature reconstruction*):** `pytorch_tabnet.pretraining.TabNetPretrainer` com `pretraining_ratio = 0.8`, sem uso de rótulos de classe.
   - **Arquitetura Canônica:** `n_d=16`, `n_a=16`, `n_steps=4`, `gamma=1.3`, `n_independent=2`, `n_shared=2`, `mask_type="sparsemax"`, `lr=0.02`, `weight_decay=1e-5`, `pretrain_epochs=30`, `batch_size=64`, dimensão latente $d_{\text{latent}} = n_d = 16$ (`latent = sum(steps_out)`).
3. **Datasets Utilizados e Ausência Total de Exposição aos Destinos:**
   - Nos experimentos anteriores (`configs/scratch_vs_tl.yaml`), o pool de pré-treinamento era `["UORED", "HUST", "Mechanical_Gear", "Electric_Motor"]`. **Nenhum dos 5 datasets de destino (`CWRU_12k`, `CWRU_48k`, `PU`, `HUST_Gearbox`, `UOEMD`) jamais participou do pré-treinamento**.
   - Na matriz atual (`configs/da_dg_tl_matrix.yaml`), o `TabNetPretrainer` é ajustado exclusivamente sobre os dados de treinamento das fontes ativas de cada configuração (`S1`: `UORED`; `S2`: `HUST`; `S3`: `MFPT`; `S4`: `UORED + HUST + MFPT`).
4. **Compatibilidade com a Cabeça Customizada:**
   - O `TabNetPretrainer.network` expõe `embedder` (`EmbeddingGenerator`) e `encoder` (`TabNetEncoder`), cuja saída `steps_out, _ = encoder(embedder(x))` consiste em $n_{\text{steps}} = 4$ tensores de dimensão `(batch_size, n_d=16)`. A soma `latent = torch.sum(torch.stack(steps_out, dim=0), dim=0)` é idêntica à representação interna que o `TabNetClassifier` entrega à sua camada `final_mapping = nn.Linear(n_d, n_classes)`. Portanto, acoplar uma cabeça `nn.Linear(16, K)` sobre `latent` é **100% compatível arquiteturalmente** com o encoder do `TabNetPretrainer` sem qualquer alteração no backbone.
5. **Preservação do Protocolo TL (Seção 11):**
   - Em **TL**, o encoder **não** é assumido como sempre congelado: preservamos integralmente os três modos auditados de [`src/experiments/run_scratch_vs_tl_benchmark.py`](file:///home/vfrocha/VibNet-1D/src/experiments/run_scratch_vs_tl_benchmark.py) sobre os 66 folds LOCO dos 5 destinos (`CWRU_12k` [4], `CWRU_48k` [12], `PU` [4], `HUST_Gearbox` [30], `UOEMD` [16]):
     - **`Scratch`**: `TabNetClassifier` inicializado do zero (`from_unsupervised=None`), encoder e cabeça livres;
     - **`Fine-Tuning`**: `TabNetClassifier` inicializado com `from_unsupervised=pretrainer`, **encoder descongelado** e ajustado junto com a cabeça nativa do destino nas $K-1$ condições de treino;
     - **`Linear Probing`**: encoder pré-treinado **congelado** (`backbone_param_max_abs_diff == 0.0`), treinando apenas a cabeça linear no vocabulário nativo do destino.

---

## 3. Auditoria das Referências RMS, Conversão para $\text{m/s}^2$ e Normalização DG/DA (Seções 5 e 6)

### 3.1. Conversão de Unidades Físicas ($1\,g = 9{,}80665\,\text{m/s}^2$) e Média Geométrica em DG
As referências RMS das classes normais das fontes (tanto com o valor pré-decimação informado na seção 5 quanto com o valor harmonizado em disco após a decimação de `MFPT` para $48.828\text{ Hz}$) são convertidas para $\text{m/s}^2$ antes de qualquer agregação:

| Fonte | Unidade Original | $\text{RMS}_{\text{ref}}$ Informado (Pré-Decimação) | $\text{RMS}_{\text{ref}}$ em $\text{m/s}^2$ (Informado) | $\text{RMS}_{\text{ref}}$ Harmonizado em Disco ($1{,}0\text{ s}$) | $\text{RMS}_{\text{ref}}$ em $\text{m/s}^2$ (Harmonizado em Disco) |
| :--- | :---: | ---: | ---: | ---: | ---: |
| **`UORED`** (`Class_Normal`, 180 jan.) | $\text{m/s}^2$ | $5{,}85\text{ m/s}^2$ | **$5{,}850000\text{ m/s}^2$** | $5{,}850288\text{ m/s}^2$ | **$5{,}850288\text{ m/s}^2$** |
| **`HUST`** (`Class_Normal`, 150 jan.) | $\text{m/s}^2$ | $0{,}181\text{ m/s}^2$ | **$0{,}181000\text{ m/s}^2$** | $0{,}181141\text{ m/s}^2$ | **$0{,}181141\text{ m/s}^2$** |
| **`MFPT`** (`Class_23`, 18 jan. 48,8 kHz) | $g$ | $0{,}874\,g$ *(36 jan. 0,5 s)* | **$8{,}571012\text{ m/s}^2$** | $0{,}767595\,g$ *(18 jan. 1,0 s)* | **$7{,}527536\text{ m/s}^2$** |

Aplicando a média geométrica $\text{RMS}_{\text{global}} = \exp\left(\frac{1}{|S|}\sum_{i \in S} \ln(\text{RMS}_{\text{ref}, i}^{(\text{m/s}^2)})\right)$ exclusivamente sobre as fontes ativas de cada configuração:

| Configuração | Fontes Ativas | $\text{RMS}_{\text{global}}$ Geométrico em $\text{m/s}^2$ (Valores Informados) | $\text{RMS}_{\text{global}}$ Geométrico em $\text{m/s}^2$ (Harmonizado em Disco) | Razão $\max/\min$ em $\text{m/s}^2$ |
| :---: | :--- | ---: | ---: | ---: |
| **`S1`** | `UORED` | **$5{,}850000\text{ m/s}^2$** | **$5{,}850288\text{ m/s}^2$** | $1{,}00\times$ |
| **`S2`** | `HUST` | **$0{,}181000\text{ m/s}^2$** | **$0{,}181141\text{ m/s}^2$** | $1{,}00\times$ |
| **`S3`** | `MFPT` | **$8{,}571012\text{ m/s}^2$** | **$7{,}527536\text{ m/s}^2$** | $1{,}00\times$ |
| **`S4`** | `UORED + HUST + MFPT` | **$2{,}085865\text{ m/s}^2$** | **$1{,}998071\text{ m/s}^2$** | $47{,}35\times$ *(informado)* / $41{,}56\times$ *(disco)* |

> [!WARNING]
> **Diagnóstico Físico de Comparabilidade de Escalas (Seção 5):**
> Mesmo após converter `MFPT` de $g$ para $\text{m/s}^2$, a amplitude saudável do `HUST` ($0{,}1811\text{ m/s}^2$) é **$\approx 32{,}3\times$ menor** que a do `UORED` ($5{,}8503\text{ m/s}^2$) e **$\approx 41{,}6\times$ menor** que a do `MFPT` ($7{,}5275\text{ m/s}^2$), além de o destino `PU` estar documentado em `docs/dataset_mapping.md` como saída em Volts ($V$, $\text{RMS}_{\text{ref}} = 0{,}2764$).
> - Como **126 das 141 features ($89{,}4\%$)** são matematicamente invariantes a qualquer fator escalar positivo $c > 0$, o valor de $\text{RMS}_{\text{global}}$ em DG afeta **exclusivamente as 15 features sensíveis a escala**.
> - Em [`fit_dg_rms_normalizer()`](file:///home/vfrocha/VibNet-1D/src/vibnet/evaluation/da_dg_tl_protocol.py), implementamos tanto o modo `geometric_mean_ms2` (que converte $1\,g = 9{,}80665\text{ m/s}^2$ e calcula $\exp(\text{mean}(\log(\text{RMS}_{\text{ref}, i})))$) quanto o modo de trava `strict_check_geometric_ms2` (que exige aprovação explícita quando a razão em $\text{m/s}^2$ excede `max_allowed_ratio`).

### 3.2. Diagnóstico da Calibração Externa `CWRU_12k -> CWRU_48k` em DA (Seção 6 e Entregável 8)
A auditoria física e estatística ([`audit_cwru_12k_48k_calibration_compatibility()`](file:///home/vfrocha/VibNet-1D/src/vibnet/evaluation/da_dg_tl_protocol.py)) confirmou:
1. **Mesma bancada, mesmo sensor, mesmo canal e mesma unidade física ($g$):** Tanto `CWRU_12k` quanto `CWRU_48k` foram adquiridos no acelerômetro do mancal *Drive End (DE)* do motor Reliance Electric de $2\text{ HP}$ da Case Western Reserve University (rolamento `6205-2RS JEM SKF`), sob as mesmas 4 cargas (`0, 1, 2, 3 HP`), diferenciando-se apenas pela taxa de amostragem do conversor A/D ($12.000\text{ Hz}$ vs. $48.000\text{ Hz}$).
2. **Estabilidade do RMS Normal entre cargas em `CWRU_12k`:**
   - `Load_0HP`: $0{,}075655 \pm 0{,}003277\,g$ (2 gravações: `s00000`, `s00001`)
   - `Load_1HP`: $0{,}066215 \pm 0{,}001410\,g$ (2 gravações: `s00002`, `s00003`)
   - `Load_2HP`: $0{,}064112 \pm 0{,}001181\,g$ (2 gravações: `s00004`, `s00005`)
   - `Load_3HP`: $0{,}069971 \pm 0{,}005358\,g$ (2 gravações: `s00006`, `s00007`)
   - Média global: $0{,}068036\,g$ (variação entre cargas $< 11\%$).
3. **Compatibilidade de escala nas classes de falha (`Class_1..3`):** Em `CWRU_12k`, o RMS médio das falhas varia entre $0{,}195\,g$ e $0{,}271\,g$; em `CWRU_48k`, varia na mesma ordem de grandeza ($0{,}114\,g$ a $0{,}746\,g$).
4. **Implementação em `fit_da_rms_normalizer()`:** Para `CWRU_48k`, utiliza-se **exclusivamente as 2 gravações normais da partição de adaptação de 20% de `CWRU_12k`** (da mesma seed ativa), bloqueando programaticamente qualquer gravação da partição de teste de `CWRU_12k` (`cross_reference_forbidden_recording_keys`) e registrando `calibration_mode = "external_related_domain_adaptation_normal"`.

---

## 4. Identificação Matemática das 126 Features Invariantes e 15 Sensíveis a Escala (Seção 5 e Entregável 6)

O catálogo completo e auditável de 141 itens está implementado em [`get_141_feature_scale_invariance_catalog()`](file:///home/vfrocha/VibNet-1D/src/vibnet/evaluation/da_dg_tl_protocol.py). Sob qualquer transformação de escala positiva $\tilde{x} = c \cdot x$ ($c > 0$):

### 4.1. As 15 Features Sensíveis a Escala (`get_scale_sensitive_15_indices()`)
Total: **9 em `src/features/extractors_v2.py` (`0..15`) + 6 em `src/features/signalai_wrapper.py` (`16..140`) = 15 features**:

| # | Índice Global (`0..140`) | Nome Canônico da Feature | Domínio / Módulo | Lei de Escala ($x \mapsto c \cdot x$) | Justificativa Matemática |
| ---: | :---: | :--- | :--- | :---: | :--- |
| 1 | **`0`** | `v2_time_abs_mean` | Tempo (`extractors_v2.py:L30`) | $c^1 \cdot f(x)$ | $\frac{1}{L}\sum |c x_t| = c \frac{1}{L}\sum |x_t|$ (homogênea de grau 1). |
| 2 | **`2`** | `v2_time_hilbert_envelope_mean` | Tempo (`extractors_v2.py:L34-L36`) | $c^1 \cdot f(x)$ | A transformada de Hilbert $\mathcal{H}$ é linear $\implies |c x + j\mathcal{H}(c x)| = c A(t)$. |
| 3 | **`3`** | `v2_time_hilbert_envelope_rms` | Tempo (`extractors_v2.py:L37`) | $c^1 \cdot f(x)$ | $\sqrt{\frac{1}{L}\sum (c A_t)^2} = c \cdot \text{RMS}(A)$. |
| 4 | **`5`** | `v2_time_tkeo_mean` | Tempo (`extractors_v2.py:L41-L42`) | $c^2 \cdot f(x)$ | Operador Teager-Kaiser $\Psi[c x_n] = (c x_n)^2 - (c x_{n-1})(c x_{n+1}) = c^2 \Psi[x_n]$. |
| 5 | **`6`** | `v2_time_tkeo_std` | Tempo (`extractors_v2.py:L43`) | $c^2 \cdot f(x)$ | Desvio-padrão de $\Psi[c x_n] = c^2 \Psi[x_n]$ escala com $c^2$. |
| 6 | **`8`** | `v2_freq_welch_peak_mag` | Frequência (`extractors_v2.py:L57`) | $c^2 \cdot f(x)$ | O periodograma de Welch $P_{xx}(f)$ é quadrático na amplitude $\implies \max_f P_{c x}(f) = c^2 \max_f P_x(f)$. |
| 7 | **`9`** | `v2_freq_welch_total_power` | Frequência (`extractors_v2.py:L59`) | $c^2 \cdot f(x)$ | $\sum_f P_{c x}(f) = c^2 \sum_f P_x(f)$. |
| 8 | **`12`** | `v2_freq_welch_legacy_band_power` | Frequência (`extractors_v2.py:L70`) | $c^2 \cdot f(x)$ | Potência espectral acumulada na banda escala com $c^2$. |
| 9 | **`13`** | `v2_tf_log_stft_power_mean` | Tempo-Freq. (`extractors_v2.py:L79-L83`) | $f(x) + 2\ln c$ | $\text{mean}(\ln(|c Z_{t,f}|^2)) = \text{mean}(\ln(|Z_{t,f}|^2)) + 2\ln c$ (deslocamento aditivo). |
| 10 | **`16`** | `signai_time_rms` | Tempo (`signalai_wrapper.py:L37`) | $c^1 \cdot f(x)$ | $\text{RMS}(c x) = c \cdot \text{RMS}(x)$. |
| 11 | **`17`** | `signai_time_peak` | Tempo (`signalai_wrapper.py:L37`) | $c^1 \cdot f(x)$ | $\max_t |c x_t| = c \max_t |x_t|$. |
| 12 | **`18`** | `signai_time_peak_to_peak` | Tempo (`signalai_wrapper.py:L37`) | $c^1 \cdot f(x)$ | $\max(c x) - \min(c x) = c (\max x - \min x)$. |
| 13 | **`19`** | `signai_time_variance` | Tempo (`signalai_wrapper.py:L37`) | $c^2 \cdot f(x)$ | $\text{Var}(c x) = c^2 \text{Var}(x)$. |
| 14 | **`27`** | `signai_freq_mean_spectral_power` | Frequência (`signalai_wrapper.py:L37`) | $c^2 \cdot f(x)$ | Potência espectral média $\frac{1}{K}\sum_k P(f_k)$ escala com $c^2$. |
| 15 | **`114`** | `signai_wpd_db4_total_energy` | WPD (`signalai_wrapper.py:L37`) | $c^2 \cdot f(x)$ | Energia total dos coeficientes WPD $\sum_{j=0}^{63} \|d_{6,j}\|_2^2$ escala com $c^2$. |

### 4.2. As 126 Features Invariantes a Escala (`get_scale_invariant_126_indices()`)
Total: **7 em `extractors_v2.py` + 119 em `SignalAIWrapper` = 126 features** onde $f(c \cdot x) = f(x)$ para todo $c > 0$:
1. **Tempo (`extractors_v2.py`, 2 features — índices `1, 4`):** Entropia de Shannon do histograma normalizado (`1`) e Curtose do envelope de Hilbert (`4`, momento padronizado de 4ª ordem $\mathbb{E}[(A-\mu)^4]/\sigma^4$, grau 0).
2. **Frequência e STFT (`extractors_v2.py`, 5 features — índices `7, 10, 11, 14, 15`):** Frequência de pico Welch (`7`, $\arg\max_f c^2 P_{xx}(f)$), razão de potência relativa em baixa frequência $P_{\text{low}}/P_{\text{total}}$ (`10`, cancela $c^2$), razão alta/baixa frequência $P_{\text{high}}/P_{\text{low}}$ (`11`, cancela $c^2$), desvio-padrão do log-espectrograma STFT (`14`, $\text{std}(\ln|Z|^2 + 2\ln c) = \text{std}(\ln|Z|^2)$) e frequência de pico na banda $1\text{–}200\text{ Hz}$ (`15`).
3. **Tempo Adimensional (`SignalAIWrapper`, 7 features — índices `20..26`):** `Skewness` (`20`), `Kurtosis` (`21`), `Crest Factor` ($\text{Peak}/\text{RMS}$, `22`), `Shape Factor` ($\text{RMS}/\text{MeanAbs}$, `23`), `Impulse Factor` ($\text{Peak}/\text{MeanAbs}$, `24`), `Clearance Factor` ($\text{Peak}/\text{MeanSqrtAbs}^2$, `25`) e `Zero-Crossing Rate` (`26`, pois $\text{sign}(c x_t) = \text{sign}(x_t)$).
4. **Frequência Normalizada (`SignalAIWrapper`, 14 features — índices `28..41`):** Centróide espectral, frequência RMS, espalhamento espectral, assimetria e curtose espectral, *spectral flatness* (média geométrica / média aritmética), *spectral rolloff 85%*, entropia espectral, frequência dominante, potências relativas nas bandas `B1..B4` e razão de energia harmônica (todas divididas pela energia espectral total, cancelando $c^2$).
5. **Wavelet Packet Decomposition (`SignalAIWrapper`, 72 features — índices `42..113`):** As 64 energias relativas das sub-bandas de nível 6 (`42..105`, $E_{6,k} / \sum_j E_{6,j}$), entropia de Shannon WPD (`106`), razão macro-banda alta/baixa (`107`), curtose/assimetria dos coeficientes WPD (`108..112`) e razão máxima de sub-banda (`113`).
6. **Empirical Mode Decomposition (`SignalAIWrapper`, 26 features — índices `115..140`):** Como a interpolação por splines cúbicas dos extremos locais no *sifting* da EMD é homogênea de grau 1 ($\text{IMF}_m(c x) = c \cdot \text{IMF}_m(x)$), todas as 26 features — energias relativas das 6 IMFs (`115..120`), correlação de Pearson $\text{corr}(\text{IMF}_m, x)$ (`121..126`), curtose (`127..132`), assimetria (`133..138`), entropia de energia EMD (`139`) e ZCR da IMF1 (`140`) — cancelam $c$ exatamente.

> **Proteção de Ablação (`select_feature_subset_for_ablation`):** Mantemos as **141 features** como configuração principal (`mode="full_141"`). Quando `mode="invariant_126"` é selecionado, a função extrai os 126 índices invariantes e **bloqueia com `ValueError` qualquer tentativa de carregar um checkpoint TabNet de 141 entradas**.

---

## 5. Preparação e Validação Isolada do MFPT (Seção 7 e Entregável 7)

Executada de forma isolada e transacional via [`prepare_isolated_mfpt_detrended_rms()`](file:///home/vfrocha/VibNet-1D/src/vibnet/evaluation/da_dg_tl_protocol.py):

1. **Preservação Integral dos Originais:**
   - Os 114 arquivos originais em `data/processed/MFPT` foram 100% preservados sem alteração e copiados para backup dedicado em `data/backups/remediation_mfpt_20261118_192304/processed_MFPT_original_114_backup/`.
2. **Identificação e Decimação Anti-Aliasing dos 6 Arquivos de $270\text{ lbs}$ ($97.656\text{ Hz} \to 48.828\text{ Hz}$):**
   - Gravações identificadas em `Load_270`:
     - `Class_23` (`Normal`): `s00000`, `s00001`, `s00002` (cada uma tinha 12 janelas de $0{,}5\text{ s}$ = $585.936$ pontos a $97.656\text{ Hz}$);
     - `Class_25` (`Outer`): `s00003`, `s00004`, `s00005` (cada uma tinha 12 janelas de $0{,}5\text{ s}$ = $585.936$ pontos a $97.656\text{ Hz}$).
   - Cada gravação contínua de $585.936$ amostras foi decimada por fator 2 com filtro FIR polifásico anti-aliasing (`scipy.signal.resample_poly(sig, up=1, down=2)` $\to 292.968$ amostras a $48.828\text{ Hz}$), submetida a `detrend` linear e segmentada em **6 janelas exatas de $1{,}0\text{ s}$ (`48.828` amostras)** (`w00..w05`).
3. **Resultado Efetivo e Validação (`scripts/validate_dataset_tl.py`):**
   - **Total de gravações únicas:** `20` gravações (`6` decimadas de $97.656\text{ Hz}$ + `14` nativas a $48.828\text{ Hz}$).
   - **Total de janelas de $1{,}0\text{ s}$ (`shape == (48828,)`):** **`78` janelas** (confirmando exatamente a expectativa física sem forçar contagens):
     - `Class_23` (`Normal`): **18 janelas** (3 gravações $\times$ 6 janelas em `Load_270`);
     - `Class_24` (`Inner`): **21 janelas** (7 gravações $\times$ 3 janelas em `Load_0..300`);
     - `Class_25` (`Outer`): **39 janelas** (3 gravações $\times$ 6 janelas em `Load_270` + 7 gravações $\times$ 3 janelas em `Load_25..300`).
   - **Normalização `detrended_rms/MFPT` e Manifesto:**
     - $\text{RMS}_{\text{ref}}$ calculado sobre as 18 janelas `Class_23` de $1{,}0\text{ s}$: **`0.7675950299980164 g`** ($7{,}527536\text{ m/s}^2$).
     - Média do RMS de `Class_23` após normalização: **`1.0000000000000002`** (`0` falhas matemáticas, `0` shape/dtype mismatches nos 10 datasets em `data/processed_tl/validation/dataset_validation.csv` e `reference_validation.csv`).

---

## 6. Manifestos Reproduzíveis das Partições DA e DG (Seção 8 e Entregável 9)

Gerados e exportados para:
- [`docs/experiment_planning/partition_manifests/da_dg_recording_partition_manifests.json`](file:///home/vfrocha/VibNet-1D/docs/experiment_planning/partition_manifests/da_dg_recording_partition_manifests.json)
- [`docs/experiment_planning/partition_manifests/da_dg_recording_partition_manifests.csv`](file:///home/vfrocha/VibNet-1D/docs/experiment_planning/partition_manifests/da_dg_recording_partition_manifests.csv) (**6.489 registros de gravações** auditados nas seeds `42`, `123` e `2026`, com **zero sobreposição de gravações** entre partições).

### Resumo das Partições por Dataset / Configuração (por Seed)

| Protocolo | Dataset / Config | Classes Supervisionadas | Gravações Totais | Partição `train` / `adapt` (Gravações / Janelas) | Partição `val` / `test` (Gravações / Janelas) | Proporção Efetiva (`Gravações` / `Janelas`) |
| :---: | :--- | :--- | ---: | :--- | :--- | :--- |
| **DA** | **`CWRU_12k`** | `Normal, Ball, Inner, Outer` (4) | **232** | **`adapt`: 46 grav.** / **500 jan.**<br>*(`Normal`: **2 grav.** integrais [40–80 jan.], `Ball`: 15, `Inner`: 15, `Outer`: 14; `adapt_val = 0`)* | **`test`: 186 grav.** / **2.020 jan.**<br>*(`Normal`: **6 grav.**, `Ball`: 61, `Inner`: 61, `Outer`: 58)* | **19,83% grav.** / **19,84%–20,63% jan.** |
| **DA** | **`CWRU_48k`** | `Ball, Inner, Outer` (3) | **70** | **`adapt`: 14 grav.** / **111–128 jan.**<br>*(`Ball`: 5, `Inner`: 4, `Outer`: 5; `adapt_val = 0`)* | **`test`: 56 grav.** / **474–491 jan.**<br>*(`Ball`: 19, `Inner`: 18, `Outer`: 19)* | **20,00% grav.** / **18,44%–21,26% jan.** |
| **DA** | **`PU`** | `Normal, Outer, Inner & Outer` (3) | **1.627** | **`adapt`: 326 grav.** / **1.298 jan.**<br>*(`Normal`: 96, `Outer`: 192, `Inner & Outer`: 38; `adapt_val = 0`)* | **`test`: 1.301 grav.** / **5.178 jan.**<br>*(`Normal`: 384, `Outer`: 767, `Inner & Outer`: 150)* | **20,04% grav.** / **20,04% jan.** |
| **DG** | **`S1` (`UORED`)** | `Normal, Ball` (2) | **28** | **`source_train`: 22 grav.** / **220 jan.**<br>*(`Normal`: 14, `Ball`: 8)* | **`source_val`: 6 grav.** / **60 jan.**<br>*(`Normal`: 4, `Ball`: 2)* | **78,57% train** / **21,43% val** |
| **DG** | **`S2` (`HUST`)** | Todas as 5 classes (5) | **72** | **`source_train`: 58 grav.** / **562–568 jan.** | **`source_val`: 14 grav.** / **132–138 jan.** | **80,56% train** / **19,44% val** |
| **DG** | **`S3` (`MFPT`)** | `Inner, Outer` (2) | **17** | **`source_train`: 14 grav.** / **48–51 jan.**<br>*(`Inner`: 6, `Outer`: 8)* | **`source_val`: 3 grav.** / **9–12 jan.**<br>*(`Inner`: 1, `Outer`: 2)* | **82,35% train** / **17,65% val** |
| **DG** | **`S4` (`Combinado`)**| Todas as 5 classes (5) | **117** | **`source_train`: 94 grav.** / **830–839 jan.** | **`source_val`: 23 grav.** / **201–210 jan.** | **80,34% train** / **19,66% val** |

---

## 7. Baselines Clássicos RF e SVM (Seção 10 e Entregável 10)

Implementados em [`train_eval_classical_da_dg_baseline()`](file:///home/vfrocha/VibNet-1D/src/vibnet/evaluation/da_dg_tl_protocol.py):
1. **DG (`variant="Source-Only"`):**
   - `RandomForest` e `SVM` treinados exclusivamente em `source_train`, com pesos de balanceamento hierárquico (`compute_hierarchical_source_balance`).
   - `StandardScaler` ajustado exclusivamente em `X_source_train_raw` e aplicado sem reajuste aos destinos.
   - O espaço de classes predito corresponde exatamente às classes presentes na configuração de origem (`S1`: 2 classes; `S2`: 5 classes; `S3`: 2 classes; `S4`: 5 classes), contabilizando classes desconhecidas do destino como erro.
2. **DA `Target-Only 20%` (`variant="Target-Only 20%"`):**
   - `RandomForest` e `SVM` treinados exclusivamente nos 20% rotulados da partição de adaptação do destino (`adapt`).
   - `StandardScaler` ajustado exclusivamente nos 20% de adaptação do destino e aplicado aos 80% de teste.
3. **DA `Source + Target 20%` (`variant="Source + Target 20%"`):**
   - `RandomForest` e `SVM` treinados no conjunto combinado `source_train + target_adapt_20%`.
   - `StandardScaler` ajustado exclusivamente na união de treinamento (`source_train + target_adapt_20%`).
   - **Nota Metodológica Registrada (`joint_learning_note`):** `Source + Target 20%` constitui um baseline de **aprendizado conjunto (*pooled joint learning*)** no espaço de features, não sendo equivalente à transferência explícita de representação e expansão cirúrgica da cabeça linear sobre o encoder congelado do TabNet.
4. **Padronização em `SVM` vs. `RandomForest`:**
   - Para `SVM` (kernel RBF), o `StandardScaler` ajustado exclusivamente no treino é indispensável para evitar dominância de features de maior escala numérica nas distâncias euclidianas $\|x_i - x_j\|_2^2$.
   - Para `RandomForest`, como árvores de decisão são invariantes a transformações lineares monotônicas por atributo, o `StandardScaler` ajustado exclusivamente no treino é mantido por **uniformidade estrita de pipeline**, sem alterar as partições de limiar das árvores.
5. **Preservação do Protocolo TL dos Baselines Clássicos:**
   - O módulo [`src/experiments/run_scratch_vs_tl_classical_baselines.py`](file:///home/vfrocha/VibNet-1D/src/experiments/run_scratch_vs_tl_classical_baselines.py) permanece intacto, reutilizando os 396 manifestos de fold LOCO com `StandardScaler` e tuning interno 3-fold CV restritos às $K-1$ condições de treino do destino.
