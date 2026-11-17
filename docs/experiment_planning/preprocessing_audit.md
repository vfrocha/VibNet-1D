# Auditoria do Pipeline de Pré-processamento, Extração de Características e Normalização (Estudo Preliminar D)

## 1. Objetivo

Auditar a implementação real de pré-processamento de sinais ([`src/preprocessing/normalize_dataset_tl.py`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/src/preprocessing/normalize_dataset_tl.py)), extração das **141 características do TabVibNet** ([`src/features/extractors_v2.py`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/src/features/extractors_v2.py), [`src/features/signalai_wrapper.py`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/src/features/signalai_wrapper.py), [`src/vibnet/tabular/features.py`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/src/vibnet/tabular/features.py)) e padronização tabular (`StandardScaler`), verificando potenciais pontos de **vazamento de informação (*data leakage*)** nos protocolos **TL**, **DA** e **DG estrito**.

---

## 2. Auditoria das 141 Características (`TabVibNet` / `extract_fusion_features`)

### 2.1. Composição Exata do Vetor de 141 Atributos
A inspeção de [`src/features/extractors_v2.py`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/src/features/extractors_v2.py), [`src/features/signalai_wrapper.py`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/src/features/signalai_wrapper.py) e [`src/vibnet/tabular/features.py`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/src/vibnet/tabular/features.py) confirma que `extract_fusion_features(window)` produz sempre um vetor `float64` de dimensão fixa **$D = 141$**, independente do número de pontos $L$ da janela de $1{,}0\text{ s}$:

| Bloco de Atributos | Qtd. | Índices (`0..140`) | Descrição Técnica | Dependência de Escala de Amplitude ($w \mapsto c \cdot w$) |
| :--- | ---: | :---: | :--- | :--- |
| **Tempo Clássico (`extractors_v2` + `SignAI`)** | **18** | `0..6` (v2) + `16..26` (SignAI) | RMS, Peak, Peak-to-Peak, Mean Abs, Variance, Std, Energy, Zero-Crossing Rate, Kurtosis, Skewness, Crest Factor, Shape Factor, Impulse Factor, Clearance Factor, Entropy | **8 dependentes de escala** (RMS, Peak, P2P, MeanAbs, Std, Var, Energy); **10 invariantes a escala** (Kurtosis, Skewness, Crest, Shape, Impulse, Clearance, ZCR, Entropy) |
| **Frequência Clássico (`extractors_v2` + `SignAI`)** | **24** | `7..15` (v2) + `27..41` (SignAI) | Centróide Espectral, Freq. RMS, Desvio Espectral, Assimetria/Curtose Espectral, Entropia de Welch, Energia nas 3 primeiras harmônicas de rotação, Momentos espectrais | **18 invariantes a escala** (frequências normalizadas pelo espectro de potência total e entropia); **6 dependentes de escala** (energias absolutas / potência média) |
| **Wavelet Packet Decomposition (`WPD` db4, nível 6)** | **73** | `42..114` | Energias relativas das $2^6 = 64$ sub-bandas (`64`), Entropia de Shannon WPD (`1`), mais razão de energia, curtose e assimetria das 8 macro-bandas (`8`) | **72 invariantes a escala** (as 64 sub-bandas são divididas por `total_energy`; curtose/assimetria são adimensionais); **1 dependente de escala** |
| **Empirical Mode Decomposition (`EMD`, 6 IMFs)** | **26** | `115..140` | Energias relativas das 6 primeiras IMFs (`6`), correlação com sinal original (`6`), curtose (`6`), assimetria (`6`), entropia de energia EMD (`1`) e taxa de cruzamento por zero da IMF1 (`1`) | **26 invariantes a escala** ($100\%$ invariantes a multiplicação por escalar positivo $c > 0$) |
| **Total (`TabVibNet`)** | **141** | `0..140` | Fusão Tempo + Frequência + WPD + EMD | **126 invariantes a escala ($89{,}4\%$)** / **15 dependentes de escala ($10{,}6\%$)** |

### 2.2. Compatibilidade Multi-Taxa de Amostragem ($f_s \in [12.000, 64.000]\text{ Hz}$)
1. **Resolução espectral idêntica ($\Delta f = 1{,}0\text{ Hz}$):** Como todas as janelas têm duração física de exatamente **$1{,}0\text{ s}$** ($L = f_s$, após a harmonização do `MFPT` para $48.828\text{ Hz}$), a transformada de Fourier `np.fft.rfft(sig)` com `fs = len(sig)` possui resolução espectral exata de $1{,}0\text{ Hz/bin}$ em todos os datasets.
2. **Limitação física inerente (Frequência de Nyquist $f_N = f_s / 2$):**
   - A árvore WPD de nível 6 divide o intervalo $[0, f_s/2]$ em 64 sub-bandas uniformes de largura $\Delta B = f_s / 128$. Assim, a sub-banda $k$ representa a mesma **fração relativa da banda de Nyquist** em cada bancada, mas frequências absolutas distintas ($93{,}75\text{ Hz/banda}$ em `CWRU_12k` vs. $500\text{ Hz/banda}$ em `PU`). Esta é uma limitação física natural de transferência entre sensores com taxas distintas e deve ser explicitamente registrada como limitação metodológica nos artigos.

---

## 3. Auditoria do Pré-processamento de Referência (`detrended_rms`)

### 3.1. Definição Matemática
Para uma janela bruta $w \in \mathbb{R}^L$:
1. **Remoção de tendência linear (*Linear Detrending*):**
   $$\tilde{w} = \text{detrend}_{\text{linear}}(w)$$
2. **Normalização pela referência saudável ($\text{RMS}_{\text{ref}}$):**
   $$w_{\text{norm}} = \frac{\tilde{w}}{\text{RMS}_{\text{ref}}}, \quad \text{onde } \text{RMS}_{\text{ref}} = \frac{1}{|\mathcal{N}_{\text{ref}}|} \sum_{u \in \mathcal{N}_{\text{ref}}} \sqrt{\frac{1}{L}\sum_{t=1}^{L} \tilde{u}_t^2}$$
   sendo $\mathcal{N}_{\text{ref}}$ o conjunto autorizado de janelas da classe `Normal` (saudável).

### 3.2. Problema de Vazamento Identificado na Implementação Atual (`normalize_dataset_tl.py`)
No script atual [`src/preprocessing/normalize_dataset_tl.py`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/src/preprocessing/normalize_dataset_tl.py) (funções `compute_healthy_reference_rms` e `normalize_dataset`), o valor $\text{RMS}_{\text{ref}}^{(d)}$ é pré-calculado uma única vez sobre **todas as janelas normais de todas as condições do dataset $d$** (incluindo a condição que será usada como teste em LOCO e incluindo os domínios-alvo que deveriam ser desconhecidos em DG):
- Valores pré-calculados atualmente em `data/processed_tl/detrended_rms/`:
  - `CWRU_12k` (e `CWRU_48k`): $\text{RMS}_{\text{ref}} = 0{,}0735$
  - `UORED`: $\text{RMS}_{\text{ref}} = 0{,}1047$
  - `HUST`: $\text{RMS}_{\text{ref}} = 2{,}1743$
  - `PU`: $\text{RMS}_{\text{ref}} = 0{,}2828$
  - `UOEMD`: $\text{RMS}_{\text{ref}} = 0{,}1066$
  - `HUST_Gearbox`: $\text{RMS}_{\text{ref}} = 0{,}5887$

Embora o RMS da classe normal varie pouco entre condições de uma mesma bancada, **utilizar janelas normais da condição de teste (em TL/DA) ou utilizar o $\text{RMS}_{\text{ref}}$ do dataset-alvo (em DG estrito) viola o isolamento metodológico entre treino e teste**.

### 3.3. Propriedade Matemática Chave para Reaproveitamento do Cache de Features!
Observe como uma mudança no escalar $\text{RMS}_{\text{ref}}$ afeta as 141 características:
- Se uma janela já foi normalizada por $\text{RMS}_{\text{ref}}^{\text{(global)}}$ e queremos re-normalizá-la por $\text{RMS}_{\text{ref}}^{\text{(estrito)}}$, o sinal no tempo é simplesmente multiplicado pelo fator escalar positivo:
  $$\alpha = \frac{\text{RMS}_{\text{ref}}^{\text{(global)}}}{\text{RMS}_{\text{ref}}^{\text{(estrito)}}} > 0$$
- Das 141 características:
  - **126 características ($89{,}4\%$)** são matematicamente invariantes a $\alpha$ ($f(\alpha w) = f(w)$);
  - **As 15 características dependentes de amplitude** escalam de forma exata por $\alpha^1$ (RMS, Peak, P2P, MeanAbs, Std) ou $\alpha^2$ (Variance, Energy, Potências espectrais absolutas)!
- Contudo, se extrairmos as 141 features diretamente das janelas em `data/processed/<dataset>` (que possuem apenas `detrend` linear ou sinal bruto) aplicando o $\text{RMS}_{\text{ref}}$ restrito a cada fold/protocolo — ou se reescalarmos exatamente o sinal detrended antes da extração — eliminamos 100% de qualquer vazamento! Mais ainda: como o `StandardScaler` ($z = (x - \mu_{\text{train}}) / \sigma_{\text{train}}$) é ajustado no conjunto de treino de cada fold, dentro de um mesmo dataset qualquer multiplicador constante $\alpha$ idêntico no treino e no teste de um fold cancela-se exatamente no `StandardScaler` para todos os atributos homogêneos de grau 1 ou 2!
- **Onde o $\text{RMS}_{\text{ref}}$ faz diferença real (não cancelada pelo `StandardScaler`)?** Quando **múltiplos datasets com sensores/unidades distintas são combinados** (como `UORED + HUST + MFPT` ou `Origem vs. Alvo` em DA/DG), pois cada dataset tem seu próprio $\text{RMS}_{\text{ref}}^{(d)}$!

---

## 4. Regras de Normalização sem Vazamento para TL, DA e DG Estrito

### 4.1. Protocolo Transfer Learning (TL)
1. **Pré-treinamento não supervisionado (`TabNetPretrainer` na Origem `UORED + HUST + MFPT`):**
   - Cada dataset de origem $s \in \{\text{UORED}, \text{HUST}, \text{MFPT}\}$ tem suas janelas detrended divididas pelo seu próprio $\text{RMS}_{\text{ref}}^{(s)}$ (calculado sobre as janelas `Normal` daquele dataset de origem $s$).
   - O `StandardScaler` do pré-treinamento ($\mathcal{S}_{\text{src}}$) é ajustado (`fit`) exclusivamente sobre as 141 características do pool de origem.
2. **Fine-tuning / Treinamento Supervisionado no Alvo (LOCO no Dataset-Alvo $T$):**
   - Em cada fold LOCO (onde $C_{\text{test}}$ é a condição de teste retida e $\mathcal{C}_{\text{train}} = \mathcal{C} \setminus \{C_{\text{test}}\}$ são as condições de treinamento do alvo):
     - $\text{RMS}_{\text{ref}}^{(T, \text{fold})}$ é calculado **exclusivamente sobre as janelas `Normal` das condições de treinamento $\mathcal{C}_{\text{train}}$** (para `CWRU_48k`, utiliza-se o $\text{RMS}_{\text{ref}}$ das condições de treinamento correspondentes em `CWRU_12k`).
     - Tanto as janelas de treino ($\mathcal{C}_{\text{train}}$) quanto as de teste ($C_{\text{test}}$) do fold são divididas por esse mesmo $\text{RMS}_{\text{ref}}^{(T, \text{fold})}$.
     - Para `RF`, `SVM` e `TabNet Scratch`, o `StandardScaler` é ajustado (`fit`) exclusivamente nas features de $\mathcal{C}_{\text{train}}$ do fold. Para o `TabNet Fine-tuning`, documenta-se claramente o uso do scaler ajustado em $\mathcal{C}_{\text{train}}$ do fold (garantindo a mesma distribuição de entrada do espaço pré-treinado já normalizado por $\text{RMS}_{\text{ref}}$).

### 4.2. Protocolo Domain Adaptation Supervisionado (DA — 20% Adaptação / 80% Teste no Alvo)
1. **Particionamento livre de vazamento temporal (Split por Gravação Bruta `sXXXXX`):**
   - Como janelas de uma mesma gravação (`s00001_w00` a `s00001_w09`) são contíguas no tempo, **o split de 20% adaptação / 80% teste NUNCA deve ser feito aleatoriamente no nível de janelas individuais**.
   - O particionamento $20\% / 80\%$ deve ser realizado **no nível de gravações (`recording_id` = prefixo `sXXXXX`)**, estratificado por `(condição, classe)` (com semente fixa por repetição/seed), garantindo que todas as janelas de uma mesma gravação caiam integralmente nos $20\%$ de adaptação ou integralmente nos $80\%$ de teste.
2. **Cálculo de $\text{RMS}_{\text{ref}}$ e `StandardScaler` em DA:**
   - **Nos domínios de origem (`UORED`, `HUST`, `MFPT`):** Cada dataset de origem usa seu $\text{RMS}_{\text{ref}}^{(s)}$ calculado nas suas próprias janelas `Normal`.
   - **No domínio-alvo $T$:** O valor $\text{RMS}_{\text{ref}}^{(T, \text{DA})}$ é calculado **exclusivamente sobre as janelas da classe `Normal` pertencentes à partição de 20% de adaptação do alvo** (para `CWRU_48k`, que não possui classe `Normal` própria, utiliza-se o $\text{RMS}_{\text{ref}}$ de 20% das gravações normais de `CWRU_12k`, documentando explicitamente essa referência de bancada).
   - O `StandardScaler` de adaptação é ajustado sem tocar nos 80% de teste.

### 4.3. Protocolo Domain Generalization Estrito (DG — 0% de Estatísticas do Alvo)
No protocolo **DG estrito** (Seção 2, Requisito 7), **nenhuma informação do domínio-alvo $T$ pode ser utilizada no treinamento, seleção de hiperparâmetros ou cálculo de estatísticas de normalização** — nem mesmo o $\text{RMS}_{\text{ref}}^{(T)}$ das janelas normais do alvo ou a média/desvio-padrão das features do alvo!

#### Análise Física do Desafio em DG Estrito:
Se não podemos usar janelas normais do domínio-alvo $T$ para calcular $\text{RMS}_{\text{ref}}^{(T)}$, como normalizar uma janela bruta de teste $w \in \mathbb{R}^L$ proveniente do sensor do domínio-alvo $T$?
Analisamos duas opções rigorosas e compatíveis com o requisito de zero estatísticas do dataset-alvo:

1. **Opção DG-Primária Recomendada — Normalização `detrended_rms` no Treino de Origem + Referência Macro de Origem no Teste (`Source-Macro RMS_ref`):**
   - **Nos datasets de origem ($s \in \{\text{UORED}, \text{HUST}, \text{MFPT}\}$):** Cada dataset de origem $s$ é normalizado pelo seu próprio $\text{RMS}_{\text{ref}}^{(s)}$ saudável (ou pelo $\text{RMS}_{\text{ref}}^{(\text{src, macro})}$) e o `StandardScaler` $\mathcal{S}_{\text{src}}$ é ajustado (`fit`) **exclusivamente sobre a união dos datasets de origem**.
   - **No dataset-alvo de teste ($T$, 100% cego):** Cada janela de teste $w \in T$ sofre `detrend` linear e é dividida exclusivamente pela **média geométrica (ou macro-média ponderada igualmente por dataset) dos $\text{RMS}_{\text{ref}}$ dos domínios de origem**:
     $$\text{RMS}_{\text{ref}}^{(\text{src, geo})} = \left( \prod_{s \in \mathcal{S}_{\text{src}}} \text{RMS}_{\text{ref}}^{(s)} \right)^{1/|\mathcal{S}_{\text{src}}|}$$
     seguida da aplicação (`transform`) do `StandardScaler` $\mathcal{S}_{\text{src}}$ ajustado na origem.
   - **Por que a média geométrica/macro por domínio de origem é essencial?** Porque `HUST` possui $\text{RMS}_{\text{ref}} = 2{,}1743$ (951 janelas), enquanto `UORED` possui $\text{RMS}_{\text{ref}} = 0{,}1047$ (370 janelas). Uma média ponderada por número de janelas seria dominada pela escala de voltagem do `HUST`, enquanto a ponderação igualitária por domínio dá peso $1/3$ a cada bancada de origem.
   - **O que acontece com as 141 features no alvo sob essa regra?** Como demonstrado na Seção 2.1, **126 das 141 características ($89{,}4\%$)** — incluindo fatores de forma/crista/impulso, curtose, assimetria, frequências normalizadas, todas as 64 energias relativas WPD e todas as 26 features EMD — são **invariantes a escala positiva**, portanto seu valor em uma janela de teste $w$ é **exatamente o mesmo** qualquer que seja a constante escalar $\text{RMS}_{\text{ref}} > 0$ utilizada! Apenas as 15 features de amplitude absoluta refletirão a diferença de ganho físico entre o sensor de origem e o sensor-alvo.

2. **Opção DG-Ablação Complementar — Normalização por Janela Individual (`Window-RMS` / `Detrended Z-Score por Janela`) em DG:**
   - Como uma única janela de teste $w$ está disponível no momento da inferência (o que não constitui estatística agregada do domínio-alvo, mas sim auto-normalização da própria amostra de $1\text{ s}$), dividir $\tilde{w}$ pelo seu próprio $\text{RMS}(\tilde{w})$ equivale a fixar $\text{RMS}(w_{\text{norm}}) = 1$ (tornando 100% das features invariantes ao ganho do acelerômetro, ao custo de remover a razão de energia global entre rolamento saudável e defeituoso).
   - **Recomendação:** Adotar a **Opção DG-Primária (`detrended_rms` com $\text{RMS}_{\text{ref}}$ e `StandardScaler` calculados estritamente na origem)** como protocolo principal (cumprindo literalmente os Requisitos 6 e 7 da Seção 2), e registrar a sensibilidade das 15 features dependentes de escala como ponto de análise na discussão do artigo de DG.

