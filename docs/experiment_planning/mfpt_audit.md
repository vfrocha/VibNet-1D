# Auditoria Técnica do Dataset MFPT (Estudo Preliminar B)

## 1. Objetivo

Avaliar a integridade e a compatibilidade metodológica do dataset **MFPT (Machinery Failure Prevention Technology)** como candidato a dataset de origem (*source domain*) para:
1. **Pré-treinamento não supervisionado do TabNet** (`TabNetPretrainer` sobre as 141 características);
2. **Treinamento supervisionado da cabeça de classificação de origem** nos protocolos de **Domain Adaptation (DA)** e **Domain Generalization (DG)**.

O inventário detalhado dos 20 arquivos brutos encontra-se em [`docs/experiment_planning/mfpt_file_inventory.csv`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/docs/experiment_planning/mfpt_file_inventory.csv).

---

## 2. Auditoria dos Arquivos Brutos (`vibdata/raw/MFPT/MFPT.csv`)

A inspeção direta dos metadados e adaptadores de `MFPT_raw` na biblioteca `vibdata` revelou exatamente **20 arquivos `.mat`** distribuídos em **3 classes de rolamento**, todas adquiridas com velocidade nominal de eixo de **$25\text{ Hz}$ ($1.500\text{ RPM}$)**:

| Índice | Arquivo Bruto (`.mat`) | Rótulo `vibdata` | Classe Conceitual Global | Carga (`lbs`) | Rotação (`Hz`) | $f_s$ Real (`Hz`) | Duração Aprox. (`s`) | Amostras por Arquivo | Janelas sem Reamostragem *(Buggy)* | Janelas Harmonizadas a $48.828\text{ Hz}$ (`1.0 s`) |
| ---: | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0–2 | `baseline_1.mat` .. `baseline_3.mat` (3 arq.) | `23` (`Normal`) | **`Normal`** | 270 | 25 | **97.656** | ~6,0 s | 585.936 | 36 *(0,5 s cada)* | **18** *(6 por arq.)* |
| 3–9 | `InnerRaceFault_vload_1.mat` .. `7.mat` (7 arq.) | `24` (`Inner Race`) | **`Inner`** | 0, 50, 100, 150, 200, 250, 300 | 25 | **48.828** | ~3,0 s | 146.484 | 21 *(1,0 s cada)* | **21** *(3 por arq.)* |
| 10–12 | `OuterRaceFault_1.mat` .. `3.mat` (3 arq.) | `25` (`Outer Race`) | **`Outer`** | 270 | 25 | **97.656** | ~6,0 s | 585.936 | 36 *(0,5 s cada)* | **18** *(6 por arq.)* |
| 13–19 | `OuterRaceFault_vload_1.mat` .. `7.mat` (7 arq.) | `25` (`Outer Race`) | **`Outer`** | 25, 50, 100, 150, 200, 250, 300 | 25 | **48.828** | ~3,0 s | 146.484 | 21 *(1,0 s cada)* | **21** *(3 por arq.)* |
| **Total** | **20 arquivos `.mat`** | **3 classes** | **`Normal`, `Inner`, `Outer`** | **9 níveis** | **25 Hz** | **Mista (97.656 / 48.828)** | — | — | **114** *(inconsistente)* | **78** *(janelas exatas de 1,0 s)* |

*(Nota: Em processamentos históricos anteriores registrados em `docs/datasets_and_protocols.md`, foram contabilizadas 126 janelas [`Normal=18, Inner=63, Outer=45`] devido a variações de fatiamento; já no manifesto `run_20261105_141743/fold_manifest.csv`, o `MFPT` não estava presente no diretório `data/processed_tl/detrended_rms/`).*

---

## 3. Diagnóstico da Discrepância de Taxa de Amostragem ($97.656\text{ Hz}$ vs. $48.828\text{ Hz}$)

### 3.1. Problema no Pipeline Legado (`src/data/make_dataset.py` e `src/data/make_dataset_tl.py`)
Nos scripts atuais [`src/data/make_dataset.py`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/src/data/make_dataset.py) (linha 272) e [`src/data/make_dataset_tl.py`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/src/data/make_dataset_tl.py) (linha 188), o pipeline define um tamanho fixo de janela em número de pontos:
```python
"MFPT": {"window": 48828}
```
e aplica `T.Sequential([T.SimpleSplit(48828)])` diretamente sobre o sinal bruto, sem verificar o campo `sample_rate` nos metadados de cada arquivo:

1. **Nos 14 arquivos `*_vload_*.mat` ($f_s = 48.828\text{ Hz}$):** Uma janela de $48.828$ pontos corresponde exatamente a **$1{,}0\text{ s}$** de sinal físico.
2. **Nos 6 arquivos de carga constante a $270\text{ lbs}$ (`baseline_1..3.mat` e `OuterRaceFault_1..3.mat`, com $f_s = 97.656\text{ Hz}$):** Uma janela de $48.828$ pontos sem decimação prévia corresponde a apenas **$0{,}5\text{ s}$** de sinal físico!
3. **Consequência sobre as 141 características (`extract_fusion_features`):**
   - No extrator de frequência de [`src/features/extractors_v2.py`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/src/features/extractors_v2.py), a FFT utiliza `fs = len(sig)` (assumindo janela de $1{,}0\text{ s}$). Se uma janela de $0{,}5\text{ s}$ ($48.828$ amostras a $97.656\text{ Hz}$) é passada para `extract_fusion_features`, a resolução espectral real é de $2\text{ Hz/bin}$, mas o extrator interpreta como $1\text{ Hz/bin}$, **deslocando todas as frequências características (Centróide Espectral, Frequência RMS, Harmônicas de Rotação e Bandas WPD) por um fator de 2 ($100\%$ de erro de escala espectral)** justamente em todas as amostras da classe `Normal` e em metade das amostras da classe `Outer`!

---

## 4. Procedimento de Correção e Harmonização Recomendado

Como $97.656 = 2 \times 48.828$ de forma exata, a harmonização física para a taxa única $f_s^{\text{alvo}} = 48.828\text{ Hz}$ é matematicamente limpa e livre de artefatos de interpolação fracionária:

1. **Inspeção de metadados por gravação:** Para cada item retornado por `MFPT_raw`, ler `fs_real = int(meta["sample_rate"])`.
2. **Decimação polifásica com filtro anti-aliasing (`scipy.signal.resample_poly`):**
   - Se `fs_real == 97656`, aplicar `sig = scipy.signal.resample_poly(sig, up=1, down=2)` antes da divisão em janelas.
   - Isso aplica um filtro passa-baixas FIR anti-aliasing com corte em $f_N = 24.414\text{ Hz}$ e reduz cada gravação de $6{,}0\text{ s}$ ($585.936$ pontos) para $292.968$ pontos a $48.828\text{ Hz}$.
3. **Fatiamento em janelas de $1{,}0\text{ s}$ ($48.828$ pontos):**
   - `baseline_1..3.mat` (3 arq. $\times$ 6 janelas) $\rightarrow$ **18 janelas `Normal`** de $1{,}0\text{ s}$ a $48.828\text{ Hz}$;
   - `InnerRaceFault_vload_1..7.mat` (7 arq. $\times$ 3 janelas) $\rightarrow$ **21 janelas `Inner`** de $1{,}0\text{ s}$ a $48.828\text{ Hz}$;
   - `OuterRaceFault_1..3.mat` (3 arq. $\times$ 6 janelas) + `OuterRaceFault_vload_1..7.mat` (7 arq. $\times$ 3 janelas) $\rightarrow$ **$18 + 21 = 39$ janelas `Outer`** de $1{,}0\text{ s}$ a $48.828\text{ Hz}$.
4. **Normalização de referência (`detrended_rms`):**
   - O $\text{RMS}_{\text{ref}}^{(\text{MFPT})}$ passa a ser calculado sobre as **18 janelas `Normal` harmonizadas a $48.828\text{ Hz}$**, eliminando a distorção de banda de frequência.

---

## 5. Parecer Final sobre o MFPT

- **Classificação:** **Apto com processamento adicional obrigatório** (decimação anti-aliasing `97.656 Hz -> 48.828 Hz` nos 6 arquivos de $270\text{ lbs}$).
- **Adequação ao Pré-treinamento Não Supervisionado do TabNet (TL, DA, DG):** **Sim.** Adiciona 78 janelas genuínas de $1{,}0\text{ s}$ em 9 níveis de carga radial ($0$ a $300\text{ lbs}$), enriquecendo a diversidade de condições mecânicas do conjunto de origem (`UORED + HUST + MFPT`).
- **Adequação ao Treinamento Supervisionado de Origem (DA e DG):** **Sim.** Contribui diretamente para 3 das 5 classes conceituais globais (`Normal`: 18, `Inner`: 21, `Outer`: 39), complementando perfeitamente o `HUST` (que fornece `Normal`, `Ball`, `Inner`, `Outer`, `Inner & Outer`) e o `UORED` (`Normal`, `Ball`).

