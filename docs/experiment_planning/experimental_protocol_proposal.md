# Proposta Consolidada de Protocolo Experimental — TL, DA e DG

## 1. Visão Geral e Premissas Fixadas

Este documento consolida o desenho metodológico para os experimentos definitivos de **Transfer Learning (TL)**, **Domain Adaptation (DA)** e **Domain Generalization (DG)** do projeto **VibNet-1D**, fundamentado nos achados da auditoria de datasets ([`dataset_audit.md`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/docs/experiment_planning/dataset_audit.md)), MFPT ([`mfpt_audit.md`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/docs/experiment_planning/mfpt_audit.md)), HUST Gearbox ([`hust_gearbox_preliminary.md`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/docs/experiment_planning/hust_gearbox_preliminary.md)), compatibilidade de classes ([`class_compatibility.md`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/docs/experiment_planning/class_compatibility.md)) e pré-processamento ([`preprocessing_audit.md`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/docs/experiment_planning/preprocessing_audit.md)).

### Decisões Fixadas para Todos os Protocolos:
1. **Arquitetura principal de Aprendizado Profundo:** `TabNet` (`TabNetPretrainer` para pré-treinamento não supervisionado por reconstrução das características + `TabNetClassifier` para a tarefa supervisionada).
2. **Baselines Clássicos:** `Random Forest (RF)` e `Support Vector Machine (SVM)`, com otimização interna de hiperparâmetros (`StratifiedKFold` ou `StratifiedGroupKFold` interno) restrita exclusivamente à partição de treinamento de cada fold/experimento. `XGBoost` e `MLP` estão excluídos desta etapa.
3. **Representação de Entrada:** As **141 características do TabVibNet** (`extract_fusion_features`: 18 Tempo + 24 Frequência + 73 WPD + 26 EMD) extraídas de janelas de $1{,}0\text{ s}$ sem sobreposição.
4. **Pré-processamento de Referência:** `detrended_rms` (remoção linear de tendência seguida de divisão pelo $\text{RMS}_{\text{ref}}$ calculado sobre sinais normais autorizados pelo protocolo).
5. **Métricas de Avaliação:**
   - **Métrica primária:** `Macro-F1` (média e desvio-padrão entre folds e entre seeds).
   - **Métricas secundárias:** `Balanced Accuracy`, `Accuracy` global, `ROC-AUC` (one-vs-rest) e matriz de confusão agregada.
6. **Seeds para Reprodutibilidade:** 5 sementes aleatórias independentes (`seeds = [42, 100, 2024, 7, 999]`) para inicialização do `TabNetPretrainer`, `TabNetClassifier`, `RF`, `SVM` e sorteio de partições de validação/adaptação.

---

## 2. Preparação Prévia Obrigatória dos Datasets de Origem (`UORED`, `HUST`, `MFPT`)

Antes de iniciar as rodadas definitivas de TL, DA e DG:
1. **Harmonização do `MFPT` (`97.656 Hz -> 48.828 Hz`):** Aplicar decimação anti-aliasing por fator 2 (`scipy.signal.resample_poly(sig, up=1, down=2)`) nos 6 arquivos gravados a $97.656\text{ Hz}$ (`baseline_1..3.mat` e `OuterRaceFault_1..3.mat`), gerando **78 janelas exatas de $1{,}0\text{ s}$** a $48.828\text{ Hz}$ (`Normal`: 18, `Inner`: 21, `Outer`: 39).
2. **Decisão sobre Re-extração de `UORED` e `PU`:**
   - **Opção A (Congelar dados atuais em disco para `UORED` e `PU` + incluir `MFPT` harmonizado):** Mantém compatibilidade direta com os manifestos anteriores (`UORED` = 370 janelas; `HUST` = 951 janelas; `PU` = 6.476 janelas), adicionando `MFPT` (78 janelas). Mesmo sem `Inner`/`Outer` no `UORED` em disco, a união `UORED + HUST + MFPT` cobre todas as 5 classes globais (`Normal`: 348, `Ball`: 200, `Inner`: 171, `Outer`: 189, `Inner & Outer`: 150 = **1.058 janelas supervisionadas** e **1.399 janelas para pré-treinamento não supervisionado**).
   - **Opção B (Re-extrair `UORED` e `PU` corrigindo a leitura dos `.mat` de `Inner`/`Outer`):** Recupera +230 janelas em `UORED` (`Inner`: 100, `Outer`: 100) e +3.520 janelas em `PU` (`Class_28` / `Inner`). Esta decisão deve ser aprovada pelo pesquisador antes da execução.

---

## 3. Protocolo 1 — Transfer Learning (TL)

### 3.1. Objetivo Científico
Avaliar se o **pré-treinamento não supervisionado multi-domínio (`TabNetPretrainer`)** sobre um pool diversificado de bancadas de origem (`UORED + HUST + MFPT`) melhora a eficiência amostral, a estabilidade e o `Macro-F1` quando transferido para domínios-alvo distintos sob mudança de condição operacional.

### 3.2. Configuração Experimental de TL
- **Pool de Pré-treinamento Não Supervisionado (Origem):** Todas as janelas de `UORED + HUST + MFPT` ($1.399$ janelas usando disco atual + MFPT, ou $1.629$ com UORED completo). Cada dataset de origem é normalizado pelo seu próprio $\text{RMS}_{\text{ref}}^{(s)}$, e o `TabNetPretrainer` reconstrói asfeatures mascaradas (`pretraining_ratio = 0.8`).
- **Datasets-Alvo (5 alvos) e Estratégia de Folds:**
  1. **`CWRU_12k` (4 folds LOCO):** `Load_0HP`, `Load_1HP`, `Load_2HP`, `Load_3HP` (4 classes: `Normal`, `Inner`, `Outer`, `Ball`; avaliado em Detecção Binária e Diagnóstico Multiclasse).
  2. **`CWRU_48k` (Diagnóstico Multiclasse — 3 classes: `Inner`, `Outer`, `Ball`):**
     - *Recomendação metodológica:* Avaliar tanto no **LOCO agrupado por Carga (`4 Macro-Folds`: `Load_0HP`..`Load_3HP`, onde todas as 3 classes estão presentes em todos os folds)** quanto no LOCO de 12 condições (`Load_x_Sev_y`, registrando que `Load_0HP_Sev_0.014` não possui `Class_1`).
  3. **`PU` (4 folds LOCO):** `C1`, `C2`, `C3`, `C4` (avaliado em Detecção Binária e Diagnóstico Multiclasse nas classes presentes no alvo).
  4. **`UOEMD` (6 folds LOCO):** `Load_{0,1}_{Const,Linear,Step}_Speed` (8 classes de motor/rolamento; avaliado em Detecção Binária e Diagnóstico Multiclasse).
  5. **`HUST_Gearbox` (6 Macro-Folds de Regime de Velocidade — `LOSO-Regime`):** `20Hz`, `25Hz`, `30Hz`, `35Hz`, `40Hz` e `VS_0_40Hz` (150 janelas de teste e 750 de treino por fold, conforme justificado no Estudo A [`hust_gearbox_preliminary.md`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/docs/experiment_planning/hust_gearbox_preliminary.md)), mais análise destacada do fold extremo de carga máxima **`Load_4`**.
- **Modelos Comparados em cada Fold de cada Dataset-Alvo:**
  1. `Random Forest (RF)` *(From Scratch no treino do alvo, com tuning interno 3-fold CV)*;
  2. `SVM` *(From Scratch no treino do alvo, com tuning interno 3-fold CV)*;
  3. `TabNet — From Scratch` *(inicialização aleatória, treinado apenas nas condições de treino do alvo)*;
  4. `TabNet — TL Fine-Tuning` *(inicializado com o encoder pré-treinado em `UORED + HUST + MFPT`, com ajuste completo de pesos nas condições de treino do alvo)*;
  5. `TabNet — TL Linear Probing` *(encoder de origem congelado, treinando apenas a camada linear final nas condições de treino do alvo)*.
- **Isolamento de Normalização em TL:** Em cada fold do alvo $T$, $\text{RMS}_{\text{ref}}^{(T,\text{fold})}$ e `StandardScaler` são calculados **exclusivamente sobre as condições de treinamento do fold**, sem tocar na condição de teste.

---

## 4. Protocolo 2 — Domain Adaptation Supervisionado (DA)

### 4.1. Objetivo Científico
Avaliar a adaptação de modelos entre diferentes bancadas de rolamentos operando sobre a **ontologia conceitual global de 5 classes (`0: Normal`, `1: Ball`, `2: Inner`, `3: Outer`, `4: Inner & Outer`)**, quando apenas uma pequena fração rotulada (**20% das gravações do domínio-alvo**) está disponível para adaptação e **80% das gravações do domínio-alvo** são retidas para teste.

### 4.2. Configuração Experimental de DA
- **Domínios de Origem:** `UORED + HUST + MFPT` (1.058 janelas supervisionadas nas 5 classes globais; 1.399 janelas para o `TabNetPretrainer`).
- **Domínios-Alvo (3 alvos de rolamentos):**
  1. **`CWRU_12k`** (classes presentes: `Normal`, `Ball`, `Inner`, `Outer`);
  2. **`CWRU_48k`** (classes presentes: `Ball`, `Inner`, `Outer`);
  3. **`PU`** (classes presentes em disco: `Normal`, `Outer`, `Inner & Outer`; mais `Inner` caso re-extraído).
- **Protocolo de Particionamento 20% Adaptação / 80% Teste (Sem Vazamento de Gravação):**
  - Para cada uma das 5 seeds (`[42, 100, 2024, 7, 999]`), as **gravações brutas (`recording_id` = `sXXXXX`)** do dataset-alvo são divididas de forma estratificada por `(condição, classe)` em:
    - **$20\%$ das gravações ($\mathcal{D}_{\text{adapt}}^{(T)}$):** usadas para adaptação supervisionada e cálculo de $\text{RMS}_{\text{ref}}^{(T,\text{DA})}$;
    - **$80\%$ das gravações ($\mathcal{D}_{\text{test}}^{(T)}$):** totalmente isoladas para avaliação final.
  - Jamais dividir janelas da mesma gravação entre os 20% e os 80%.
- **Braços Experimentais Comparados em DA:**
  1. **`Target-Only 20%` (RF, SVM, TabNet Scratch):** Treinados exclusivamente nos 20% rotulados do domínio-alvo $\mathcal{D}_{\text{adapt}}^{(T)}$ (mede o limite inferior sem transferência de origem).
  2. **`Source + Target 20% Pooled` (RF, SVM, TabNet Scratch):** Treinados na união dos dados supervisionados de origem (`UORED + HUST + MFPT`) + os 20% do alvo $\mathcal{D}_{\text{adapt}}^{(T)}$, na ontologia global de 5 classes.
  3. **`TabNet — Unsupervised Pretrain (Source) + Supervised Fine-Tune (Source -> Target 20%)`:**
     - Etapa 1: Pré-treinamento não supervisionado (`TabNetPretrainer`) em `UORED + HUST + MFPT`;
     - Etapa 2: Treinamento supervisionado da cabeça global de 5 classes na origem (`UORED + HUST + MFPT`);
     - Etapa 3: Fine-tuning supervisionado nos 20% rotulados do domínio-alvo ($\mathcal{D}_{\text{adapt}}^{(T)}$), avaliado nos 80% de teste ($\mathcal{D}_{\text{test}}^{(T)}$).

---

## 5. Protocolo 3 — Domain Generalization Estrito (DG)

### 5.1. Objetivo Científico
Avaliar a capacidade de generalização *zero-shot* (*out-of-domain*) para uma bancada de rolamentos **totalmente inédita**, sob a restrição estrita de que **nenhuma amostra, nenhum rótulo, nenhuma seleção de hiperparâmetros e nenhuma estatística de normalização ($\text{RMS}_{\text{ref}}$ ou `StandardScaler`) do domínio-alvo** seja utilizada antes da inferência final.

### 5.2. Configuração Experimental de DG Estrito
- **Conjunto de Treinamento e Validação (100% restrito à Origem `UORED + HUST + MFPT`):**
  - **Pré-treinamento não supervisionado (`TabNetPretrainer`):** Todas as janelas de `UORED + HUST + MFPT`.
  - **Treinamento e validação supervisionada (5 classes globais):** As 1.058 janelas de `UORED + HUST + MFPT` pertencentes às 5 classes globais (`Normal`: 348, `Ball`: 200, `Inner`: 171, `Outer`: 189, `Inner & Outer`: 150).
  - **Seleção de hiperparâmetros e *Early Stopping* na Origem:** Realizados exclusivamente via validação cruzada agrupada por gravação/condição **dentro dos domínios de origem (`UORED + HUST + MFPT`)**, com ponderação balanceada de classes (`class_weight='balanced'` ou amostragem balanceada por classe/domínio).
- **Normalização Estrita em DG (Zero Estatísticas do Alvo):**
  - Cada dataset de origem $s \in \{\text{UORED}, \text{HUST}, \text{MFPT}\}$ é normalizado pelo seu $\text{RMS}_{\text{ref}}^{(s)}$ saudável.
  - O `StandardScaler` global $\mathcal{S}_{\text{src}}$ é ajustado (`fit`) exclusivamente sobre as 141 features do conjunto de treinamento de origem.
  - Na avaliação de qualquer janela bruta $w$ de um dataset-alvo $T \in \{\text{CWRU\_12k}, \text{CWRU\_48k}, \text{PU}\}$ (100% do dataset-alvo é avaliado como teste):
    1. Aplica-se `detrend` linear em $w$;
    2. Divide-se por $\text{RMS}_{\text{ref}}^{(\text{src, geo})} = (\text{RMS}_{\text{ref}}^{(\text{UORED})} \cdot \text{RMS}_{\text{ref}}^{(\text{HUST})} \cdot \text{RMS}_{\text{ref}}^{(\text{MFPT})})^{1/3}$ (zero uso do dataset-alvo);
    3. Extraem-se as 141 features e aplica-se $\mathcal{S}_{\text{src}}\text{.transform}(x)$.
- **Modelos Comparados em DG Estrito (avaliados em 100% de `CWRU_12k`, `CWRU_48k` e `PU`):**
  1. `Random Forest (RF)` *(treinado e otimizado exclusivamente em `UORED + HUST + MFPT`)*;
  2. `SVM` *(treinado e otimizado exclusivamente em `UORED + HUST + MFPT`)*;
  3. `TabNet — Supervised Source Only` *(sem pré-treinamento reconstrutivo, treinado supervisionadamente em `UORED + HUST + MFPT`)*;
  4. `TabNet — Unsupervised Pretrained + Supervised Source` *(pré-treinado via `TabNetPretrainer` em todas as janelas de origem e ajustado supervisionadamente nas 5 classes globais de origem)*.
- **Ablação de Origens em DG:** Além do pool completo `UORED + HUST + MFPT`, avaliar subconjuntos de origem (ex.: `HUST` isolado — que possui as 5 classes — vs. `HUST + MFPT` vs. `UORED + HUST + MFPT`) para quantificar o ganho da diversidade multi-domínio. Quando um subconjunto de origem não possuir uma classe presente no alvo (ex.: `UORED + MFPT` testado em `PU`, onde falta `Inner & Outer`), aplica-se rigorosamente a regra de contabilização como erro (Seção 2, Requisito 9).

