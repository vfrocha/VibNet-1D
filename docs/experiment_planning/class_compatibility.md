# Compatibilidade de Classes e Ontologia Global (Estudo Preliminar C)

## 1. Objetivo e Escopo

Nos protocolos de **Domain Adaptation (DA)** e **Domain Generalization (DG)** entre diferentes bancadas de rolamentos, o modelo precisa operar sobre um **vocabulário conceitual global unificado**. Já no protocolo de **Transfer Learning (TL)**, o pré-treinamento no domínio de origem é **não supervisionado** (`TabNetPretrainer` via reconstrução das 141 características), e a cabeça supervisionada final é treinada e avaliada especificamente no espaço de classes nativo de cada dataset-alvo (`CWRU_12k`, `CWRU_48k`, `PU`, `UOEMD`, `HUST_Gearbox`).

Este documento formaliza o mapeamento exato de cada classe local dos datasets de rolamentos (`UORED`, `HUST`, `MFPT`, `CWRU_12k`, `CWRU_48k`, `PU`) para as **5 classes conceituais globais** estabelecidas para DA e DG, bem como a regra estrita de contabilização de erro para classes não representadas na cabeça de classificação.

A matriz completa de compatibilidade está disponível em [`docs/experiment_planning/class_compatibility_matrix.csv`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/docs/experiment_planning/class_compatibility_matrix.csv).

---

## 2. Ontologia Conceitual Global de 5 Classes (DA e DG)

Conforme as decisões metodológicas estabelecidas (Seção 2, Requisitos 8 e 9), o espaço global de classes para diagnóstico de rolamentos em DA e DG possui exatamente **5 classes mutuamente exclusivas**:

| ID Global | Nome Conceitual Global | Descrição Física da Falha | Regra de Composição |
| :---: | :--- | :--- | :--- |
| **`0`** | **`Normal`** | Rolamento saudável (*Healthy / Baseline*) | Sem defeito estrutural |
| **`1`** | **`Ball`** | Falha no elemento rolante (*Ball / Roller Fault*) | Defeito simples na esfera/rolo |
| **`2`** | **`Inner`** | Falha na pista interna (*Inner Race / Inner Ring Fault*) | Defeito simples na pista interna |
| **`3`** | **`Outer`** | Falha na pista externa (*Outer Race / Outer Ring Fault*) | Defeito simples na pista externa |
| **`4`** | **`Inner & Outer`** | Falha composta simultânea nas pistas interna e externa | **Classe independente** (NÃO decompor em `Inner` e `Outer`) |

---

## 3. Matriz de Mapeamento por Dataset (Bruto vs. Processado em Disco)

A tabela a seguir cruza os rótulos oficiais de `vibdata/raw/labels.csv` com as pastas efetivamente presentes em `data/processed_tl/detrended_rms/` (conforme verificado em [`results/scratch_vs_tl/run_20261105_141743/fold_manifest.csv`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/results/scratch_vs_tl/run_20261105_141743/fold_manifest.csv)):

| Dataset | Papel | Classe Global `0: Normal` | Classe Global `1: Ball` | Classe Global `2: Inner` | Classe Global `3: Outer` | Classe Global `4: Inner & Outer` | Classes Fora da Ontologia de 5 Classes |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`UORED`** *(em disco)* | Origem | `Class_Normal` (**180** jan.) | `Class_Ball` (**100** jan.) | **0 em disco** *(100 brutas em `I_1..5`)* | **0 em disco** *(100 brutas em `O_6..10`)* | Ausente (0) | `Class_Cage` (**90** jan. em disco / 100 brutas) |
| **`HUST`** | Origem | `Class_Normal` (**150** jan.) | `Class_43` (**100** jan.) | `Class_41` (**150** jan.) | `Class_42` (**150** jan.) | `Class_45` (**150** jan.) | `Class_44` (`Inner & Ball`, **120** jan.); `Class_46` (`Outer & Ball`, **131** jan.) |
| **`MFPT`** *(harmonizado)* | Origem | `Class_23` / `Normal` (**18** jan.) | Ausente (0) | `Class_24` / `Inner` (**21** jan. / 63 hist.) | `Class_25` / `Outer` (**39** jan. / 45 hist.) | Ausente (0) | Nenhuma |
| **`CWRU_12k`** | Alvo | `Class_0` (**280** jan.) | `Class_3` (**760** jan.) | `Class_1` (**760** jan.) | `Class_2` (**720** jan.) | Ausente (0) | Nenhuma |
| **`CWRU_48k`** | Alvo | Ausente em 48 kHz (0) | `Class_3` (**204** jan.) | `Class_1` (**194** jan.) | `Class_2` (**204** jan.) | Ausente (0) | Nenhuma |
| **`PU`** *(em disco)* | Alvo | `Class_Normal` (**1.918** jan.) | Ausente (0) | **0 em disco** *(3.520 brutas em `Class_28` / `KI*`)* | `Class_27` (**3.811** jan.) | `Class_29` (**747** jan.) | Nenhuma |

### Alertas Críticos de Mapeamento Identificados na Auditoria:
1. **`CWRU_12k` e `CWRU_48k` (`Class_2` vs. `Class_3`):**
   - Em `vibdata/raw/labels.csv`, o rótulo `2` é **`Outer Race`** (`Class_2`) e o rótulo `3` é **`Ball`** (`Class_3`).
   - Qualquer script de DA ou DG deve mapear explicitamente:
     - `Class_0 -> 0 (Normal)`
     - `Class_1 -> 2 (Inner)`
     - `Class_2 -> 3 (Outer)`
     - `Class_3 -> 1 (Ball)`
2. **Classes Compostas Adicionais de Origem (`Class_Cage` no `UORED` e `Class_44` / `Class_46` no `HUST`):**
   - `Class_Cage` (90 janelas em `UORED`), `Class_44` (`Inner & Ball`, 120 janelas em `HUST`) e `Class_46` (`Outer & Ball`, 131 janelas em `HUST`) **não pertencem às 5 classes globais** e **não existem em nenhum dataset-alvo** (`CWRU_12k`, `CWRU_48k`, `PU`).
   - **Regra recomendada:**
     - **No pré-treinamento não supervisionado (`TabNetPretrainer`):** **MANTER** todas as janelas (`Class_Cage`, `Class_44`, `Class_46`), pois a reconstrução auto-supervisionada de características se beneficia de maior diversidade de assinaturas impulsivas de rolamentos (+341 janelas sem usar rótulos).
     - **No treinamento supervisionado de origem (DA e DG):** **FILTRAR** `Class_Cage`, `Class_44` e `Class_46` do conjunto de treinamento supervisionado, treinando o classificador de origem estritamente sobre as 5 classes conceituais globais (`0: Normal`, `1: Ball`, `2: Inner`, `3: Outer`, `4: Inner & Outer`).

---

## 4. Análise de Cobertura de Classes entre Origem e Alvo

### 4.1. Treinamento com Origem Combinada (`HUST + UORED + MFPT`)
Quando o conjunto de origem combina `HUST + UORED + MFPT` (com os dados atualmente em disco + `MFPT` harmonizado):
- **`0: Normal`**: $150\text{ (HUST)} + 180\text{ (UORED)} + 18\text{ (MFPT)} = \mathbf{348\text{ janelas}}$;
- **`1: Ball`**: $100\text{ (HUST)} + 100\text{ (UORED)} + 0\text{ (MFPT)} = \mathbf{200\text{ janelas}}$;
- **`2: Inner`**: $150\text{ (HUST)} + 0\text{ (UORED disco)} + 21\text{ (MFPT)} = \mathbf{171\text{ janelas}}$ *(ou $271$ se `UORED` for re-extraído)*;
- **`3: Outer`**: $150\text{ (HUST)} + 0\text{ (UORED disco)} + 39\text{ (MFPT)} = \mathbf{189\text{ janelas}}$ *(ou $289$ se `UORED` for re-extraído)*;
- **`4: Inner & Outer`**: $150\text{ (HUST)} + 0\text{ (UORED)} + 0\text{ (MFPT)} = \mathbf{150\text{ janelas}}$.

**Conclusão fundamental:** A união dos datasets de origem (`HUST + UORED + MFPT`) **cobre 100% das 5 classes conceituais globais** com pelo menos 150 janelas por classe (`Total supervisionado = 1.058 janelas` nas 5 classes)! Assim, em **Domain Generalization (DG)** multi-source (`UORED + HUST + MFPT` $\rightarrow$ Alvo), **nenhuma classe dos datasets-alvo (`CWRU_12k`, `CWRU_48k`, `PU`) fica sem representação na cabeça de classificação!**

### 4.2. Cenários de Subconjunto de Origem ou Adaptação Parcial (Classes Não Representadas na Cabeça)
Caso seja executado um experimento de origem única (ex.: `MFPT` $\rightarrow$ `CWRU_12k`, onde a origem não possui `Ball`, ou `UORED` $\rightarrow$ `PU`, onde a origem não possui `Inner & Outer`), ou caso em **DA** a cabeça seja restrita às classes vistas:

#### Regra Metodológica Obrigatória (Seção 2, Requisito 9):
> **Se uma amostra de teste pertencer a uma classe não representada na cabeça de classificação, ela DEVE permanecer no conjunto de avaliação e ser contabilizada obrigatoriamente como erro ($\text{TP}_c = 0$, $\text{FN}_c = N_c \implies \text{Recall}_c = 0$, $\text{F1}_c = 0$). É estritamente proibido excluir amostras de teste para inflar artificialmente as métricas.**

#### Implementação Matemática do Cálculo de Métricas com Classes Não Representadas:
Seja $\mathcal{Y}_{\text{eval}}$ o conjunto de classes verdadeiras presentes nas amostras de teste do domínio-alvo (por exemplo, $\mathcal{Y}_{\text{eval}} = \{0, 1, 2, 3\}$ no `CWRU_12k`, $\mathcal{Y}_{\text{eval}} = \{1, 2, 3\}$ no `CWRU_48k`, e $\mathcal{Y}_{\text{eval}} = \{0, 3, 4\}$ no `PU` atual em disco).
1. **Codificação fixa em $\{0, 1, 2, 3, 4\}$:** Todos os rótulos verdadeiros $y_{\text{true}}$ e predições $\hat{y}_{\text{pred}}$ são representados pelos IDs inteiros globais $\{0, 1, 2, 3, 4\}$.
2. **Cálculo de `Macro-F1` e `Balanced Accuracy` sobre o suporte verdadeiro do teste ($\mathcal{Y}_{\text{eval}}$):**
   - `f1_score(y_true, y_pred, labels=sorted(np.unique(y_true)), average='macro', zero_division=0)`
   - ` recall_score(y_true, y_pred, labels=sorted(np.unique(y_true)), average='macro', zero_division=0)` (que equivale exatamente à `Balanced Accuracy` sobre todas as classes reais presentes no conjunto de teste).
   - **Atenção:** Se o modelo prever uma classe $c' \notin \mathcal{Y}_{\text{eval}}$ (por exemplo, prever `4: Inner & Outer` ao avaliar `CWRU_12k`), essa predição é automaticamente um **Falso Negativo** para a classe verdadeira da amostra (reduzindo o Recall e o F1 da classe verdadeira). Para que o Falso Positivo na classe fantasma $c'$ não crie uma divisão por $(|\mathcal{Y}_{\text{eval}}| + 1)$ com $\text{F1}_{c'}=0$ quando $N_{c'}=0$, a média macro deve ser tomada exatamente sobre `labels = sorted(np.unique(y_true))` (todas as classes que possuem amostras reais no teste), enquanto qualquer classe real $c \in \mathcal{Y}_{\text{eval}}$ que o modelo jamais consiga prever recebe $\text{F1}_c = 0{,}0$ e penaliza diretamente a média!
3. **Relato transparente de cobertura:** Todo relatório de DA e DG deve reportar:
   - Métrica principal **Global (com todas as amostras de teste e penalidade integral para classes não representadas)**;
   - Opcionalmente, como métrica diagnóstica secundária claramente rotulada, o desempenho restrito às classes compartilhadas (*Shared-Class Macro-F1*), nunca substituindo a métrica global.

