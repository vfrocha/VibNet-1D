# Resumo Executivo da Auditoria e Questões para Decisão (`audit_summary.md`)

## 1. Síntese Executiva

A auditoria completa dos datasets, metadados brutos (`vibdata`), scripts de preparação e resultados preliminares do **VibNet-1D** foi concluída sem alterar nenhum resultado experimental existente. Os seis relatórios técnicos detalhados e quatro tabelas CSV de suporte foram gerados em `docs/experiment_planning/`:

- **Documentos Técnicos:**
  1. [`dataset_audit.md`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/docs/experiment_planning/dataset_audit.md) — Inventário completo de todos os datasets e discrepâncias entre código, documentação e disco.
  2. [`mfpt_audit.md`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/docs/experiment_planning/mfpt_audit.md) — Estudo Preliminar B: auditoria dos 20 arquivos do `MFPT` e solução para a taxa mista ($97.656\text{ Hz}$ vs. $48.828\text{ Hz}$).
  3. [`hust_gearbox_preliminary.md`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/docs/experiment_planning/hust_gearbox_preliminary.md) — Estudo Preliminar A: análise das 30 condições do `HUST_Gearbox` e proposta dos protocolos Macro-LOCO por Regime.
  4. [`class_compatibility.md`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/docs/experiment_planning/class_compatibility.md) — Estudo Preliminar C: mapeamento exato para a ontologia global de 5 classes e regra de penalização para classes não representadas.
  5. [`preprocessing_audit.md`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/docs/experiment_planning/preprocessing_audit.md) — Estudo Preliminar D: auditoria das 141 características (`TabVibNet`), invariância de escala e isolamento de $\text{RMS}_{\text{ref}}$ e `StandardScaler`.
  6. [`experimental_protocol_proposal.md`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/docs/experiment_planning/experimental_protocol_proposal.md) — Desenho experimental consolidado para TL, DA e DG.
- **Tabelas CSV Auditadas:**
  - [`dataset_inventory_summary.csv`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/docs/experiment_planning/dataset_inventory_summary.csv)
  - [`mfpt_file_inventory.csv`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/docs/experiment_planning/mfpt_file_inventory.csv)
  - [`hust_gearbox_split_analysis.csv`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/docs/experiment_planning/hust_gearbox_split_analysis.csv)
  - [`class_compatibility_matrix.csv`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/docs/experiment_planning/class_compatibility_matrix.csv)

---

## 2. Principais Descobertas e Alertas Metodológicos

1. **Extração Parcial Silenciosa em `PU` e `UORED` nos Dados Atuais em Disco:**
   - Em **`PU`**, todos os 880 arquivos brutos do rótulo `28` (`Inner Ring Fault`, rolamentos `KI01..KI21`) falharam silenciosamente durante a execução original de `make_dataset.py` (`except Exception: continue`). Portanto, `data/processed_tl/detrended_rms/PU/` contém **6.476 janelas** de 3 classes (`Normal`: 1.918, `Outer`: 3.811, `Inner & Outer`: 747) e **0 janelas de `Inner` (`Class_28`)**.
   - Em **`UORED`**, 23 dos 60 arquivos `.mat` (incluindo todos os 10 arquivos de `Inner` e todos os 10 de `Outer`) falharam na busca da chave MATLAB interna em `UORED_raw.__getitem__`. Assim, `data/processed_tl/detrended_rms/UORED/` possui **370 janelas** em 3 classes (`Normal`: 180, `Ball`: 100, `Cage`: 90).
2. **Cobertura Completa das 5 Classes Globais pela Origem (`HUST + UORED + MFPT`):**
   - O dataset **`HUST` (951 janelas)** está 100% íntegro e é o único dataset de origem que contém **todas as 5 classes conceituais globais** (`Normal`: 150, `Ball`: 100, `Inner`: 150, `Outer`: 150, `Inner & Outer`: 150, além de 251 janelas em falhas compostas com esfera `Class_44` e `Class_46`).
   - Combinando `HUST + UORED + MFPT`, todas as 5 classes globais possuem entre 150 e 348 janelas de treinamento na origem (`Total = 1.058 janelas supervisionadas`), garantindo que **nenhuma classe dos alvos `CWRU_12k`, `CWRU_48k` ou `PU` fique órfã no treinamento multi-origem de DA e DG**.
3. **Verificação Crítica dos Rótulos de `CWRU_12k` e `CWRU_48k`:**
   - Confirmado em `vibdata/raw/labels.csv`: `Class_0 = Normal`, `Class_1 = Inner`, **`Class_2 = Outer`** e **`Class_3 = Ball`**. O comentário legado em `scripts/exp_tl_cross_dataset_mlp_tabnet.py` invertia `Class_2` e `Class_3`; corrigir esse mapeamento é essencial para não trocar `Outer` por `Ball` em DA e DG.
4. **Parecer do `MFPT` (Estudo B):**
   - **Apto com reamostragem obrigatória:** 6 arquivos de $270\text{ lbs}$ (`baseline_1..3` e `OuterRaceFault_1..3`) têm $f_s = 97.656\text{ Hz}$, enquanto os 14 arquivos `vload` têm $f_s = 48.828\text{ Hz}$. Sem decimação por 2, as janelas de $48.828$ pontos dos 6 arquivos duram $0{,}5\text{ s}$ e dobram artificialmente as frequências na FFT. Com `resample_poly(sig, up=1, down=2)`, obtêm-se **78 janelas exatas de $1{,}0\text{ s}$ a $48.828\text{ Hz}$** (`Normal`: 18, `Inner`: 21, `Outer`: 39).
5. **Parecer do `HUST_Gearbox` (Estudo A):**
   - O LOCO padrão de 30 condições é saturado (24/30 folds com `F1 = 1.0000`) porque 29 condições interpolam a condição de teste. A degradação concentra-se na fronteira de **Carga Máxima (`Load_4`, F1 médio `~0.82`)** e **Baixa Velocidade sob Carga Máxima (`Cond_20_4` com F1 `0.42–0.56` e `Cond_25_4` com F1 `0.56–0.58`)**, além da queda de representação linear no regime não estacionário **`VS_0_40Hz` (`F1 = 0.6287` em Linear Probing)**. Propõe-se o protocolo **Macro-LOCO de 6 Folds por Regime de Velocidade (`20Hz`, `25Hz`, `30Hz`, `35Hz`, `40Hz`, `VS_0_40Hz`)** complementado pelo fold extremo de carga **`Load_4`**.
6. **Invariância de Escala das 141 Características em DG Estrito (Estudo D):**
   - Das 141 características do `TabVibNet`, **126 ($89{,}4\%$)** — incluindo todas as 64 sub-bandas relativas WPD, todas as 26 features EMD e os fatores adimensionais no tempo/frequência — são **matematicamente invariantes a qualquer escalar positivo $\text{RMS}_{\text{ref}} > 0$**. Isso viabiliza o protocolo **DG estrito** usando exclusivamente o $\text{RMS}_{\text{ref}}$ e o `StandardScaler` dos domínios de origem.

---

## 3. Decisões Pendentes que Requerem Validação do Pesquisador

Antes de implementarmos os scripts definitivos de TL, DA e DG, solicitamos sua definição sobre os **4 pontos de decisão** abaixo:

1. **Re-extração Bruta de `PU` (`Class_28 / Inner`) e `UORED` (`Inner` e `Outer`) vs. Manutenção dos Dados Atuais em Disco:**
   - *Opção 1A (Recomendada se quiser completude física total de `PU` e `UORED`):* Corrigir o carregamento dos arquivos `.mat` que falharam silenciosamente e re-extrair `PU` (incluindo `Class_28 = Inner`) e `UORED` (incluindo `Inner` e `Outer`).
   - *Opção 1B (Mais rápida e compatível com execuções anteriores):* Manter `PU` (3 classes em disco: `Normal`, `Outer`, `Inner & Outer`) e `UORED` (370 janelas) exatamente como estão em disco, e processar apenas a correção de reamostragem do `MFPT` (78 janelas).
2. **Protocolo de Folds para `HUST_Gearbox` em TL:**
   - Aprova a adoção do **Macro-LOCO de 6 Folds por Regime de Velocidade (`20Hz`, `25Hz`, `30Hz`, `35Hz`, `40Hz`, `VS_0_40Hz`)** + destaque para o fold de fronteira de carga máxima **`Load_4`** em substituição (ou adição) aos 30 folds individuais?
3. **Protocolo de Folds para `CWRU_48k` em TL:**
   - Como a condição `Load_0HP_Sev_0.014` possui 0 amostras da `Class_1` (`Inner`), prefere avaliar o `CWRU_48k` agrupado pelas **4 cargas de motor (`Load_0HP`..`Load_3HP`, onde todas as 3 classes têm suporte completo em todos os folds)**, ou manter os 12 folds `Load_x_Sev_y`?
4. **Normalização de Amplitude em DG Estrito:**
   - Confirma a adoção da **Opção DG-Primária** (`detrend` linear dividido pela média geométrica $\text{RMS}_{\text{ref}}^{(\text{src, geo})}$ dos domínios de origem + `StandardScaler` ajustado exclusivamente nos domínios de origem, preservando 0% de uso de estatísticas do alvo)?

