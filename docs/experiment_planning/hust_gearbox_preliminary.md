# Estudo Preliminar do Dataset HUST Gearbox (Estudo Preliminar A)

## 1. Contexto e Motivação

Em experimentos preliminares de **Leave-One-Condition-Out (LOCO)** com 30 folds individuais no dataset **`HUST_Gearbox`**, os modelos clássicos (**Random Forest** e **SVM**) e o **TabNet From Scratch** atingiram desempenho próximo da saturação ($\text{Macro-F1} \approx 0{,}96$ na média global, com **24 dos 30 folds cravando $\text{Macro-F1} = 1{,}0000$**).

Um benchmark onde o modelo *From Scratch* e os classificadores clássicos já atingem $1{,}0000$ na maioria das condições possui **efeito teto (*ceiling effect*)**, impedindo observar ganhos estatisticamente significativos do **Transfer Learning (TL)**. Este estudo investiga a estrutura física das 30 condições do `HUST_Gearbox` e propõe recortes experimentais legítimos, fisicamente motivados e sem vazamento de dados nos quais o problema é genuinamente desafiador.

Os dados completos por condição e regime estão salvos em [`docs/experiment_planning/hust_gearbox_split_analysis.csv`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/docs/experiment_planning/hust_gearbox_split_analysis.csv).

---

## 2. Estrutura Física do Dataset `HUST_Gearbox`

A auditoria do manifesto [`results/scratch_vs_tl/run_20261105_141743/fold_manifest.csv`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/results/scratch_vs_tl/run_20261105_141743/fold_manifest.csv) e de `vibdata/raw/HUST_gearbox/HUST_gearbox.csv` confirmou a seguinte estrutura exata:

- **Taxa de amostragem:** $f_s = 25.600\text{ Hz}$, janelas de $1{,}0\text{ s}$ ($25.600$ amostras), sem sobreposição (`overlap = 0`).
- **Total de gravações e janelas:** **90 gravações brutas** $\rightarrow$ **900 janelas** (exatamente 10 janelas por gravação).
- **Classes (3 classes perfeitamente balanceadas):**
  - `Class_Normal` (Engrenagem saudável): **300 janelas** (30 gravações, 1 por condição);
  - `Class_Broken` (Dente quebrado / *Broken Tooth*): **300 janelas** (30 gravações, 1 por condição);
  - `Class_Missing` (Dente faltante / *Missing Tooth*): **300 janelas** (30 gravações, 1 por condição).
- **Condições Operacionais (30 condições, 30 janelas por condição = 10 por classe):**
  1. **Grade de Velocidade Constante (25 condições — `Cond_<Speed>_<Load>`):**
     - **5 velocidades de rotação:** `20 Hz`, `25 Hz`, `30 Hz`, `35 Hz`, `40 Hz`;
     - **5 níveis de carga:** `0`, `1`, `2`, `3`, `4`;
     - Total: $5 \times 5 = 25$ condições (750 janelas).
  2. **Regime de Velocidade Variável / Run-up & Run-down (5 condições — `Cond_L<Load>_VS_0_40_0`):**
     - Variação contínua de velocidade `0 -> 40 -> 0 Hz` sob os **5 níveis de carga** (`L0`, `L1`, `L2`, `L3`, `L4`);
     - Total: 5 condições (150 janelas).

### Por que o LOCO padrão de 30 folds é trivial na maioria dos folds?
No protocolo LOCO de 30 folds (treino em 29 condições = 870 janelas; teste em 1 condição = 30 janelas), quando a condição de teste é intermediária (por exemplo, `Cond_30_2`), o conjunto de treinamento contém simultaneamente a mesma velocidade nas cargas vizinhas (`Cond_30_1`, `Cond_30_3`) e a mesma carga nas velocidades vizinhas (`Cond_25_2`, `Cond_35_2`). O classificador realiza **interpolação densa na grade velocidade $\times$ carga**, e não extrapolação de domínio.

---

## 3. Análise Quantitativa de Desempenho por Regime Operacional (`detrended_rms`, 141 Features)

Utilizando as execuções reais em [`results/baseline_loco_results_20261003_052352.csv`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/results/baseline_loco_results_20261003_052352.csv) (RF e SVM) e [`results/scratch_vs_tl/run_20261105_141743/results.csv`](file:///c:/Users/victor.rocha/Documents/VibNet-1D/results/scratch_vs_tl/run_20261105_141743/results.csv) (TabNet Scratch, TL Fine-tune e TL Linear), agregamos o desempenho por **Grupo de Carga (`Load_Group`)**, **Grupo de Velocidade (`Speed_Group`)** e **Condições Críticas Individuais**:

### 3.1. Desempenho Agregado por Nível de Carga (`Load_Group` — 6 condições / 180 janelas por carga)

| Grupo de Carga | Nº Condições | Janelas de Teste | RF Macro-F1 (Média ± DP) | SVM Macro-F1 (Média ± DP) | TabNet Scratch F1 | TabNet TL Fine-Tune F1 | TabNet TL Linear F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`Load_0`** (Sem carga) | 6 | 180 | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | `0.9889 ± 0.0272` | `1.0000 ± 0.0000` | `0.9033 ± 0.1458` |
| **`Load_1`** (Carga baixa) | 6 | 180 | `0.9704 ± 0.0726` | `0.9944 ± 0.0136` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | `0.8926 ± 0.1726` |
| **`Load_2`** (Carga média) | 6 | 180 | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | `0.9776 ± 0.0548` | `0.9721 ± 0.0683` | `0.9163 ± 0.1190` |
| **`Load_3`** (Carga alta) | 6 | 180 | `1.0000 ± 0.0000` | `0.9944 ± 0.0136` | `0.9944 ± 0.0136` | `0.9889 ± 0.0172` | `0.9052 ± 0.1710` |
| **`Load_4` (Carga máxima — Fronteira)** | **6** | **180** | **`0.8281 ± 0.2182`** | **`0.8193 ± 0.2440`** | **`0.8125 ± 0.2458`** | **`0.8229 ± 0.2422`** | **`0.8168 ± 0.2335`** |

### 3.2. Desempenho Agregado por Regime de Velocidade (`Speed_Group` — 5 condições / 150 janelas por velocidade)

| Grupo de Velocidade | Nº Condições | Janelas de Teste | RF Macro-F1 (Média ± DP) | SVM Macro-F1 (Média ± DP) | TabNet Scratch F1 | TabNet TL Fine-Tune F1 | TabNet TL Linear F1 |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **`20Hz` (Velocidade mínima — Fronteira)** | **5** | **150** | **`0.9111 ± 0.1987`** | **`0.8923 ± 0.2409`** | **`0.8848 ± 0.2575`** | **`0.8962 ± 0.2320`** | **`0.8950 ± 0.2348`** |
| **`25Hz` (Velocidade baixa)** | **5** | **150** | **`0.8756 ± 0.1948`** | **`0.9111 ± 0.1987`** | **`0.9157 ± 0.1886`** | **`0.9111 ± 0.1987`** | **`0.9222 ± 0.1740`** |
| **`30Hz`** (Velocidade intermediária) | 5 | 150 | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` | `1.0000 ± 0.0000` |
| **`35Hz`** (Velocidade alta) | 5 | 150 | `0.9715 ± 0.0638` | `0.9933 ± 0.0149` | `0.9745 ± 0.0571` | `0.9679 ± 0.0459` | `0.9613 ± 0.0865` |
| **`40Hz`** (Velocidade máxima) | 5 | 150 | `1.0000 ± 0.0000` | `0.9933 ± 0.0149` | `0.9867 ± 0.0298` | `0.9800 ± 0.0447` | `0.9798 ± 0.0451` |
| **`VS_0_40Hz` (Velocidade Variável)** | **5** | **150** | `1.0000 ± 0.0000` | `0.9800 ± 0.0447` | `0.9665 ± 0.0562` | `0.9866 ± 0.0299` | **`0.6287 ± 0.0539`** |

### 3.3. Todas as Condições com Queda de Desempenho ($\text{Macro-F1} < 1{,}00$)

| Condição de Teste | Regime | Velocidade | Carga | RF Macro-F1 | SVM Macro-F1 | TabNet Scratch F1 | TabNet TL Fine-Tune F1 | TabNet TL Linear F1 |
| :--- | :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: |
| **`Cond_20_4`** | Constante | `20Hz` | **`Load_4`** | **`0.5556`** | **`0.4613`** | **`0.4242`** | **`0.4815`** | **`0.4747`** |
| **`Cond_25_4`** | Constante | `25Hz` | **`Load_4`** | **`0.5556`** | **`0.5556`** | **`0.5783`** | **`0.5556`** | **`0.6111`** |
| **`Cond_35_4`** | Constante | `35Hz` | **`Load_4`** | **`0.8573`** | `0.9666` | **`0.8724`** | `0.8977` | `0.8066` |
| **`Cond_25_1`** | Constante | `25Hz` | `Load_1` | **`0.8222`** | `0.9666` | `1.0000` | `1.0000` | `1.0000` |
| **`Cond_L4_VS_0_40_0`**| Variável | `VS_0_40Hz` | **`Load_4`** | `1.0000` | **`0.9000`** | `1.0000` | `1.0000` | **`0.6085`** |
| **`Cond_L2_VS_0_40_0`**| Variável | `VS_0_40Hz` | `Load_2` | `1.0000` | `1.0000` | **`0.8657`** | **`0.8325`** | **`0.6844`** |
| **`Cond_L0_VS_0_40_0`**| Variável | `VS_0_40Hz` | `Load_0` | `1.0000` | `1.0000` | **`0.9333`** | `1.0000` | **`0.6119`** |

---

## 4. Interpretação Física e Metodológica

A análise revela dois fenômenos físicos claros:
1. **Degradação severa no regime de Carga Máxima (`Load_4`) combinada com Baixa Velocidade (`20Hz` e `25Hz`):**
   - Em `Cond_20_4` e `Cond_25_4`, o `Macro-F1` cai para **`0.42–0.56`** em todos os modelos (RF, SVM e TabNet), mesmo quando outras 29 condições estão no treino! Isso ocorre porque sob baixa rotação ($20\text{–}25\text{ Hz}$) e torque frenante máximo (`Load_4`), a energia dos impactos de engrenamento (*gear mesh*) é fortemente amortecida pelo carregamento torsional, alterando a assinatura impulsiva entre `Broken Tooth` e `Missing Tooth`.
2. **Deslocamento de representação no regime Não Estacionário (`VS_0_40Hz`):**
   - Enquanto árvores de decisão (RF) conseguem memorizar limiares se outras 4 cargas de `VS_0_40Hz` estiverem no conjunto de treinamento (como ocorre no LOCO de 30 folds, onde `Cond_L0..L3_VS_0_40_0` treinam o modelo para testar `Cond_L4_VS_0_40_0`), o **TabNet TL Linear** cai para **`Macro-F1 = 0.6287`** em todas as 5 condições `VS_0_40Hz`.
   - Mais importante: no LOCO de 30 folds, quando testamos uma condição `VS_0_40Hz`, **as outras 4 condições `VS_0_40Hz` estão no conjunto de treino**! Se avaliarmos a **transferência real de Velocidade Constante (25 condições) $\rightarrow$ Velocidade Variável (5 condições)**, nenhuma condição `VS_0_40Hz` é vista no treinamento!

---

## 5. Protocolos Recomendados para Substituir ou Complementar o LOCO de 30 Folds em `HUST_Gearbox`

Para evitar *cherry-picking* arbitrário de condições isoladas e, ao mesmo tempo, eliminar a interpolação trivial do LOCO de 30 folds, recomendamos **dois protocolos estruturados por blocos físicos (Macro-LOCO por Regime)**:

### Recomendação Principal — Protocolo `HUST_Gearbox` Macro-Regime (6 Folds Agrupados por Velocidade / Estacionariedade — `LOSO-Regime`)
Agrupar as 30 condições nos **6 regimes físicos de velocidade** (cada fold retém todas as 5 cargas `Load 0..4` daquele regime de velocidade para teste):
- **Fold 1 (`Speed_20Hz` — Extrapolação de Baixa Velocidade):** Treino em 25 condições (`750` janelas) $\rightarrow$ Teste nas 5 condições de `20Hz` (`150` janelas, 50 por classe).
- **Fold 2 (`Speed_25Hz`):** Treino em 25 condições (`750` janelas) $\rightarrow$ Teste nas 5 condições de `25Hz` (`150` janelas, 50 por classe).
- **Fold 3 (`Speed_30Hz`):** Treino em 25 condições (`750` janelas) $\rightarrow$ Teste nas 5 condições de `30Hz` (`150` janelas, 50 por classe).
- **Fold 4 (`Speed_35Hz`):** Treino em 25 condições (`750` janelas) $\rightarrow$ Teste nas 5 condições de `35Hz` (`150` janelas, 50 por classe).
- **Fold 5 (`Speed_40Hz` — Extrapolação de Alta Velocidade):** Treino em 25 condições (`750` janelas) $\rightarrow$ Teste nas 5 condições de `40Hz` (`150` janelas, 50 por classe).
- **Fold 6 (`Speed_VS_0_40Hz` — Extrapolação Estacionário $\rightarrow$ Não Estacionário):** Treino exclusivamente nas 25 condições de velocidade constante (`750` janelas) $\rightarrow$ Teste nas 5 condições de velocidade variável `Cond_L0..L4_VS_0_40_0` (`150` janelas, 50 por classe).

**Vantagens científicas deste protocolo de 6 Macro-Folds:**
1. **Zero *cherry-picking*:** Todas as 900 janelas e todas as 30 condições são avaliadas exatamente uma vez no teste (6 folds $\times$ 150 janelas = 900 janelas), mantendo **6x menos custo computacional** que os 30 folds.
2. **Elimina o vazamento entre condições de mesmo regime de velocidade:** Ao testar `20Hz` ou `VS_0_40Hz`, o modelo **não tem nenhuma condição naquela velocidade (nem em rampa variável) no conjunto de treino**, forçando extrapolação real de frequência de engrenamento (*gear mesh frequency*).
3. **Recorte de Fronteira Complementar (`Load-Level Macro-LOCO`, 5 Folds por Carga `Load_0`..`Load_4`):** Pode ser reportado como análise complementar, destacando especificamente o fold de fronteira de carga máxima **`Load_4`** (180 janelas de teste, onde o `Macro-F1` cai para `~0.81–0.82`).

