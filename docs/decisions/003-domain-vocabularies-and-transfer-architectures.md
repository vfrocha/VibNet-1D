# ADR 003: Preservação dos Vocabulários de Classes por Domínio e Arquiteturas de Transferência (`Multi-Head MLP`, `SupCon MLP` e `Unsupervised TabNet`)

- **Status**: Aceito e Validado Experimentalmente
- **Contexto**: Transferência de aprendizado entre datasets com espaços de rótulos heterogêneos (*Heterogeneous Label-Space Transfer Learning*)
- **Artefatos Relacionados**:
  - `src/models/build_tabnet_resnet_balance.py`
  - `src/models/build_mlp_contrastive.py`
  - `scripts/exp_tl_contrastive_mlp.py`
  - `scripts/exp_tl_cross_dataset_mlp_tabnet.py`

## 1. Problema Científico

Os 10 datasets do **VibNet-1D** possuem diferentes componentes mecânicos (rolamentos, engrenagens, motores) e diferentes taxonomias de falha:
- `CWRU_12k`: 4 classes (`Class_0` a `Class_3`); `CWRU_48k`: 3 classes (`Class_1` a `Class_3`).
- `HUST`: 7 classes incluindo falhas compostas (`IO`, `IB`, `OB`).
- `PU`: 6 classes distinguindo danos artificiais (`KA0`, `KI0`) e reais/múltiplos (`KA1-3`, `KI1-2`, `KB2`).
- `UOEMD` e `UORED`: 5 classes incluindo gaiola (`C`).
- `HUST_Gearbox` e `Mechanical_Gear`: 4 classes de falha em dentes de engrenagem (`Broken`, `Miss`, `Root`).

Forçar um vocabulário único global entre todos os datasets exigiria colapsar classes distintas (perdendo granularidade diagnóstica) ou assumir equivalências físicas falsas (por exemplo, equiparar trinca na raiz de dente de engrenagem a falha na pista interna de rolamento).

## 2. Decisão Arquitetural

Preservar integralmente o **vocabulário de classes nativo de cada dataset** (`LabelEncoder` independente por domínio $d$, com $|\mathcal{Y}_d|$ classes) e acoplar os domínios exclusivamente através de um **extrator/encoder de representações compartilhado** $f_\theta: \mathbb{R}^{141} \to \mathbb{R}^{d_{\text{emb}}}$:

1. **Arquitetura 1 — `MLP` Multi-Cabeça Supervisionado (`HybridDLModel` / `ce`)**:
   - **Encoder compartilhado**: `Linear(141, 128) -> BatchNorm1d -> ReLU -> Dropout(0.3) -> Linear(128, 64) -> BatchNorm1d -> ReLU -> Dropout(0.3)` (ou perfil `256 -> 128`).
   - **Pré-treinamento Multi-Fonte**: Para cada domínio fonte $s \in \mathcal{S}$, instancia-se uma cabeça linear dedicada $g_{\phi_s}: \mathbb{R}^{d_{\text{emb}}} \to \mathbb{R}^{|\mathcal{Y}_s|}$. Em cada época de pré-treinamento, o otimizador (`Adam`, `lr=0.001`) atualiza $\theta$ e $\phi_s$ minimizando a entropia cruzada de cada domínio fonte no seu próprio espaço de classes.
   - **Fine-tuning no Domínio Alvo $t$**: As cabeças de fonte $\{\phi_s\}$ são descartadas; acopla-se uma nova cabeça $g_{\phi_t}: \mathbb{R}^{d_{\text{emb}}} \to \mathbb{R}^{|\mathcal{Y}_t|}$ e treina-se o conjunto $(\theta, \phi_t)$ nas $K-1$ condições de treino do alvo balanceadas via SMOTE.
2. **Arquitetura 2 — `MLP` Contrastivo Supervisionado por Domínio (`SupCon`)**:
   - Durante o pré-treinamento nas fontes, acopla-se uma cabeça de projeção $h_\psi: \mathbb{R}^{d_{\text{emb}}} \to \mathbb{S}^{63}$ (normalizada em $L_2$) e calcula-se a perda `SupConLoss` ($\tau = 0{,}07$) **dentro dos mini-batches de cada domínio fonte separadamente**. Assim, amostras de domínios diferentes nunca são tratadas como falsos positivos de mesma classe nem forçadas a compartilhar rótulos.
3. **Arquitetura 3 — `TabNet` (`TabNetPretrainer` Não Supervisionado + `TabNetClassifier`)**:
   - Como o `pytorch-tabnet` não suporta múltiplas cabeças supervisionadas simultâneas nativamente, as matrizes padronizadas dos domínios fonte são concatenadas ($X_{\text{sources}} \in \mathbb{R}^{N_{\text{total}} \times 141}$) para treinar um `TabNetPretrainer` auto-supervisionado (reconstrução de atributos mascarados com `pretraining_ratio=0.8`).
   - Os pesos pré-treinados do encoder de atenção são então transferidos via `from_unsupervised=pretrainer` para o `TabNetClassifier` supervisionado no domínio alvo.

## 3. Consequências e Resultados Observados

- Essa formulação permitiu transferir conhecimento de domínios fonte heterogêneos (`UORED`, `HUST`, `Mechanical_Gear`, `Electric_Motor`) para **5 domínios alvo** (`CWRU_12k`, `CWRU_48k`, `HUST_Gearbox`, `PU`, `UOEMD`), alcançando `Macro-F1` de **$0{,}9630 \pm 0{,}1106$** em `HUST_Gearbox` (`MLP` com fonte `UORED`) e **$0{,}9235 \pm 0{,}0131$** em `CWRU_48k` (`MLP` contrastivo `joint + expanded + asymmetric` com fontes `CWRU_12k|MFPT`).
- A mesma separação entre `encoder` compartilhado e cabeças específicas por domínio (`ModuleDict` de cabeças) será reutilizada diretamente na futura etapa `src/vibnet/signal/` para redes convolucionais 1D no domínio do tempo.
