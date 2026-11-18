# Auditoria e Protocolo Experimental da Matriz DA, DG e TL — VibNet-1D

- **Data:** 18/11/2026
- **Ambiente Auditado:** Máquina `andromeda` (`/home/vfrocha/VibNet-1D`, branch `main`, commit base `ed466ae`)
- **Escopo:** Auditoria completa da infraestrutura, datasets, gravações originais, pré-treinamento TabNet, compatibilidade da cabeça de 5 classes, normalização sem vazamento, balanceamento hierárquico e preparação modular da matriz **Domain Adaptation (DA)**, **Domain Generalization (DG)** e **Transfer Learning (TL)**.

---

## 1. Estrutura Atual do Projeto e Componentes Reutilizáveis

| Módulo / Script | Responsabilidade Atual | Reutilização na Nova Matriz (DA / DG / TL) |
| :--- | :--- | :--- |
| [`src/vibnet/data/transactional_io.py`](file:///home/vfrocha/VibNet-1D/src/vibnet/data/transactional_io.py) | Operações transacionais em staging com validação 1:1, backup e rollback coordenado. | Reutilizado integralmente para qualquer futura atualização de dados ou manifestos. |
| [`src/preprocessing/normalize_dataset_tl.py`](file:///home/vfrocha/VibNet-1D/src/preprocessing/normalize_dataset_tl.py) | Normalização `detrended_rms` por dataset completo. | Reutilizado para as fontes (`UORED`, `HUST`, `MFPT`). **Não utilizado diretamente nos destinos de DA/DG/TL**, onde o cálculo de $\text{RMS}_{\text{ref}}$ agora é isolado por partição (`fit_da_rms_normalizer` / `fit_dg_rms_normalizer`). |
| [`src/features/extractors_v2.py`](file:///home/vfrocha/VibNet-1D/src/features/extractors_v2.py) e [`src/features/signalai_wrapper.py`](file:///home/vfrocha/VibNet-1D/src/features/signalai_wrapper.py) | Extração canônica do vetor de **141 features** (`16` VibNet-1D + `125` SignAI = `141`). | Reutilizado integralmente via `default_141_feature_extractor`. |
| [`src/experiments/run_baseline_loco_v2.py`](file:///home/vfrocha/VibNet-1D/src/experiments/run_baseline_loco_v2.py) | Split LOCO, `scale_loco_fold_strict`, `apply_smote_with_audit`, validação interna e métricas. | Reutilizado para o protocolo **TL** (LOCO nos 5 destinos). |
| [`src/experiments/run_scratch_vs_tl_benchmark.py`](file:///home/vfrocha/VibNet-1D/src/experiments/run_scratch_vs_tl_benchmark.py) | `pretrain_tabnet_on_sources`, `train_eval_tabnet_variant` (`Scratch`, `Fine-Tuning`, `Linear Probing`) com verificação de congelamento (`_snapshot_module_tensors`). | Reutilizado e estendido em [`src/vibnet/evaluation/da_dg_tl_protocol.py`](file:///home/vfrocha/VibNet-1D/src/vibnet/evaluation/da_dg_tl_protocol.py) para a cabeça fixa de 5 classes em DA e DG. |
| [`src/experiments/run_scratch_vs_tl_classical_baselines.py`](file:///home/vfrocha/VibNet-1D/src/experiments/run_scratch_vs_tl_classical_baselines.py) | Treinamento e tuning interno 3-fold CV de `RandomForest` e `SVM`. | Reutilizado para os baselines clássicos em TL e adaptável para DG/DA. |
| [`src/vibnet/evaluation/provenance.py`](file:///home/vfrocha/VibNet-1D/src/vibnet/evaluation/provenance.py) | Coleta de hashes SHA-256, commit Git, versões de pacotes e gravação de `run_metadata.json`. | Reutilizado em todas as execuções da nova matriz. |

---

## 2. Inventário Real de Datasets, Classes, Gravações (`sNNNNN`) e Janelas em Disco (Andromeda)

A inspeção direta de `data/processed/` e `data/processed_tl/detrended_rms/` na máquina **Andromeda** (após o saneamento do `HUST` para 951 janelas) confirmou que cada janela `.npy` segue o padrão `<dataset>/<condição>/<classe>/sNNNNN_wNN.npy`, onde `sNNNNN` identifica inequivocamente o índice da gravação original no adaptador `vibdata` (`0` gravações compartilhadas entre condições ou classes distintas):

| Dataset | Papel na Matriz | $f_s$ (Hz) | Unidade Física | Condições | Gravações Únicas (`sNNNNN`) | Janelas em `processed/` | Janelas em `detrended_rms/` | Distribuição por Pasta de Classe (`Janelas` / `Gravações`) |
| :--- | :--- | ---: | :---: | ---: | ---: | ---: | ---: | :--- |
| **`UORED`** | Fonte (`S1`, `S4`) | 42.000 | $\text{m/s}^2$ | 18 | **37** | 370 | 370 | `Class_Normal`: 180 (18 grav.)<br>`Class_Ball`: 100 (10 grav.)<br>`Class_Cage`: 90 (9 grav., *fora das 5 classes*) |
| **`HUST`** *(HUST Bearing)* | Fonte (`S2`, `S4`) | 51.200 | $\text{m/s}^2$ | 3 (`0W, 200W, 400W`) | **99** | **951** | **951** | `Class_Normal`: 150 (15 grav.)<br>`Class_43` (`Ball`): 100 (12 grav.)<br>`Class_41` (`Inner`): 150 (15 grav.)<br>`Class_42` (`Outer`): 150 (15 grav.)<br>`Class_45` (`Inner & Outer`): 150 (15 grav.)<br>`Class_44` (`IB`): 120 (12 grav., *fora*)<br>`Class_46` (`OB`): 131 (15 grav., *fora*) |
| **`MFPT`** | Fonte (`S3`, `S4`) | 48.828 *(97.656 bruto em 270 lbs)* | $g$ | 9 (`Load_0..300`) | **20** | **114** *(pré-harmonização)* | **Ausente** *(0)* | `Class_23` (`Normal`): 36 jan. (3 grav.; *18 jan. após decimação 2x*)<br>`Class_24` (`Inner`): 21 jan. (7 grav.)<br>`Class_25` (`Outer`): 57 jan. (10 grav.; *39 jan. após decimação 2x*) |
| **`CWRU_12k`** | Destino (`DA, DG, TL`) | 12.000 | $g$ | 4 (`Load_0..3HP`) | **232** | 2.520 | 2.520 | `Class_0` (`Normal`): 280 (**8 grav.**, 2/carga)<br>`Class_1` (`Inner`): 760 (76 grav.)<br>`Class_2` (`Outer`): 720 (72 grav.)<br>`Class_3` (`Ball`): 760 (76 grav.) |
| **`CWRU_48k`** | Destino (`DA, DG, TL`) | 48.000 | $g$ | 12 (`Load x Sev`) | **70** | 602 | 602 | `Class_1` (`Inner`): 194 (22 grav.)<br>`Class_2` (`Outer`): 204 (24 grav.)<br>`Class_3` (`Ball`): 204 (24 grav.)<br>*(Sem `Class_0` / `Normal` nativo)* |
| **`PU`** | Destino (`DA, DG, TL`) | 64.000 | $\text{m/s}^2$ | 4 (`C1..C4`) | **1.627** | 6.476 | 6.476 | `Class_Normal`: 1.918 (480 grav.)<br>`Class_27` (`Outer`): 3.811 (959 grav.)<br>`Class_29` (`Inner & Outer`): 747 (188 grav.)<br>*(`Class_28` / `Inner`: 0 em disco)* |
| **`HUST_Gearbox`** | Destino (`TL`) | 25.600 | $g$ | 30 | **90** | 900 | 900 | `Class_Normal`: 300 (30 grav.)<br>`Class_Broken`: 300 (30 grav.)<br>`Class_Missing`: 300 (30 grav.) |
| **`UOEMD`** | Destino (`TL`) | 42.000 | $g$ | 16 | **128** | 1.280 | 1.280 | 8 classes $\times$ 160 janelas (16 grav./classe) |

---

## 3. Auditoria Específica dos Checkpoints e do Pré-Treinamento TabNet (Seção 9)

1. **Existência de Checkpoints em Disco:**
   - A busca exaustiva no repositório (`*.pt`, `*.pth`, `*.zip`) confirmou que **não existem arquivos binários de checkpoint TabNet persistidos em disco**.
   - Nos experimentos anteriores (`run_scratch_vs_tl_benchmark.py` e `build_tabnet_resnet_balance.py`), o `TabNetPretrainer` era treinado em memória RAM no início da execução (`_PRETRAINER_CACHE`) e repassado em memória para o `TabNetClassifier` (`from_unsupervised=unsupervised_model`) ou para o `Linear Probing` (`copy.deepcopy(pretrained_tabnet.network)`).
2. **Procedimento e Hiperparâmetros do Pré-Treinamento Existente:**
   - **Classe utilizada:** `pytorch_tabnet.pretraining.TabNetPretrainer` (**auto-supervisionado**, reconstrução de atributos mascarados com `pretraining_ratio = 0.8`, sem uso de rótulos).
   - **Arquitetura em `run_scratch_vs_tl_benchmark.py` (`configs/scratch_vs_tl.yaml`):** `n_d=16`, `n_a=16`, `n_steps=4`, `gamma=1.3`, `n_independent=2`, `n_shared=2`, `mask_type="sparsemax"`, `lr=0.02`, `pretrain_epochs=30`, `batch_size=64`, dimensão latente $d_{\text{latent}} = n_d = 16$.
   - *(Observação: o módulo legado `src/models/build_tabnet_resnet.py` utilizava `n_steps=5, gamma=1.5, mask_type='entmax'`. Na nova matriz padronizamos os hiperparâmetros auditados de `configs/scratch_vs_tl.yaml`: `n_d=16, n_a=16, n_steps=4, gamma=1.3, mask_type='sparsemax'`).*
3. **Datasets que Participaram do Pré-Treinamento Anterior vs. Nova Matriz:**
   - Nos experimentos anteriores (`configs/scratch_vs_tl.yaml`), o pool de pré-treinamento era `["UORED", "HUST", "Mechanical_Gear", "Electric_Motor"]`. **Nenhum dos 5 datasets de destino (`CWRU_12k`, `CWRU_48k`, `PU`, `HUST_Gearbox`, `UOEMD`) jamais participou do pré-treinamento** (risco zero de vazamento de destino).
   - Na nova matriz (`configs/da_dg_tl_matrix.yaml`), o pré-treinamento auto-supervisionado do `TabNetPretrainer` será executado especificamente sobre cada configuração de origem solicitada (**`S1`: `UORED`**, **`S2`: `HUST`**, **`S3`: `MFPT`**, **`S4`: `UORED + HUST + MFPT`**) e salvo em disco (`checkpoints/tabnet_pretrainer_<S_k>_seed<SEED>.pt`) junto com seu manifesto de proveniência (`features`, ordem das 141 features, `StandardScaler` e hash SHA-256).
4. **Ordem e Dimensão das 141 Features (`input_dim = 141`):**
   - `indices 0..15` (`16` features): `src/features/extractors_v2.py::extract_advanced_features` (`7` Tempo + `6` PSD/Welch + `3` STFT).
   - `indices 16..140` (`125` features): `src/features/signalai_wrapper.py::SignalAIWrapper(pipeline_name='all')` (`11` Tempo + `15` Frequência + `73` WPD `db4` + `26` EMD).

---

## 4. Diagnóstico de Compatibilidade das Configurações `S1..S4` com a Cabeça de 5 Classes (Seção 3)

### 4.1. Cobertura de Classes por Configuração de Origem

| Config. | Fontes | `0: Normal` | `1: Ball` | `2: Inner` | `3: Outer` | `4: Inner & Outer` | Classes Ausentes na Fonte |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`S1`** | `UORED` | **180 jan.** (18 grav.) | **100 jan.** (10 grav.) | **0** | **0** | **0** | `Inner`, `Outer`, `Inner & Outer` (3 ausentes) |
| **`S2`** | `HUST` | **150 jan.** (15 grav.) | **100 jan.** (12 grav.) | **150 jan.** (15 grav.) | **150 jan.** (15 grav.) | **150 jan.** (15 grav.) | **Nenhuma (5/5 classes cobertas)** |
| **`S3`** | `MFPT` | *0 sup.* *(36/18 p/ $\text{RMS}_{\text{ref}}$)* | **0** | **21 jan.** (7 grav.) | **57/39 jan.** (10 grav.) | **0** | `Normal`*, `Ball`, `Inner & Outer` (3 ausentes) |
| **`S4`** | `UORED + HUST + MFPT` | **330 jan.** (33 grav.) | **200 jan.** (22 grav.) | **171 jan.** (22 grav.) | **207/189 jan.** (25 grav.) | **150 jan.** (15 grav.) | **Nenhuma (5/5 classes cobertas)** |

### 4.2. Impacto no Treinamento e Transferência dos Pesos da Cabeça (`nn.Linear(16, 5)`) e Solução Metodológica Implementada

Em `S1` (`UORED`, apenas classes `{0, 1}`) e `S3` (`MFPT`, apenas classes `{2, 3}`), uma camada linear `nn.Linear(16, 5)` treinada com `CrossEntropyLoss` nas fontes nunca recebe exemplos positivos das classes ausentes $\mathcal{C}_{\text{absent}}$:
- Se os pesos e vieses das linhas $c \in \mathcal{C}_{\text{absent}}$ permanecerem com sua inicialização aleatória sem restrição, ou se sofrerem apenas gradiente negativo, o comportamento dessas saídas em **DG** seria arbitrário.
- **Solução Metodológica Defensável Implementada em [`src/vibnet/evaluation/da_dg_tl_protocol.py`](file:///home/vfrocha/VibNet-1D/src/vibnet/evaluation/da_dg_tl_protocol.py):**
  1. **Espaço de saída estritamente fixo em 5 classes (`head.out_features == 5`):** A cabeça sempre possui 5 neurônios de saída correspondentes a `(0: Normal, 1: Ball, 2: Inner, 3: Outer, 4: Inner & Outer)`.
  2. **Rastreamento explícito de classes observadas (`observed_class_ids`) e Mascaramento Estrutural de Logits para Classes Não Aprendidas (`-1e9`):**
     - O checkpoint da cabeça (`FiveClassHeadCheckpoint`) registra `observed_class_ids` (as classes com suporte real no treinamento).
     - Em **DG** (onde não há adaptação ao destino), qualquer classe $c \notin \text{observed\_class\_ids}$ recebe máscara estrutural de logit $-\infty$ (`-1e9`), garantindo probabilidade $P(Y=c \mid x) = 0$. Assim, o modelo **jamais "adivinha" por ruído aleatório uma classe que nunca viu nas fontes**, e toda amostra real do destino pertencente a uma classe não aprendida pela fonte resulta deterministicamente em $\text{Recall}_c = 0$ e $\text{F1}_c = 0$.
     - Em **DA** (onde a cabeça do destino é inicializada com cópia de `head.state_dict()` da fonte e adaptada nos 20% rotulados do destino), os pesos das classes já aprendidas na fonte (`observed_source`) são transferidos como *warm-start*, enquanto as classes novas presentes nos 20% de adaptação do destino passam a integrar `combined_observed = observed_source | observed_target_adapt` e têm seus pesos aprendidos durante a adaptação com o encoder congelado!

---

## 5. Diagnóstico de Normalização (`DA` e `DG`) e Balanceamento Hierárquico

### 5.1. Incompatibilidade Física da Média Simples de $\text{RMS}_{\text{ref}}$ em DG (Seção 4.2)
A auditoria quantitativa dos sinais detrended em `data/processed/` na máquina Andromeda revelou:
- **`UORED`**: unidade física $\text{m/s}^2$, $f_s = 42.000\text{ Hz}$, $\text{RMS}_{\text{ref}}(\text{Normal}) = \mathbf{5{,}850288}$ (amplitude máxima $1.072{,}49\text{ m/s}^2$);
- **`HUST`**: unidade física $\text{m/s}^2$, $f_s = 51.200\text{ Hz}$, $\text{RMS}_{\text{ref}}(\text{Normal}) = \mathbf{0{,}181141}$ (amplitude máxima $51{,}49\text{ m/s}^2$);
- **`MFPT`**: unidade física $g$ ($\approx 9{,}80665\text{ m/s}^2$), $f_s = 48.828\text{ Hz}$, $\text{RMS}_{\text{ref}}(\text{Normal}) = \mathbf{0{,}873542\text{ g}}$.

**Problema verificado:**
1. As unidades físicas **não são homogêneas** (`UORED` e `HUST` estão em $\text{m/s}^2$, enquanto `MFPT` e `CWRU` estão em $g$).
2. A razão entre o maior e o menor $\text{RMS}_{\text{ref}}$ nas fontes é de **$32{,}30\times$** ($5{,}850288 / 0{,}181141$). A média aritmética simples ($\text{RMS}_{\text{ref}}^{\text{arith}} = 2{,}301657$) é dominada em $84{,}7\%$ pelo `UORED`.
3. **Como implementamos em `fit_dg_rms_normalizer()`:** Conforme exigido pela Seção 4.2 (*"Caso as referências não sejam comparáveis, não implemente silenciosamente a média"*), o modo padrão `strict_check_arithmetic` executa `assess_source_rms_comparability()` e **bloqueia a média aritmética silenciosa** lançando `IncomparableSourceRMSReferenceError` com o relatório completo de unidades e razão de escala ($32{,}30\times$).
4. **Alternativas Metodológicas Disponíveis para Sua Decisão (Seção 7 abaixo):**
   - **Alternativa DG-1 (Média Geométrica das Fontes `explicit_geometric_mean` = $0{,}974586$):** Invariante à dominância de uma única ordem de grandeza; como **126 das 141 features ($89{,}4\%$)** são matematicamente invariantes a qualquer escalar positivo $c > 0$, 126 features permanecem exatas e 15 features de energia absoluta usam a referência geométrica das fontes (zero dados do destino).
   - **Alternativa DG-2 (Subconjunto das 126 Features Invariantes a Escala em DG):** Utilizar exclusivamente as 126 features adimensionais/relativas (ou normalizar cada janela individual pelo seu próprio RMS apenas em DG), tornando o modelo 100% imune à diferença de ganho entre acelerômetros ($g$ vs. $\text{m/s}^2$) com zero estatísticas do destino.

### 5.2. Gargalo de Gravações em `CWRU_12k` (`Class_0`) e `CWRU_48k` (Ausência de `Normal`) para DA (Seção 4.1 e 5.1)
1. **`CWRU_12k` (`Class_0 / Normal` possui apenas 8 gravações no total — 2 por carga):**
   - Se reservarmos $20\%$ das 8 gravações de `Class_0` para adaptação, $\text{round}(8 \times 0{,}20) = \mathbf{2\text{ gravações}}$ para adaptação (e $6$ para teste).
   - Ao dividir essas 2 gravações de adaptação em treino e validação por gravação, a única divisão possível sem quebrar gravações é **1 gravação (12,5% = 20–40 janelas) para `adapt_train` e 1 gravação (12,5% = 20–40 janelas) para `adapt_val`** (enquanto `Class_1..3` têm 72–76 gravações = 15 gravações na adaptação -> 12 `adapt_train` e 3 `adapt_val`).
   - Nossa função `create_da_recording_split()` suporta exatamente `min_adapt_train_recordings_per_class=1` e `min_adapt_val_recordings_per_class=1` (permitindo `1 train + 1 val + 6 test` em `Class_0` de `CWRU_12k` sem jamais dividir janelas da mesma gravação!) e interrompe com `InsufficientRecordingsError` se qualquer classe tiver menos de 2 gravações na adaptação.
2. **`CWRU_48k` (0 gravações da classe `Normal`):**
   - Como `CWRU_48k` possui apenas `Class_1` (`Inner`), `Class_2` (`Outer`) e `Class_3` (`Ball`), não há sinais normais próprios na partição de adaptação de 20% de `CWRU_48k`.
   - Em `fit_da_rms_normalizer()`, exigimos que seja passado explicitamente `cross_reference_normal_windows` vindo **exclusivamente das gravações normais da partição de adaptação de 20% do dataset irmão `CWRU_12k`** (mesma bancada física CWRU), sem tocar nas gravações de teste.

---

## 6. Comandos para Executar Futuramente Cada Configuração

```bash
# 1. Executar a suíte de testes unitários e sintéticos (11 requisitos da Seção 13)
python3 -m unittest tests/test_da_dg_tl_protocol.py -v

# 2. Inspecionar o plano completo da matriz DA, DG e TL e diagnóstico de cobertura/RMS em modo dry-run
python3 src/experiments/run_da_dg_tl_matrix.py --dry-run

# 3. Inspecionar apenas um protocolo ou configuração específica (ex.: DG com S1..S4)
python3 src/experiments/run_da_dg_tl_matrix.py --protocols DG --source-configs S1 S2 S3 S4 --dry-run
```

