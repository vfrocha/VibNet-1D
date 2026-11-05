# Relatório de Auditoria de Compatibilidade e Reprodutibilidade — Benchmark Scratch vs. Transfer Learning (`scratch_vs_tl`)

- **Data/Hora UTC da Auditoria**: `2026-11-05T16:54:20.003854+00:00`
- **Commit Git (`HEAD`)**: `77c773b1580fb730468793ab1722d2837f4e9d14` (branch `main`, `git_dirty=True`)
- **Diretório de Referência Inicial (`baseline_loco_v2`)**: `/Users/vrocha/Documents/VibNet-1D/results/baseline_loco_v2/run_20261104_160031` (`present_on_disk=False`)

---

## 1. Resumo Executivo da Auditoria (14 Dimensões Inspecionadas)

Antes de implementar ou executar o benchmark **Scratch vs. Transfer Learning (`MLP` e `TabNet`)**, foram inspecionados todos os scripts, modelos, configurações e artefatos de resultados existentes no repositório **VibNet-1D**:

1. **Implementação atual do `baseline_loco_v2`** ([`src/experiments/run_baseline_loco_v2.py`](../../../src/experiments/run_baseline_loco_v2.py)):
   - Avalia os 5 datasets alvo (`CWRU_12k`: 4 folds, `CWRU_48k`: 12 folds, `HUST_Gearbox`: 30 folds, `UOEMD`: 16 folds, `PU`: 4 folds; total = **66 folds LOCO**), variante `detrended_rms` (`data/processed_tl/detrended_rms/`), 141 features (`extract_fusion_features` + `extract_advanced_features`), estratégias `["none", "smote"]` e seeds `[42, 123, 2026]`.
   - O diretório `results/baseline_loco_v2/run_20261104_160031/` foi executado no servidor Linux após o commit `77c773b`.
2. **Implementações existentes de `MLP Scratch` e `TabNet Scratch`**:
   - Em `run_baseline_loco_v2.py`: `MLP Scratch` usa `HybridDLModel` (`141->256->128->64` + `Bottleneck 64->128->64` + `Linear(64->C_T)`) treinado com `AdamW(lr=1e-3, weight_decay=1e-4)` e `StepLR(10, 0.5)` por 30 épocas (`batch_size=64`), com proteção `trim_single_remainder_sample`. `TabNet Scratch` usa `TabNetClassifier(n_d=16, n_a=16, n_steps=4, gamma=1.3, mask_type='sparsemax')` com validação interna agrupada por condição operacional (`patience=15`, `max_epochs=30`) seguida de *refit* no treino completo permitido.
   - Em `run_baseline_experiments.py` (legado): Usava `data/processed/`, omitia `UOEMD`, vazava `X_test` no `eval_set` do `TabNet` e continha bug de deslocamento de colunas no `MLP`. **Incompatível e descartado para comparação.**
3. **Implementações existentes de `MLP` e `TabNet` Transfer Learning (`Fine-Tuning`)**:
   - Em [`scripts/exp_tl_cross_dataset_mlp_tabnet.py`](../../../scripts/exp_tl_cross_dataset_mlp_tabnet.py) (`results/tl_cross_dataset_mlp_tabnet/run_20261028_173310`): Avaliou os 5 alvos (66 folds) e 15 subconjuntos das 4 fontes dedicadas (`UORED`, `HUST`, `Mechanical_Gear`, `Electric_Motor`) com 3 seeds (`42, 123, 2026`).
   - **Incompatibilidades identificadas em `run_20261028_173310`**:
     - Avaliou **apenas** `balance_strategy = "smote"` (ausência total de `balance_strategy = "none"`).
     - Avaliou **apenas** `Fine-Tuning` (ausência total de `Linear Probing`).
     - **Não exportou** predições por amostra (`predictions.csv`) nem manifesto de folds (`fold_manifest.csv`).
     - Apresentou **divergência de taxa de amostragem ($f_s$) em 2 fontes**: `Mechanical_Gear` foi extraído com `fs=10000` (canônico: `5000 Hz`) e `Electric_Motor` com `fs=42000` (canônico: `50000 Hz`), reescalonando as features espectrais FFT/PSD dessas duas fontes.
     - No alvo `PU` (fold `C4_1500rpm_0.7Nm_400N`), o `TabNet` falhou nas 45 avaliações (`8577 % 64 == 1` no `BatchNorm1d`) por ausência de `trim_single_remainder_sample`.
     - Na Etapa 1 do `MLP`, o pré-treinamento misturava lotes das fontes com lotes de `X_target_train_bal` (treinamento conjunto fonte+alvo em vez de pré-treinamento estrito nas fontes seguido de adaptação no alvo).
     - No `TabNet`, os hiperparâmetros (`n_steps=5, gamma=1.5, mask_type='entmax'`) e o critério de parada (`eval_set=[(X_target_train_bal, y_target_train_bal)]`) diferiam do `TabNet Scratch` de `baseline_loco_v2` (`n_steps=4, gamma=1.3, mask_type='sparsemax'` com validação interna agrupada por condição).
4. **Implementações existentes de `Fine-Tuning` vs. `Linear Probing`**:
   - Em [`src/experiments/run_tabvibnet_linear_probing.py`](../../../src/experiments/run_tabvibnet_linear_probing.py) e [`src/models/build_tabnet_resnet.py`](../../../src/models/build_tabnet_resnet.py) (`results/results_tabvibnet_linear_probing.csv`):
   - **Incompatibilidades críticas identificadas**:
     - Carregou janelas de `data/processed/` em vez de `data/processed_tl/detrended_rms/`.
     - Omitiu o dataset alvo `PU` (avaliou apenas 4 dos 5 alvos).
     - Contaminou o pool de fontes com os demais datasets alvo (ex.: usou `CWRU_12k` como fonte ao avaliar `CWRU_48k` e vice-versa).
     - Executou apenas `balance_strategy = "none"` (sem `smote`) e apenas 1 seed (sem fixar `torch.manual_seed` no ramo `MLP`).
     - Não calculou `Accuracy` e não exportou `predictions.csv` nem `fold_manifest.csv`.

---

## 2. Viabilidade Arquitetural (`MLP` e `TabNet`: `Scratch`, `Fine-Tuning`, `Linear Probing`)

| Arquitetura | Método | Viabilidade Técnica | Detalhamento da Implementação e Garantias |
|---|---|:---:|---|
| **`MLP`** | **`Scratch`** | **Sim (Nativo)** | `HybridDLModel` (`MLPEncoder 141->256->128->64` + `Bottleneck 64->128->64` + `Linear(64->C_T)`) inicializado aleatoriamente e treinado exclusivamente no fold de treino do alvo. |
| **`MLP`** | **`Fine-Tuning`** | **Sim (Nativo)** | Pré-treina `MLPEncoder + Bottleneck` (com cabeças independentes por fonte $H_s$) **exclusivamente nos datasets fonte**. No alvo, anexa nova cabeça `Linear(64->C_T)` e atualiza `encoder + bottleneck + head` no fold de treino do alvo. |
| **`MLP`** | **`Linear Probing`** | **Sim (Nativo)** | Carrega exatamente o mesmo `MLPEncoder + Bottleneck` pré-treinado nas fontes, congela todos os seus parâmetros (`requires_grad=False`) e buffers (`eval()`), e treina **exclusivamente** a cabeça linear `Linear(64->C_T)`. Verificado programaticamente (`max_abs_diff == 0.0`). |
| **`TabNet`** | **`Scratch`** | **Sim (Nativo)** | `TabNetClassifier` (`n_d=16, n_a=16, n_steps=4, gamma=1.3, mask_type='sparsemax'`, `from_unsupervised=None`) com seleção de época via validação interna agrupada por condição nas $K-1$ condições de treino + *refit* no treino completo. |
| **`TabNet`** | **`Fine-Tuning`** | **Sim (Nativo)** | Pré-treina `TabNetPretrainer` (auto-supervisionado, `pretraining_ratio=0.8`) **exclusivamente nos datasets fonte** com a mesma arquitetura (`n_d=16, n_a=16, n_steps=4, gamma=1.3`). Transfere via `from_unsupervised` para `TabNetClassifier` e ajusta todos os parâmetros no fold de treino do alvo. |
| **`TabNet`** | **`Linear Probing`** | **Sim (Com Adaptação Mínima Transparente)** | `TabNetClassifier.fit()` do `pytorch-tabnet` não possui flag nativa para congelar o encoder durante `.fit()`. A menor adaptação tecnicamente rigorosa consiste em extrair o `embedder` e o `encoder` (`TabNetEncoder`) pré-treinados do `TabNetPretrainer.network`, congelá-los integralmente (`requires_grad=False` e `.eval()`), extrair a representação latente canônica do TabNet $z = \sum_{t=1}^{n_{\text{steps}}} h_t(x) \in \mathbb{R}^{16}$ e treinar apenas a camada linear `final_mapping = nn.Linear(16, C_T)`. Verificado programaticamente (`max_abs_diff == 0.0`). |

---

## 3. Seleção de Datasets Fonte e Compatibilidade de Rótulos

- **Datasets Alvo (`5`)**: `CWRU_12k` (4 folds), `CWRU_48k` (12 folds), `HUST_Gearbox` (30 folds), `UOEMD` (16 folds), `PU` (4 folds) $\rightarrow$ **66 folds LOCO**.
- **Pool Canônico de Datasets Fonte (`4` fontes dedicadas)**: `["UORED", "HUST", "Mechanical_Gear", "Electric_Motor"]` (definido em `configs/scratch_vs_tl.yaml`).
  - **Por que não usar `CWRU_12k + MFPT` para todos os alvos**: `CWRU_12k` é um dos 5 datasets alvo (usá-lo como fonte para si mesmo violaria a separação fonte/alvo, e usá-lo para `CWRU_48k` compartilha a mesma bancada física CWRU). Já o pool `["UORED", "HUST", "Mechanical_Gear", "Electric_Motor"]` é **100% disjunto** dos 5 datasets alvo.
  - **Política de Rótulos entre Fontes e Alvo ([`ADR-003`](../../decisions/003-domain-vocabularies-and-transfer-architectures.md))**: Cada dataset preserva seu vocabulário nativo de classes (`LabelEncoder` local por domínio). Não há mapeamento artificial entre modos de falha distintos:
    - No `MLP`, cada domínio fonte possui sua própria cabeça linear $H_s$ durante o pré-treinamento supervisionado multi-head, e cada alvo $T$ recebe uma nova cabeça $H_T: \mathbb{R}^{64} \to \mathbb{R}^{C_T}$.
    - No `TabNet`, o pré-treinamento (`TabNetPretrainer`) é auto-supervisionado (reconstrução de atributos mascarados) sobre as matrizes de fontes padronizadas individualmente por domínio, e o alvo $T$ recebe uma cabeça linear $\mathbb{R}^{16} \to \mathbb{R}^{C_T}$.

---

## 4. Decisão de Reuso vs. Re-execução

1. **Experimentos que DEVEM ser executados no pipeline unificado `scratch_vs_tl`**:
   - Todos os pares `(Architecture, Method)` $\in \{\text{MLP}, \text{TabNet}\} \times \{\text{Scratch}, \text{Fine-Tuning}, \text{Linear Probing}\} $ devem ser executados sob o mesmo runner independente (`src/experiments/run_scratch_vs_tl_benchmark.py`) para garantir:
     - Mesma assinatura de partição (`split_signature`) e pré-processamento (`preprocessing_signature`) em `fold_manifest.csv`;
     - Exportação completa de predições e probabilidades por amostra em `predictions.csv`;
     - Hiperparâmetros e critério de seleção de épocas estritamente alinhados entre `Scratch`, `Fine-Tuning` e `Linear Probing`;
     - Taxas de amostragem canônicas em todas as fontes (`Mechanical_Gear = 5000 Hz`, `Electric_Motor = 50000 Hz`);
     - Cobertura completa de `BALANCE_STRATEGIES = ["none", "smote"]` e `SEEDS = [42, 123, 2026]` sem falhas de `BatchNorm1d` no fold `C4` de `PU`.
   - **Nota sobre reuso opcional de `baseline_loco_v2`**: Caso um diretório `results/baseline_loco_v2/run_*` contenha `predictions.csv` completo e assinaturas de partição idênticas, o runner suporta validação cruzada contra seus resultados, mas por padrão executa os 6 métodos de forma pareada para garantir comparabilidade estrita no mesmo loop LOCO.
2. **Tamanho Exato do Benchmark Completo**:
   - $66\text{ folds LOCO} \times 3\text{ seeds} \times 2\text{ estratégias de balanceamento} \times 2\text{ arquiteturas} \times 3\text{ métodos} = \mathbf{2.376\text{ avaliações}}$ (sendo $396$ avaliações por par `[Arquitetura, Método]`).

---

## 5. Rastreabilidade Git e Hashes SHA-256 dos Scripts Auditados

| Arquivo | SHA-256 |
|---|---|
| `src/experiments/run_baseline_loco_v2.py` | `83c59b95d092ac42b80a9c8e66c884969d5d146a7df2b1ef7fc94146e6061c47` |
| `src/experiments/run_baseline_experiments.py` | `498f69f8cbe15d2e74714051aeef7a706ef043724ceaac03547a5891580e93fc` |
| `src/experiments/run_tabvibnet_linear_probing.py` | `7a6242a06b0387bb694083c63241961b75d28cc92d622b8f1167707579548c34` |
| `scripts/exp_tl_cross_dataset_mlp_tabnet.py` | `f8de393558ce514e32a43eafbe9df1ca69eaa6dc967a991367d3071cc0a39733` |
| `scripts/exp_tl_balance_strategies.py` | `f374677aedca6cb5fc65ac07a69b08172606ed4b40a452f65a53836f99fb7cc3` |
| `scripts/exp_tl_contrastive_mlp.py` | `db41c05540a198699822e19fa20d70944571832dd37fefaa3e691febe0f3190b` |
| `src/models/build_tabnet_resnet.py` | `db9e53eccc168107a6fc6ca87a40da4fccf26077d053dbbbd1e6c3ad8899cb2c` |
| `src/models/build_tabnet_resnet_balance.py` | `b7500731135cba9f84a468464686fed3824bfd39a841cd1947f0a31aa61b17b3` |
| `src/models/build_tabular.py` | `5252b8d5317f0e758851fa17cf63626f2aceccea516d237ca9ddf23262169bd4` |
| `src/models/build_sklearn.py` | `8889573a53fb8db2683150e9ed90b1b68daf11b44070ff4225b3d31bdeec37c7` |
| `src/models/build_pytorch_baselines.py` | `6a79b355749ee3f239ccc80c920da36f86810988917a1869b13da07334f8484d` |
| `src/features/extractors_v2.py` | `6b98a00c705bee0ae4adb8963e8c9fb3129b78ff5a25e210ca8f1253dd3442b2` |
| `src/features/signalai_wrapper.py` | `a17832990a3f999cf3062f0e213acebf6bcad2de4ea66e93050b0c69a3e0a4be` |
| `src/preprocessing/normalize_dataset_tl.py` | `630c221c1f82fa82159794d34acaa8959b3506a67d699de707c108a66bd4237d` |
| `src/vibnet/data/registry.py` | `b78d492985abd40507e4cec123b4ac838fd1bb88ec600d1c12e7805f9f95c2eb` |
| `configs/datasets.yaml` | `98e9a822f3f461abcbecae25105695d1652e09314999b0d892d3caf5508523c2` |
