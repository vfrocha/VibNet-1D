# VibNet-1D — Cross-Dataset & Cross-Condition Machinery Fault Diagnosis

**VibNet-1D** is a research repository for rotating machinery fault diagnosis from vibration signals under variable operating conditions and across heterogeneous datasets, developed using **Multi-Source Transfer Learning**, **Supervised Contrastive Learning**, and strict **Leave-One-Condition-Out (LOCO)** evaluation protocols.

The project is organized into two complementary scientific stages:

1. **TabVibNet (Completed & Audited Stage):** Multi-domain statistical, spectral, time-frequency, and intrinsic mode feature extraction (**141 features** per window), evaluated with domain-specific **Multi-Head MLPs**, **Supervised Contrastive MLPs (`SupConLoss`)**, **TabNet** (`TabNetPretrainer` + `TabNetClassifier`), and classical baselines.
2. **VibNet-1D Time-Domain Stage (Modular Architecture Prepared):** Standardized data contracts (`TimeDomainWindowBatch`), LOCO partition alignment (`TimeDomainLOCOAdapter`), 1D time-domain transforms (`src/data/transforms_1d.py` for detrend, polyphase resampling, and Z-score/RMS normalization), PyTorch 1D model interfaces (`src/models/build_1d_time.py`), and configuration templates (`configs/time_domain/`) prepared for end-to-end 1D neural architectures (`1D-CNN`, `1D-ResNet`, `WDCNN`) operating directly on 1-second time-domain windows.

---

## Repository Structure

```text
VibNet-1D/
├── pyproject.toml                          # Editable package configuration (pip install -e .)
├── requirements.txt                        # Consolidated scientific, DL & Git dependencies
├── configs/                                # Declarative YAML configurations
│   ├── datasets.yaml                       # Canonical metadata & sampling rates for all 10 datasets
│   ├── tabvibnet/                          # TabVibNet experiment configurations
│   │   ├── tl_cross_dataset_mlp_tabnet.yaml
│   │   ├── tl_contrastive_mlp.yaml
│   │   └── tl_variants_and_balance.yaml
│   └── time_domain/                        # Template for upcoming 1D time-domain experiments
│       └── template_1d_loco.yaml
├── docs/                                   # Scientific & reproducibility documentation
│   ├── project_overview.md                 # Project evolution, architecture, and roadmap
│   ├── datasets_and_protocols.md           # Dataset specifications, 141 features, and LOCO protocol
│   ├── reproducibility.md                  # Provenance schema (run_metadata.json) & reproduction guide
│   ├── experiments/
│   │   └── baseline_loco_v2.md             # Leakage-free paired LOCO supervised baseline specification
│   ├── results/
│   │   └── tabvibnet_summary.md            # Consolidated & audited TabVibNet empirical results
│   └── decisions/                          # Architecture Decision Records (ADRs)
│       ├── 001-rms-reference-normalization-and-detrending.md
│       ├── 002-loco-isolation-and-target-balancing.md
│       ├── 003-domain-vocabularies-and-transfer-architectures.md
│       └── 004-planned-modular-architecture-vibnet.md
├── experiments/                            # Modular experiment entrypoints by family
│   ├── tabvibnet/
│   │   ├── README.md
│   │   ├── run_cross_dataset_tl.py         # Multi-source cross-dataset TL runner (15 source subsets x 5 targets)
│   │   ├── run_contrastive_tl.py           # Supervised Contrastive Multi-Head MLP runner
│   │   └── run_variants_and_balance.py     # Preprocessing variants, aux features & target balance runner
│   └── time_domain/
│       └── README.md                       # Contracts for fair TabVibNet vs. 1D time-domain comparison
├── reports/
│   └── reproducibility_audit/              # Automated inventory & anomaly audit outputs
│       ├── EXPERIMENTAL_RESULTS_REPORT.md
│       ├── results_inventory.csv
│       ├── metrics_long.csv
│       ├── metrics_summary.csv
│       └── anomalies_and_warnings.csv
├── results/                                # Preserved raw experimental outputs (CSV, JSON, logs, plots)
├── scripts/                                # Audit, verification, and standalone experiment scripts
│   ├── audit_experimental_results.py       # Automated results inventory, anomaly detector & Markdown reporter
│   ├── validate_dataset_tl.py              # Dataset integrity & reference RMS validation CLI
│   ├── verify_cross_dataset_mlp_tabnet.py  # AST, compilation & combinatorics verification suite
│   ├── verify_mlp_contrastive.py           # AST, CSV consistency & tensor verification suite
│   ├── exp_tl_balance_strategies.py        # Target-only class balancing ablation
│   ├── exp_tl_contrastive_mlp.py           # Supervised Contrastive Multi-Head MLP ablation
│   └── exp_tl_cross_dataset_mlp_tabnet.py  # Full cross-dataset multi-source TL benchmark
├── src/
│   ├── config.py                           # Central path & environment variable manager (RAW_DATA_DIR, DATA_ROOT, RESULTS_DIR)
│   ├── vibnet/                             # Unified modular package
│   │   ├── data/                           # Canonical dataset registry (DatasetSpec), YAML loader & validation
│   │   ├── preprocessing/                  # Reference RMS normalization, linear detrending & leak-free scaling
│   │   ├── evaluation/                     # LOCO splits, source combinatorics, balancing, metrics & provenance
│   │   ├── tabular/                        # 141-feature TabVibNet spec & model facades
│   │   └── signal/                         # Time-domain window contracts & LOCO alignment verification
│   ├── data/                               # Dataset preparation, dataloader, validation & 1D transforms (transforms_1d.py)
│   ├── preprocessing/                      # Dataset-level RMS normalization implementation
│   ├── features/                           # 141-feature extractors (Time, FFT/PSD, WPD db4, EMD)
│   ├── models/                             # Multi-Head MLP, SupCon MLP, TabNet, baselines & 1D time models (build_1d_time.py)
│   └── experiments/                        # Experiment runners (including run_baseline_loco_v2.py)
└── tests/                                  # Unit test suites (stdlib-compatible & ML tensor tests)
    ├── test_audit_experimental_results.py
    ├── test_baseline_loco_v2.py
    ├── test_vibnet_modular_architecture.py
    └── test_mlp_contrastive.py
```

---

## Datasets & Canonical Registry

The repository integrates **10 rotating machinery vibration datasets**. Canonical metadata is maintained in [`configs/datasets.yaml`](configs/datasets.yaml) and [`src/vibnet/data/registry.py`](src/vibnet/data/registry.py):

| Dataset | Default Role | Sampling Rate (`fs_hz`) | Conditions (LOCO Folds) | Classes | Healthy Reference (`RMS_ref`) | Total Windows |
| :--- | :--- | :---: | :---: | :---: | :--- | :---: |
| **`CWRU_12k`** | Target / Source | `12,000 Hz` | `4` | `4` | `CWRU_12k` (`Class_0`) | `2,013` |
| **`CWRU_48k`** | Target | `48,000 Hz` | `12` | `4` (`3` active) | `CWRU_12k` (`Class_0`)* | `4,524` |
| **`PU`** | Target | `64,000 Hz` | `4` | `4` (`6` raw) | `PU` (`Healthy` / `K00`) | `10,002` |
| **`HUST_Gearbox`** | Target | `25,600 Hz` | `30` | `4` | `HUST_Gearbox` (`Normal`) | `18,946` |
| **`UOEMD`** | Target | `42,000 Hz` | `16` | `8` (`5` active) | `UOEMD` (`H` / `H_1`) | `6,560` |
| **`UORED`** | Source | `42,000 Hz` | `20` (1 cls/cond) | `5` | `UORED` (`Healthy` / `H`) | `8,200` |
| **`HUST`** | Source | `51,200 Hz` | `11` | `9` (`7` active) | `HUST` (`Normal` / `N`) | `49,500` |
| **`Mechanical_Gear`** | Source | `5,000 Hz` | `4` | `3` (`4` raw) | `Mechanical_Gear` (`Normal`) | `3,502` |
| **`Electric_Motor`** | Source | `50,000 Hz` | `1` | `4` (`3` active) | `Electric_Motor` (`Healthy`) | `9,600` |
| **`MFPT`** | Source | `48,828 Hz` | `13` (1 cls/cond) | `3` | `MFPT` (`Class_0` / `Normal`) | `1,553` |

*\*Note: `CWRU_48k` uses the healthy baseline RMS of `CWRU_12k` (`Class_0`) because isolated healthy 48 kHz files are absent from the local repository (`ADR-001`).*

---

## Scientific Protocol & Methodological Guarantees

1. **Signal Preprocessing (`detrended_rms`, [ADR-001](docs/decisions/001-rms-reference-normalization-and-detrending.md)):**
   - Each window undergoes linear detrending to remove thermal/sensor drift, followed by division by the mean RMS of the dataset's healthy reference condition (`RMS_ref`). Unlike per-window z-score normalization, `detrended_rms` preserves relative fault-to-healthy energy severity while aligning baseline amplitude across datasets.
2. **141-Feature Multi-Domain Vector (`TabVibNet`):**
   - **Time Domain (`18` features):** Mean, Std, Variance, RMS, Peak, Peak-to-Peak, Skewness, Kurtosis, Crest Factor, Shape Factor, Impulse Factor, Margin Factor, Energy, Entropy, Zero-Crossing Rate, Quantiles (`Q25`, `Q50`, `Q75`).
   - **Frequency Domain (`24` features):** FFT and Welch PSD spectral centroids, bandwidth, roll-off (`85%`, `95%`), spectral flatness/skewness/kurtosis/entropy, dominant frequency, and 5 normalized band energies.
   - **Wavelet Packet Decomposition (`73` features):** `db4` wavelet at level `3` (`8` terminal sub-bands), extracting relative energy, Shannon entropy, statistical moments, and band ratios.
   - **Empirical Mode Decomposition (`26` features):** First `3` Intrinsic Mode Functions (`IMF 1..3`) plus residue (`8` descriptors per IMF + `2` residue descriptors).
3. **Strict Leave-One-Condition-Out (LOCO) Isolation ([ADR-002](docs/decisions/002-loco-isolation-and-target-balancing.md)):**
   - For a target dataset with $K$ operating conditions, fold $k \in \{1, \dots, K\}$ holds out **100% of windows from condition $c_k$** as the test set and trains exclusively on the remaining $K - 1$ conditions.
   - Feature standardization (`StandardScaler`) and class balancing (`SMOTE`) are fitted **exclusively on the target training split (`K - 1` conditions)**. Source domains are standardized independently per dataset and are never resampled.
4. **Domain-Specific Classification Heads ([ADR-003](docs/decisions/003-domain-vocabularies-and-transfer-architectures.md)):**
   - Because each dataset has its own native fault taxonomy (`3` to `9` classes across bearings, gears, and stator/rotor faults), transfer models share a common latent encoder (`z in R^64`) while maintaining independent classification heads $H_s$ for each source domain and $H_T$ for the target domain.

---

## Summary of Audited Experimental Results (`TabVibNet`)

Full fold-level and summary tables are documented in [`docs/results/tabvibnet_summary.md`](docs/results/tabvibnet_summary.md) and [`reports/reproducibility_audit/EXPERIMENTAL_RESULTS_REPORT.md`](reports/reproducibility_audit/EXPERIMENTAL_RESULTS_REPORT.md).

### 1. Preprocessing Variants & Feature Representation (`CWRU_12k|MFPT -> CWRU_48k`, 12 LOCO Folds)

| Experiment Stage | Configuration | `MLP` Macro-F1 | `TabNet` Macro-F1 | `Macro_F1_Selection_Score` |
| :--- | :--- | :---: | :---: | :---: |
| **Signal Variants** (`tl_four_variants`) | **`detrended_rms` (Selected)** | **`0.9202 ± 0.1353`** | **`0.8909 ± 0.1774`** | **`0.9055`** |
| Signal Variants (`tl_four_variants`) | `raw_rms` | `0.8655 ± 0.1818` | `0.8775 ± 0.1782` | `0.8715` |
| Signal Variants (`tl_four_variants`) | `detrended` | `0.8669 ± 0.1753` | `0.8432 ± 0.1984` | `0.8550` |
| Signal Variants (`tl_four_variants`) | `raw` | `0.8560 ± 0.1723` | `0.8087 ± 0.2116` | `0.8323` |
| **Aux Energy Features** (`tl_aux_features`) | **`detrended_rms__base` (141 dims)** | **`0.9202 ± 0.1353`** | **`0.8909 ± 0.1774`** | **`0.9055`** |
| Aux Energy Features (`tl_aux_features`) | `detrended_rms__rms_b1_b2` (144 dims) | `0.8799 ± 0.1725` | `0.7966 ± 0.2194` | `0.8383` |
| **Target Balancing** (`tl_balance_strategies`) | **`smote` / `hybrid` (Selected)** | **`0.9389 ± 0.1289`** | `0.8851 ± 0.1687` | **`0.9120`** |

### 2. Supervised Contrastive Multi-Head MLP (`CWRU_12k|MFPT -> CWRU_48k`, 3 Seeds `[42, 123, 2026]`)

| Ablation Stage | Contrastive Mode | Encoder Architecture | Projection Head | Mean `Macro-F1` (36 Folds) | Mean `Accuracy` |
| :--- | :--- | :--- | :--- | :---: | :---: |
| Stage 1 (Mode) | `baseline` (CE only) | `baseline` (`128->64`) | `asymmetric` | `0.9056 ± 0.0101` | `0.9182` |
| Stage 1 (Mode) | `joint` (`CE + 0.5*SupCon`) | `baseline` (`128->64`) | `asymmetric` | `0.9097 ± 0.0104` | `0.9218` |
| Stage 1 (Mode) | `pretrain` (`SupCon -> CE`) | `baseline` (`128->64`) | `asymmetric` | `0.9133 ± 0.0116` | `0.9255` |
| **Stage 2 & 3 (Best)** | **`joint` (`CE + 0.5*SupCon`)** | **`expanded` (`256->128->64`)** | **`asymmetric` (`64->64->32`)** | **`0.9235 ± 0.0131`** | **`0.9350`** |
| Stage 3 (Head) | `joint` (`CE + 0.5*SupCon`) | `expanded` (`256->128->64`) | `symmetric` (`64->64`) | `0.8863 ± 0.0150` | `0.9012` |

### 3. Cross-Dataset Multi-Source Benchmark (`15` Source Subsets $\times$ `5` Targets = `5,940` Evaluations)

Evaluated in `results/tl_cross_dataset_mlp_tabnet/run_20261028_173310` across all `15` non-empty subsets of `['UORED', 'HUST', 'Mechanical_Gear', 'Electric_Motor']` and `3` seeds (`[42, 123, 2026]`):

| Target Dataset | LOCO Folds | Best Source Subset (`MLP`) | `MLP` Macro-F1 | Best Source Subset (`TabNet`) | `TabNet` Macro-F1 |
| :--- | :---: | :--- | :---: | :--- | :---: |
| **`CWRU_12k`** | `4` | `UORED\|Mechanical_Gear\|Electric_Motor` | **`0.9554 ± 0.0445`** | `UORED\|Mechanical_Gear\|Electric_Motor` | `0.9180 ± 0.0725` |
| **`CWRU_48k`** | `12` | `Mechanical_Gear` | **`0.9293 ± 0.1400`** | `UORED\|Mechanical_Gear` | `0.8712 ± 0.1656` |
| **`PU`** | `4` | `HUST\|Mechanical_Gear` | `0.4836 ± 0.1669` | `Mechanical_Gear`* | **`0.5566 ± 0.0566`** (`3/4` folds)* |
| **`HUST_Gearbox`** | `30` | `UORED` | **`0.4673 ± 0.1235`** | `UORED` | `0.4534 ± 0.1190` |
| **`UOEMD`** | `16` | `UORED\|HUST` | **`0.3210 ± 0.1066`** | `UORED\|Electric_Motor` | `0.2644 ± 0.0970` |

*\*Methodological Note on `PU`: In `run_20261028_173310`, `TabNet` failed on `PU` fold `C4_1500rpm_0.7Nm_400N` because SMOTE produced `8,577` samples (`8,577 % 64 = 1`), triggering a 1-sample final batch error in PyTorch `BatchNorm1d`. The `TabNet` mean on `PU` is therefore computed over `3/4` folds (`C1..C3`), whereas `MLP` is evaluated across all `4/4` folds. This issue is resolved in [`src/vibnet/evaluation/balancing.py`](src/vibnet/evaluation/balancing.py) via [`trim_single_remainder_sample`](src/vibnet/evaluation/balancing.py#L58-L85). Across all `75` paired `(Target, Source_Subset)` comparisons, **`MultiHeadMLP` outperforms `TabNet` in `62/75` cases (`82.7%`)**.*

---

## Quickstart & Usage

### 1. Environment Setup (Linux / macOS / Servers)

Create a virtual environment, install dependencies, and register the package (`src`) in editable mode so no `sys.path.append` is needed:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

*(Note: All modular orchestration, YAML loading, results auditing, and unit tests in `src/vibnet/` and `tests/` are designed to run even in a bare Python 3.9+ standard-library environment).*

### 2. Google Colab Setup (3-Line Snippet)

To mount Google Drive, point the dataset path to your Drive folder, and install the project in Google Colab:

```python
from google.colab import drive; drive.mount('/content/drive')
%env VIBNET_PROCESSED_DIR=/content/drive/MyDrive/VibNet_Project/data/processed
!pip install -q -r requirements.txt && pip install -q -e .
```

> **Supported Environment Variables (`src/config.py`):**
> - `VIBNET_RAW_DIR`: directory for raw `vibdata` datasets (falls back to `/home/vfrocha/VibNet_Project/raw_data` if present on the server, or `<repo_root>/data/raw`).
> - `VIBNET_PROCESSED_DIR`: directory for processed 1D `.npy` windows (default: `<repo_root>/data/processed`).
> - `VIBNET_PROCESSED_TL_DIR`: directory for Transfer Learning preprocessing variants (default: `<repo_root>/data/processed_tl`).
> - `VIBNET_RESULTS_DIR`: directory for experiment logs, CSVs, and figures (default: `<repo_root>/results`).

### 3. Run Unit Tests & Verification Suites

From the repository root:

```bash
# Run unit tests for results auditing and the src/vibnet/ modular architecture
python -m unittest tests/test_audit_experimental_results.py tests/test_vibnet_modular_architecture.py -v

# Run static AST, compilation, and published artifact verification scripts
python scripts/verify_cross_dataset_mlp_tabnet.py
python scripts/verify_mlp_contrastive.py
```

When running inside the full ML environment (`numpy`, `torch`, `scikit-learn` installed), run the complete test suite including tensor tests:

```bash
python -m unittest discover -s tests -v
```

### 4. Regenerate the Automated Experimental Results Audit

To inventory all files under `results/`, extract fold-level and summary metrics, flag anomalies, and regenerate `reports/reproducibility_audit/EXPERIMENTAL_RESULTS_REPORT.md`:

```bash
python scripts/audit_experimental_results.py
```

### 5. Inspect & Run Modular Experiments (`experiments/tabvibnet/`)

Validate any experiment configuration and inspect its execution plan and SHA-256 provenance manifest in `--dry-run` mode:

```bash
python experiments/tabvibnet/run_cross_dataset_tl.py --dry-run
python experiments/tabvibnet/run_contrastive_tl.py --dry-run
python experiments/tabvibnet/run_variants_and_balance.py --stage variants --dry-run
```

To execute full training runs (requires processed datasets in `data/processed_tl/` and ML dependencies):

```bash
# 1. Preprocessing variants, auxiliary energy features, or target balance strategies
python experiments/tabvibnet/run_variants_and_balance.py --stage variants
python experiments/tabvibnet/run_variants_and_balance.py --stage aux_features
python experiments/tabvibnet/run_variants_and_balance.py --stage balance

# 2. Supervised Contrastive Multi-Head MLP ablation
python experiments/tabvibnet/run_contrastive_tl.py

# 3. Cross-Dataset Multi-Source Transfer Learning benchmark (MLP vs. TabNet)
python experiments/tabvibnet/run_cross_dataset_tl.py

# 4. Leakage-free paired LOCO supervised baseline benchmark (baseline_loco_v2)
python -m src.experiments.run_baseline_loco_v2
```

---

## Documentation Index

- **Project Overview & Evolution:** [`docs/project_overview.md`](docs/project_overview.md)
- **Datasets, 141 Features & LOCO Protocol:** [`docs/datasets_and_protocols.md`](docs/datasets_and_protocols.md)
- **Leakage-Free Paired LOCO Baseline (`baseline_loco_v2`):** [`docs/experiments/baseline_loco_v2.md`](docs/experiments/baseline_loco_v2.md)
- **Reproducibility & Provenance Guide:** [`docs/reproducibility.md`](docs/reproducibility.md)
- **Consolidated `TabVibNet` Results:** [`docs/results/tabvibnet_summary.md`](docs/results/tabvibnet_summary.md)
- **Automated Audit Report:** [`reports/reproducibility_audit/EXPERIMENTAL_RESULTS_REPORT.md`](reports/reproducibility_audit/EXPERIMENTAL_RESULTS_REPORT.md)
- **Architecture Decision Records (ADRs):**
  - [`ADR-001`](docs/decisions/001-rms-reference-normalization-and-detrending.md): Reference RMS Normalization and Linear Detrending (`detrended_rms`)
  - [`ADR-002`](docs/decisions/002-loco-isolation-and-target-balancing.md): Strict LOCO Isolation, Target-Only Balancing, and `BatchNorm1d` Protection
  - [`ADR-003`](docs/decisions/003-domain-vocabularies-and-transfer-architectures.md): Native Domain Vocabularies, Multi-Head MLP, and Supervised Contrastive Learning
  - [`ADR-004`](docs/decisions/004-planned-modular-architecture-vibnet.md): Modular Architecture Separating `TabVibNet` (`tabular`) and `VibNet-1D` (`signal`)
