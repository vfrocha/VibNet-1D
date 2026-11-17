# Relatório de Diagnóstico da Instalação — VibNet-1D

- **Data/Hora (UTC)**: `2026-11-17T13:32:08.291510+00:00`
- **Status Geral do Ambiente**: **`APROVADO (OK)`**

---

## 1. Sistema Operacional, Python e Hardware

- **Sistema Operacional**: `Linux-7.0.0-42-generic-x86_64-with-glibc2.39` (`Ubuntu 24.04.5 LTS`)
- **Versão do Python**: `3.12.3` (`/home/vfrocha/VibNet-1D/.venv/bin/python`)
- **Ambiente Virtual Isolado (`.venv`)**: `Sim` (`/home/vfrocha/VibNet-1D/.venv`)
- **Processador (CPU)**: `Intel(R) Core(TM) i9-14900K` (`32` núcleos lógicos)
- **Memória RAM**: Total `62.56 GB` | Disponível `58.62 GB`

## 2. Aceleração por GPU e Runtime PyTorch / CUDA

- **GPU Detectada [0]**: `NVIDIA RTX A6000` | Driver NVIDIA `580.178.04` | CUDA máx. do driver `13.0` | VRAM `49140.0 MB`
- **PyTorch Instalado**: `True` (versão `2.6.0+cu124`, CUDA compilado `12.4`)
- **`torch.cuda.is_available()`**: `True`
- **Teste Multiplicação Matricial CPU (`100x100`)**: `OK`
- **Teste Multiplicação Matricial CUDA (`100x100`)**: `OK`
- **Diagnóstico PyTorch**: CUDA ativo e verificado com multiplicação matricial 100x100 em NVIDIA RTX A6000.

## 3. Diretórios Configurados

| Variável | Caminho Resolvido | Gravável |
|---|---|:---:|
| `VIBNET_RAW_DIR` | `/home/vfrocha/VibNet-1D/data/raw` | `Sim` |
| `VIBNET_PROCESSED_DIR` | `/home/vfrocha/VibNet-1D/data/processed` | `Sim` |
| `VIBNET_PROCESSED_TL_DIR` | `/home/vfrocha/VibNet-1D/data/processed_tl` | `Sim` |
| `VIBNET_RESULTS_DIR` | `/home/vfrocha/VibNet-1D/results` | `Sim` |

## 4. Dependências Instaladas por Grupo

| Grupo | Pacote | Versão Instalada |
|---|---|---|
| `scientific_base` | `numpy` | `2.5.5` |
| `scientific_base` | `scipy` | `1.18.1` |
| `scientific_base` | `pandas` | `2.3.3` |
| `scientific_base` | `scikit-learn` | `1.9.1` |
| `signal_processing` | `PyWavelets` | `1.10.0` |
| `deep_learning` | `torch` | `2.6.0+cu124` |
| `deep_learning` | `pytorch-tabnet` | `4.1.0` |
| `classical_ml` | `xgboost` | `3.4.1` |
| `classical_ml` | `imbalanced-learn` | `0.14.2` |
| `download_and_utils` | `gdown` | `4.7.3` |
| `download_and_utils` | `tqdm` | `4.70.1` |
| `download_and_utils` | `PyYAML` | `6.0.3` |
| `download_and_utils` | `rarfile` | `4.5` |
| `visualization` | `matplotlib` | `3.11.2` |
| `visualization` | `seaborn` | `0.13.2` |
| `vibration_ecosystem` | `vibdata` | `1.0.0` |

## 5. Inventário de Datasets (Brutos e Processados)

- **Datasets Canônicos no Registro (10)**: `CWRU_12k, CWRU_48k, MFPT, PU, UORED, UOEMD, HUST, HUST_Gearbox, Mechanical_Gear, Electric_Motor`
- **Datasets Brutos Presentes (9)**: `CWRU_12k, CWRU_48k, MFPT, PU, UORED, UOEMD, HUST_Gearbox, Mechanical_Gear, Electric_Motor`
- **Datasets Processados em Alguma Variante (10)**: `CWRU_12k, CWRU_48k, MFPT, PU, UORED, UOEMD, HUST, HUST_Gearbox, Mechanical_Gear, Electric_Motor`

| Dataset | Bruto Presente | `processed_1d` | `raw` | `detrended` | `raw_rms` | `detrended_rms` |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `CWRU_12k` | Sim | 2520 win | 2520 win | 2520 win | 2520 win | 2520 win |
| `CWRU_48k` | Sim | 602 win | 602 win | 602 win | 602 win | 602 win |
| `MFPT` | Sim | 114 win | 78 win | 78 win | 78 win | 78 win |
| `PU` | Sim | 6476 win | 10181 win | 10181 win | 10181 win | 12099 win |
| `UORED` | Sim | 370 win | 230 win | 230 win | 230 win | 600 win |
| `UOEMD` | Sim | 1280 win | 1280 win | 1280 win | 1280 win | 2560 win |
| `HUST` | Não | 1101 win | — | — | — | 1101 win |
| `HUST_Gearbox` | Sim | 900 win | 900 win | 900 win | 900 win | 1780 win |
| `Mechanical_Gear` | Sim | 180 win | 180 win | 180 win | 180 win | 360 win |
| `Electric_Motor` | Sim | 66 win | 66 win | 66 win | 66 win | 130 win |

## 6. Verificação de Scripts e Testes Automatizados

- **Compilação Sintática dos Scripts de Experimento**: `OK`
- **Suítes de Testes Unitários Executadas**: `OK`

## 7. Avisos e Recomendações

- Nenhum aviso registrado.
