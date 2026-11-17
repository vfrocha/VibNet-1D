# Relatório de Diagnóstico da Instalação — VibNet-1D

- **Data/Hora (UTC)**: `2026-11-17T12:49:10.203681+00:00`
- **Status Geral do Ambiente**: **`PENDÊNCIAS DETECTADAS`**

---

## 1. Sistema Operacional, Python e Hardware

- **Sistema Operacional**: `Linux-6.6.114.1-microsoft-standard-WSL2-x86_64-with-glibc2.39` (`Ubuntu 24.04.1 LTS`)
- **Versão do Python**: `3.12.3` (`/home/vfrocha/vibnet-experiments/venv/bin/python`)
- **Ambiente Virtual Isolado (`.venv`)**: `Sim` (`/home/vfrocha/vibnet-experiments/venv`)
- **Processador (CPU)**: `Intel(R) Core(TM) i7-8700 CPU @ 3.20GHz` (`12` núcleos lógicos)
- **Memória RAM**: Total `7.46 GB` | Disponível `6.49 GB`

## 2. Aceleração por GPU e Runtime PyTorch / CUDA

- **GPU Detectada [0]**: `NVIDIA GeForce GTX 1050` | Driver NVIDIA `560.94` | CUDA máx. do driver `12.6` | VRAM `2048.0 MB`
- **PyTorch Instalado**: `True` (versão `2.6.0+cu124`, CUDA compilado `12.4`)
- **`torch.cuda.is_available()`**: `True`
- **Teste Multiplicação Matricial CPU (`100x100`)**: `OK`
- **Teste Multiplicação Matricial CUDA (`100x100`)**: `OK`
- **Diagnóstico PyTorch**: CUDA ativo e verificado com multiplicação matricial 100x100 em NVIDIA GeForce GTX 1050.

## 3. Diretórios Configurados

| Variável | Caminho Resolvido | Gravável |
|---|---|:---:|
| `VIBNET_RAW_DIR` | `/mnt/c/Users/victor.rocha/Documents/VibNet-1D/data/raw` | `Sim` |
| `VIBNET_PROCESSED_DIR` | `/mnt/c/Users/victor.rocha/Documents/VibNet-1D/data/processed` | `Sim` |
| `VIBNET_PROCESSED_TL_DIR` | `/mnt/c/Users/victor.rocha/Documents/VibNet-1D/data/processed_tl` | `Sim` |
| `VIBNET_RESULTS_DIR` | `/mnt/c/Users/victor.rocha/Documents/VibNet-1D/results` | `Sim` |

## 4. Dependências Instaladas por Grupo

| Grupo | Pacote | Versão Instalada |
|---|---|---|
| `scientific_base` | `numpy` | `2.4.6` |
| `scientific_base` | `scipy` | `1.17.1` |
| `scientific_base` | `pandas` | `3.0.3` |
| `scientific_base` | `scikit-learn` | `1.8.0` |
| `signal_processing` | `PyWavelets` | `AUSENTE` |
| `deep_learning` | `torch` | `2.6.0+cu124` |
| `deep_learning` | `pytorch-tabnet` | `AUSENTE` |
| `classical_ml` | `xgboost` | `AUSENTE` |
| `classical_ml` | `imbalanced-learn` | `AUSENTE` |
| `download_and_utils` | `gdown` | `6.1.0` |
| `download_and_utils` | `tqdm` | `4.67.3` |
| `download_and_utils` | `PyYAML` | `6.0.3` |
| `download_and_utils` | `rarfile` | `4.2` |
| `visualization` | `matplotlib` | `3.10.9` |
| `visualization` | `seaborn` | `0.13.2` |
| `vibration_ecosystem` | `vibdata` | `1.1.1` |

## 5. Inventário de Datasets (Brutos e Processados)

- **Datasets Canônicos no Registro (10)**: `CWRU_12k, CWRU_48k, MFPT, PU, UORED, UOEMD, HUST, HUST_Gearbox, Mechanical_Gear, Electric_Motor`
- **Datasets Brutos Presentes (0)**: `Nenhum`
- **Datasets Processados em Alguma Variante (0)**: `Nenhum`

| Dataset | Bruto Presente | `processed_1d` | `raw` | `detrended` | `raw_rms` | `detrended_rms` |
|---|:---:|:---:|:---:|:---:|:---:|:---:|
| `CWRU_12k` | Não | — | — | — | — | — |
| `CWRU_48k` | Não | — | — | — | — | — |
| `MFPT` | Não | — | — | — | — | — |
| `PU` | Não | — | — | — | — | — |
| `UORED` | Não | — | — | — | — | — |
| `UOEMD` | Não | — | — | — | — | — |
| `HUST` | Não | — | — | — | — | — |
| `HUST_Gearbox` | Não | — | — | — | — | — |
| `Mechanical_Gear` | Não | — | — | — | — | — |
| `Electric_Motor` | Não | — | — | — | — | — |

## 6. Verificação de Scripts e Testes Automatizados

- **Compilação Sintática dos Scripts de Experimento**: `OK`
- **Suítes de Testes Unitários Executadas**: `OK`

## 7. Avisos e Recomendações

- Nenhum aviso registrado.

## 8. Falhas Críticas Detectadas

- **FALHA**: Dependências obrigatórias ausentes: ['PyWavelets', 'pytorch-tabnet', 'xgboost', 'imbalanced-learn']
