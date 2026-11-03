# ADR 001: Remoção de Tendência Linear e Normalização por RMS Saudável de Referência (`detrended_rms`)

- **Status**: Aceito e Validado Experimentalmente
- **Contexto**: Etapa TabVibNet (`data/processed_tl/`, `tl_four_variants`, `tl_aux_features`)
- **Artefatos Relacionados**:
  - `src/data/make_dataset_tl.py`
  - `src/preprocessing/normalize_dataset_tl.py`
  - `scripts/validate_dataset_tl.py`
  - `results/tl_four_variants/run_20261005_163159/`
  - `results/tl_aux_features/run_20261012_093659/`

## 1. Problema e Contexto Físico

Os 10 datasets de vibração do repositório foram adquiridos em diferentes bancadas, com sensores e condicionadores distintos (acelerômetros em $g$, $m/s^2$ ou tensão em Volts) e diferentes níveis de *offset* CC ou deriva térmica de baixa frequência:
- O RMS médio da condição saudável varia em duas ordens de grandeza entre bancadas: de $\text{RMS}_{\text{ref}} = 0{,}019937$ em `UOEMD` e $0{,}067780$ em `CWRU_12k` até $2{,}096993$ em `Electric_Motor` e $2{,}173862$ em `HUST`.
- Nos experimentos preliminares de normalização (`src/experiments/exp_normalization_*.py`), testou-se dividir cada janela $w$ pelo seu próprio RMS instantâneo ($\tilde{w} = w / \text{RMS}(w)$). Essa abordagem **destruiu a discriminação entre condição normal e falha**, pois forçava todas as janelas (saudáveis ou severamente danificadas) a terem $\text{RMS}(\tilde{w}) \equiv 1{,}0$, anulando todos os atributos dependentes de energia (`RMS`, `Std`, `Peak`, energias espectrais e WPD).
- Além disso, o pipeline inicial `src/data/make_dataset.py` aplicava `scipy.signal.detrend(w, type='linear')` diretamente ao gerar `data/processed/`, impedindo isolar o efeito da remoção de tendência linear do efeito da normalização de amplitude.

## 2. Decisão Metodológica

1. **Geração de 4 Variantes Independentes (`data/processed_tl/`)**:
   - `raw`: janelas de 1 s sem remoção de tendência e sem reescalonamento.
   - `detrended`: aplicação de `scipy.signal.detrend(w, type='linear')` janela a janela.
   - `raw_rms`: divisão de cada janela de `raw` pelo $\text{RMS}_{\text{ref}}^{(d)}$ da classe saudável de `raw`.
   - `detrended_rms`: divisão de cada janela de `detrended` pelo $\text{RMS}_{\text{ref}}^{(d)}$ da classe saudável de `detrended`.
2. **Referência Cruzada Documentada para `CWRU_48k`**:
   - Como os arquivos de 48 kHz processados em `CWRU_48k` contêm apenas as 3 classes de falha (`Class_1`, `Class_2`, `Class_3`), definiu-se em `src/preprocessing/normalize_dataset_tl.py` o uso do $\text{RMS}_{\text{ref}}$ da classe saudável `Class_0` do dataset irmão `CWRU_12k` ($\text{RMS}_{\text{ref}} = 0{,}067780$), adquirido na mesma bancada física CWRU.
3. **Exclusão das Features Auxiliares Redundantes (`Aux_Window_RMS`, `B1`, `B2`)**:
   - Após o experimento de ablação `tl_aux_features` (`run_20261012_093659`), fixou-se `include_window_rms_feature = False` (141 features base), pois o escalonamento linear por $\text{RMS}_{\text{ref}}^{(d)}$ já preserva a energia relativa nos atributos de amplitude existentes no vetor de 141 features.

## 3. Evidência Experimental Observada

No alvo `CWRU_48k` (12 folds LOCO, fontes `CWRU_12k|MFPT`):
- **`tl_four_variants` (`run_20261005_163159`)**: A variante `detrended_rms` atingiu `Macro-F1` de **$0{,}9202 \pm 0{,}1353$** (`MLP`) e **$0{,}8909 \pm 0{,}1774$** (`TABNET`, `Selection_Score` = $0{,}9055$), superando `raw_rms` ($0{,}8667$ / $0{,}8762$, `Selection_Score` = $0{,}8715$), `detrended` ($0{,}8977$ / $0{,}8124$, `Selection_Score` = $0{,}8550$) e `raw` ($0{,}8305$ / $0{,}8341$, `Selection_Score` = $0{,}8323$).
- **`tl_aux_features` (`run_20261012_093659`)**: Em `detrended_rms`, o vetor base de 141 features (`detrended_rms__base`: `MLP` $0{,}9202$, `TABNET` $0{,}8909$) superou a versão com 144 features (`detrended_rms__rms_b1_b2`: `MLP` $0{,}8799$, `TABNET` $0{,}7966$).

## 4. Implicações para a Futura Etapa de Sinal no Domínio do Tempo (`src/vibnet/signal/`)

Os modelos 1D no domínio do tempo (`1D-CNN`, `ResNet-1D`) deverão consumir por padrão a variante **`detrended_rms`** de `data/processed_tl/`, garantindo que a amplitude das formas de onda de entrada já esteja alinhada pela energia saudável de referência de cada bancada sem suprimir a assinatura energética das falhas.
