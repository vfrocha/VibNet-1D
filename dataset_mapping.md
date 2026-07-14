

# Compendium of Vibration Datasets for Intelligent Fault Diagnosis

This document describes the physical and structural properties of the vibration datasets used in the _Transfer Learning_ experiments of the **VibNet-1D** architecture.

## 1. Methodological Premises

To ensure the integrity of the evaluation and remove the Similarity Bias, the preprocessing pipeline follows these strict rules:

-   **Base Sampling:** Strict windowing of 1 second in duration.
    
-   **Overlap:** No _gap_ and no _overlap_ (shift equals the window size).
    
-   **Validation:** LOCO (_Leave-One-Condition-Out_) strategy, where the model is tested in a completely unseen load/speed domain during training.
    

## 2. Test Sets (Target Domains)

These datasets have high variability in operational conditions and will be used to validate the model's robustness (Knowledge Transfer and LOCO Validation).

| Dataset | Component | Fs (Hz) | Window (Points) | Classes | Operating Conditions | Special Notes |
|---|---:|---:|---:|---:|---|---|
| **CWRU 12K** | Bearing | 12,000 | 12,000 | 4 | 4 (Loads: 0, 1, 2, 3 HP) | Classic benchmark in the field. |
| **CWRU 48K** | Bearing | 48,000 | 48,000 | 3 | 12 (Load × Severity) | *Not applicable for Fault Detection (absence of normal class in some severities).* |
| **UOEMD** | Electric Motor | 42,000 | 42,000 | 8 | 8 (Load × Speed) | Used only Accelerometer 1 (Channel 1). Includes speed transitions. |
| **HUST Gearbox** | Gearbox | 25,600 | 25,600 | 3 | 30 (Load × Speed) | Extensive data with 3 gearbox health states. |

## 3. Training Sets (Source Domains)

These datasets make up the repository of prior knowledge (_Pre-training_). They provide the massive amount of data necessary for the neural network to learn generic feature extraction before _Fine-Tuning_.


| Dataset | Component | Fs (Hz) | Window (Points) | Classes | Operating Conditions | Special Notes |
|---|---:|---:|---:|---:|---|---|
| **PU** | Bearing | 64,000 | 64,000 | 3 | 4 (Load × Speed) | Paderborn University. Includes artificial and real faults. |
| **IMS** | Bearing | 20,000 | 20,000 | - | Temporal Evolution (Degradation) | NASA. Focused on continuous *Run-to-Failure*. |
| **MFPT** | Bearing | 48,828 | 48,828 | 3 | 3 (Baseline, Outer, Inner) | Massive load variation in inner race faults. |
| **HUST Bearing** | Bearing | 51,200 | 51,200 | 7 | Multiple | Unlike the Gearbox, Fs is doubled. |
| **UORED** | Bearing | 42,000 | 42,000 | 5 | Degradation Stages | Allows *Leave-One-Bearing-Out* validation. |
| **Mechanical Gear** | Gearbox | 5,000 | 5,000 | 6 | 6 (Load × Speed) | Kaggle dataset. 2 sensors (X and Y). Focus on X-axis. |
| **Electric Motor** | Electric Motor | 50,000 | 50,000 | 4 | 30 Unique Experiments | Zenodo dataset. 3 Axes (X, Y, Z). Focus on Accelerometer X. |
| **UOC** | Gearbox | 20,000 | 2,048* | 9 | Multiple Positions / Defects | *Signals in "burst" format of 0.18 s. Windowing reduced to 2048 points.* |

## 4. LOCO (Leave-One-Condition-Out) Validation Dynamics

The LOCO validation will be applied such that, given a set of conditions $C = \{c_1, c_2, ..., c_n\}$ from a target dataset:

1.  The model is trained on data from conditions $C_{train} = C \setminus \{c_i\}$.
    
2.  The model is strictly tested on the data of condition $c_i$.
    
3.  The process is repeated $n$ times to cover all conditions as testing.
    

### Justification for Avoiding Similarity Bias

By forcing prediction on an unseen condition $c_i$ (e.g., evaluating a network trained only on motors at 15Hz and 30Hz on a motor running at 60Hz with Maximum Load), it is proven that the model has learned the **fundamental signatures of the physical fault** (e.g., resonance frequencies or impact modulations), and not merely the "stationary texture" or background noise of the machine at a specific rotation.
