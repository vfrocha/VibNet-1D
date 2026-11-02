"""
VibNet-1D Modular Architecture (`src.vibnet`).

Organizes the VibNet-1D research framework into reusable components for:
- `src.vibnet.data`: Canonical dataset registry (`DatasetSpec`), sampling rates,
  reference RMS mapping, and dataset validation interfaces.
- `src.vibnet.preprocessing`: Signal RMS reference normalization and leak-free
  feature standardization (`StandardScaler` fitted strictly on training folds).
- `src.vibnet.evaluation`: LOCO evaluation protocol contracts, source subset
  combinatorics, target-only balancing with `BatchNorm1d` remainder-1 protection,
  standardized metrics (`Macro_F1`, `Macro_F1_Selection_Score`, `ROC_AUC_Valid`),
  and full execution provenance (`run_metadata.json`).
- `src.vibnet.tabular`: TabVibNet 141-feature statistical/spectral/WPD/EMD
  extraction and tabular Transfer Learning / Contrastive Learning model facades.
- `src.vibnet.signal`: Time-domain signal contracts (`TimeDomainWindowBatch`,
  `TimeDomainLOCOAdapter`) ensuring fair LOCO comparison with `TabVibNet`
  without prematurely coupling specific 1D neural architectures.
"""

__version__ = "0.2.0"

__all__ = [
    "__version__",
]
