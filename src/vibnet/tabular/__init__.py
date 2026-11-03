"""
TabVibNet tabular feature extraction and model facades (`src.vibnet.tabular`).
"""

from src.vibnet.tabular.features import (
    AUX_ENERGY_FEATURE_DIM,
    TABVIBNET_FEATURE_DIM,
    TabVibNetFeatureSpec,
    extract_tabvibnet_features,
    get_tabvibnet_feature_spec,
)
from src.vibnet.tabular.models import (
    TabularModelOutput,
    normalize_model_return_tuple,
    run_contrastive_mlp_transfer,
    run_multihead_mlp_transfer,
    run_tabnet_transfer,
)

__all__ = [
    "AUX_ENERGY_FEATURE_DIM",
    "TABVIBNET_FEATURE_DIM",
    "TabVibNetFeatureSpec",
    "TabularModelOutput",
    "extract_tabvibnet_features",
    "get_tabvibnet_feature_spec",
    "normalize_model_return_tuple",
    "run_contrastive_mlp_transfer",
    "run_multihead_mlp_transfer",
    "run_tabnet_transfer",
]
