"""Data processing modules."""

from inflamm_debate_fm.data.load import load_adatas, load_combined_adatas
from inflamm_debate_fm.data.preprocessing import (
    fill_and_transform_infl_categories,
    preprocess_all_datasets,
)
from inflamm_debate_fm.data.transforms import (
    transform_adata_to_X_y_acute,
    transform_adata_to_X_y_acute_and_subacute,
    transform_adata_to_X_y_acute_subacute_to_chronic,
    transform_adata_to_X_y_acute_to_chronic,
    transform_adata_to_X_y_all,
    transform_adata_to_X_y_chronic,
    transform_adata_to_X_y_subacute,
    transform_adata_to_X_y_takao,
)

__all__ = [
    "load_adatas",
    "load_combined_adatas",
    "preprocess_all_datasets",
    "fill_and_transform_infl_categories",
    "transform_adata_to_X_y_all",
    "transform_adata_to_X_y_takao",
    "transform_adata_to_X_y_acute",
    "transform_adata_to_X_y_subacute",
    "transform_adata_to_X_y_acute_and_subacute",
    "transform_adata_to_X_y_chronic",
    "transform_adata_to_X_y_acute_to_chronic",
    "transform_adata_to_X_y_acute_subacute_to_chronic",
]
