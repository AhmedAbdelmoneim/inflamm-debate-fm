"""Data transformation functions for creating X, y pairs from AnnData."""

from typing import Tuple

import anndata as ad
import numpy as np
import pandas as pd


def transform_adata_to_X_y_all(
    adata: ad.AnnData,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Transform AnnData to X, X_emb, y, groups for all samples.

    Args:
        adata: AnnData object.

    Returns:
        Tuple of (X_raw, X_emb, y, groups).
    """
    X = adata.X.copy()
    X_emb = adata.obsm["X_bulkformer"].copy()
    y = adata.obs["group"].map({"inflammation": 1, "control": 0}).values.astype(int)
    groups = adata.obs.get("dataset", pd.Series(np.zeros(len(adata)), index=adata.obs.index))
    return X, X_emb, y, groups


def transform_adata_to_X_y_takao(
    adata: ad.AnnData,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Transform AnnData to X, X_emb, y, groups for Takao subset.

    Args:
        adata: AnnData object.

    Returns:
        Tuple of (X_raw, X_emb, y, groups).
    """
    adata_sub = adata[~adata.obs["takao_status"].isna()].copy()
    X = adata_sub.X.copy()
    X_emb = adata_sub.obsm["X_bulkformer"].copy()
    y = (
        adata_sub.obs["takao_status"]
        .map({"takao_inflamed": 1, "takao_control": 0})
        .values.astype(int)
    )
    groups = adata_sub.obs.get(
        "dataset", pd.Series(np.zeros(len(adata_sub)), index=adata_sub.obs.index)
    )
    return X, X_emb, y, groups


def transform_adata_to_X_y_acute(
    adata: ad.AnnData,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Transform AnnData to X, X_emb, y, groups for acute inflammation.

    Args:
        adata: AnnData object.

    Returns:
        Tuple of (X_raw, X_emb, y, groups).
    """
    adata_sub = adata[(adata.obs["group"] == "control") | (adata.obs["infl_acute"])].copy()
    X = adata_sub.X.copy()
    X_emb = adata_sub.obsm["X_bulkformer"].copy()
    y = adata_sub.obs["infl_acute"].map({True: 1, False: 0}).values.astype(int)
    groups = adata_sub.obs.get(
        "dataset", pd.Series(np.zeros(len(adata_sub)), index=adata_sub.obs.index)
    )
    return X, X_emb, y, groups


def transform_adata_to_X_y_subacute(
    adata: ad.AnnData,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Transform AnnData to X, X_emb, y, groups for subacute inflammation.

    Args:
        adata: AnnData object.

    Returns:
        Tuple of (X_raw, X_emb, y, groups).
    """
    adata_sub = adata[(adata.obs["group"] == "control") | (adata.obs["infl_subacute"])].copy()
    X = adata_sub.X.copy()
    X_emb = adata_sub.obsm["X_bulkformer"].copy()
    y = adata_sub.obs["infl_subacute"].map({True: 1, False: 0}).values.astype(int)
    groups = adata_sub.obs.get(
        "dataset", pd.Series(np.zeros(len(adata_sub)), index=adata_sub.obs.index)
    )
    return X, X_emb, y, groups


def transform_adata_to_X_y_acute_and_subacute(
    adata: ad.AnnData,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Transform AnnData to X, X_emb, y, groups for acute and subacute inflammation.

    Args:
        adata: AnnData object.

    Returns:
        Tuple of (X_raw, X_emb, y, groups).
    """
    adata_sub = adata[
        (adata.obs["group"] == "control")
        | (adata.obs["infl_acute"])
        | (adata.obs["infl_subacute"])
    ].copy()
    X = adata_sub.X.copy()
    X_emb = adata_sub.obsm["X_bulkformer"].copy()
    y = (
        (adata_sub.obs["infl_acute"] | adata_sub.obs["infl_subacute"])
        .map({True: 1, False: 0})
        .values.astype(int)
    )
    groups = adata_sub.obs.get(
        "dataset", pd.Series(np.zeros(len(adata_sub)), index=adata_sub.obs.index)
    )
    return X, X_emb, y, groups


def transform_adata_to_X_y_chronic(
    adata: ad.AnnData,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Transform AnnData to X, X_emb, y, groups for chronic inflammation.

    Args:
        adata: AnnData object.

    Returns:
        Tuple of (X_raw, X_emb, y, groups).
    """
    adata_sub = adata[(adata.obs["group"] == "control") | (adata.obs["infl_chronic"])].copy()
    X = adata_sub.X.copy()
    X_emb = adata_sub.obsm["X_bulkformer"].copy()
    y = adata_sub.obs["infl_chronic"].map({True: 1, False: 0}).values.astype(int)
    groups = adata_sub.obs.get(
        "dataset", pd.Series(np.zeros(len(adata_sub)), index=adata_sub.obs.index)
    )
    return X, X_emb, y, groups


def transform_adata_to_X_y_acute_to_chronic(
    adata: ad.AnnData,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Transform AnnData to X, X_emb, y, groups for acute vs chronic inflammation.

    Args:
        adata: AnnData object.

    Returns:
        Tuple of (X_raw, X_emb, y, groups).
    """
    adata_sub = adata[(adata.obs["infl_acute"]) | (adata.obs["infl_chronic"])].copy()
    X = adata_sub.X.copy()
    X_emb = adata_sub.obsm["X_bulkformer"].copy()
    y = adata_sub.obs["infl_chronic"].map({True: 0, False: 1}).values.astype(int)
    groups = adata_sub.obs.get(
        "dataset", pd.Series(np.zeros(len(adata_sub)), index=adata_sub.obs.index)
    )
    return X, X_emb, y, groups


def transform_adata_to_X_y_acute_subacute_to_chronic(
    adata: ad.AnnData,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Transform AnnData to X, X_emb, y, groups for acute+subacute vs chronic inflammation.

    Args:
        adata: AnnData object.

    Returns:
        Tuple of (X_raw, X_emb, y, groups).
    """
    adata_sub = adata[
        (adata.obs["infl_acute"]) | (adata.obs["infl_subacute"]) | (adata.obs["infl_chronic"])
    ].copy()
    X = adata_sub.X.copy()
    X_emb = adata_sub.obsm["X_bulkformer"].copy()
    y = adata_sub.obs["infl_chronic"].map({True: 0, False: 1}).values.astype(int)
    groups = adata_sub.obs.get(
        "dataset", pd.Series(np.zeros(len(adata_sub)), index=adata_sub.obs.index)
    )
    return X, X_emb, y, groups
