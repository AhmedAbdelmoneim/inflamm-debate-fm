"""Data transformation functions for creating X, y pairs from AnnData."""

from collections.abc import Callable

import anndata as ad
import numpy as np
import pandas as pd


def _transform_adata_to_X_y(
    adata: ad.AnnData,
    filter_func: Callable[[ad.AnnData], ad.AnnData],
    label_func: Callable[[ad.AnnData], np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Generic transform function to extract X, X_emb, y, groups from AnnData.

    Args:
        adata: AnnData object.
        filter_func: Function to filter AnnData object.
        label_func: Function to extract labels from AnnData object.

    Returns:
        Tuple of (X_raw, X_emb, y, groups).
    """
    adata_sub = filter_func(adata)
    X = adata_sub.X.copy()
    X_emb = adata_sub.obsm["X_bulkformer"].copy()
    y = label_func(adata_sub)
    groups = adata_sub.obs.get(
        "dataset", pd.Series(np.zeros(len(adata_sub)), index=adata_sub.obs.index)
    )
    return X, X_emb, y, groups


def transform_adata_to_X_y_all(
    adata: ad.AnnData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Transform AnnData to X, X_emb, y, groups for all samples.

    Args:
        adata: AnnData object.

    Returns:
        Tuple of (X_raw, X_emb, y, groups).
    """
    return _transform_adata_to_X_y(
        adata=adata,
        filter_func=lambda x: x,
        label_func=lambda x: x.obs["group"]
        .map({"inflammation": 1, "control": 0})
        .values.astype(int),
    )


def transform_adata_to_X_y_takao(
    adata: ad.AnnData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Transform AnnData to X, X_emb, y, groups for Takao subset.

    Args:
        adata: AnnData object.

    Returns:
        Tuple of (X_raw, X_emb, y, groups).
    """
    return _transform_adata_to_X_y(
        adata=adata,
        filter_func=lambda x: x[~x.obs["takao_status"].isna()].copy(),
        label_func=lambda x: x.obs["takao_status"]
        .map({"takao_inflamed": 1, "takao_control": 0})
        .values.astype(int),
    )


def transform_adata_to_X_y_acute(
    adata: ad.AnnData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Transform AnnData to X, X_emb, y, groups for acute inflammation.

    Args:
        adata: AnnData object.

    Returns:
        Tuple of (X_raw, X_emb, y, groups).
    """
    return _transform_adata_to_X_y(
        adata=adata,
        filter_func=lambda x: x[(x.obs["group"] == "control") | (x.obs["infl_acute"])].copy(),
        label_func=lambda x: x.obs["infl_acute"].map({True: 1, False: 0}).values.astype(int),
    )


def transform_adata_to_X_y_subacute(
    adata: ad.AnnData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Transform AnnData to X, X_emb, y, groups for subacute inflammation.

    Args:
        adata: AnnData object.

    Returns:
        Tuple of (X_raw, X_emb, y, groups).
    """
    return _transform_adata_to_X_y(
        adata=adata,
        filter_func=lambda x: x[(x.obs["group"] == "control") | (x.obs["infl_subacute"])].copy(),
        label_func=lambda x: x.obs["infl_subacute"].map({True: 1, False: 0}).values.astype(int),
    )


def transform_adata_to_X_y_acute_and_subacute(
    adata: ad.AnnData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Transform AnnData to X, X_emb, y, groups for acute and subacute inflammation.

    Args:
        adata: AnnData object.

    Returns:
        Tuple of (X_raw, X_emb, y, groups).
    """
    return _transform_adata_to_X_y(
        adata=adata,
        filter_func=lambda x: x[
            (x.obs["group"] == "control") | (x.obs["infl_acute"]) | (x.obs["infl_subacute"])
        ].copy(),
        label_func=lambda x: (x.obs["infl_acute"] | x.obs["infl_subacute"])
        .map({True: 1, False: 0})
        .values.astype(int),
    )


def transform_adata_to_X_y_chronic(
    adata: ad.AnnData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Transform AnnData to X, X_emb, y, groups for chronic inflammation.

    Args:
        adata: AnnData object.

    Returns:
        Tuple of (X_raw, X_emb, y, groups).
    """
    return _transform_adata_to_X_y(
        adata=adata,
        filter_func=lambda x: x[(x.obs["group"] == "control") | (x.obs["infl_chronic"])].copy(),
        label_func=lambda x: x.obs["infl_chronic"].map({True: 1, False: 0}).values.astype(int),
    )


def transform_adata_to_X_y_acute_to_chronic(
    adata: ad.AnnData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Transform AnnData to X, X_emb, y, groups for acute vs chronic inflammation.

    Args:
        adata: AnnData object.

    Returns:
        Tuple of (X_raw, X_emb, y, groups).
    """
    return _transform_adata_to_X_y(
        adata=adata,
        filter_func=lambda x: x[(x.obs["infl_acute"]) | (x.obs["infl_chronic"])].copy(),
        label_func=lambda x: x.obs["infl_chronic"].map({True: 0, False: 1}).values.astype(int),
    )


def transform_adata_to_X_y_acute_subacute_to_chronic(
    adata: ad.AnnData,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
    """Transform AnnData to X, X_emb, y, groups for acute+subacute vs chronic inflammation.

    Args:
        adata: AnnData object.

    Returns:
        Tuple of (X_raw, X_emb, y, groups).
    """
    return _transform_adata_to_X_y(
        adata=adata,
        filter_func=lambda x: x[
            (x.obs["infl_acute"]) | (x.obs["infl_subacute"]) | (x.obs["infl_chronic"])
        ].copy(),
        label_func=lambda x: x.obs["infl_chronic"].map({True: 0, False: 1}).values.astype(int),
    )
