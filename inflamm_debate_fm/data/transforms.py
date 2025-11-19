"""Data transformation functions for creating X, y pairs from AnnData."""

from collections.abc import Callable

import anndata as ad
import numpy as np
import pandas as pd


def _transform_adata_to_X_y(
    adata: ad.AnnData,
    filter_func: Callable[[ad.AnnData], ad.AnnData],
    label_func: Callable[[ad.AnnData], np.ndarray],
    embedding_keys: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, pd.Series]:
    """Generic transform function to extract X, X_emb, y, groups from AnnData.

    Args:
        adata: AnnData object.
        filter_func: Function to filter AnnData object.
        label_func: Function to extract labels from AnnData object.
        embedding_keys: List of obsm keys to extract embeddings from. If None, detects available.

    Returns:
        Tuple of (X_raw, X_embeddings_dict, y, groups).
        X_embeddings_dict maps embedding key to embedding array.
    """
    adata_sub = filter_func(adata)
    X = adata_sub.X.copy()

    # Detect available embeddings if not specified
    if embedding_keys is None:
        embedding_keys = [key for key in adata_sub.obsm.keys() if key.startswith("X_")]

    # Extract all requested embeddings
    X_embeddings = {}
    for key in embedding_keys:
        if key in adata_sub.obsm:
            X_embeddings[key] = adata_sub.obsm[key].copy()

    y = label_func(adata_sub)
    groups = adata_sub.obs.get(
        "dataset", pd.Series(np.zeros(len(adata_sub)), index=adata_sub.obs.index)
    )
    return X, X_embeddings, y, groups


def transform_adata_to_X_y_all(
    adata: ad.AnnData,
    embedding_keys: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, pd.Series]:
    """Transform AnnData to X, X_emb, y, groups for all samples.

    Args:
        adata: AnnData object.
        embedding_keys: List of obsm keys to extract embeddings from. If None, detects available.

    Returns:
        Tuple of (X_raw, X_embeddings_dict, y, groups).
    """
    return _transform_adata_to_X_y(
        adata=adata,
        filter_func=lambda x: x,
        label_func=lambda x: x.obs["group"]
        .map({"inflammation": 1, "control": 0})
        .values.astype(int),
        embedding_keys=embedding_keys,
    )


def transform_adata_to_X_y_takao(
    adata: ad.AnnData,
    embedding_keys: list[str] | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray, pd.Series]:
    """Transform AnnData to X, X_emb, y, groups for Takao subset.

    Args:
        adata: AnnData object.
        embedding_keys: List of obsm keys to extract embeddings from. If None, detects available.

    Returns:
        Tuple of (X_raw, X_embeddings_dict, y, groups).
    """
    return _transform_adata_to_X_y(
        adata=adata,
        filter_func=lambda x: x[~x.obs["takao_status"].isna()].copy(),
        label_func=lambda x: x.obs["takao_status"]
        .map({"takao_inflamed": 1, "takao_control": 0})
        .values.astype(int),
        embedding_keys=embedding_keys,
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
