"""Dimensionality analysis functions."""

import numpy as np


def intrinsic_dimensionality(X_emb: np.ndarray) -> float:
    """Calculate intrinsic dimensionality using participation ratio.

    Participation ratio = (sum(variance)^2) / sum(variance^2)

    Args:
        X_emb: Embedding matrix of shape (n_samples, n_features).

    Returns:
        Participation ratio (intrinsic dimensionality estimate).
    """
    cov = np.cov(X_emb.T)
    eigvals = np.linalg.eigvalsh(cov)
    pr = (eigvals.sum() ** 2) / (eigvals**2).sum()
    return pr


def participation_ratio(X: np.ndarray) -> float:
    """Calculate participation ratio (alias for intrinsic_dimensionality).

    Args:
        X: Embedding matrix of shape (n_samples, n_features).

    Returns:
        Participation ratio.
    """
    return intrinsic_dimensionality(X)
