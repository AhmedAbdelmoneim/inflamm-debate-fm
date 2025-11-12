"""Inflammation vector calculation and bootstrapping."""

from loguru import logger
import numpy as np
from sklearn.utils import resample

from inflamm_debate_fm.config import get_config


def calculate_inflammation_vector(embeddings: np.ndarray, labels: np.ndarray) -> np.ndarray | None:
    """Calculate the mean(inflamed) - mean(control) vector.

    Args:
        embeddings: Embedding matrix of shape (n_samples, n_features).
        labels: Binary labels (0=control, 1=inflamed).

    Returns:
        Inflammation vector of shape (1, n_features) or None if one class is missing.
    """
    control_mask = labels == 0
    inflamed_mask = labels == 1

    if np.sum(control_mask) == 0 or np.sum(inflamed_mask) == 0:
        logger.warning("One class is missing. Cannot calculate vector.")
        return None

    mean_control = np.mean(embeddings[control_mask], axis=0)
    mean_inflamed = np.mean(embeddings[inflamed_mask], axis=0)

    # Ensure vector is 2D for compatibility
    return (mean_inflamed - mean_control).reshape(1, -1)


def bootstrap_vector(
    embeddings: np.ndarray,
    labels: np.ndarray,
    n_bootstraps: int = 20,
) -> tuple[np.ndarray | None, list[float]]:
    """Generate bootstrapped inflammation vectors and calculate similarities.

    Args:
        embeddings: Embedding matrix of shape (n_samples, n_features).
        labels: Binary labels (0=control, 1=inflamed).
        n_bootstraps: Number of bootstrap iterations.

    Returns:
        Tuple of (original_vector, list of cosine similarities to original).
    """
    config = get_config()
    n_bootstraps = config.get("model", {}).get("n_bootstraps", n_bootstraps)

    bootstrapped_vectors = []

    control_indices = np.where(labels == 0)[0]
    inflamed_indices = np.where(labels == 1)[0]

    original_vector = calculate_inflammation_vector(embeddings, labels)
    if original_vector is None:
        return None, []

    similarities = []

    logger.info(f"Running {n_bootstraps} bootstraps...")
    for i in range(n_bootstraps):
        # Resample indices with replacement
        bs_control_indices = resample(control_indices)
        bs_inflamed_indices = resample(inflamed_indices)

        # Get bootstrapped embeddings
        bs_embeddings_control = embeddings[bs_control_indices]
        bs_embeddings_inflamed = embeddings[bs_inflamed_indices]

        # Calculate vector for this bootstrap sample
        mean_bs_control = np.mean(bs_embeddings_control, axis=0)
        mean_bs_inflamed = np.mean(bs_embeddings_inflamed, axis=0)
        bs_vector = (mean_bs_inflamed - mean_bs_control).reshape(1, -1)

        bootstrapped_vectors.append(bs_vector)

        # Calculate cosine similarity to original vector
        from sklearn.metrics.pairwise import cosine_similarity

        sim = cosine_similarity(original_vector, bs_vector)[0, 0]
        similarities.append(sim)

    return original_vector, similarities
