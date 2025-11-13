"""Functions for loading pre-computed embeddings."""

from pathlib import Path

import anndata as ad
from loguru import logger
import numpy as np

from inflamm_debate_fm.config import DATA_DIR, get_config


def load_embedding(
    dataset_name: str,
    embeddings_dir: Path | str | None = None,
    flavor: str = "default",
) -> np.ndarray:
    """Load embedding for a specific dataset.

    Args:
        dataset_name: Name of the dataset.
        embeddings_dir: Directory containing embedding files. If None, uses config default.
        flavor: Embedding flavor (e.g., 'default', 'normalized'). Used to construct filename.

    Returns:
        Embedding array.
    """
    config = get_config()

    if embeddings_dir is None:
        base_dir = DATA_DIR / config["paths"]["embeddings_dir"]
        if flavor != "default":
            embeddings_dir = base_dir / flavor
        else:
            embeddings_dir = base_dir
    else:
        embeddings_dir = Path(embeddings_dir)

    # Construct filename based on flavor
    if flavor == "default":
        filename = f"{dataset_name}_transcriptome_embeddings.npy"
    else:
        filename = f"{dataset_name}_transcriptome_embeddings_{flavor}.npy"

    embedding_path = embeddings_dir / filename

    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

    logger.info(f"Loading embedding from {embedding_path}")
    return np.load(embedding_path)


def load_embeddings_for_adatas(
    adatas: dict[str, ad.AnnData],
    embeddings_dir: Path | str | None = None,
    flavor: str = "default",
    embedding_key: str = "X_bulkformer",
) -> None:
    """Load and add embeddings to AnnData objects.

    Args:
        adatas: Dictionary of AnnData objects.
        embeddings_dir: Directory containing embedding files. If None, uses config default.
        flavor: Embedding flavor (e.g., 'default', 'normalized').
        embedding_key: Key to store embedding in AnnData.obsm.
    """
    config = get_config()

    if embeddings_dir is None:
        base_dir = DATA_DIR / config["paths"]["embeddings_dir"]
        if flavor != "default":
            embeddings_dir = base_dir / flavor
        else:
            embeddings_dir = base_dir
    else:
        embeddings_dir = Path(embeddings_dir)

    for name, adata in adatas.items():
        try:
            embedding = load_embedding(name, embeddings_dir, flavor)
            adata.obsm[embedding_key] = embedding
            logger.info(f"Added {flavor} embeddings to {name}")
        except FileNotFoundError as e:
            logger.warning(f"Could not load embeddings for {name}: {e}")
