"""Functions for generating embeddings from transcriptome data."""

from pathlib import Path
from typing import Dict

import anndata as ad
from loguru import logger
import numpy as np

from inflamm_debate_fm.config import DATA_DIR
from inflamm_debate_fm.config.config import get_config


def generate_embeddings(
    adata: ad.AnnData,
    model_name: str = "bulkformer",
    flavor: str = "default",
    batch_size: int = 32,
    device: str = "cpu",
    output_dir: Path | str | None = None,
) -> np.ndarray:
    """Generate embeddings for an AnnData object.

    This is a placeholder function. In practice, this would:
    1. Load the embedding model (e.g., bulkformer)
    2. Process the transcriptome data (X) through the model
    3. Return the embeddings

    Args:
        adata: AnnData object with transcriptome data in .X.
        model_name: Name of the embedding model to use.
        flavor: Embedding flavor identifier (for different preprocessing/versions).
        batch_size: Batch size for embedding generation.
        device: Device to use ('cpu' or 'cuda').
        output_dir: Directory to save embeddings. If None, uses config default.

    Returns:
        Embedding array of shape (n_samples, n_embedding_dim).

    Raises:
        NotImplementedError: This function is a placeholder and needs to be implemented.
    """
    config = get_config()
    embedding_config = config.get("embedding", {})

    model_name = embedding_config.get("model_name", model_name)
    batch_size = embedding_config.get("batch_size", batch_size)
    device = embedding_config.get("device", device)

    logger.info(f"Generating {flavor} embeddings for {adata.shape[0]} samples using {model_name}")
    logger.warning(
        "generate_embeddings is a placeholder. Implement actual embedding generation logic."
    )

    # Placeholder: return zeros
    # In practice, this would:
    # 1. Load model from checkpoint or hub
    # 2. Process adata.X through model in batches
    # 3. Return embeddings
    n_samples = adata.shape[0]
    n_dim = 512  # Placeholder dimension

    embeddings = np.zeros((n_samples, n_dim))

    # Save embeddings if output_dir is provided
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset_name = adata.uns.get("dataset_name", "unknown")
        if flavor == "default":
            filename = f"{dataset_name}_transcriptome_embeddings.npy"
        else:
            filename = f"{dataset_name}_transcriptome_embeddings_{flavor}.npy"

        output_path = output_dir / filename
        np.save(output_path, embeddings)
        logger.info(f"Saved embeddings to {output_path}")

    return embeddings


def generate_embeddings_for_datasets(
    adatas: Dict[str, ad.AnnData],
    model_name: str = "bulkformer",
    flavor: str = "default",
    batch_size: int = 32,
    device: str = "cpu",
    output_dir: Path | str | None = None,
) -> Dict[str, np.ndarray]:
    """Generate embeddings for multiple datasets.

    Args:
        adatas: Dictionary of AnnData objects.
        model_name: Name of the embedding model to use.
        flavor: Embedding flavor identifier.
        batch_size: Batch size for embedding generation.
        device: Device to use ('cpu' or 'cuda').
        output_dir: Directory to save embeddings. If None, uses config default.

    Returns:
        Dictionary mapping dataset names to embedding arrays.
    """
    config = get_config()

    if output_dir is None:
        base_dir = DATA_DIR / config["paths"]["embeddings_dir"]
        if flavor != "default":
            output_dir = base_dir / flavor
        else:
            output_dir = base_dir
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = {}
    for name, adata in adatas.items():
        logger.info(f"Generating embeddings for {name}")
        adata.uns["dataset_name"] = name
        embedding = generate_embeddings(adata, model_name, flavor, batch_size, device, output_dir)
        embeddings[name] = embedding

    return embeddings
