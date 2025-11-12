"""Functions for generating embeddings from transcriptome data."""

from pathlib import Path

import anndata as ad
from loguru import logger
import numpy as np

from inflamm_debate_fm.config import (
    BULKFORMER_DATA_DIR,
    BULKFORMER_MODEL_DIR,
    DATA_DIR,
    get_config,
)


def generate_embeddings(
    adata: ad.AnnData,
    model_name: str = "bulkformer",
    flavor: str = "default",
    batch_size: int = 32,
    device: str = "cpu",
    output_dir: Path | str | None = None,
    model_dir: Path | None = None,
    data_dir: Path | None = None,
    aggregate_type: str = "max",
) -> np.ndarray:
    """Generate embeddings for an AnnData object.

    Args:
        adata: AnnData object with transcriptome data in .X.
        model_name: Name of the embedding model to use.
        flavor: Embedding flavor identifier (for different preprocessing/versions).
        batch_size: Batch size for embedding generation.
        device: Device to use ('cpu' or 'cuda').
        output_dir: Directory to save embeddings. If None, uses config default.
        model_dir: Directory containing model files. If None, uses config default.
        data_dir: Directory containing data files. If None, uses config default.
        aggregate_type: Aggregation method for BulkFormer ('max', 'mean', 'median', 'all').

    Returns:
        Embedding array of shape (n_samples, n_embedding_dim).

    Raises:
        ValueError: If model_name is not supported.
    """
    config = get_config()
    embedding_config = config.get("embedding", {})

    model_name = embedding_config.get("model_name", model_name)
    batch_size = embedding_config.get("batch_size", batch_size)
    device = embedding_config.get("device", device)

    logger.info(f"Generating {flavor} embeddings for {adata.shape[0]} samples using {model_name}")

    if model_name == "bulkformer":
        from inflamm_debate_fm.bulkformer.generate import generate_bulkformer_embeddings

        # Determine output path
        output_path = None
        if output_dir is not None:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            # Output path will be set by the caller typically
        elif flavor != "default":
            output_dir = DATA_DIR / config["paths"]["embeddings_dir"] / flavor
            output_dir.mkdir(parents=True, exist_ok=True)

        # Use configurable model and data directories
        if model_dir is None:
            model_dir = BULKFORMER_MODEL_DIR
        if data_dir is None:
            data_dir = BULKFORMER_DATA_DIR

        embeddings = generate_bulkformer_embeddings(
            adata=adata,
            model_dir=model_dir,
            data_dir=data_dir,
            device=device,
            batch_size=batch_size,
            aggregate_type=aggregate_type,
            output_path=output_path,
        )
    else:
        raise ValueError(f"Unsupported model_name: {model_name}. Supported: 'bulkformer'")

    embeddings = np.array(embeddings)

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
    adatas: dict[str, ad.AnnData],
    model_name: str = "bulkformer",
    flavor: str = "default",
    batch_size: int = 32,
    device: str = "cpu",
    output_dir: Path | str | None = None,
) -> dict[str, np.ndarray]:
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
