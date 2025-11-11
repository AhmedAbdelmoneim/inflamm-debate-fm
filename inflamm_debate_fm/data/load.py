"""Data loading functions."""

import os
from pathlib import Path
from typing import Dict

import anndata as ad
from loguru import logger
import numpy as np

from inflamm_debate_fm.config import DATA_DIR
from inflamm_debate_fm.config.config import get_config


def load_adatas(
    ann_data_dir: Path | str | None = None,
    embeddings_dir: Path | str | None = None,
    load_embeddings: bool = True,
) -> Dict[str, ad.AnnData]:
    """Load AnnData files and optionally add embeddings.

    Args:
        ann_data_dir: Directory containing AnnData files. If None, uses config default.
        embeddings_dir: Directory containing embedding files. If None, uses config default.
        load_embeddings: Whether to load and add embeddings to AnnData objects.

    Returns:
        Dictionary mapping dataset names to AnnData objects.
    """
    config = get_config()

    if ann_data_dir is None:
        ann_data_dir = DATA_DIR / config["paths"]["ann_data_dir"]
    else:
        ann_data_dir = Path(ann_data_dir)

    if embeddings_dir is None:
        embeddings_dir = DATA_DIR / config["paths"]["embeddings_dir"]
    else:
        embeddings_dir = Path(embeddings_dir)

    logger.info(f"Loading AnnData files from {ann_data_dir}")
    adatas = {}

    for f in sorted(os.listdir(ann_data_dir)):
        if f.endswith(".h5ad"):
            name = f.replace("_orthologs.h5ad", "")
            path = ann_data_dir / f
            adatas[name] = ad.read_h5ad(path)
            logger.info(f"Loaded {name}: {adatas[name].shape}")

            # Add embeddings if requested
            if load_embeddings:
                embedding_path = embeddings_dir / f"{name}_transcriptome_embeddings.npy"
                if embedding_path.exists():
                    embedding = np.load(embedding_path)
                    adatas[name].obsm["X_bulkformer"] = embedding
                    logger.info(f"Added embeddings to {name}")
                else:
                    logger.warning(f"Embedding file not found: {embedding_path}")

    return adatas


def load_combined_adatas(
    combined_data_dir: Path | str | None = None,
) -> Dict[str, ad.AnnData]:
    """Load combined human and mouse AnnData files.

    Args:
        combined_data_dir: Directory containing combined AnnData files.
            If None, uses config default.

    Returns:
        Dictionary with 'human' and 'mouse' keys containing combined AnnData objects.
    """
    config = get_config()

    if combined_data_dir is None:
        combined_data_dir = DATA_DIR / config["paths"]["combined_data_dir"]
    else:
        combined_data_dir = Path(combined_data_dir)

    logger.info(f"Loading combined AnnData files from {combined_data_dir}")

    human_path = combined_data_dir / "human_combined.h5ad"
    mouse_path = combined_data_dir / "mouse_combined.h5ad"

    if not human_path.exists():
        raise FileNotFoundError(f"Human combined data not found: {human_path}")
    if not mouse_path.exists():
        raise FileNotFoundError(f"Mouse combined data not found: {mouse_path}")

    human_adata = ad.read_h5ad(human_path)
    mouse_adata = ad.read_h5ad(mouse_path)

    logger.info(f"Loaded human_combined: {human_adata.shape}")
    logger.info(f"Loaded mouse_combined: {mouse_adata.shape}")

    return {"human": human_adata, "mouse": mouse_adata}


def load_embedding(
    dataset_name: str,
    embeddings_dir: Path | str | None = None,
    embedding_key: str = "X_bulkformer",
) -> np.ndarray:
    """Load embedding for a specific dataset.

    Args:
        dataset_name: Name of the dataset.
        embeddings_dir: Directory containing embedding files. If None, uses config default.
        embedding_key: Key to store embedding in AnnData.obsm.

    Returns:
        Embedding array.
    """
    config = get_config()

    if embeddings_dir is None:
        embeddings_dir = DATA_DIR / config["paths"]["embeddings_dir"]
    else:
        embeddings_dir = Path(embeddings_dir)

    embedding_path = embeddings_dir / f"{dataset_name}_transcriptome_embeddings.npy"

    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding file not found: {embedding_path}")

    logger.info(f"Loading embedding from {embedding_path}")
    return np.load(embedding_path)


def combine_adatas(adatas: Dict[str, ad.AnnData], species_prefix: str) -> ad.AnnData:
    """Combine AnnData objects for a given species.

    Args:
        adatas: Dictionary of AnnData objects.
        species_prefix: Prefix to filter datasets (e.g., 'human' or 'mouse').

    Returns:
        Combined AnnData object.
    """
    species_adatas = [adatas[k] for k in sorted(adatas.keys()) if k.startswith(species_prefix)]
    species_keys = [k for k in sorted(adatas.keys()) if k.startswith(species_prefix)]

    if not species_adatas:
        raise ValueError(f"No datasets found with prefix '{species_prefix}'")

    logger.info(f"Combining {len(species_adatas)} {species_prefix} datasets: {species_keys}")

    combined = ad.concat(species_adatas, join="outer", label="dataset", keys=species_keys)

    logger.info(f"Combined {species_prefix} dataset shape: {combined.shape}")
    return combined
