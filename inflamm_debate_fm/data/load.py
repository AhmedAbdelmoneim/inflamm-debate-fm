"""Data loading functions."""

from pathlib import Path

import anndata as ad
import numpy as np


def load_adatas(
    ann_data_dir: Path, embeddings_dir: Path, load_embeddings: bool = True
) -> dict[str, ad.AnnData]:
    """Load AnnData files and optionally add embeddings."""
    adatas = {}
    for f in sorted(ann_data_dir.glob("*.h5ad")):
        name = f.stem.replace("_orthologs", "")
        adatas[name] = ad.read_h5ad(f)
        if load_embeddings:
            embedding_path = embeddings_dir / f"{name}_transcriptome_embeddings.npy"
            if embedding_path.exists():
                adatas[name].obsm["X_bulkformer"] = np.load(embedding_path)
    return adatas


def load_cleaned_adatas(
    anndata_cleaned_dir: Path, embeddings_dir: Path | None = None, load_embeddings: bool = False
) -> dict[str, ad.AnnData]:
    """Load cleaned AnnData files.

    Args:
        anndata_cleaned_dir: Directory containing cleaned AnnData files.
        embeddings_dir: Optional directory containing embeddings.
        load_embeddings: Whether to load embeddings if available.

    Returns:
        Dictionary of dataset name to AnnData object.
    """
    adatas = {}
    for f in sorted(anndata_cleaned_dir.glob("*.h5ad")):
        name = f.stem
        adatas[name] = ad.read_h5ad(f)
        if load_embeddings and embeddings_dir is not None:
            embedding_path = embeddings_dir / f"{name}_transcriptome_embeddings.npy"
            if embedding_path.exists():
                adatas[name].obsm["X_bulkformer"] = np.load(embedding_path)
    return adatas


def load_combined_adatas(combined_data_dir: Path) -> dict[str, ad.AnnData]:
    """Load combined human and mouse AnnData files."""
    return {
        "human": ad.read_h5ad(combined_data_dir / "human_combined.h5ad"),
        "mouse": ad.read_h5ad(combined_data_dir / "mouse_combined.h5ad"),
    }


def combine_adatas(adatas: dict[str, ad.AnnData], species_prefix: str) -> ad.AnnData:
    """Combine AnnData objects for a given species."""
    species_keys = [k for k in sorted(adatas.keys()) if k.startswith(species_prefix)]
    if len(species_keys) == 0:
        raise ValueError(f"No datasets found with prefix '{species_prefix}'")
    species_adatas = [adatas[k] for k in species_keys]
    if len(species_adatas) == 1:
        # If only one dataset, return it directly with dataset label
        result = species_adatas[0].copy()
        result.obs["dataset"] = species_keys[0]
        return result
    return ad.concat(
        species_adatas,
        join="outer",
        label="dataset",
        keys=species_keys,
    )
