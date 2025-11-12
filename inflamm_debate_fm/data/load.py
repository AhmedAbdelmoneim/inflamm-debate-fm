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


def load_combined_adatas(combined_data_dir: Path) -> dict[str, ad.AnnData]:
    """Load combined human and mouse AnnData files."""
    return {
        "human": ad.read_h5ad(combined_data_dir / "human_combined.h5ad"),
        "mouse": ad.read_h5ad(combined_data_dir / "mouse_combined.h5ad"),
    }


def combine_adatas(adatas: dict[str, ad.AnnData], species_prefix: str) -> ad.AnnData:
    """Combine AnnData objects for a given species."""
    species_adatas = [adatas[k] for k in sorted(adatas.keys()) if k.startswith(species_prefix)]
    return ad.concat(
        species_adatas,
        join="outer",
        label="dataset",
        keys=[k for k in sorted(adatas.keys()) if k.startswith(species_prefix)],
    )
