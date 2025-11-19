"""Data preparation functions for fine-tuning."""

import json
from pathlib import Path
from typing import Literal

import anndata as ad
from loguru import logger
import numpy as np
import pandas as pd

from inflamm_debate_fm.config import DATA_DIR, get_config
from inflamm_debate_fm.data.load import combine_adatas

SpeciesLiteral = Literal["human", "mouse", "combined", "universal"]


def prepare_finetuning_data(
    species: SpeciesLiteral,
    n_inflammation: int = 32,
    n_control: int = 32,
    random_seed: int = 42,
    output_dir: Path | None = None,
) -> tuple[ad.AnnData, pd.DataFrame]:
    """Prepare data for fine-tuning with balanced inflammation/control samples.

    Args:
        species: Which species to use ('human', 'mouse', 'combined', or 'universal').
        n_inflammation: Number of inflammation samples to use.
        n_control: Number of control samples to use.
        random_seed: Random seed for reproducibility.
        output_dir: Directory to save metadata about which samples were used.
                    If None, uses default from config.

    Returns:
        Tuple of (AnnData with selected samples, DataFrame with sample metadata).
    """
    np.random.seed(random_seed)

    config = get_config()

    # Load individual datasets from anndata_cleaned (files like human_burn.h5ad, mouse_burn.h5ad)
    cleaned_data_dir = DATA_DIR / config["paths"]["anndata_cleaned_dir"]

    logger.info(f"Loading data for {species} fine-tuning from {cleaned_data_dir}")

    # Load all individual dataset files
    adatas = {}
    for f in sorted(cleaned_data_dir.glob("*.h5ad")):
        name = f.stem
        adatas[name] = ad.read_h5ad(f)

    if len(adatas) == 0:
        raise ValueError(
            f"No AnnData files found in {cleaned_data_dir}. "
            f"Expected files like human_burn.h5ad, human_sepsis.h5ad, mouse_burn.h5ad, etc."
        )

    logger.info(f"Found {len(adatas)} datasets: {list(adatas.keys())}")

    if species in {"combined", "universal"}:
        # Combine human and mouse
        human_adata = combine_adatas(adatas, "human")
        mouse_adata = combine_adatas(adatas, "mouse")
        adata = ad.concat(
            [human_adata, mouse_adata],
            join="outer",
            label="species",
            keys=["human", "mouse"],
        )
        # Fill NaN values created by outer join (genes not present in one species)
        # This is expected when combining human and mouse data
        # Convert to dense if sparse, then check for NaN
        X_dense = adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X
        if np.isnan(X_dense).any():
            nan_count = np.isnan(X_dense).sum()
            logger.info(
                f"Filling {nan_count} NaN values in combined data "
                "(genes not present in one species). This is expected."
            )
            # Fill NaN with -10 (BulkFormer's standard padding value for missing genes)
            # This matches what align_genes_to_bulkformer does for missing genes
            X_dense = np.nan_to_num(X_dense, nan=-10.0, posinf=-10.0, neginf=-10.0)
            adata.X = X_dense
    elif species == "human":
        adata = combine_adatas(adatas, "human")
    elif species == "mouse":
        adata = combine_adatas(adatas, "mouse")
    else:
        raise ValueError(f"Invalid species: {species}")

    logger.info(f"Total samples in {species} data: {adata.shape[0]}")

    # Extract labels
    if "group" not in adata.obs.columns:
        raise ValueError("'group' column not found in adata.obs")

    # Get inflammation and control indices
    inflammation_mask = adata.obs["group"] == "inflammation"
    control_mask = adata.obs["group"] == "control"

    inflammation_indices = np.where(inflammation_mask)[0]
    control_indices = np.where(control_mask)[0]

    logger.info(
        f"Available samples - Inflammation: {len(inflammation_indices)}, "
        f"Control: {len(control_indices)}"
    )

    # Validate we have enough samples
    if len(inflammation_indices) < n_inflammation:
        raise ValueError(
            f"Not enough inflammation samples: {len(inflammation_indices)} < {n_inflammation}"
        )
    if len(control_indices) < n_control:
        raise ValueError(f"Not enough control samples: {len(control_indices)} < {n_control}")

    # Sample the required number
    selected_inflammation = np.random.choice(
        inflammation_indices, size=n_inflammation, replace=False
    )
    selected_control = np.random.choice(control_indices, size=n_control, replace=False)

    # Combine and sort
    selected_indices = np.sort(np.concatenate([selected_inflammation, selected_control]))

    # Create subset
    adata_subset = adata[selected_indices].copy()

    # Create metadata DataFrame tracking which samples were used
    sample_metadata = pd.DataFrame(
        {
            "sample_index": selected_indices,
            "original_index": adata_subset.obs_names,
            "label": adata_subset.obs["group"].values,
            "dataset": adata_subset.obs.get("dataset", "unknown").values,
        }
    )

    if "species" in adata_subset.obs.columns:
        sample_metadata["species"] = adata_subset.obs["species"].values

    # Calculate percentage of data used
    total_samples = len(inflammation_indices) + len(control_indices)
    used_samples = len(selected_indices)
    percentage_used = (used_samples / total_samples) * 100

    logger.info(
        f"Selected {used_samples} samples ({percentage_used:.1f}% of available data): "
        f"{n_inflammation} inflammation, {n_control} control"
    )

    # Save metadata if output_dir is provided
    if output_dir is not None:
        save_finetuning_metadata(
            sample_metadata=sample_metadata,
            species=species,
            n_inflammation=n_inflammation,
            n_control=n_control,
            percentage_used=percentage_used,
            output_dir=output_dir,
        )

    return adata_subset, sample_metadata


def save_finetuning_metadata(
    sample_metadata: pd.DataFrame,
    species: str,
    n_inflammation: int,
    n_control: int,
    percentage_used: float,
    output_dir: Path,
) -> None:
    """Save metadata about which samples were used for fine-tuning.

    Args:
        sample_metadata: DataFrame with sample information.
        species: Species used ('human', 'mouse', 'combined', or 'universal').
        n_inflammation: Number of inflammation samples.
        n_control: Number of control samples.
        percentage_used: Percentage of available data used.
        output_dir: Directory to save metadata.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save CSV with sample details
    metadata_path = output_dir / f"finetuning_samples_{species}.csv"
    sample_metadata.to_csv(metadata_path, index=False)
    logger.info(f"Saved sample metadata to {metadata_path}")

    # Save summary JSON
    summary = {
        "species": species,
        "n_inflammation": n_inflammation,
        "n_control": n_control,
        "total_samples": len(sample_metadata),
        "percentage_of_available_data": percentage_used,
        "metadata_file": str(metadata_path),
    }

    summary_path = output_dir / f"finetuning_summary_{species}.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")


def load_finetuning_metadata(species: str, metadata_dir: Path) -> tuple[pd.DataFrame, dict]:
    """Load metadata about which samples were used for fine-tuning.

    Args:
        species: Species ('human', 'mouse', 'combined', or 'universal').
        metadata_dir: Directory containing metadata files.

    Returns:
        Tuple of (sample metadata DataFrame, summary dictionary).
    """
    metadata_dir = Path(metadata_dir)

    # Load CSV
    metadata_path = metadata_dir / f"finetuning_samples_{species}.csv"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    sample_metadata = pd.read_csv(metadata_path)

    # Load summary
    summary_path = metadata_dir / f"finetuning_summary_{species}.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    with open(summary_path, "r") as f:
        summary = json.load(f)

    return sample_metadata, summary
