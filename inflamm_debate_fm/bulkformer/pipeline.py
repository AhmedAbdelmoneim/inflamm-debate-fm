"""BulkFormer embedding generation pipeline with different configurations."""

from pathlib import Path

import anndata as ad
from loguru import logger
import numpy as np
import pandas as pd

from inflamm_debate_fm.bulkformer.embed import (
    extract_embeddings_from_adata,
    load_bulkformer_model,
)
from inflamm_debate_fm.config import DATA_ROOT, get_config
from inflamm_debate_fm.data.orthologs import load_orthology_mapping
from inflamm_debate_fm.utils.wandb_utils import init_wandb, log_results

# Dataset names by species
HUMAN_DATASETS = ["human_burn", "human_trauma", "human_sepsis"]
MOUSE_DATASETS = ["mouse_burn", "mouse_trauma", "mouse_sepsis", "mouse_infection"]


def filter_to_ortholog_genes(adata: ad.AnnData, orthology_mapping: pd.DataFrame) -> ad.AnnData:
    """Filter AnnData to only genes present in mouse ortholog mapping.

    Args:
        adata: AnnData object with 'ensembl_id' in var.
        orthology_mapping: DataFrame with 'human_ensembl' column.

    Returns:
        Filtered AnnData object.
    """
    if "ensembl_id" not in adata.var.columns:
        adata.var["ensembl_id"] = adata.var.index

    # Get set of human Ensembl IDs that have mouse orthologs
    ortholog_genes = set(orthology_mapping["human_ensembl"].dropna().unique())

    # Filter to genes present in ortholog mapping
    mask = adata.var["ensembl_id"].isin(ortholog_genes)
    adata_filtered = adata[:, mask].copy()

    logger.info(f"Filtered to ortholog genes: {adata.shape[1]} -> {adata_filtered.shape[1]} genes")

    return adata_filtered


def generate_embeddings_for_config(
    config_name: str,
    datasets: list[str],
    output_dir: Path,
    device: str = "cpu",
    batch_size: int = 256,
    filter_orthologs: bool = False,
    use_wandb: bool = False,
) -> None:
    """Generate embeddings for a specific configuration.

    Args:
        config_name: Name of the configuration (e.g., 'human_only', 'mouse_only').
        datasets: List of dataset names to process.
        output_dir: Directory to save embeddings.
        device: Device to run inference on.
        batch_size: Batch size for inference.
        filter_orthologs: If True, filter human datasets to ortholog genes only.
        use_wandb: If True, log to wandb.
    """
    config = get_config()
    ann_data_dir = DATA_ROOT / config["paths"]["ann_data_dir"]
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb if requested
    wandb_run = None
    if use_wandb:
        try:
            wandb_run = init_wandb(
                project=config.get("wandb", {}).get("project", "inflamm-debate-fm"),
                tags=config.get("wandb", {}).get("tags", []) + ["embeddings", config_name],
                config={
                    "config_name": config_name,
                    "datasets": datasets,
                    "device": device,
                    "batch_size": batch_size,
                    "filter_orthologs": filter_orthologs,
                },
            )
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            use_wandb = False

    # Load orthology mapping if needed
    orthology_mapping = None
    if filter_orthologs:
        logger.info("Loading orthology mapping for gene filtering...")
        orthology_mapping = load_orthology_mapping()

    # Load model once for all datasets
    logger.info(f"Loading BulkFormer model for {config_name} configuration...")
    model = load_bulkformer_model(device=device)

    # Process each dataset
    for dataset_name in datasets:
        logger.info(f"Processing {dataset_name} for {config_name} configuration...")

        adata_path = ann_data_dir / f"{dataset_name}_orthologs.h5ad"
        if not adata_path.exists():
            logger.warning(f"Dataset not found: {adata_path}, skipping...")
            continue

        adata = ad.read_h5ad(adata_path)
        logger.info(f"Loaded {dataset_name}: {adata.shape}")

        # Filter to ortholog genes if requested
        if filter_orthologs and dataset_name.startswith("human"):
            adata = filter_to_ortholog_genes(adata, orthology_mapping)

        # Generate embeddings
        embeddings = extract_embeddings_from_adata(
            adata=adata,
            model=model,
            device=device,
            batch_size=batch_size,
        )

        # Save embeddings
        output_filename = f"{dataset_name}_{config_name}_embeddings.npy"
        output_path = output_dir / output_filename
        np.save(output_path, embeddings)
        logger.success(f"Saved embeddings to {output_path}")

        # Log to wandb
        if use_wandb and wandb_run:
            log_results(
                {
                    "dataset": dataset_name,
                    "n_samples": adata.shape[0],
                    "n_genes": adata.shape[1],
                    "embedding_shape": embeddings.shape,
                },
                prefix=f"{config_name}/{dataset_name}",
            )

    # Log completion
    if use_wandb and wandb_run:
        log_results(
            {"status": "completed", "n_datasets": len(datasets)},
            prefix=config_name,
        )

    logger.success(f"Completed {config_name} configuration")


def generate_all_embeddings(
    output_base_dir: Path | str,
    device: str = "cpu",
    batch_size: int = 256,
    use_wandb: bool = False,
) -> None:
    """Generate embeddings for all configurations.

    Args:
        output_base_dir: Base directory for output embeddings.
        device: Device to run inference on.
        batch_size: Batch size for inference.
        use_wandb: If True, log to wandb.
    """
    output_base_dir = Path(output_base_dir)

    # Configuration 1: Human datasets only
    logger.info("=" * 60)
    logger.info("Configuration 1: Human datasets only")
    logger.info("=" * 60)
    generate_embeddings_for_config(
        config_name="human_only",
        datasets=HUMAN_DATASETS,
        output_dir=output_base_dir / "human_only",
        device=device,
        batch_size=batch_size,
        use_wandb=use_wandb,
    )

    # Configuration 2: Mouse datasets only
    logger.info("=" * 60)
    logger.info("Configuration 2: Mouse datasets only")
    logger.info("=" * 60)
    generate_embeddings_for_config(
        config_name="mouse_only",
        datasets=MOUSE_DATASETS,
        output_dir=output_base_dir / "mouse_only",
        device=device,
        batch_size=batch_size,
        use_wandb=use_wandb,
    )

    # Configuration 3: Human datasets filtered to ortholog genes
    logger.info("=" * 60)
    logger.info("Configuration 3: Human datasets with ortholog gene filtering")
    logger.info("=" * 60)
    generate_embeddings_for_config(
        config_name="human_ortholog_filtered",
        datasets=HUMAN_DATASETS,
        output_dir=output_base_dir / "human_ortholog_filtered",
        device=device,
        batch_size=batch_size,
        filter_orthologs=True,
        use_wandb=use_wandb,
    )

    logger.success("All embedding configurations completed!")
