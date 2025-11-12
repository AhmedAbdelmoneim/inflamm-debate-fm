"""Embedding generation commands."""

from pathlib import Path

import anndata as ad
from loguru import logger
import numpy as np
import typer

from inflamm_debate_fm.config import DATA_DIR, get_config
from inflamm_debate_fm.embeddings.generate import generate_embeddings

app = typer.Typer(help="Embedding generation commands")


@app.command("generate")
def embed_generate(
    dataset_name: str,
    ann_data_dir: Path | None = None,
    output_dir: Path | None = None,
    model_name: str = "bulkformer",
    flavor: str = "default",
    batch_size: int = 256,
    device: str = "cpu",
    aggregate_type: str = typer.Option(
        "max", help="Aggregation method for BulkFormer (max, mean, median, all)"
    ),
    model_dir: Path | None = None,
    data_dir: Path | None = None,
) -> None:
    """Generate embeddings for a dataset using BulkFormer.

    Args:
        dataset_name: Name of the dataset to process.
        ann_data_dir: Directory containing AnnData files.
        output_dir: Directory to save embeddings.
        model_name: Name of the embedding model (currently only 'bulkformer').
        flavor: Embedding flavor identifier.
        batch_size: Batch size for embedding generation (default: 256 for BulkFormer).
        device: Device to use ('cpu' or 'cuda').
        aggregate_type: Aggregation method for BulkFormer embeddings ('max', 'mean', 'median', 'all').
        model_dir: Directory containing model files. If None, uses config default.
        data_dir: Directory containing data files. If None, uses config default.
    """
    config = get_config()

    if ann_data_dir is None:
        ann_data_dir = DATA_DIR / config["paths"]["ann_data_dir"]
    else:
        ann_data_dir = Path(ann_data_dir)

    if output_dir is None:
        output_dir = DATA_DIR / config["paths"]["embeddings_dir"]
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load AnnData
    adata_path = ann_data_dir / f"{dataset_name}_orthologs.h5ad"
    if not adata_path.exists():
        raise FileNotFoundError(f"AnnData file not found: {adata_path}")

    adata = ad.read_h5ad(adata_path)
    logger.info(f"Loaded {dataset_name}: {adata.shape}")

    # Generate embeddings
    logger.info(f"Generating embeddings for {dataset_name} using {model_name}...")
    embeddings = generate_embeddings(
        adata=adata,
        model_name=model_name,
        flavor=flavor,
        batch_size=batch_size,
        device=device,
        output_dir=None,  # We'll save manually below
        model_dir=model_dir,
        data_dir=data_dir,
        aggregate_type=aggregate_type,
    )

    # Save embeddings
    output_path = output_dir / f"{dataset_name}_transcriptome_embeddings.npy"
    np.save(output_path, embeddings)
    logger.success(f"Saved embeddings to {output_path}")
