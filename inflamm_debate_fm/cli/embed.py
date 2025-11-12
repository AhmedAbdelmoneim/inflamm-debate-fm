"""Embedding generation commands."""

import anndata as ad
from loguru import logger
import numpy as np
import typer

from inflamm_debate_fm.bulkformer.generate import generate_bulkformer_embeddings
from inflamm_debate_fm.config import DATA_ROOT, get_config

app = typer.Typer(help="Embedding generation commands")


@app.command("generate")
def embed_generate(dataset_name: str, batch_size: int = 256, device: str = "cpu") -> None:
    """Generate embeddings for a dataset using BulkFormer."""
    config = get_config()
    ann_data_dir = DATA_ROOT / config["paths"]["ann_data_dir"]
    output_dir = DATA_ROOT / config["paths"]["embeddings_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    adata_path = ann_data_dir / f"{dataset_name}_orthologs.h5ad"
    adata = ad.read_h5ad(adata_path)
    logger.info(f"Loaded {dataset_name}: {adata.shape}")

    logger.info(f"Generating embeddings for {dataset_name}...")
    embeddings = generate_bulkformer_embeddings(adata=adata, batch_size=batch_size, device=device)

    output_path = output_dir / f"{dataset_name}_transcriptome_embeddings.npy"
    np.save(output_path, embeddings)
    logger.success(f"Saved embeddings to {output_path}")
