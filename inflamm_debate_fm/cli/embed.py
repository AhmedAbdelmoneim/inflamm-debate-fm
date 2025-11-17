"""Embedding generation commands."""

from pathlib import Path

from loguru import logger
import typer

from inflamm_debate_fm.bulkformer.embed import extract_embeddings_from_adata
from inflamm_debate_fm.bulkformer.pipeline import generate_all_embeddings
from inflamm_debate_fm.config import DATA_ROOT, get_config

app = typer.Typer(help="Embedding generation commands")


@app.command("generate")
def embed_generate(
    dataset_name: str,
    batch_size: int = 256,
    device: str = "cpu",
    output_dir: str | None = None,
    chunk_size: int | None = typer.Option(
        None,
        "--chunk-size",
        help="Process samples in chunks to reduce memory usage. Recommended: 1000-5000.",
    ),
) -> None:
    """Generate embeddings for a single dataset using BulkFormer."""
    config = get_config()
    ann_data_dir = DATA_ROOT / config["paths"]["anndata_cleaned_dir"]

    if output_dir is None:
        output_dir = DATA_ROOT / config["paths"]["embeddings_dir"]
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    adata_path = ann_data_dir / f"{dataset_name}.h5ad"
    if not adata_path.exists():
        logger.error(f"Dataset not found: {adata_path}")
        raise typer.Exit(1)

    import anndata as ad
    import numpy as np

    # Load AnnData in backed mode to reduce memory usage
    try:
        adata = ad.read_h5ad(adata_path, backed="r")
        logger.info(f"Loaded {dataset_name} in backed mode: {adata.shape}")
    except Exception:
        adata = ad.read_h5ad(adata_path)
        logger.info(f"Loaded {dataset_name}: {adata.shape}")

    output_path = output_dir / f"{dataset_name}_embeddings.npy"

    logger.info(f"Generating embeddings for {dataset_name}...")
    # Use incremental saving to avoid OOM
    extract_embeddings_from_adata(
        adata=adata,
        device=device,
        batch_size=batch_size,
        output_path=output_path,
        chunk_size=chunk_size,
    )

    # Verify output
    if output_path.exists():
        embeddings_mmap = np.load(output_path, mmap_mode="r")
        logger.success(f"Saved embeddings to {output_path} (shape: {embeddings_mmap.shape})")
        del embeddings_mmap
    else:
        logger.error(f"Failed to save embeddings to {output_path}")
        raise typer.Exit(1)


@app.command("all-configs")
def embed_all_configs(
    output_dir: str | None = None,
    device: str = "cpu",
    batch_size: int = 256,
    use_wandb: bool = typer.Option(False, "--use-wandb", help="Enable wandb logging"),
    chunk_size: int | None = typer.Option(
        None,
        "--chunk-size",
        help="Process samples in chunks to reduce memory usage. Recommended: 1000-5000 for large datasets.",
    ),
) -> None:
    """Generate embeddings for all configurations (human-only, mouse-only, human-ortholog-filtered)."""
    config = get_config()

    if output_dir is None:
        output_dir = DATA_ROOT / config["paths"]["embeddings_dir"]
    else:
        output_dir = Path(output_dir)

    generate_all_embeddings(
        output_base_dir=output_dir,
        device=device,
        batch_size=batch_size,
        use_wandb=use_wandb,
        chunk_size=chunk_size,
    )
