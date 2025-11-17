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

    adata = ad.read_h5ad(adata_path)
    logger.info(f"Loaded {dataset_name}: {adata.shape}")

    logger.info(f"Generating embeddings for {dataset_name}...")
    embeddings = extract_embeddings_from_adata(adata=adata, device=device, batch_size=batch_size)

    output_path = output_dir / f"{dataset_name}_embeddings.npy"
    np.save(output_path, embeddings)
    logger.success(f"Saved embeddings to {output_path}")


@app.command("all-configs")
def embed_all_configs(
    output_dir: str | None = None,
    device: str = "cpu",
    batch_size: int = 256,
    use_wandb: bool = typer.Option(False, "--use-wandb", help="Enable wandb logging"),
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
    )
