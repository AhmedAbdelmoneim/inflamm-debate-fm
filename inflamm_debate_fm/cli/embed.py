"""Embedding generation commands."""

from pathlib import Path

from loguru import logger
import typer

from inflamm_debate_fm.bulkformer.embed import extract_embeddings_from_adata
from inflamm_debate_fm.bulkformer.pipeline import generate_all_embeddings
from inflamm_debate_fm.config import DATA_ROOT, get_config
from inflamm_debate_fm.embeddings.multi_model import (
    add_multi_model_embeddings_to_adata,
    detect_available_models,
)

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


@app.command("multi-model")
def embed_multi_model(
    dataset_name: str = typer.Argument(
        "all", help="Dataset name (without .h5ad) or 'all' to process all datasets"
    ),
    batch_size: int = 4,
    device: str = "cpu",
    output_dir: str | None = None,
) -> None:
    """Generate mean-pooled sample-level embeddings from all available models.

    Automatically detects available models (zero-shot + fine-tuned) and generates
    mean-pooled sample-level embeddings for each. This is designed for probing analysis.

    The embeddings are saved to the AnnData file in obsm with keys:
    - 'X_zero_shot': Zero-shot model embeddings
    - 'X_human': Human fine-tuned model embeddings (if available)
    - 'X_mouse': Mouse fine-tuned model embeddings (if available)
    - 'X_combined': Combined fine-tuned model embeddings (if available)

    Args:
        dataset_name: Name of the dataset (without .h5ad extension) or 'all' to process all datasets.
        batch_size: Batch size for inference (default: 4 for CUDA memory optimization).
        device: Device to run inference on ('cpu' or 'cuda').
        output_dir: Optional directory to save updated AnnData files.
                    If None, overwrites the original files.
    """
    import anndata as ad

    config = get_config()
    ann_data_dir = DATA_ROOT / config["paths"]["anndata_cleaned_dir"]

    # Detect available models once (shared across all datasets)
    available_models = detect_available_models()
    if len(available_models) == 0:
        logger.error("No models available for embedding extraction")
        raise typer.Exit(1)

    logger.info(f"Found {len(available_models)} models: {list(available_models.keys())}")

    # Determine which datasets to process
    if dataset_name.lower() == "all":
        # Process all .h5ad files in the directory
        dataset_paths = sorted(ann_data_dir.glob("*.h5ad"))
        if len(dataset_paths) == 0:
            logger.error(f"No datasets found in {ann_data_dir}")
            raise typer.Exit(1)
        logger.info(f"Processing {len(dataset_paths)} datasets...")
    else:
        # Process single dataset
        dataset_paths = [ann_data_dir / f"{dataset_name}.h5ad"]
        if not dataset_paths[0].exists():
            logger.error(f"Dataset not found: {dataset_paths[0]}")
            raise typer.Exit(1)

    # Process each dataset
    for adata_path in dataset_paths:
        dataset_name_current = adata_path.stem
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Processing {dataset_name_current}...")
        logger.info(f"{'=' * 60}")

        try:
            # Load AnnData
            logger.info(f"Loading {dataset_name_current}...")
            adata = ad.read_h5ad(adata_path)
            logger.info(f"Loaded {dataset_name_current}: {adata.shape}")

            # Extract embeddings from all models
            adata = add_multi_model_embeddings_to_adata(
                adata=adata,
                device=device,
                batch_size=batch_size,
                models=available_models,
            )

            # Determine output path
            if output_dir is None:
                output_path = adata_path
            else:
                output_dir_path = Path(output_dir)
                output_dir_path.mkdir(parents=True, exist_ok=True)
                output_path = output_dir_path / adata_path.name

            logger.info(f"Saving updated AnnData to {output_path}...")
            adata.write_h5ad(output_path)
            logger.success(f"Saved embeddings to {output_path}")

            # Print summary
            logger.info("Embedding summary:")
            for key in sorted(adata.obsm.keys()):
                if key.startswith("X_"):
                    logger.info(f"  {key}: {adata.obsm[key].shape}")

        except Exception as e:
            logger.error(f"Failed to process {dataset_name_current}: {e}")
            logger.exception(e)
            # Continue with next dataset instead of failing completely
            continue

    logger.success(f"\nCompleted processing {len(dataset_paths)} dataset(s)")
