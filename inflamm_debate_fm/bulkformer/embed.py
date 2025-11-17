"""BulkFormer embedding extraction functions."""

from collections import OrderedDict
from pathlib import Path
import sys

import anndata as ad
from loguru import logger
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.typing import SparseTensor
from tqdm import tqdm

from inflamm_debate_fm.config import BULKFORMER_DATA_DIR, BULKFORMER_MODEL_DIR, MODELS_ROOT

# Add BulkFormer to path for imports
# Use MODELS_ROOT from config to handle environment variable overrides
BULKFORMER_BASE = MODELS_ROOT / "BulkFormer"
if BULKFORMER_BASE.exists():
    sys.path.insert(0, str(BULKFORMER_BASE))
    logger.debug(f"Added BulkFormer to Python path: {BULKFORMER_BASE}")
else:
    logger.warning(
        f"BulkFormer directory not found at {BULKFORMER_BASE}. "
        "Run 'make bulkformer-setup' to clone the repository."
    )

try:
    from model.config import model_params
    from utils.BulkFormer import BulkFormer
except ImportError as e:
    logger.error(
        f"Failed to import BulkFormer: {e}\n"
        f"Make sure BulkFormer repository is cloned at {BULKFORMER_BASE}\n"
        "Run 'make bulkformer-setup' to set up BulkFormer."
    )
    raise


def estimate_memory_usage(
    n_samples: int, n_genes: int, batch_size: int, embedding_dim: int = 512
) -> dict[str, float]:
    """Estimate memory usage for embedding generation.

    Args:
        n_samples: Number of samples to process.
        n_genes: Number of genes (after alignment to BulkFormer).
        batch_size: Batch size for inference.
        embedding_dim: Embedding dimension (default 512 for BulkFormer).

    Returns:
        Dictionary with memory estimates in GB for different components.
    """
    # Expression array memory (float32)
    expr_memory_gb = (n_samples * n_genes * 4) / (1024**3)

    # Embedding output memory (float32)
    embedding_memory_gb = (n_samples * n_genes * embedding_dim * 4) / (1024**3)

    # Batch processing memory (one batch at a time)
    batch_expr_memory_gb = (batch_size * n_genes * 4) / (1024**3)
    batch_embedding_memory_gb = (batch_size * n_genes * embedding_dim * 4) / (1024**3)

    # Model memory (rough estimate, actual depends on model architecture)
    model_memory_gb = 2.0  # Approximate for BulkFormer

    return {
        "expression_array": expr_memory_gb,
        "embedding_output": embedding_memory_gb,
        "batch_expression": batch_expr_memory_gb,
        "batch_embedding": batch_embedding_memory_gb,
        "model": model_memory_gb,
        "total_peak": batch_expr_memory_gb + batch_embedding_memory_gb + model_memory_gb,
        "total_if_accumulated": expr_memory_gb + embedding_memory_gb + model_memory_gb,
    }


def load_bulkformer_model(device: str = "cpu") -> torch.nn.Module:
    """Load BulkFormer model.

    Args:
        device: Device to load model on ('cpu' or 'cuda').

    Returns:
        Loaded BulkFormer model in eval mode.
    """
    device = torch.device(device)
    logger.info(f"Loading BulkFormer model from {BULKFORMER_MODEL_DIR} on {device}")

    model_graph_path = BULKFORMER_MODEL_DIR / "G_gtex.pt"
    model_graph_weights_path = BULKFORMER_MODEL_DIR / "G_gtex_weight.pt"
    model_gene_embedding_path = BULKFORMER_MODEL_DIR / "esm2_feature_concat.pt"
    model_checkpoint_path = BULKFORMER_MODEL_DIR / "checkpoint.pt"

    # Try alternative checkpoint name
    if not model_checkpoint_path.exists():
        model_checkpoint_path = BULKFORMER_MODEL_DIR / "Bulkformer_ckpt_epoch_29.pt"
        if not model_checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found in {BULKFORMER_MODEL_DIR}")

    graph = torch.load(model_graph_path, map_location="cpu", weights_only=False)
    weights = torch.load(model_graph_weights_path, map_location="cpu", weights_only=False)
    graph = SparseTensor(row=graph[1], col=graph[0], value=weights).t().to(device)

    gene_emb = torch.load(model_gene_embedding_path, map_location="cpu", weights_only=False)

    model_config = model_params.copy()
    model_config["graph"] = graph
    model_config["gene_emb"] = gene_emb

    model = BulkFormer(**model_config).to(device)

    ckpt_model = torch.load(model_checkpoint_path, map_location=device, weights_only=False)
    new_state_dict = OrderedDict()
    for key, value in ckpt_model.items():
        new_key = key[7:] if key.startswith("module.") else key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def load_bulkformer_gene_data() -> dict:
    """Load BulkFormer gene data files.

    Returns:
        Dictionary with 'bulkformer_gene_list' and 'gene_length_dict'.
    """
    gene_length_file = BULKFORMER_DATA_DIR / "gene_length_df.csv"
    gene_info_file = BULKFORMER_DATA_DIR / "bulkformer_gene_info.csv"

    gene_length_df = pd.read_csv(gene_length_file)
    gene_length_dict = gene_length_df.set_index("ensg_id")["length"].to_dict()
    bulkformer_gene_info = pd.read_csv(gene_info_file)
    bulkformer_gene_list = bulkformer_gene_info["ensg_id"].to_list()

    return {
        "gene_length_dict": gene_length_dict,
        "bulkformer_gene_list": bulkformer_gene_list,
    }


def align_genes_to_bulkformer(
    expr_df: pd.DataFrame, bulkformer_gene_list: list
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align gene expression matrix to BulkFormer gene list.

    Args:
        expr_df: Expression DataFrame with genes as columns.
        bulkformer_gene_list: List of Ensembl gene IDs to align to.

    Returns:
        Tuple of (aligned DataFrame, var DataFrame with mask column).
    """
    to_fill_columns = list(set(bulkformer_gene_list) - set(expr_df.columns))

    padding_df = pd.DataFrame(
        np.full((expr_df.shape[0], len(to_fill_columns)), -10),
        columns=to_fill_columns,
        index=expr_df.index,
    )

    aligned_df = pd.concat([expr_df, padding_df], axis=1)[bulkformer_gene_list]

    var = pd.DataFrame(index=aligned_df.columns)
    var["mask"] = [1 if i in to_fill_columns else 0 for i in list(var.index)]

    return aligned_df, var


def adata_to_dataframe(adata: ad.AnnData) -> pd.DataFrame:
    """Convert AnnData to DataFrame with Ensembl IDs as columns.

    Args:
        adata: AnnData object with 'ensembl_id' in var.

    Returns:
        DataFrame with samples as rows and genes (ensembl_id) as columns.
    """
    if "ensembl_id" not in adata.var.columns:
        # Try to use 'ensembl' column if available (from cleaned files)
        if "ensembl" in adata.var.columns:
            adata.var["ensembl_id"] = adata.var["ensembl"]
        else:
            adata.var["ensembl_id"] = adata.var.index

    df = pd.DataFrame(
        adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
        index=adata.obs_names,
        columns=adata.var["ensembl_id"],
    )

    return df.loc[:, df.columns.notna()]


def extract_embeddings(
    model: torch.nn.Module,
    expr_array: np.ndarray,
    device: str = "cpu",
    batch_size: int = 256,
    output_path: Path | None = None,
) -> np.ndarray | None:
    """Extract gene-level embeddings from BulkFormer model.

    This function extracts embeddings without filtering or aggregation.
    Returns gene-level embeddings for all genes in the model.

    Args:
        model: Loaded BulkFormer model.
        expr_array: Expression array of shape [N_samples, N_genes].
        device: Device to run inference on.
        batch_size: Batch size for inference.
        output_path: Optional path to save embeddings incrementally. If provided,
                     embeddings are saved to disk and None is returned. Otherwise,
                     embeddings are returned as numpy array.

    Returns:
        Embeddings array of shape [N_samples, N_genes, embedding_dim] if output_path is None,
        otherwise None (embeddings saved to disk).
    """
    device_obj = torch.device(device)
    model.eval()

    n_samples = expr_array.shape[0]

    # For GPU, check available memory and warn if low
    if device_obj.type == "cuda":
        if torch.cuda.is_available():
            free_mem = torch.cuda.get_device_properties(
                0
            ).total_memory - torch.cuda.memory_allocated(0)
            free_gb = free_mem / (1024**3)
            logger.info(f"GPU memory: {free_gb:.2f} GB free")
            if free_gb < 2.0:
                logger.warning(
                    f"Low GPU memory ({free_gb:.2f} GB free). "
                    f"Consider using a smaller batch_size (current: {batch_size})"
                )
        # Keep data on CPU and move batches to GPU to save memory
        dataset = TensorDataset(torch.tensor(expr_array, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    else:
        expr_tensor = torch.tensor(expr_array, dtype=torch.float32, device=device_obj)
        dataset = TensorDataset(expr_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # If saving to disk, use memory-mapped array for incremental writes
    if output_path is not None:
        # Process first batch to determine embedding dimensions
        first_batch = next(iter(dataloader))[0]
        with torch.no_grad():
            first_batch = first_batch.to(device_obj)
            _, emb = model(first_batch, [2])
            emb_shape = emb[2].shape  # [batch_size, n_genes, embedding_dim]
            embedding_dim = emb_shape[-1]
            n_genes = emb_shape[-2]
            # Clean up first batch
            del first_batch, emb
            if device_obj.type == "cuda":
                torch.cuda.empty_cache()

        # Create memory-mapped array for incremental writes
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Creating memory-mapped array at {output_path} for shape ({n_samples}, {n_genes}, {embedding_dim})"
        )

        # Use temporary file first, then move to final location
        temp_path = output_path.with_suffix(".tmp.npy")
        mmap_array = np.lib.format.open_memmap(
            temp_path, mode="w+", dtype=np.float32, shape=(n_samples, n_genes, embedding_dim)
        )

        # Reset dataloader to process from beginning
        dataloader = DataLoader(
            dataset, batch_size=batch_size, shuffle=False, pin_memory=(device_obj.type == "cuda")
        )

        sample_idx = 0
        with torch.no_grad():
            for batch_idx, (X,) in enumerate(
                tqdm(dataloader, total=len(dataloader), desc="Extracting embeddings")
            ):
                X = X.to(device_obj)
                output, emb = model(X, [2])
                # Extract embeddings from layer 2 (gene-level)
                emb = emb[2].detach().cpu().numpy()

                # Write batch to memory-mapped array
                batch_size_actual = emb.shape[0]
                mmap_array[sample_idx : sample_idx + batch_size_actual] = emb
                sample_idx += batch_size_actual

                # Clear GPU cache after each batch to avoid fragmentation
                if device_obj.type == "cuda":
                    del X, output, emb
                    torch.cuda.empty_cache()

                    # Log memory usage every 10 batches
                    if (batch_idx + 1) % 10 == 0:
                        allocated = torch.cuda.memory_allocated(0) / (1024**3)
                        reserved = torch.cuda.memory_reserved(0) / (1024**3)
                        logger.debug(
                            f"Batch {batch_idx + 1}: GPU memory allocated={allocated:.2f} GB, reserved={reserved:.2f} GB"
                        )
                else:
                    del emb

        # Flush and close memory-mapped array
        del mmap_array
        # Move temp file to final location
        temp_path.rename(output_path)
        logger.success(f"Saved embeddings to {output_path}")
        return None
    else:
        # Original behavior: accumulate in memory
        all_emb_list = []
        with torch.no_grad():
            for batch_idx, (X,) in enumerate(
                tqdm(dataloader, total=len(dataloader), desc="Extracting embeddings")
            ):
                X = X.to(device_obj)
                output, emb = model(X, [2])
                # Extract embeddings from layer 2 (gene-level)
                emb = emb[2].detach().cpu().numpy()
                all_emb_list.append(emb)

                # Clear GPU cache after each batch to avoid fragmentation
                if device_obj.type == "cuda":
                    del X, output
                    torch.cuda.empty_cache()

                    # Log memory usage every 10 batches
                    if (batch_idx + 1) % 10 == 0:
                        allocated = torch.cuda.memory_allocated(0) / (1024**3)
                        reserved = torch.cuda.memory_reserved(0) / (1024**3)
                        logger.debug(
                            f"Batch {batch_idx + 1}: GPU memory allocated={allocated:.2f} GB, reserved={reserved:.2f} GB"
                        )

        return np.vstack(all_emb_list)


def extract_embeddings_from_adata(
    adata: ad.AnnData,
    model: torch.nn.Module | None = None,
    device: str = "cpu",
    batch_size: int = 256,
    output_path: Path | None = None,
    chunk_size: int | None = None,
) -> np.ndarray | None:
    """Extract BulkFormer embeddings from an AnnData object.

    Args:
        adata: AnnData object with 'ensembl_id' in var.
        model: Pre-loaded BulkFormer model. If None, will load it.
        device: Device to run inference on.
        batch_size: Batch size for inference.
        output_path: Optional path to save embeddings incrementally. If provided,
                     embeddings are saved to disk and None is returned. Otherwise,
                     embeddings are returned as numpy array.
        chunk_size: Optional chunk size for processing large datasets. If provided,
                    processes samples in chunks to reduce memory usage. Only used
                    when output_path is provided.

    Returns:
        Embeddings array of shape [N_samples, N_genes, embedding_dim] if output_path is None,
        otherwise None (embeddings saved to disk).
    """
    if model is None:
        model = load_bulkformer_model(device=device)

    gene_data = load_bulkformer_gene_data()

    if "ensembl_id" not in adata.var.columns:
        # Try to use 'ensembl' column if available (from cleaned files)
        if "ensembl" in adata.var.columns:
            adata.var["ensembl_id"] = adata.var["ensembl"]
        else:
            adata.var["ensembl_id"] = adata.var.index

    # Process in chunks if requested and output_path is provided
    if chunk_size is not None and output_path is not None and adata.shape[0] > chunk_size:
        logger.info(f"Processing {adata.shape[0]} samples in chunks of {chunk_size}")
        return _extract_embeddings_chunked(
            adata=adata,
            model=model,
            gene_data=gene_data,
            device=device,
            batch_size=batch_size,
            output_path=output_path,
            chunk_size=chunk_size,
        )

    # Process entire dataset
    expr_df = adata_to_dataframe(adata)
    aligned_df, var = align_genes_to_bulkformer(expr_df, gene_data["bulkformer_gene_list"])

    logger.info(f"Aligned expression matrix: {expr_df.shape} -> {aligned_df.shape}")

    # Estimate memory usage and warn if high
    if output_path is None:
        # Only estimate if we're accumulating in memory
        mem_estimate = estimate_memory_usage(
            n_samples=aligned_df.shape[0],
            n_genes=aligned_df.shape[1],
            batch_size=batch_size,
        )
        if mem_estimate["total_if_accumulated"] > 10.0:
            logger.warning(
                f"Estimated memory usage: {mem_estimate['total_if_accumulated']:.2f} GB. "
                f"Consider using --chunk-size or output_path to avoid OOM."
            )

    # Clear intermediate dataframes to free memory
    del expr_df
    import gc

    gc.collect()

    embeddings = extract_embeddings(
        model=model,
        expr_array=aligned_df.values,
        device=device,
        batch_size=batch_size,
        output_path=output_path,
    )

    # Clear aligned_df after use
    del aligned_df
    gc.collect()

    if embeddings is not None:
        logger.success(f"Generated BulkFormer embeddings: {embeddings.shape}")
    return embeddings


def _extract_embeddings_chunked(
    adata: ad.AnnData,
    model: torch.nn.Module,
    gene_data: dict,
    device: str,
    batch_size: int,
    output_path: Path,
    chunk_size: int,
) -> None:
    """Extract embeddings in chunks for memory efficiency.

    Args:
        adata: AnnData object with 'ensembl_id' in var.
        model: Pre-loaded BulkFormer model.
        gene_data: Dictionary with 'bulkformer_gene_list' and 'gene_length_dict'.
        device: Device to run inference on.
        batch_size: Batch size for inference.
        output_path: Path to save embeddings.
        chunk_size: Number of samples to process per chunk.
    """
    n_samples = adata.shape[0]
    output_path = Path(output_path)

    # Process first chunk to determine dimensions
    first_chunk = adata[: min(chunk_size, n_samples)]
    expr_df = adata_to_dataframe(first_chunk)
    aligned_df, _ = align_genes_to_bulkformer(expr_df, gene_data["bulkformer_gene_list"])

    # Get embedding dimensions from first batch
    device_obj = torch.device(device)
    test_tensor = torch.tensor(aligned_df.values[:batch_size], dtype=torch.float32)
    with torch.no_grad():
        test_tensor = test_tensor.to(device_obj)
        _, emb = model(test_tensor, [2])
        emb_shape = emb[2].shape
        embedding_dim = emb_shape[-1]
        n_genes = emb_shape[-2]

    del test_tensor, emb, aligned_df, expr_df, first_chunk
    if device_obj.type == "cuda":
        torch.cuda.empty_cache()
    import gc

    gc.collect()

    # Create memory-mapped array for all samples
    temp_path = output_path.with_suffix(".tmp.npy")
    logger.info(
        f"Creating memory-mapped array at {output_path} for shape ({n_samples}, {n_genes}, {embedding_dim})"
    )
    mmap_array = np.lib.format.open_memmap(
        temp_path, mode="w+", dtype=np.float32, shape=(n_samples, n_genes, embedding_dim)
    )

    # Process in chunks
    sample_idx = 0
    for chunk_start in range(0, n_samples, chunk_size):
        chunk_end = min(chunk_start + chunk_size, n_samples)
        logger.info(
            f"Processing chunk {chunk_start}:{chunk_end} ({chunk_end - chunk_start} samples)"
        )

        # Extract chunk
        chunk_adata = adata[chunk_start:chunk_end]
        expr_df = adata_to_dataframe(chunk_adata)
        aligned_df, _ = align_genes_to_bulkformer(expr_df, gene_data["bulkformer_gene_list"])

        # Generate embeddings for chunk
        chunk_embeddings = extract_embeddings(
            model=model,
            expr_array=aligned_df.values,
            device=device,
            batch_size=batch_size,
            output_path=None,  # Return array, don't save
        )

        # Write to memory-mapped array
        chunk_size_actual = chunk_embeddings.shape[0]
        mmap_array[sample_idx : sample_idx + chunk_size_actual] = chunk_embeddings
        sample_idx += chunk_size_actual

        # Clear chunk data
        del chunk_adata, expr_df, aligned_df, chunk_embeddings
        gc.collect()
        if device_obj.type == "cuda":
            torch.cuda.empty_cache()

    # Flush and close
    del mmap_array
    temp_path.rename(output_path)
    logger.success(f"Saved chunked embeddings to {output_path}")
