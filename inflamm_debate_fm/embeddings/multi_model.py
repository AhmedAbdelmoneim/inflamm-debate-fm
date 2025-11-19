"""Multi-model embedding extraction with mean-pooling."""

from pathlib import Path

import anndata as ad
from loguru import logger
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from inflamm_debate_fm.bulkformer.embed import (
    adata_to_dataframe,
    align_genes_to_bulkformer,
    load_bulkformer_gene_data,
    load_bulkformer_model,
)
from inflamm_debate_fm.config import MODELS_ROOT
from inflamm_debate_fm.finetuning.lora import load_lora_checkpoint


def detect_available_models(models_root: Path | None = None) -> dict[str, Path]:
    """Detect available models (zero-shot + fine-tuned).

    Args:
        models_root: Root directory for models. If None, uses MODELS_ROOT.

    Returns:
        Dictionary mapping model names to checkpoint paths.
        Keys: 'zero_shot', 'human', 'mouse', 'combined', 'universal' (where available)
    """
    if models_root is None:
        models_root = MODELS_ROOT

    available_models = {}

    # Zero-shot model is always available (BulkFormer base model)
    bulkformer_model_dir = models_root / "BulkFormer" / "model"
    if bulkformer_model_dir.exists():
        available_models["zero_shot"] = bulkformer_model_dir

    # Check for fine-tuned models
    finetuned_dir = models_root / "finetuned_lora"
    if finetuned_dir.exists():
        for species in ["human", "mouse", "combined", "universal"]:
            species_dir = finetuned_dir / species
            if species_dir.exists():
                # Check for checkpoint_best or checkpoint_final
                checkpoint_best = species_dir / "checkpoint_best"
                checkpoint_final = species_dir / "checkpoint_final"
                if checkpoint_best.exists():
                    available_models[species] = checkpoint_best
                elif checkpoint_final.exists():
                    available_models[species] = checkpoint_final

    return available_models


def extract_embeddings_with_mean_pooling(
    model: nn.Module,
    expr_array: np.ndarray,
    device: str,
    batch_size: int,
    gb_repeat: int = 3,
) -> np.ndarray:
    """Extract gene-level embeddings and mean-pool to sample-level.

    Args:
        model: Loaded BulkFormer model (zero-shot or fine-tuned).
        expr_array: Expression array of shape [N_samples, N_genes].
        device: Device to run inference on.
        batch_size: Batch size for inference.
        gb_repeat: Number of graph blocks (default 3 for BulkFormer).

    Returns:
        Sample-level embeddings of shape [N_samples, embedding_dim] after mean-pooling.
    """
    device_obj = torch.device(device)
    model.eval()

    # Keep data on CPU and move batches to GPU to save memory
    if device_obj.type == "cuda":
        dataset = TensorDataset(torch.tensor(expr_array, dtype=torch.float32))
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, pin_memory=True)
    else:
        expr_tensor = torch.tensor(expr_array, dtype=torch.float32, device=device_obj)
        dataset = TensorDataset(expr_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    sample_embeddings_list = []

    with torch.no_grad():
        for (X_batch,) in tqdm(dataloader, desc="Extracting embeddings", total=len(dataloader)):
            X_batch = X_batch.to(device_obj)

            # Get gene-level embeddings from the last graph block
            # Use inputs_embeds for PEFT compatibility
            if hasattr(model, "base_model"):
                # PEFT-wrapped model
                output, hidden = model(inputs_embeds=X_batch, repr_layers=[gb_repeat - 1])
            else:
                # Standard BulkFormer model
                output, hidden = model(X_batch, repr_layers=[gb_repeat - 1])

            gene_embeddings = hidden[gb_repeat - 1]  # [batch_size, n_genes, embedding_dim]

            # Mean-pool across genes to get sample-level embeddings
            sample_embeddings = gene_embeddings.mean(dim=1)  # [batch_size, embedding_dim]

            # Move to CPU and store
            sample_embeddings_list.append(sample_embeddings.cpu().numpy())

            # Aggressive memory cleanup for CUDA
            if device_obj.type == "cuda":
                del X_batch, output, hidden, gene_embeddings, sample_embeddings
                torch.cuda.empty_cache()

    # Concatenate all batches
    sample_embeddings = np.concatenate(sample_embeddings_list, axis=0)
    return sample_embeddings


def extract_multi_model_embeddings(
    adata: ad.AnnData,
    models: dict[str, Path],
    device: str = "cpu",
    batch_size: int = 4,
) -> dict[str, np.ndarray]:
    """Extract embeddings from multiple models and mean-pool to sample-level.

    Args:
        adata: AnnData object with expression data.
        models: Dictionary mapping model names to checkpoint paths.
        device: Device to run inference on.
        batch_size: Batch size for inference.

    Returns:
        Dictionary mapping model names to sample-level embeddings.
    """
    # Prepare expression data
    gene_data = load_bulkformer_gene_data()
    expr_df = adata_to_dataframe(adata)
    aligned_df, var = align_genes_to_bulkformer(expr_df, gene_data["bulkformer_gene_list"])

    # Check for NaN/Inf values and fill them
    nan_count = np.isnan(aligned_df.values).sum()
    inf_count = np.isinf(aligned_df.values).sum()
    if nan_count > 0 or inf_count > 0:
        logger.warning(f"Found {nan_count} NaN and {inf_count} Inf values. Filling with -10.")
        aligned_df = aligned_df.fillna(-10.0)
        aligned_df = aligned_df.replace([np.inf, -np.inf], -10.0)

    expr_array = aligned_df.values.astype(np.float32)

    # Get gb_repeat from model config (default 3 for BulkFormer)
    gb_repeat = 3

    all_embeddings = {}

    # Process zero-shot model first
    if "zero_shot" in models:
        logger.info("Extracting embeddings from zero-shot model...")
        model = load_bulkformer_model(device=device)
        embeddings = extract_embeddings_with_mean_pooling(
            model=model,
            expr_array=expr_array,
            device=device,
            batch_size=batch_size,
            gb_repeat=gb_repeat,
        )
        all_embeddings["zero_shot"] = embeddings
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    # Process fine-tuned models
    for model_name, checkpoint_path in models.items():
        if model_name == "zero_shot":
            continue

        logger.info(f"Extracting embeddings from {model_name} fine-tuned model...")
        base_model = load_bulkformer_model(device=device)
        model = load_lora_checkpoint(
            checkpoint_path=checkpoint_path,
            base_model=base_model,
            device=device,
        )
        embeddings = extract_embeddings_with_mean_pooling(
            model=model,
            expr_array=expr_array,
            device=device,
            batch_size=batch_size,
            gb_repeat=gb_repeat,
        )
        all_embeddings[model_name] = embeddings
        del model, base_model
        if device == "cuda":
            torch.cuda.empty_cache()

    return all_embeddings


def add_multi_model_embeddings_to_adata(
    adata: ad.AnnData,
    device: str = "cpu",
    batch_size: int = 4,
    models: dict[str, Path] | None = None,
) -> ad.AnnData:
    """Add multi-model embeddings to AnnData object.

    Args:
        adata: AnnData object to add embeddings to.
        device: Device to run inference on.
        batch_size: Batch size for inference.
        models: Optional dictionary of models. If None, will detect available models.

    Returns:
        AnnData object with embeddings added to obsm.
    """
    if models is None:
        models = detect_available_models()

    if len(models) == 0:
        logger.warning("No models found for embedding extraction")
        return adata

    logger.info(f"Found {len(models)} models: {list(models.keys())}")

    # Extract embeddings from all models
    all_embeddings = extract_multi_model_embeddings(
        adata=adata,
        models=models,
        device=device,
        batch_size=batch_size,
    )

    # Add embeddings to obsm with standardized keys
    for model_name, embeddings in all_embeddings.items():
        obsm_key = f"X_{model_name}"
        adata.obsm[obsm_key] = embeddings
        logger.info(f"Added {obsm_key}: {embeddings.shape}")

    return adata
