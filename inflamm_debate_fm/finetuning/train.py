"""Training functions for LoRA fine-tuning."""

from pathlib import Path

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
from inflamm_debate_fm.config import MODELS_ROOT, get_config
from inflamm_debate_fm.finetuning.data import prepare_finetuning_data, save_finetuning_metadata
from inflamm_debate_fm.finetuning.lora import apply_lora_to_bulkformer, save_lora_checkpoint
from inflamm_debate_fm.utils.wandb_utils import init_wandb


def train_lora_model(
    species: str,
    n_inflammation: int = 32,
    n_control: int = 32,
    n_epochs: int = 10,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    device: str = "cuda",
    output_dir: Path | None = None,
    random_seed: int = 42,
    use_wandb: bool = False,
) -> Path:
    """Train LoRA fine-tuned model for inflammation classification.

    Args:
        species: Species to train on ('human', 'mouse', or 'combined').
        n_inflammation: Number of inflammation samples.
        n_control: Number of control samples.
        n_epochs: Number of training epochs.
        batch_size: Batch size for training.
        learning_rate: Learning rate.
        weight_decay: Weight decay for optimizer.
        device: Device to train on ('cuda' or 'cpu').
        output_dir: Directory to save checkpoints and metadata.
        random_seed: Random seed for reproducibility.
        use_wandb: Whether to log to Weights & Biases.

    Returns:
        Path to saved checkpoint directory.
    """
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    device_obj = torch.device(device)
    config = get_config()

    # Set up output directory
    if output_dir is None:
        output_dir = MODELS_ROOT / "finetuned_lora" / species
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training LoRA model for {species} on {device}")
    logger.info(f"Output directory: {output_dir}")

    # Prepare data
    adata, sample_metadata = prepare_finetuning_data(
        species=species,
        n_inflammation=n_inflammation,
        n_control=n_control,
        random_seed=random_seed,
        output_dir=output_dir,
    )

    # Save metadata
    percentage_used = (
        len(sample_metadata) / (n_inflammation + n_control) * 100
        if len(sample_metadata) > 0
        else 0
    )
    save_finetuning_metadata(
        sample_metadata=sample_metadata,
        species=species,
        n_inflammation=n_inflammation,
        n_control=n_control,
        percentage_used=percentage_used,
        output_dir=output_dir,
    )

    # Prepare expression data
    gene_data = load_bulkformer_gene_data()
    expr_df = adata_to_dataframe(adata)
    aligned_df, var = align_genes_to_bulkformer(expr_df, gene_data["bulkformer_gene_list"])

    # Extract labels
    labels = adata.obs["group"].map({"inflammation": 1, "control": 0}).values.astype(int)

    # Convert to tensors
    X_tensor = torch.tensor(aligned_df.values, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)

    # Create dataset and dataloader
    dataset = TensorDataset(X_tensor, y_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Load base model and apply LoRA
    logger.info("Loading base BulkFormer model...")
    # Clear any existing CUDA cache before loading
    if device_obj.type == "cuda":
        torch.cuda.empty_cache()
        logger.info(
            f"GPU memory before loading: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    # Model is already loaded on device, so we don't need to move it again
    base_model_loaded = load_bulkformer_model(device=device)
    model = apply_lora_to_bulkformer(model=base_model_loaded, device=device)

    # Check memory after loading
    if device_obj.type == "cuda":
        logger.info(
            f"GPU memory after loading model: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB / {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
        )

    # Add classification head: aggregate gene-level embeddings to sample-level
    # Get embedding dimension and gb_repeat from model config (do this once, not in loop)
    # Access base_model if wrapped by PEFT
    unwrapped_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    # Unwrap BulkFormerPEFTWrapper if present
    if hasattr(unwrapped_model, "base_model"):
        actual_base = unwrapped_model.base_model
    else:
        actual_base = unwrapped_model
    embedding_dim = (
        actual_base.dim if hasattr(actual_base, "dim") else 640
    )  # BulkFormer embedding dimension (640)
    gb_repeat = actual_base.gb_repeat if hasattr(actual_base, "gb_repeat") else 3
    classification_head = nn.Sequential(
        nn.Linear(embedding_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 2),  # Binary classification: control vs inflammation
    ).to(device_obj)

    # Ensure model is on correct device (should already be there, but verify)
    # Only move if not already on device to avoid creating copies
    if next(model.parameters()).device != device_obj:
        model = model.to(device_obj)
    model.train()
    classification_head.train()

    # Setup optimizer and loss (optimize both model and classification head)
    all_params = list(model.parameters()) + list(classification_head.parameters())
    optimizer = torch.optim.AdamW(all_params, lr=learning_rate, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    # Setup wandb if requested
    wandb_run = None
    if use_wandb:
        try:
            wandb_run = init_wandb(
                project=config.get("wandb", {}).get("project", "inflamm-debate-fm"),
                tags=config.get("wandb", {}).get("tags", []) + ["finetuning", "lora", species],
                config={
                    "species": species,
                    "n_inflammation": n_inflammation,
                    "n_control": n_control,
                    "n_epochs": n_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "weight_decay": weight_decay,
                },
            )
            # Set run name
            if wandb_run:
                wandb_run.name = f"lora_finetune_{species}"
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            use_wandb = False

    # Training loop
    logger.info(f"Starting training for {n_epochs} epochs...")
    best_loss = float("inf")

    for epoch in range(n_epochs):
        epoch_losses = []
        model.train()

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}")
        for batch_idx, (X_batch, y_batch) in enumerate(progress_bar):
            X_batch = X_batch.to(device_obj)
            y_batch = y_batch.to(device_obj)

            # Forward pass
            optimizer.zero_grad()

            # Get gene-level embeddings from BulkFormer
            # Use repr_layers to get intermediate representations before the final head
            # Get embeddings from the last GBFormer layer (before layernorm and head)
            # Call through PEFT wrapper using inputs_embeds (PEFT-compatible parameter)
            # Only request the layer we need to minimize memory usage
            output, hidden = model(inputs_embeds=X_batch, repr_layers=[gb_repeat - 1])
            # hidden[gb_repeat - 1] is [batch_size, n_genes, embedding_dim]
            gene_embeddings = hidden[gb_repeat - 1]
            # Clear output and hidden dict to free memory immediately
            del output, hidden

            # Aggregate gene embeddings to sample-level (mean pooling)
            sample_embeddings = gene_embeddings.mean(dim=1)  # [batch_size, embedding_dim]

            # Pass through classification head
            logits = classification_head(sample_embeddings)  # [batch_size, 2]

            # Calculate loss
            loss = criterion(logits, y_batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            # Store loss value before deleting tensor
            loss_value = loss.item()
            epoch_losses.append(loss_value)
            progress_bar.set_postfix({"loss": loss_value})

            # Clear intermediate tensors to free memory
            del gene_embeddings, sample_embeddings, logits, loss
            # Clear cache more aggressively for memory-constrained scenarios
            if device_obj.type == "cuda":
                torch.cuda.empty_cache()

            # Log to wandb
            if use_wandb and wandb_run:
                wandb_run.log({"batch_loss": loss_value, "epoch": epoch})

        avg_loss = np.mean(epoch_losses)
        logger.info(f"Epoch {epoch + 1}/{n_epochs} - Average Loss: {avg_loss:.4f}")

        # Save checkpoint if best
        if avg_loss < best_loss:
            best_loss = avg_loss
            checkpoint_dir = output_dir / "checkpoint_best"
            save_lora_checkpoint(
                model=model,
                output_path=checkpoint_dir,
                metadata={
                    "epoch": epoch + 1,
                    "loss": avg_loss,
                    "species": species,
                    "n_inflammation": n_inflammation,
                    "n_control": n_control,
                },
            )
            # Also save classification head
            torch.save(
                classification_head.state_dict(),
                checkpoint_dir / "classification_head.pt",
            )

        # Log epoch metrics to wandb
        if use_wandb and wandb_run:
            wandb_run.log({"epoch_loss": avg_loss, "epoch": epoch})

    # Save final checkpoint
    final_checkpoint_dir = output_dir / "checkpoint_final"
    save_lora_checkpoint(
        model=model,
        output_path=final_checkpoint_dir,
        metadata={
            "epoch": n_epochs,
            "final_loss": avg_loss,
            "best_loss": best_loss,
            "species": species,
            "n_inflammation": n_inflammation,
            "n_control": n_control,
        },
    )
    # Also save classification head
    torch.save(
        classification_head.state_dict(),
        final_checkpoint_dir / "classification_head.pt",
    )

    logger.success(f"Training complete! Checkpoints saved to {output_dir}")

    if use_wandb and wandb_run:
        wandb_run.finish()

    return output_dir
