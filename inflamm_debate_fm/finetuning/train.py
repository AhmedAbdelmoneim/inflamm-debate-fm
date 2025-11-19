"""Training functions for LoRA fine-tuning."""

from collections import deque
from pathlib import Path

from loguru import logger
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from inflamm_debate_fm.bulkformer.embed import (
    adata_to_dataframe,
    align_genes_to_bulkformer,
    load_bulkformer_gene_data,
    load_bulkformer_model,
)
from inflamm_debate_fm.config import MODELS_ROOT, get_config
from inflamm_debate_fm.finetuning.contrastive import (
    MEMORY_BANK_SIZE,
    SPECIES_TO_ID,
    get_species_tensor,
)
from inflamm_debate_fm.finetuning.data import prepare_finetuning_data, save_finetuning_metadata
from inflamm_debate_fm.finetuning.lora import apply_lora_to_bulkformer, save_lora_checkpoint
from inflamm_debate_fm.finetuning.model_utils import (
    build_classification_head,
    get_model_embedding_dim,
    get_model_gb_repeat,
)
from inflamm_debate_fm.finetuning.training_loops import (
    train_classification_epoch,
    train_contrastive_epoch,
)
from inflamm_debate_fm.utils.wandb_utils import init_wandb


def train_lora_model(
    species: str,
    n_inflammation: int = 32,
    n_control: int = 32,
    n_epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    weight_decay: float = 0.01,
    device: str = "cuda",
    output_dir: Path | None = None,
    random_seed: int = 42,
    use_wandb: bool = False,
    early_stopping_patience: int = 7,
    contrastive_weight: float = 1.0,
    contrastive_temperature: float = 0.07,
) -> Path:
    """Train LoRA fine-tuned model for inflammation classification.

    Args:
        species: Species to train on ('human', 'mouse', 'combined', or 'universal').
        n_inflammation: Number of inflammation samples.
        n_control: Number of control samples.
        n_epochs: Number of training epochs (default: 50).
        batch_size: Batch size for training.
        learning_rate: Learning rate.
        weight_decay: Weight decay for optimizer.
        device: Device to train on ('cuda' or 'cpu').
        output_dir: Directory to save checkpoints and metadata.
        random_seed: Random seed for reproducibility.
        use_wandb: Whether to log to Weights & Biases.
        early_stopping_patience: Number of epochs to wait without improvement before stopping (default: 7).
        contrastive_weight: Weight applied to cross-species InfoNCE loss (used for 'universal' mode).
        contrastive_temperature: Temperature used inside the InfoNCE loss.

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

    training_mode = "cross_contrastive" if species == "universal" else "classification"

    logger.info(f"Training LoRA model for {species} on {device}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(
        f"Training mode: {training_mode}"
        + (
            f" (contrastive_weight={contrastive_weight}, temperature={contrastive_temperature})"
            if training_mode == "cross_contrastive"
            else ""
        )
    )

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

    # Check for NaN/Inf values and fill them (critical for combined human+mouse data)
    nan_count = np.isnan(aligned_df.values).sum()
    inf_count = np.isinf(aligned_df.values).sum()
    if nan_count > 0 or inf_count > 0:
        logger.warning(
            f"Found {nan_count} NaN and {inf_count} Inf values in aligned data. "
            "Filling with -10 (BulkFormer's standard padding value). "
            "This is expected when combining human and mouse data."
        )
        # Use -10 to match BulkFormer's padding for missing genes
        aligned_df = aligned_df.fillna(-10.0)
        aligned_df = aligned_df.replace([np.inf, -np.inf], -10.0)

    # Extract labels
    labels = adata.obs["group"].map({"inflammation": 1, "control": 0}).values.astype(int)

    # Convert to tensors
    X_tensor = torch.tensor(aligned_df.values, dtype=torch.float32)
    y_tensor = torch.tensor(labels, dtype=torch.long)

    # Final validation: ensure no NaN/Inf in tensors
    if torch.isnan(X_tensor).any() or torch.isinf(X_tensor).any():
        logger.error("NaN/Inf values detected in input tensor after conversion!")
        raise ValueError("Input tensor contains NaN or Inf values")

    # Create dataset and dataloader
    fallback_species = (
        "human" if species == "human" else "mouse" if species == "mouse" else "unknown"
    )
    species_tensor = get_species_tensor(adata, fallback_species)

    if training_mode == "cross_contrastive":
        human_count = int((species_tensor == SPECIES_TO_ID["human"]).sum().item())
        mouse_count = int((species_tensor == SPECIES_TO_ID["mouse"]).sum().item())
        if human_count == 0 or mouse_count == 0:
            raise ValueError(
                "Universal contrastive mode requires both human and mouse samples. "
                "Please ensure the combined dataset includes both species."
            )
        logger.info(
            f"Batchable samples - Human: {human_count}, Mouse: {mouse_count}. "
            "Only cross-species inflammation pairs will be treated as positives."
        )

    dataset = TensorDataset(X_tensor, y_tensor, species_tensor)
    if training_mode == "cross_contrastive":
        dataloader = None
    else:
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

    # Get model configuration
    gb_repeat = get_model_gb_repeat(model)

    # Prepare trainable parameters and optional classification head
    trainable_params = list(model.parameters())
    classification_head: nn.Module | None = None
    criterion: nn.Module | None = None
    if training_mode == "classification":
        embedding_dim = get_model_embedding_dim(model)
        classification_head = build_classification_head(embedding_dim, device_obj)
        criterion = nn.CrossEntropyLoss()
        trainable_params += list(classification_head.parameters())

    # Ensure model is on correct device (should already be there, but verify)
    # Only move if not already on device to avoid creating copies
    if next(model.parameters()).device != device_obj:
        model = model.to(device_obj)
    model.train()

    # Setup optimizer
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate, weight_decay=weight_decay)

    # Gradient clipping to prevent gradient explosion (helps with NaN losses)
    max_grad_norm = 1.0

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
                    "training_mode": training_mode,
                    "contrastive_weight": contrastive_weight,
                    "contrastive_temperature": contrastive_temperature,
                },
            )
            # Set run name
            if wandb_run:
                wandb_run.name = f"lora_finetune_{species}"
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            use_wandb = False

    # Setup memory banks for universal contrastive training
    human_bank: deque[torch.Tensor] | None = None
    mouse_bank: deque[torch.Tensor] | None = None
    if training_mode == "cross_contrastive":
        human_bank = deque(maxlen=MEMORY_BANK_SIZE)
        mouse_bank = deque(maxlen=MEMORY_BANK_SIZE)

    # Training loop with early stopping
    logger.info(
        f"Starting training for {n_epochs} epochs (early stopping patience: {early_stopping_patience})..."
    )
    best_loss = float("inf")
    epochs_without_improvement = 0
    best_epoch = 0

    for epoch in range(n_epochs):
        if training_mode == "cross_contrastive":
            if human_bank is None or mouse_bank is None:
                raise RuntimeError("Contrastive mode requires initialized memory banks.")
            avg_loss, avg_contrastive_loss = train_contrastive_epoch(
                model=model,
                optimizer=optimizer,
                X_tensor=X_tensor,
                y_tensor=y_tensor,
                species_tensor=species_tensor,
                batch_size=batch_size,
                random_seed=random_seed,
                epoch=epoch,
                n_epochs=n_epochs,
                device=device_obj,
                gb_repeat=gb_repeat,
                max_grad_norm=max_grad_norm,
                human_bank=human_bank,
                mouse_bank=mouse_bank,
                use_wandb=use_wandb,
                wandb_run=wandb_run,
                contrastive_weight=contrastive_weight,
                contrastive_temperature=contrastive_temperature,
                trainable_params=trainable_params,
            )
        else:
            if dataloader is None or classification_head is None or criterion is None:
                raise RuntimeError(
                    "Classification mode requires dataloader and classification head."
                )
            avg_loss = train_classification_epoch(
                model=model,
                classification_head=classification_head,
                dataloader=dataloader,
                optimizer=optimizer,
                criterion=criterion,
                device=device_obj,
                gb_repeat=gb_repeat,
                max_grad_norm=max_grad_norm,
                use_wandb=use_wandb,
                wandb_run=wandb_run,
                epoch=epoch,
                n_epochs=n_epochs,
                trainable_params=trainable_params,
            )
            avg_contrastive_loss = None

        logger.info(f"Epoch {epoch + 1}/{n_epochs} - Average Loss: {avg_loss:.4f}")
        if avg_contrastive_loss is not None:
            logger.info(f"  Contrastive Loss (avg): {avg_contrastive_loss:.4f}")

        # Check for improvement
        if avg_loss < best_loss:
            improvement = best_loss - avg_loss
            best_loss = avg_loss
            best_epoch = epoch + 1
            epochs_without_improvement = 0
            logger.info(f"  ✓ New best loss: {best_loss:.4f} (improvement: {improvement:.4f})")

            # Save checkpoint if best
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
                    "training_mode": training_mode,
                    "contrastive_weight": contrastive_weight,
                    "contrastive_temperature": contrastive_temperature,
                },
            )
            if classification_head is not None:
                torch.save(
                    classification_head.state_dict(),
                    checkpoint_dir / "classification_head.pt",
                )
        else:
            epochs_without_improvement += 1
            logger.info(
                f"  No improvement ({epochs_without_improvement}/{early_stopping_patience} epochs)"
            )

        # Log epoch metrics to wandb
        if use_wandb and wandb_run:
            epoch_log = {
                "epoch_loss": avg_loss,
                "best_loss": best_loss,
                "epochs_without_improvement": epochs_without_improvement,
                "epoch": epoch,
            }
            if avg_contrastive_loss is not None:
                epoch_log["epoch_contrastive_loss"] = avg_contrastive_loss
            wandb_run.log(epoch_log)

        # Early stopping check
        if epochs_without_improvement >= early_stopping_patience:
            logger.info(
                f"Early stopping triggered: no improvement for {early_stopping_patience} epochs. "
                f"Best loss: {best_loss:.4f} at epoch {best_epoch}"
            )
            break

    # Save final checkpoint (last epoch, even if not best)
    final_checkpoint_dir = output_dir / "checkpoint_final"
    save_lora_checkpoint(
        model=model,
        output_path=final_checkpoint_dir,
        metadata={
            "epoch": epoch
            + 1,  # Use actual epoch reached (may be less than n_epochs if early stopped)
            "final_loss": avg_loss,
            "best_loss": best_loss,
            "best_epoch": best_epoch,
            "early_stopped": epochs_without_improvement >= early_stopping_patience,
            "species": species,
            "n_inflammation": n_inflammation,
            "n_control": n_control,
            "training_mode": training_mode,
            "contrastive_weight": contrastive_weight,
            "contrastive_temperature": contrastive_temperature,
        },
    )
    if classification_head is not None:
        torch.save(
            classification_head.state_dict(),
            final_checkpoint_dir / "classification_head.pt",
        )

    logger.success(
        f"Training complete! Best loss: {best_loss:.4f} at epoch {best_epoch}. "
        f"Checkpoints saved to {output_dir}"
    )

    if use_wandb and wandb_run:
        wandb_run.finish()

    return output_dir
