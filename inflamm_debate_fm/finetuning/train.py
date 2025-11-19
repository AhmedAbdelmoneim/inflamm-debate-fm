"""Training functions for LoRA fine-tuning."""

from collections import deque
from pathlib import Path

from loguru import logger
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

SPECIES_TO_ID = {"human": 0, "mouse": 1}
MEMORY_BANK_SIZE = 1024


def _get_species_tensor(adata, fallback_label: str) -> torch.Tensor:
    """Return tensor encoding species IDs aligned with adata rows."""
    fallback = fallback_label.lower()
    if "species" in adata.obs.columns:
        species_series = adata.obs["species"].astype(str)
        species_series = species_series.replace("nan", fallback_label)
        labels = species_series.str.lower().tolist()
    else:
        labels = [fallback] * adata.shape[0]

    mapped = [SPECIES_TO_ID.get(label, -1) for label in labels]
    return torch.tensor(mapped, dtype=torch.long)


def _match_positions(
    selected_indices: torch.Tensor, all_indices: torch.Tensor
) -> torch.Tensor | None:
    """Map selected global indices to their positions inside a reference index tensor."""
    if all_indices.numel() == 0 or selected_indices.numel() == 0:
        return None

    eq = selected_indices.unsqueeze(1) == all_indices.unsqueeze(0)
    if not torch.all(eq.any(dim=1)):
        return None

    return eq.to(dtype=torch.int64).argmax(dim=1)


def _shuffle_indices(indices: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    if indices.numel() == 0:
        return indices
    perm = torch.randperm(indices.numel(), generator=generator)
    return indices[perm]


def _build_cross_species_batch_indices(
    labels: torch.Tensor,
    species_ids: torch.Tensor,
    batch_size: int,
    seed: int,
) -> list[list[int]]:
    if batch_size % 2 != 0:
        raise ValueError("Cross-species contrastive mode requires an even batch size.")

    generator = torch.Generator()
    generator.manual_seed(seed)

    human_mask = species_ids == SPECIES_TO_ID["human"]
    mouse_mask = species_ids == SPECIES_TO_ID["mouse"]

    def split_indices(label_value: int, mask: torch.Tensor) -> torch.Tensor:
        return torch.where(mask & (labels == label_value))[0]

    human_infl = _shuffle_indices(split_indices(1, human_mask), generator)
    mouse_infl = _shuffle_indices(split_indices(1, mouse_mask), generator)
    human_ctrl = _shuffle_indices(split_indices(0, human_mask), generator)
    mouse_ctrl = _shuffle_indices(split_indices(0, mouse_mask), generator)

    def make_pairs(h_array: torch.Tensor, m_array: torch.Tensor):
        n_pairs = min(h_array.numel(), m_array.numel())
        pairs = [[int(h_array[i].item()), int(m_array[i].item())] for i in range(n_pairs)]
        h_remaining = h_array[n_pairs:]
        m_remaining = m_array[n_pairs:]
        return pairs, h_remaining, m_remaining

    infl_pairs, human_infl_left, mouse_infl_left = make_pairs(human_infl, mouse_infl)
    ctrl_pairs, human_ctrl_left, mouse_ctrl_left = make_pairs(human_ctrl, mouse_ctrl)

    all_pairs = infl_pairs + ctrl_pairs
    if all_pairs:
        order = torch.randperm(len(all_pairs), generator=generator).tolist()
        all_pairs = [all_pairs[i] for i in order]

    batches: list[list[int]] = []
    pairs_per_batch = batch_size // 2
    for i in range(0, len(all_pairs), pairs_per_batch):
        pair_chunk = all_pairs[i : i + pairs_per_batch]
        if not pair_chunk:
            continue
        batch = [idx for pair in pair_chunk for idx in pair]
        batches.append(batch)

    remaining = torch.cat(
        [
            human_infl_left,
            mouse_infl_left,
            human_ctrl_left,
            mouse_ctrl_left,
        ]
    )
    if remaining.numel() > 0:
        remaining = _shuffle_indices(remaining, generator)
        for i in range(0, remaining.numel(), batch_size):
            batch_tensor = remaining[i : i + batch_size]
            if batch_tensor.numel() > 0:
                batches.append(batch_tensor.tolist())

    return batches


def _iter_cross_species_batches(
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    species_tensor: torch.Tensor,
    batch_indices: list[list[int]],
    device: torch.device,
):
    for indices in batch_indices:
        idx_tensor = torch.tensor(indices, dtype=torch.long)
        yield (
            X_tensor.index_select(0, idx_tensor).to(device),
            y_tensor.index_select(0, idx_tensor).to(device),
            species_tensor.index_select(0, idx_tensor).to(device),
        )


def _compute_cross_species_contrastive_loss(
    normalized_embeddings: torch.Tensor,
    labels: torch.Tensor,
    species_ids: torch.Tensor,
    temperature: float,
    human_bank: deque | None = None,
    mouse_bank: deque | None = None,
) -> torch.Tensor | None:
    """Compute InfoNCE loss where positives are cross-species inflammation pairs."""
    device = normalized_embeddings.device
    human_mask = species_ids == SPECIES_TO_ID["human"]
    mouse_mask = species_ids == SPECIES_TO_ID["mouse"]

    if human_mask.sum() == 0 or mouse_mask.sum() == 0:
        return None

    device = normalized_embeddings.device
    human_all_idx = torch.where(human_mask)[0]
    mouse_all_idx = torch.where(mouse_mask)[0]
    human_inbatch = normalized_embeddings[human_all_idx]
    mouse_inbatch = normalized_embeddings[mouse_all_idx]

    if human_bank and len(human_bank) > 0:
        human_bank_tensor = torch.stack(list(human_bank), dim=0).to(device)
        human_all = torch.cat([human_inbatch, human_bank_tensor], dim=0)
    else:
        human_all = human_inbatch

    if mouse_bank and len(mouse_bank) > 0:
        mouse_bank_tensor = torch.stack(list(mouse_bank), dim=0).to(device)
        mouse_all = torch.cat([mouse_inbatch, mouse_bank_tensor], dim=0)
    else:
        mouse_all = mouse_inbatch

    total_loss = 0.0
    component_count = 0

    for target_label in (1, 0):  # 1 = inflammation, 0 = control
        human_subset = torch.where(human_mask & (labels == target_label))[0]
        mouse_subset = torch.where(mouse_mask & (labels == target_label))[0]

        if len(human_subset) == 0 or len(mouse_subset) == 0:
            continue

        num_pairs = min(len(human_subset), len(mouse_subset))
        if num_pairs == 0:
            continue

        perm_h = torch.randperm(len(human_subset), device=device)[:num_pairs]
        perm_m = torch.randperm(len(mouse_subset), device=device)[:num_pairs]
        selected_h = human_subset[perm_h]
        selected_m = mouse_subset[perm_m]

        labels_for_h = _match_positions(selected_m, mouse_all_idx)
        labels_for_m = _match_positions(selected_h, human_all_idx)
        if labels_for_h is None or labels_for_m is None:
            continue

        logits_h = torch.matmul(normalized_embeddings[selected_h], mouse_all.T) / temperature
        logits_m = torch.matmul(normalized_embeddings[selected_m], human_all.T) / temperature

        loss_h = F.cross_entropy(logits_h, labels_for_h)
        loss_m = F.cross_entropy(logits_m, labels_for_m)

        total_loss += 0.5 * (loss_h + loss_m)
        component_count += 1

    if component_count == 0:
        return None

    return total_loss / component_count


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
    species_tensor = _get_species_tensor(adata, fallback_species)

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
    human_bank: deque[torch.Tensor] = deque(maxlen=MEMORY_BANK_SIZE)
    mouse_bank: deque[torch.Tensor] = deque(maxlen=MEMORY_BANK_SIZE)

    # Training loop with early stopping
    logger.info(
        f"Starting training for {n_epochs} epochs (early stopping patience: {early_stopping_patience})..."
    )
    best_loss = float("inf")
    epochs_without_improvement = 0
    best_epoch = 0

    for epoch in range(n_epochs):
        epoch_losses = []
        epoch_contrastive_losses = []
        model.train()

        if training_mode == "cross_contrastive":
            batch_indices = _build_cross_species_batch_indices(
                labels=y_tensor,
                species_ids=species_tensor,
                batch_size=batch_size,
                seed=random_seed + epoch,
            )
            iterable = _iter_cross_species_batches(
                X_tensor=X_tensor,
                y_tensor=y_tensor,
                species_tensor=species_tensor,
                batch_indices=batch_indices,
                device=device_obj,
            )
            progress_bar = tqdm(
                iterable,
                total=len(batch_indices),
                desc=f"Epoch {epoch + 1}/{n_epochs}",
            )
        else:
            progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}")
        for batch_idx, (X_batch, y_batch, species_batch) in enumerate(progress_bar):
            X_batch = X_batch.to(device_obj)
            y_batch = y_batch.to(device_obj)
            species_batch = species_batch.to(device_obj)

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
            ce_loss = criterion(logits, y_batch)
            loss = ce_loss
            ce_loss_value = ce_loss.item()

            contrastive_component = None
            contrastive_loss_value = None
            normalized_embeddings = None
            if training_mode == "cross_contrastive" and contrastive_weight > 0:
                normalized_embeddings = F.normalize(sample_embeddings, p=2, dim=1)
                contrastive_component = _compute_cross_species_contrastive_loss(
                    normalized_embeddings,
                    y_batch,
                    species_batch,
                    contrastive_temperature,
                    human_bank=human_bank,
                    mouse_bank=mouse_bank,
                )
                if contrastive_component is not None:
                    loss = loss + contrastive_weight * contrastive_component
                    contrastive_loss_value = contrastive_component.item()
                    epoch_contrastive_losses.append(contrastive_loss_value)
                else:
                    logger.debug(
                        "Skipped contrastive update for this batch (insufficient cross-species inflammation pairs)."
                    )

            # Check for NaN/Inf in loss or logits before backward pass
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(
                    f"NaN/Inf loss detected at epoch {epoch + 1}, batch {batch_idx}. "
                    f"Logits stats: min={logits.min().item():.4f}, max={logits.max().item():.4f}, "
                    f"mean={logits.mean().item():.4f}, has_nan={torch.isnan(logits).any()}, "
                    f"has_inf={torch.isinf(logits).any()}"
                )
                logger.error(
                    f"Input stats: min={X_batch.min().item():.4f}, max={X_batch.max().item():.4f}, "
                    f"mean={X_batch.mean().item():.4f}, has_nan={torch.isnan(X_batch).any()}, "
                    f"has_inf={torch.isinf(X_batch).any()}"
                )
                # Skip this batch and continue
                optimizer.zero_grad()
                continue

            # Backward pass
            loss.backward()

            # Gradient clipping to prevent explosion
            torch.nn.utils.clip_grad_norm_(all_params, max_grad_norm)

            optimizer.step()

            # Store loss value before deleting tensor
            loss_value = loss.item()
            epoch_losses.append(loss_value)
            postfix = {"loss": loss_value, "cls": ce_loss_value}
            if contrastive_loss_value is not None:
                postfix["contrastive"] = contrastive_loss_value
            progress_bar.set_postfix(postfix)

            # Update memory banks with detached embeddings (CPU storage)
            if training_mode == "cross_contrastive" and normalized_embeddings is not None:
                with torch.no_grad():
                    for vec, species_id in zip(normalized_embeddings, species_batch):
                        if species_id.item() == SPECIES_TO_ID["human"]:
                            human_bank.append(vec.detach().cpu())
                        elif species_id.item() == SPECIES_TO_ID["mouse"]:
                            mouse_bank.append(vec.detach().cpu())

            # Clear intermediate tensors to free memory
            del gene_embeddings, sample_embeddings, logits, loss, ce_loss
            if contrastive_component is not None:
                del contrastive_component
            if normalized_embeddings is not None:
                del normalized_embeddings
            del species_batch
            # Clear cache more aggressively for memory-constrained scenarios
            if device_obj.type == "cuda":
                torch.cuda.empty_cache()

            # Log to wandb
            if use_wandb and wandb_run:
                log_payload = {
                    "batch_loss": loss_value,
                    "classification_loss": ce_loss_value,
                    "epoch": epoch,
                }
                if contrastive_loss_value is not None:
                    log_payload["contrastive_loss"] = contrastive_loss_value
                wandb_run.log(log_payload)

        avg_loss = np.mean(epoch_losses)
        avg_contrastive_loss = (
            float(np.mean(epoch_contrastive_losses)) if epoch_contrastive_losses else None
        )
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
            # Also save classification head
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
    # Also save classification head
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
