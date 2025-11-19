"""Training loop implementations for classification and contrastive modes."""

from collections import deque

from loguru import logger
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from inflamm_debate_fm.finetuning.batching import (
    build_cross_species_batch_indices,
    iter_cross_species_batches,
)
from inflamm_debate_fm.finetuning.contrastive import (
    SPECIES_TO_ID,
    compute_cross_species_contrastive_loss,
)


def train_classification_epoch(
    *,
    model: torch.nn.Module,
    classification_head: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    gb_repeat: int,
    max_grad_norm: float,
    use_wandb: bool,
    wandb_run,
    epoch: int,
    n_epochs: int,
    trainable_params: list[torch.nn.Parameter],
) -> float:
    """Train one epoch for classification mode.

    Returns:
        Average loss for the epoch.
    """
    epoch_losses: list[float] = []
    classification_head.train()
    model.train()

    progress_bar = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{n_epochs}")
    for batch_idx, (X_batch, y_batch, _) in enumerate(progress_bar):
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()
        output, hidden = model(inputs_embeds=X_batch, repr_layers=[gb_repeat - 1])
        gene_embeddings = hidden[gb_repeat - 1]
        del output, hidden

        sample_embeddings = gene_embeddings.mean(dim=1)
        logits = classification_head(sample_embeddings)
        loss = criterion(logits, y_batch)

        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(
                f"NaN/Inf loss detected at epoch {epoch + 1}, batch {batch_idx}. "
                f"Loss value: {loss.item()}"
            )
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
        optimizer.step()

        loss_value = loss.item()
        epoch_losses.append(loss_value)
        progress_bar.set_postfix({"loss": loss_value})

        del gene_embeddings, sample_embeddings, logits, loss
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if use_wandb and wandb_run:
            wandb_run.log(
                {"batch_loss": loss_value, "classification_loss": loss_value, "epoch": epoch}
            )

    avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
    return avg_loss


def _has_valid_cross_species_pairs(
    y_batch: torch.Tensor,
    species_batch: torch.Tensor,
) -> bool:
    """Check if batch has valid cross-species pairs for contrastive loss.

    Args:
        y_batch: Binary labels (1=inflammation, 0=control).
        species_batch: Species IDs (0=human, 1=mouse).

    Returns:
        True if batch has both human and mouse samples and at least one inflammation
        label has samples from both species.
    """
    human_mask = species_batch == SPECIES_TO_ID["human"]
    mouse_mask = species_batch == SPECIES_TO_ID["mouse"]

    # Need both species in batch
    if human_mask.sum() == 0 or mouse_mask.sum() == 0:
        return False

    # Check for cross-species inflammation pairs only (label == 1)
    human_inflamed = (human_mask & (y_batch == 1)).sum()
    mouse_inflamed = (mouse_mask & (y_batch == 1)).sum()
    return human_inflamed > 0 and mouse_inflamed > 0


def train_contrastive_epoch(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    species_tensor: torch.Tensor,
    batch_size: int,
    random_seed: int,
    epoch: int,
    n_epochs: int,
    device: torch.device,
    gb_repeat: int,
    max_grad_norm: float,
    human_bank: deque[torch.Tensor],
    mouse_bank: deque[torch.Tensor],
    use_wandb: bool,
    wandb_run,
    contrastive_weight: float,
    contrastive_temperature: float,
    trainable_params: list[torch.nn.Parameter],
) -> tuple[float, float | None]:
    """Train one epoch for contrastive mode.

    Returns:
        Tuple of (average loss, average contrastive loss).
    """
    epoch_losses: list[float] = []
    epoch_contrastive_losses: list[float] = []
    model.train()

    batch_indices = build_cross_species_batch_indices(
        labels=y_tensor,
        species_ids=species_tensor,
        batch_size=batch_size,
        seed=random_seed + epoch,
    )
    iterable = iter_cross_species_batches(
        X_tensor=X_tensor,
        y_tensor=y_tensor,
        species_tensor=species_tensor,
        batch_indices=batch_indices,
        device=device,
    )
    progress_bar = tqdm(iterable, total=len(batch_indices), desc=f"Epoch {epoch + 1}/{n_epochs}")

    for batch_idx, (X_batch, y_batch, species_batch) in enumerate(progress_bar):
        # Check if batch has valid cross-species pairs BEFORE processing
        # This saves memory by avoiding forward pass on invalid batches
        if not _has_valid_cross_species_pairs(y_batch, species_batch):
            logger.debug("Skipped batch (insufficient cross-species pairs for contrastive loss).")
            continue

        optimizer.zero_grad()
        output, hidden = model(inputs_embeds=X_batch, repr_layers=[gb_repeat - 1])
        gene_embeddings = hidden[gb_repeat - 1]
        del output, hidden

        sample_embeddings = gene_embeddings.mean(dim=1)
        normalized_embeddings = F.normalize(sample_embeddings, p=2, dim=1)
        contrastive_component = compute_cross_species_contrastive_loss(
            normalized_embeddings,
            y_batch,
            species_batch,
            contrastive_temperature,
            human_bank=human_bank,
            mouse_bank=mouse_bank,
        )

        if contrastive_component is None:
            logger.debug("Skipped batch (insufficient cross-species pairs for contrastive loss).")
            optimizer.zero_grad()
            with torch.no_grad():
                for vec, species_id in zip(normalized_embeddings, species_batch):
                    if species_id.item() == SPECIES_TO_ID["human"]:
                        human_bank.append(vec.detach().cpu())
                    elif species_id.item() == SPECIES_TO_ID["mouse"]:
                        mouse_bank.append(vec.detach().cpu())
            del gene_embeddings, sample_embeddings, normalized_embeddings
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        loss = contrastive_weight * contrastive_component
        if torch.isnan(loss) or torch.isinf(loss):
            logger.error(
                f"NaN/Inf contrastive loss at epoch {epoch + 1}, batch {batch_idx}: {loss.item()}"
            )
            optimizer.zero_grad()
            del gene_embeddings, sample_embeddings, normalized_embeddings
            if device.type == "cuda":
                torch.cuda.empty_cache()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_params, max_grad_norm)
        optimizer.step()

        loss_value = loss.item()
        contrastive_loss_value = contrastive_component.item()
        epoch_losses.append(loss_value)
        epoch_contrastive_losses.append(contrastive_loss_value)
        progress_bar.set_postfix({"loss": loss_value, "contrastive": contrastive_loss_value})

        with torch.no_grad():
            for vec, species_id in zip(normalized_embeddings, species_batch):
                if species_id.item() == SPECIES_TO_ID["human"]:
                    human_bank.append(vec.detach().cpu())
                elif species_id.item() == SPECIES_TO_ID["mouse"]:
                    mouse_bank.append(vec.detach().cpu())

        del gene_embeddings, sample_embeddings, normalized_embeddings, loss, contrastive_component
        if device.type == "cuda":
            torch.cuda.empty_cache()

        if use_wandb and wandb_run:
            wandb_run.log(
                {
                    "batch_loss": loss_value,
                    "contrastive_loss": contrastive_loss_value,
                    "epoch": epoch,
                }
            )
    avg_loss = float(np.mean(epoch_losses)) if epoch_losses else float("inf")
    avg_contrastive = (
        float(np.mean(epoch_contrastive_losses)) if epoch_contrastive_losses else None
    )
    return avg_loss, avg_contrastive
