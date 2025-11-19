"""Contrastive learning utilities for cross-species training."""

from collections import deque

import torch
import torch.nn.functional as F

SPECIES_TO_ID = {"human": 0, "mouse": 1}
MEMORY_BANK_SIZE = 1024


def get_species_tensor(adata, fallback_label: str) -> torch.Tensor:
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


def compute_cross_species_contrastive_loss(
    normalized_embeddings: torch.Tensor,
    labels: torch.Tensor,
    species_ids: torch.Tensor,
    temperature: float,
    human_bank: deque[torch.Tensor] | None = None,
    mouse_bank: deque[torch.Tensor] | None = None,
) -> torch.Tensor | None:
    """Compute InfoNCE loss where positives are cross-species pairs (inflammation-inflammation and control-control).

    Args:
        normalized_embeddings: L2-normalized sample embeddings [batch_size, embedding_dim].
        labels: Binary labels (1=inflammation, 0=control) [batch_size].
        species_ids: Species IDs (0=human, 1=mouse) [batch_size].
        temperature: Temperature scaling for InfoNCE.
        human_bank: Optional memory bank of human embeddings for negatives.
        mouse_bank: Optional memory bank of mouse embeddings for negatives.

    Returns:
        Contrastive loss tensor or None if insufficient cross-species pairs.
    """
    device = normalized_embeddings.device
    human_mask = species_ids == SPECIES_TO_ID["human"]
    mouse_mask = species_ids == SPECIES_TO_ID["mouse"]

    if human_mask.sum() == 0 or mouse_mask.sum() == 0:
        return None

    human_inbatch_idx = torch.where(human_mask)[0]
    mouse_inbatch_idx = torch.where(mouse_mask)[0]
    human_inbatch = normalized_embeddings[human_inbatch_idx]  # [H, D]
    mouse_inbatch = normalized_embeddings[mouse_inbatch_idx]  # [M, D]

    # Build negative pools by concatenating bank tensors if available
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

        labels_for_h = _match_positions(selected_m, mouse_inbatch_idx)
        labels_for_m = _match_positions(selected_h, human_inbatch_idx)
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
