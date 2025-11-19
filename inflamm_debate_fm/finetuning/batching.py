"""Batch construction utilities for cross-species contrastive training."""

import torch

from inflamm_debate_fm.finetuning.contrastive import SPECIES_TO_ID


def _shuffle_indices(indices: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    """Shuffle indices using a generator for reproducibility."""
    if indices.numel() == 0:
        return indices
    perm = torch.randperm(indices.numel(), generator=generator)
    return indices[perm]


def build_cross_species_batch_indices(
    labels: torch.Tensor,
    species_ids: torch.Tensor,
    batch_size: int,
    seed: int,
) -> list[list[int]]:
    """Build batch indices ensuring each batch contains cross-species pairs.

    Args:
        labels: Binary labels (1=inflammation, 0=control).
        species_ids: Species IDs (0=human, 1=mouse).
        batch_size: Batch size (must be even).
        seed: Random seed for reproducibility.

    Returns:
        List of batch index lists, each containing pairs of human-mouse samples.
    """
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
            batch = remaining[i : i + batch_size].tolist()
            if len(batch) == batch_size:
                batches.append(batch)

    return batches


def iter_cross_species_batches(
    X_tensor: torch.Tensor,
    y_tensor: torch.Tensor,
    species_tensor: torch.Tensor,
    batch_indices: list[list[int]],
    device: torch.device,
):
    """Iterate over batches constructed from cross-species pairs.

    Args:
        X_tensor: Expression tensor [n_samples, n_genes].
        y_tensor: Labels tensor [n_samples].
        species_tensor: Species IDs tensor [n_samples].
        batch_indices: List of batch index lists.
        device: Device to move tensors to.

    Yields:
        Tuples of (X_batch, y_batch, species_batch) tensors.
    """
    for batch_idx_list in batch_indices:
        X_batch = X_tensor[batch_idx_list].to(device)
        y_batch = y_tensor[batch_idx_list].to(device)
        species_batch = species_tensor[batch_idx_list].to(device)
        yield X_batch, y_batch, species_batch
