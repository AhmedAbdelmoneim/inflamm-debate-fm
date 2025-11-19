"""Model setup utilities for fine-tuning."""

import torch
import torch.nn as nn


def get_model_gb_repeat(model: torch.nn.Module) -> int:
    """Extract gb_repeat from BulkFormer model.

    Args:
        model: BulkFormer model (may be wrapped by PEFT or BulkFormerPEFTWrapper).

    Returns:
        Number of graph blocks (gb_repeat), default 3 if not found.
    """
    # Access base_model if wrapped by PEFT
    unwrapped_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    # Unwrap BulkFormerPEFTWrapper if present
    if hasattr(unwrapped_model, "base_model"):
        actual_base = unwrapped_model.base_model
    else:
        actual_base = unwrapped_model
    return actual_base.gb_repeat if hasattr(actual_base, "gb_repeat") else 3


def get_model_embedding_dim(model: torch.nn.Module) -> int:
    """Extract embedding dimension from BulkFormer model.

    Args:
        model: BulkFormer model (may be wrapped by PEFT or BulkFormerPEFTWrapper).

    Returns:
        Embedding dimension, default 640 if not found.
    """
    # Access base_model if wrapped by PEFT
    unwrapped_model = model.get_base_model() if hasattr(model, "get_base_model") else model
    # Unwrap BulkFormerPEFTWrapper if present
    if hasattr(unwrapped_model, "base_model"):
        actual_base = unwrapped_model.base_model
    else:
        actual_base = unwrapped_model
    return actual_base.dim if hasattr(actual_base, "dim") else 640


def build_classification_head(embedding_dim: int, device: torch.device) -> nn.Module:
    """Build a classification head for binary classification.

    Args:
        embedding_dim: Input embedding dimension.
        device: Device to place head on.

    Returns:
        Classification head module.
    """
    return nn.Sequential(
        nn.Linear(embedding_dim, 128),
        nn.ReLU(),
        nn.Dropout(0.1),
        nn.Linear(128, 2),  # Binary classification: control vs inflammation
    ).to(device)
