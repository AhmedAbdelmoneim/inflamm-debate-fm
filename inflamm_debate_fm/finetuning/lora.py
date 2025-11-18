"""LoRA (Low-Rank Adaptation) implementation for BulkFormer fine-tuning."""

from pathlib import Path
from typing import Any

from loguru import logger
import torch
import torch.nn as nn

try:
    from peft import LoraConfig, TaskType, get_peft_model

    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("peft library not available. Install with: pip install peft")

from inflamm_debate_fm.bulkformer.embed import load_bulkformer_model


def apply_lora_to_bulkformer(
    model: nn.Module | None = None,
    r: int = 8,
    lora_alpha: int = 16,
    lora_dropout: float = 0.1,
    target_modules: list[str] | None = None,
    device: str = "cpu",
) -> nn.Module:
    """Apply LoRA to BulkFormer model.

    Args:
        model: Pre-loaded BulkFormer model. If None, will load it.
        r: LoRA rank (lower = fewer parameters).
        lora_alpha: LoRA alpha scaling parameter.
        lora_dropout: LoRA dropout rate.
        target_modules: List of module names to apply LoRA to.
                       If None, applies to linear layers in GBFormer blocks.
        device: Device to load model on.

    Returns:
        Model with LoRA adapters applied.
    """
    if not PEFT_AVAILABLE:
        raise ImportError("peft library is required for LoRA. Install with: pip install peft")

    if model is None:
        model = load_bulkformer_model(device=device)

    # Default target modules: linear layers in GBFormer blocks and projection layers
    # PEFT requires exact module names or simple patterns
    if target_modules is None:
        # Dynamically find all Linear layers in the model
        import re

        linear_modules = []
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                linear_modules.append(name)

        # Filter to key modules: projections and transformer attention/FFN layers
        # Exclude the final head layer (we'll train that separately)
        target_modules = [
            name
            for name in linear_modules
            if (
                name.startswith("gene_emb_proj.")
                or name.startswith("x_proj.")
                or name.startswith("ae_enc.")
                or re.match(
                    r"gb_formers\.\d+\.f\.\d+\.net\.layers\.\d+\.\d+\.fn\.(to_q|to_k|to_v|to_out)",
                    name,
                )
                or re.match(
                    r"gb_formers\.\d+\.f\.\d+\.net\.layers\.\d+\.\d+\.fn\.fn\.(w1|w2)", name
                )
            )
            and not name.startswith("head.")  # Exclude final head
        ]

        if len(target_modules) == 0:
            # Fallback: use all Linear layers except head
            target_modules = [name for name in linear_modules if not name.startswith("head.")]

        logger.info(f"Targeting {len(target_modules)} Linear layers for LoRA adaptation")

    # Create LoRA config
    lora_config = LoraConfig(
        task_type=TaskType.FEATURE_EXTRACTION,  # We're doing feature extraction
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=target_modules,
        bias="none",
    )

    # Apply LoRA
    model = get_peft_model(model, lora_config)

    # Count trainable parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_percentage = (trainable_params / total_params) * 100

    logger.info(
        f"LoRA applied: {trainable_params:,} trainable parameters "
        f"({trainable_percentage:.2f}% of {total_params:,} total)"
    )

    return model


def save_lora_checkpoint(
    model: nn.Module,
    output_path: Path,
    metadata: dict[str, Any] | None = None,
) -> None:
    """Save LoRA checkpoint.

    Args:
        model: Model with LoRA adapters.
        output_path: Path to save checkpoint.
        metadata: Optional metadata dictionary to save alongside checkpoint.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save LoRA weights
    if hasattr(model, "save_pretrained"):
        # PEFT model has save_pretrained method
        model.save_pretrained(str(output_path))
        logger.info(f"Saved LoRA checkpoint to {output_path}")
    else:
        # Fallback: save state dict
        torch.save(model.state_dict(), output_path)
        logger.info(f"Saved model state dict to {output_path}")

    # Save metadata if provided
    if metadata is not None:
        metadata_path = output_path.parent / f"{output_path.stem}_metadata.json"
        import json

        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to {metadata_path}")


def load_lora_checkpoint(
    checkpoint_path: Path,
    base_model: nn.Module | None = None,
    device: str = "cpu",
) -> nn.Module:
    """Load LoRA checkpoint.

    Args:
        checkpoint_path: Path to LoRA checkpoint.
        base_model: Base BulkFormer model. If None, will load it.
        device: Device to load on.

    Returns:
        Model with LoRA adapters loaded.
    """
    if not PEFT_AVAILABLE:
        raise ImportError("peft library is required for LoRA. Install with: pip install peft")

    checkpoint_path = Path(checkpoint_path)

    if base_model is None:
        base_model = load_bulkformer_model(device=device)

    # Load LoRA weights
    if checkpoint_path.is_dir():
        # PEFT checkpoint directory
        from peft import PeftModel

        model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
        logger.info(f"Loaded LoRA checkpoint from {checkpoint_path}")
    else:
        # State dict file
        state_dict = torch.load(checkpoint_path, map_location=device)
        base_model.load_state_dict(state_dict, strict=False)
        model = base_model
        logger.info(f"Loaded model state dict from {checkpoint_path}")

    return model
