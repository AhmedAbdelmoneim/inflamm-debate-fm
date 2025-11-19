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


class BulkFormerPEFTWrapper(nn.Module):
    """Wrapper to make BulkFormer compatible with PEFT's expected forward signature.

    PEFT expects models to accept HuggingFace-style kwargs (input_ids, etc.),
    but BulkFormer uses positional arguments. This wrapper bridges the gap.
    """

    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

    def forward(self, input_ids=None, inputs_embeds=None, repr_layers=None, **kwargs):
        """Forward pass that accepts both PEFT-style and BulkFormer-style arguments.

        Args:
            input_ids: Input tensor (PEFT may pass this)
            inputs_embeds: Input tensor (alternative PEFT parameter)
            repr_layers: Layers to return representations from
            **kwargs: Additional arguments (ignored for BulkFormer compatibility)
        """
        # Use inputs_embeds if provided (PEFT style), otherwise use input_ids
        # PEFT may pass input_ids as the first positional arg or as keyword
        if inputs_embeds is not None:
            x = inputs_embeds
        elif input_ids is not None:
            x = input_ids
        else:
            # If neither provided, check if first positional arg was passed
            # This shouldn't happen with PEFT, but handle it gracefully
            raise ValueError("Must provide either input_ids or inputs_embeds")

        # Call base model with BulkFormer's signature (only pass repr_layers if provided)
        if repr_layers is not None:
            return self.base_model(x, repr_layers=repr_layers)
        else:
            return self.base_model(x)


def apply_lora_to_bulkformer(
    model: nn.Module | None = None,
    r: int = 4,
    lora_alpha: int = 8,
    lora_dropout: float = 0.1,
    target_modules: list[str] | None = None,
    device: str = "cpu",
) -> nn.Module:
    """Apply LoRA to BulkFormer model.

    Targets only the last (highest) gb_formers block's attention layers to preserve
    pretrained representations while allowing task-specific adaptation.

    Args:
        model: Pre-loaded BulkFormer model. If None, will load it.
        r: LoRA rank (default: 4, lower = fewer parameters, less overfitting risk).
        lora_alpha: LoRA alpha scaling parameter (default: 8, typically 2*r).
        lora_dropout: LoRA dropout rate.
        target_modules: List of module names to apply LoRA to.
                       If None, automatically targets last block attention layers only.
        device: Device to load model on.

    Returns:
        Model with LoRA adapters applied.
    """
    if not PEFT_AVAILABLE:
        raise ImportError("peft library is required for LoRA. Install with: pip install peft")

    if model is None:
        model = load_bulkformer_model(device=device)

    # Store original model for finding target modules (before wrapping)
    original_model = model

    # Wrap model to make it compatible with PEFT's expected forward signature
    if not isinstance(model, BulkFormerPEFTWrapper):
        model = BulkFormerPEFTWrapper(model)

    # Default target modules: only attention layers in the LAST gb_formers block
    # This preserves pretrained embeddings and early layers while allowing
    # task-specific adaptation in the final representation layer
    if target_modules is None:
        import re

        # Find all gb_formers blocks to identify the last one
        gb_formers_blocks = []
        for name, module in original_model.named_modules():
            # Match pattern: gb_formers.{block_idx}.f.{f_idx}.net.layers...
            match = re.match(r"gb_formers\.(\d+)\.", name)
            if match:
                block_idx = int(match.group(1))
                if block_idx not in gb_formers_blocks:
                    gb_formers_blocks.append(block_idx)

        if len(gb_formers_blocks) == 0:
            raise ValueError(
                "Could not find any gb_formers blocks in the model. "
                "Please check the model architecture."
            )

        # Get the last (highest index) block
        last_block_idx = max(gb_formers_blocks)
        logger.info(
            f"Found {len(gb_formers_blocks)} gb_formers blocks. "
            f"Targeting last block (index {last_block_idx}) for LoRA adaptation."
        )

        # Find attention layers (to_q, to_v, to_out) in the last block only
        # Exclude to_k as it's less critical for task adaptation
        target_modules = []
        for name, module in original_model.named_modules():
            if isinstance(module, torch.nn.Linear):
                # Match last block attention layers: gb_formers.{last_block_idx}.f.*.net.layers.*.*.fn.(to_q|to_v|to_out)
                pattern = rf"gb_formers\.{last_block_idx}\.f\.\d+\.net\.layers\.\d+\.\d+\.fn\.(to_q|to_v|to_out)"
                if re.match(pattern, name):
                    target_modules.append(name)
                    logger.debug(f"  Targeting: {name}")

        if len(target_modules) == 0:
            # Fallback: try to find any attention layers in last block with broader pattern
            logger.warning(
                f"Could not find attention layers in last block {last_block_idx}. "
                "Trying broader pattern matching..."
            )
            for name, module in original_model.named_modules():
                if isinstance(module, torch.nn.Linear):
                    # Match last block with any attention pattern
                    pattern = rf"gb_formers\.{last_block_idx}\.f\.\d+\.net\.layers\.\d+\.\d+\.fn\.(to_q|to_k|to_v|to_out)"
                    if re.match(pattern, name):
                        target_modules.append(name)
                        logger.debug(f"  Targeting (fallback): {name}")

        if len(target_modules) == 0:
            raise ValueError(
                f"Could not find any attention layers in the last gb_formers block (index {last_block_idx}). "
                "Please check the model architecture or provide target_modules explicitly."
            )

        # Adjust target_modules to account for BulkFormerPEFTWrapper prefix
        # When PEFT wraps our wrapper, modules are accessed as "base_model.module_name"
        target_modules = [f"base_model.{name}" for name in target_modules]

        logger.info(
            f"Targeting {len(target_modules)} attention layers in last block for LoRA adaptation"
        )

    # Create LoRA config with SEQ_CLS task type for classification tasks
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,  # Sequence classification (inflammation classification)
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

    # Wrap base model to make it compatible with PEFT's expected forward signature
    # This must be done before loading PEFT checkpoint
    if not isinstance(base_model, BulkFormerPEFTWrapper):
        # Check if base_model is already wrapped by PEFT (shouldn't happen, but be safe)
        if hasattr(base_model, "base_model") and isinstance(
            base_model.base_model, BulkFormerPEFTWrapper
        ):
            wrapped_base = base_model
        else:
            wrapped_base = BulkFormerPEFTWrapper(base_model)
    else:
        wrapped_base = base_model

    # Load LoRA weights
    if checkpoint_path.is_dir():
        # PEFT checkpoint directory
        from peft import PeftModel

        model = PeftModel.from_pretrained(wrapped_base, str(checkpoint_path))

        # Verify that the wrapper is preserved in the PEFT structure
        # PEFT might access model.base_model.model, so we need to ensure that's wrapped
        if hasattr(model, "base_model") and hasattr(model.base_model, "model"):
            if not isinstance(model.base_model.model, BulkFormerPEFTWrapper):
                # PEFT unwrapped our wrapper, re-apply it
                logger.warning(
                    "PEFT unwrapped BulkFormerPEFTWrapper. Re-applying wrapper to ensure compatibility."
                )
                # Get the underlying BulkFormer model
                underlying_model = model.base_model.model
                # Re-wrap it
                wrapped_underlying = BulkFormerPEFTWrapper(underlying_model)
                # Replace it in the PEFT structure
                model.base_model.model = wrapped_underlying

        logger.info(f"Loaded LoRA checkpoint from {checkpoint_path}")
    else:
        # State dict file
        state_dict = torch.load(checkpoint_path, map_location=device)
        base_model.load_state_dict(state_dict, strict=False)
        model = base_model
        logger.info(f"Loaded model state dict from {checkpoint_path}")

    return model
