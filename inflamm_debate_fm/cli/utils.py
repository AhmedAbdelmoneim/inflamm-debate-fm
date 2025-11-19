"""Shared utilities for CLI commands."""

from inflamm_debate_fm.data.transforms import (
    transform_adata_to_X_y_all,
    transform_adata_to_X_y_takao,
)


def get_setup_transforms():
    """Get all setup transform functions.

    Returns:
        List of tuples (setup_name, transform_func) for probing experiments.
    """
    return [
        ("All Inflammation Samples vs. Control", transform_adata_to_X_y_all),
        ("Takao Subset for Inflammation vs. Control", transform_adata_to_X_y_takao),
    ]
