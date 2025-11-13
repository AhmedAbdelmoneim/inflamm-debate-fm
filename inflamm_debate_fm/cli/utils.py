"""Shared utilities for CLI commands."""

from inflamm_debate_fm.data.transforms import (
    transform_adata_to_X_y_acute,
    transform_adata_to_X_y_acute_and_subacute,
    transform_adata_to_X_y_acute_subacute_to_chronic,
    transform_adata_to_X_y_acute_to_chronic,
    transform_adata_to_X_y_all,
    transform_adata_to_X_y_chronic,
    transform_adata_to_X_y_subacute,
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
        ("Acute Inflammation vs. Control", transform_adata_to_X_y_acute),
        ("Subacute Inflammation vs. Control", transform_adata_to_X_y_subacute),
        ("Acute and Subacute Inflammation vs. Control", transform_adata_to_X_y_acute_and_subacute),
        ("Chronic Inflammation vs. Control", transform_adata_to_X_y_chronic),
        ("Acute Inflammation vs. Chronic Inflammation", transform_adata_to_X_y_acute_to_chronic),
        (
            "Acute/Subacute Inflammation vs. Chronic Inflammation",
            transform_adata_to_X_y_acute_subacute_to_chronic,
        ),
    ]
