"""Dataset-specific cleaning and preprocessing functions.

This module contains functions to clean metadata and add derived columns for each dataset.
Each function handles both cleaning and derived column generation for clarity and robustness.
"""

import anndata as ad
from loguru import logger
import numpy as np
import pandas as pd

from inflamm_debate_fm.config import get_config


def _check_columns(adata: ad.AnnData, required: list[str], dataset_name: str) -> bool:
    """Check if required columns exist in adata.obs."""
    missing = [col for col in required if col not in adata.obs.columns]
    if missing:
        logger.warning(
            f"{dataset_name}: Missing required columns: {missing}. Available: {list(adata.obs.columns)}"
        )
        return False
    return True


def set_symbol_index(adata: ad.AnnData, dataset_name: str) -> ad.AnnData:
    """Set index to symbols and ensure both symbol and ensembl are in .var.

    Args:
        adata: AnnData object with Ensembl IDs as index and symbols in .var.
        dataset_name: Name of the dataset (for logging).

    Returns:
        AnnData object with symbols as index and both symbol and ensembl in .var.
    """
    # Ensure symbol and ensembl columns exist
    if "symbol" not in adata.var.columns:
        logger.warning(f"{dataset_name}: Missing 'symbol' column in .var, using index as symbol")
        adata.var["symbol"] = adata.var.index

    if "ensembl" not in adata.var.columns:
        logger.warning(f"{dataset_name}: Missing 'ensembl' column in .var, using index as ensembl")
        adata.var["ensembl"] = adata.var.index

    # Fill missing symbols with Ensembl IDs
    adata.var["symbol"] = adata.var["symbol"].fillna(adata.var["ensembl"])

    # Store Ensembl IDs before changing index
    ensembl_values = adata.var["ensembl"].values.copy()

    # Handle duplicate symbols by appending Ensembl ID
    duplicate_mask = adata.var["symbol"].duplicated(keep=False)
    if duplicate_mask.any():
        logger.info(
            f"{dataset_name}: Found {duplicate_mask.sum()} duplicate symbols, appending Ensembl IDs"
        )
        adata.var.loc[duplicate_mask, "symbol"] = (
            adata.var.loc[duplicate_mask, "symbol"]
            + "_"
            + adata.var.loc[duplicate_mask, "ensembl"]
        )

    # Get new index (symbols)
    new_index = adata.var["symbol"].values

    # Create new var dataframe with symbols as index
    new_var = adata.var.copy()
    new_var.index = new_index
    new_var["symbol"] = new_var.index
    new_var["ensembl"] = ensembl_values

    # Create new AnnData with symbols as index
    # The X matrix columns will be automatically aligned with the new var index
    adata_new = ad.AnnData(
        X=adata.X,
        obs=adata.obs.copy(),
        var=new_var,
    )

    # Copy over other attributes
    if hasattr(adata, "obsm"):
        adata_new.obsm = adata.obsm.copy()
    if hasattr(adata, "uns"):
        adata_new.uns = adata.uns.copy()
    if hasattr(adata, "layers"):
        adata_new.layers = adata.layers.copy()

    return adata_new


def preprocess_human_burn(
    adata: ad.AnnData, acute_cutoff: int, subacute_cutoff: int
) -> ad.AnnData:
    """Clean and preprocess human burn dataset.

    Args:
        adata: AnnData object with raw metadata.
        acute_cutoff: Acute inflammation cutoff in hours.
        subacute_cutoff: Subacute inflammation cutoff in hours.

    Returns:
        Cleaned AnnData object with derived columns.
    """
    required = [
        "source_name_ch1",
        "title",
        "tissue:ch1",
        "Sex:ch1",
        "age:ch1",
        "hours_since_injury:ch1",
    ]
    if not _check_columns(adata, required, "human_burn"):
        return adata

    # Select columns
    adata.obs = adata.obs[required].copy()
    adata.obs.columns = ["group", "patient_id", "tissue", "sex", "age", "time_point_hours"]

    # Clean group column
    adata.obs["group"] = adata.obs["group"].apply(
        lambda x: "inflammation" if "Subject" in str(x) else "control"
    )

    # Extract patient ID from title
    patient_ids = adata.obs["patient_id"].str.extract(r"(\d+)")
    if patient_ids.notna().any().any():
        adata.obs["patient_id"] = (
            pd.to_numeric(patient_ids.iloc[:, 0], errors="coerce").fillna(0).astype(int)
        )
    else:
        adata.obs["patient_id"] = 0

    # Extract time point from hours_since_injury
    adata.obs["time_point_hours"] = (
        adata.obs["time_point_hours"].str.extract(r"(\d+\.?\d*)").astype(float)
    )
    adata.obs["time_point_hours"] = adata.obs["time_point_hours"].fillna(0)

    # Convert age to numeric
    adata.obs["age"] = pd.to_numeric(adata.obs["age"], errors="coerce")

    # Derived columns
    adata.obs.loc[adata.obs["group"] == "control", "time_point_hours"] = pd.NA
    adata.obs["takao_inflamed"] = adata.obs["group"] == "inflammation"
    adata.obs["takao_control"] = adata.obs["group"] == "control"
    adata.obs["takao_status"] = pd.NA
    adata.obs.loc[adata.obs["takao_inflamed"], "takao_status"] = "takao_inflamed"
    adata.obs.loc[adata.obs["takao_control"], "takao_status"] = "takao_control"
    adata.obs["takao_status"] = adata.obs["takao_status"].astype("category")

    # Inflammation categories
    adata.obs["infl_acute"] = (adata.obs["group"] != "control") & (
        adata.obs["time_point_hours"] < acute_cutoff
    )
    adata.obs["infl_subacute"] = (
        (adata.obs["group"] != "control")
        & (adata.obs["time_point_hours"] >= acute_cutoff)
        & (adata.obs["time_point_hours"] < subacute_cutoff)
    )
    adata.obs["infl_chronic"] = (adata.obs["group"] != "control") & (
        adata.obs["time_point_hours"] >= subacute_cutoff
    )

    return adata


def preprocess_human_trauma(
    adata: ad.AnnData, acute_cutoff: int, subacute_cutoff: int
) -> ad.AnnData:
    """Clean and preprocess human trauma dataset.

    Args:
        adata: AnnData object with raw metadata.
        acute_cutoff: Acute inflammation cutoff in hours.
        subacute_cutoff: Subacute inflammation cutoff in hours.

    Returns:
        Cleaned AnnData object with derived columns.
    """
    required = [
        "source_name_ch1",
        "title",
        "description",
        "tissue:ch1",
        "Sex:ch1",
        "age:ch1",
        "hours_since_injury:ch1",
    ]
    if not _check_columns(adata, required, "human_trauma"):
        return adata

    # Select columns
    adata.obs = adata.obs[required].copy()
    adata.obs.columns = [
        "group",
        "patient_id",
        "description",
        "tissue",
        "sex",
        "age",
        "time_point_hours",
    ]

    # Clean group column
    adata.obs["group"] = adata.obs["group"].apply(
        lambda x: "inflammation" if "Subject" in str(x) else "control"
    )

    # Extract patient ID from title
    patient_ids = adata.obs["patient_id"].str.extract(r"(\d+)")
    if patient_ids.notna().any().any():
        adata.obs["patient_id"] = (
            pd.to_numeric(patient_ids.iloc[:, 0], errors="coerce").fillna(0).astype(int)
        )
    else:
        adata.obs["patient_id"] = 0

    # Extract time point from hours_since_injury
    adata.obs["time_point_hours"] = (
        adata.obs["time_point_hours"].str.extract(r"(\d+\.?\d*)").astype(float)
    )
    adata.obs["time_point_hours"] = adata.obs["time_point_hours"].fillna(0)

    # Remove rows with any NaN values in expression matrix
    nan_mask = np.any(np.isnan(adata.X), axis=0)
    adata = adata[:, ~nan_mask].copy()

    # Drop description column
    adata.obs = adata.obs.drop(columns=["description"])

    # Derived columns
    adata.obs["takao_inflamed"] = (adata.obs["group"] == "inflammation") & (
        adata.obs["time_point_hours"] > (14 * 24)
    )
    adata.obs["takao_control"] = adata.obs["group"] == "control"
    adata.obs["takao_status"] = pd.NA
    adata.obs.loc[adata.obs["takao_inflamed"], "takao_status"] = "takao_inflamed"
    adata.obs.loc[adata.obs["takao_control"], "takao_status"] = "takao_control"
    adata.obs.loc[adata.obs["group"] == "control", "time_point_hours"] = pd.NA
    adata.obs["takao_status"] = adata.obs["takao_status"].astype("category")

    # Inflammation categories
    adata.obs["infl_acute"] = (adata.obs["group"] != "control") & (
        adata.obs["time_point_hours"] <= acute_cutoff
    )
    adata.obs["infl_subacute"] = (
        (adata.obs["group"] != "control")
        & (adata.obs["time_point_hours"] > acute_cutoff)
        & (adata.obs["time_point_hours"] <= subacute_cutoff)
    )
    adata.obs["infl_chronic"] = (adata.obs["group"] != "control") & (
        adata.obs["time_point_hours"] > subacute_cutoff
    )

    return adata


def preprocess_human_sepsis(adata: ad.AnnData) -> ad.AnnData:
    """Clean and preprocess human sepsis dataset.

    Args:
        adata: AnnData object with raw metadata.

    Returns:
        Cleaned AnnData object with derived columns.
    """
    required = ["tissue:ch1", "health status:ch1"]
    if not _check_columns(adata, required, "human_sepsis"):
        return adata

    # Select columns
    adata.obs = adata.obs[required].copy()
    adata.obs.columns = ["tissue", "group"]

    # Clean group column
    adata.obs["group"] = adata.obs["group"].str.lower()
    adata = adata[adata.obs["group"].isin(["sepsis", "healthy"])].copy()
    adata.obs["group"] = adata.obs["group"].map({"sepsis": "inflammation", "healthy": "control"})

    # Derived columns
    adata.obs["takao_inflamed"] = adata.obs["group"] == "inflammation"
    adata.obs["takao_control"] = adata.obs["group"] == "control"
    adata.obs["takao_status"] = pd.NA
    adata.obs.loc[adata.obs["takao_inflamed"], "takao_status"] = "takao_inflamed"
    adata.obs.loc[adata.obs["takao_control"], "takao_status"] = "takao_control"
    adata.obs["takao_status"] = adata.obs["takao_status"].astype("category")

    return adata


def preprocess_mouse_burn(
    adata: ad.AnnData, acute_cutoff: int, subacute_cutoff: int
) -> ad.AnnData:
    """Clean and preprocess mouse burn dataset.

    Args:
        adata: AnnData object with raw metadata.
        acute_cutoff: Acute inflammation cutoff in hours.
        subacute_cutoff: Subacute inflammation cutoff in hours.

    Returns:
        Cleaned AnnData object with derived columns.
    """
    required = ["title", "source_name_ch1", "characteristics_ch1", "description"]
    if not _check_columns(adata, required, "mouse_burn"):
        return adata

    # Select columns
    adata.obs = adata.obs[required].copy()
    adata.obs = adata.obs.rename(columns={"source_name_ch1": "cell_type"})

    # Split characteristics_ch1 into sex and strain
    adata.obs[["sex", "strain"]] = adata.obs["characteristics_ch1"].str.split(
        " ", n=1, expand=True
    )
    adata.obs = adata.obs.drop(columns=["characteristics_ch1", "description"])

    # Extract time_point_hours, group, and patient_id from title
    time_extract = adata.obs["title"].str.extract(r"(\d+)\s*(hr|day)")
    adata.obs["time_point_hours"] = time_extract.apply(
        lambda x: float(x[0]) if x[1] == "hr" else float(x[0]) * 24 if pd.notna(x[0]) else np.nan,
        axis=1,
    )
    adata.obs["group"] = adata.obs["title"].apply(
        lambda x: "control" if "sham" in str(x).lower() else "inflammation"
    )
    # Extract patient ID (rep number)
    patient_ids = adata.obs["title"].str.extract(r"rep\s*(\d+)")
    if patient_ids.notna().any().any():
        adata.obs["patient_id"] = (
            pd.to_numeric(patient_ids.iloc[:, 0], errors="coerce").fillna(0).astype(int)
        )
    else:
        adata.obs["patient_id"] = 0

    # Derived columns
    adata.obs["takao_inflamed"] = (adata.obs["group"] == "inflammation") & (
        adata.obs["time_point_hours"] >= 150
    )
    adata.obs["takao_control"] = adata.obs["group"] == "control"
    adata.obs["takao_status"] = pd.NA
    adata.obs.loc[adata.obs["takao_inflamed"], "takao_status"] = "takao_inflamed"
    adata.obs.loc[adata.obs["takao_control"], "takao_status"] = "takao_control"
    adata.obs.loc[adata.obs["group"] == "control", "time_point_hours"] = pd.NA
    adata.obs["takao_status"] = adata.obs["takao_status"].astype("category")

    # Inflammation categories
    adata.obs["infl_acute"] = (adata.obs["group"] != "control") & (
        adata.obs["time_point_hours"] < acute_cutoff
    )
    adata.obs["infl_subacute"] = (
        (adata.obs["group"] != "control")
        & (adata.obs["time_point_hours"] >= acute_cutoff)
        & (adata.obs["time_point_hours"] < subacute_cutoff)
    )
    adata.obs["infl_chronic"] = (adata.obs["group"] != "control") & (
        adata.obs["time_point_hours"] >= subacute_cutoff
    )

    return adata


def preprocess_mouse_trauma(
    adata: ad.AnnData, acute_cutoff: int, subacute_cutoff: int
) -> ad.AnnData:
    """Clean and preprocess mouse trauma dataset.

    Args:
        adata: AnnData object with raw metadata.
        acute_cutoff: Acute inflammation cutoff in hours.
        subacute_cutoff: Subacute inflammation cutoff in hours.

    Returns:
        Cleaned AnnData object with derived columns.
    """
    required = ["title", "source_name_ch1", "characteristics_ch1", "description"]
    if not _check_columns(adata, required, "mouse_trauma"):
        return adata

    # Select columns
    adata.obs = adata.obs[required].copy()
    adata.obs = adata.obs.rename(columns={"source_name_ch1": "cell_type"})

    # Split characteristics_ch1 into sex and strain
    adata.obs[["sex", "strain"]] = adata.obs["characteristics_ch1"].str.split(
        " ", n=1, expand=True
    )
    adata.obs = adata.obs.drop(columns=["characteristics_ch1", "description"])

    # Extract time_point_hours, group, and patient_id from title
    time_extract = adata.obs["title"].str.extract(r"(\d+)\s*(hr|day)")
    adata.obs["time_point_hours"] = time_extract.apply(
        lambda x: float(x[0]) if x[1] == "hr" else float(x[0]) * 24 if pd.notna(x[0]) else np.nan,
        axis=1,
    )
    adata.obs["group"] = adata.obs["title"].apply(
        lambda x: "control" if "sham" in str(x).lower() else "inflammation"
    )
    # Extract patient ID (rep number)
    patient_ids = adata.obs["title"].str.extract(r"rep\s*(\d+)")
    if patient_ids.notna().any().any():
        adata.obs["patient_id"] = (
            pd.to_numeric(patient_ids.iloc[:, 0], errors="coerce").fillna(0).astype(int)
        )
    else:
        adata.obs["patient_id"] = 0

    # Derived columns
    adata.obs["takao_inflamed"] = (adata.obs["group"] == "inflammation") & (
        adata.obs["time_point_hours"] > 72
    )
    adata.obs["takao_control"] = adata.obs["group"] == "control"
    adata.obs["takao_status"] = pd.NA
    adata.obs.loc[adata.obs["takao_inflamed"], "takao_status"] = "takao_inflamed"
    adata.obs.loc[adata.obs["takao_control"], "takao_status"] = "takao_control"
    adata.obs.loc[adata.obs["group"] == "control", "time_point_hours"] = pd.NA
    adata.obs["takao_status"] = adata.obs["takao_status"].astype("category")

    # Inflammation categories
    adata.obs["infl_acute"] = (adata.obs["group"] != "control") & (
        adata.obs["time_point_hours"] < acute_cutoff
    )
    adata.obs["infl_subacute"] = (
        (adata.obs["group"] != "control")
        & (adata.obs["time_point_hours"] >= acute_cutoff)
        & (adata.obs["time_point_hours"] < subacute_cutoff)
    )
    adata.obs["infl_chronic"] = (adata.obs["group"] != "control") & (
        adata.obs["time_point_hours"] >= subacute_cutoff
    )

    return adata


def preprocess_mouse_sepsis(
    adata: ad.AnnData, acute_cutoff: int, subacute_cutoff: int
) -> ad.AnnData:
    """Clean and preprocess mouse sepsis dataset.

    Args:
        adata: AnnData object with raw metadata.
        acute_cutoff: Acute inflammation cutoff in hours.
        subacute_cutoff: Subacute inflammation cutoff in hours.

    Returns:
        Cleaned AnnData object with derived columns.
    """
    required = ["infection:ch1", "strain:ch1", "time point:ch1", "title"]
    if not _check_columns(adata, required, "mouse_sepsis"):
        return adata

    # Select columns
    adata.obs = adata.obs[required].copy()
    adata.obs.columns = ["group", "strain", "time_point", "title"]

    # Clean group column
    adata.obs["group"] = adata.obs["group"].apply(
        lambda x: "control" if str(x).lower() == "none" else "inflammation"
    )

    # Extract time point and convert to float
    adata.obs["time_point"] = adata.obs["time_point"].str.extract(r"(\d+\.?\d*)").astype(float)
    adata.obs = adata.obs.rename(columns={"time_point": "time_point_hours"})

    # Extract patient ID from title (last number)
    patient_ids = adata.obs["title"].str.extract(r"(\d+)$")
    if patient_ids.notna().any().any():
        adata.obs["patient_id"] = patient_ids.iloc[:, 0].astype(int)
    else:
        adata.obs["patient_id"] = 0  # Default if no patient ID found
    adata.obs = adata.obs.drop(columns=["title"])

    # Derived columns
    has_strain = "strain" in adata.obs.columns
    if has_strain:
        adata.obs["takao_inflamed"] = (
            (adata.obs["group"] == "inflammation")
            & (adata.obs["time_point_hours"] == 4.0)
            & (adata.obs["strain"] == "C57BL/6J")
        )
        adata.obs["takao_control"] = (adata.obs["group"] == "control") & (
            adata.obs["strain"] == "C57BL/6J"
        )
    else:
        adata.obs["takao_inflamed"] = (adata.obs["group"] == "inflammation") & (
            adata.obs["time_point_hours"] == 4.0
        )
        adata.obs["takao_control"] = adata.obs["group"] == "control"

    adata.obs["takao_status"] = pd.NA
    adata.obs.loc[adata.obs["takao_inflamed"], "takao_status"] = "takao_inflamed"
    adata.obs.loc[adata.obs["takao_control"], "takao_status"] = "takao_control"
    adata.obs["takao_status"] = adata.obs["takao_status"].astype("category")

    # Inflammation categories
    adata.obs["infl_acute"] = (adata.obs["group"] != "control") & (
        adata.obs["time_point_hours"] < acute_cutoff
    )
    adata.obs["infl_subacute"] = (
        (adata.obs["group"] != "control")
        & (adata.obs["time_point_hours"] >= acute_cutoff)
        & (adata.obs["time_point_hours"] < subacute_cutoff)
    )
    adata.obs["infl_chronic"] = (adata.obs["group"] != "control") & (
        adata.obs["time_point_hours"] >= subacute_cutoff
    )

    return adata


def preprocess_mouse_infection(
    adata: ad.AnnData, acute_cutoff: int, subacute_cutoff: int
) -> ad.AnnData:
    """Clean and preprocess mouse infection dataset.

    Args:
        adata: AnnData object with raw metadata.
        acute_cutoff: Acute inflammation cutoff in hours.
        subacute_cutoff: Subacute inflammation cutoff in hours.

    Returns:
        Cleaned AnnData object with derived columns.
    """
    required = [
        "age:ch1",
        "genotype:ch1",
        "infection duration (days):ch1",
        "infection status:ch1",
        "tissue:ch1",
    ]
    if not _check_columns(adata, required, "mouse_infection"):
        return adata

    # Select columns
    adata.obs = adata.obs[required].copy()
    adata.obs.columns = [
        "age",
        "genotype",
        "infection_duration_days",
        "infection_status",
        "tissue",
    ]

    # Add group column based on infection_status
    adata.obs["group"] = adata.obs["infection_status"].apply(
        lambda x: "control" if str(x).lower() == "healthy" else "inflammation"
    )

    # Rename columns
    adata.obs = adata.obs.rename(
        columns={"infection_status": "infection_status_detail", "genotype": "strain"}
    )

    # Convert infection_duration_days to time_point_hours
    adata.obs["time_point_hours"] = adata.obs["infection_duration_days"].astype(float) * 24
    adata.obs = adata.obs.drop(columns=["infection_duration_days"])

    # Derived columns
    has_infection_status = "infection_status_detail" in adata.obs.columns
    if has_infection_status:
        adata.obs["takao_inflamed"] = (
            (adata.obs["group"] == "inflammation")
            & (adata.obs["time_point_hours"] == 24)
            & (adata.obs["infection_status_detail"] == "candida")
        )
    else:
        adata.obs["takao_inflamed"] = (adata.obs["group"] == "inflammation") & (
            adata.obs["time_point_hours"] == 24
        )

    adata.obs["takao_control"] = adata.obs["group"] == "control"
    adata.obs["takao_status"] = pd.NA
    adata.obs.loc[adata.obs["takao_inflamed"], "takao_status"] = "takao_inflamed"
    adata.obs.loc[adata.obs["takao_control"], "takao_status"] = "takao_control"
    adata.obs.loc[adata.obs["group"] == "control", "time_point_hours"] = pd.NA
    adata.obs["takao_status"] = adata.obs["takao_status"].astype("category")

    # Inflammation categories
    adata.obs["infl_acute"] = (adata.obs["group"] != "control") & (
        adata.obs["time_point_hours"] < acute_cutoff
    )
    adata.obs["infl_subacute"] = (
        (adata.obs["group"] != "control")
        & (adata.obs["time_point_hours"] >= acute_cutoff)
        & (adata.obs["time_point_hours"] < subacute_cutoff)
    )
    adata.obs["infl_chronic"] = (adata.obs["group"] != "control") & (
        adata.obs["time_point_hours"] >= subacute_cutoff
    )

    return adata


def preprocess_all_datasets(adatas: dict[str, ad.AnnData]) -> None:
    """Preprocess all datasets with cleaning and derived columns.

    Args:
        adatas: Dictionary of dataset names to AnnData objects.
    """
    config = get_config()
    time_cutoffs = config["time_cutoffs"]

    human_acute = time_cutoffs["human_acute_cutoff_hours"]
    human_subacute = time_cutoffs["human_subacute_cutoff_hours"]
    mouse_acute = time_cutoffs["mouse_acute_cutoff_hours"]
    mouse_subacute = time_cutoffs["mouse_subacute_cutoff_hours"]

    logger.info("Preprocessing datasets...")

    # Human datasets
    if "human_burn" in adatas:
        logger.info("Preprocessing human_burn...")
        adatas["human_burn"] = preprocess_human_burn(
            adatas["human_burn"], human_acute, human_subacute
        )

    if "human_trauma" in adatas:
        logger.info("Preprocessing human_trauma...")
        adatas["human_trauma"] = preprocess_human_trauma(
            adatas["human_trauma"], human_acute, human_subacute
        )

    if "human_sepsis" in adatas:
        logger.info("Preprocessing human_sepsis...")
        adatas["human_sepsis"] = preprocess_human_sepsis(adatas["human_sepsis"])

    # Mouse datasets
    if "mouse_burn" in adatas:
        logger.info("Preprocessing mouse_burn...")
        adatas["mouse_burn"] = preprocess_mouse_burn(
            adatas["mouse_burn"], mouse_acute, mouse_subacute
        )

    if "mouse_trauma" in adatas:
        logger.info("Preprocessing mouse_trauma...")
        adatas["mouse_trauma"] = preprocess_mouse_trauma(
            adatas["mouse_trauma"], mouse_acute, mouse_subacute
        )

    if "mouse_sepsis" in adatas:
        logger.info("Preprocessing mouse_sepsis...")
        adatas["mouse_sepsis"] = preprocess_mouse_sepsis(
            adatas["mouse_sepsis"], mouse_acute, mouse_subacute
        )

    if "mouse_infection" in adatas:
        logger.info("Preprocessing mouse_infection...")
        adatas["mouse_infection"] = preprocess_mouse_infection(
            adatas["mouse_infection"], mouse_acute, mouse_subacute
        )

    # Fill and transform inflammation categories
    for adata in adatas.values():
        if "infl_acute" in adata.obs.columns:
            adata.obs["infl_acute"] = adata.obs["infl_acute"].fillna(False).astype(bool)
            adata.obs["infl_subacute"] = adata.obs["infl_subacute"].fillna(False).astype(bool)
            adata.obs["infl_chronic"] = adata.obs["infl_chronic"].fillna(False).astype(bool)

    # Set index to symbols and ensure both symbol and ensembl are in .var
    for dataset_name, adata in adatas.items():
        adatas[dataset_name] = set_symbol_index(adata, dataset_name)

    logger.success("Preprocessing complete")
