"""Data preprocessing functions."""

import anndata as ad
from loguru import logger
import pandas as pd

from inflamm_debate_fm.config import get_config


def _parse_group_from_source(adata: ad.AnnData) -> None:
    """Parse group from source_name_ch1 if not present."""
    if "group" not in adata.obs.columns:
        if "source_name_ch1" in adata.obs.columns:
            adata.obs["group"] = (
                adata.obs["source_name_ch1"]
                .str.contains("control", case=False, na=False)
                .map({True: "control", False: "inflammation"})
            )
        else:
            logger.warning("Cannot parse group: source_name_ch1 not found")
            adata.obs["group"] = "unknown"


def _parse_time_point_from_source(adata: ad.AnnData) -> None:
    """Parse time_point_hours from source_name_ch1 if not present."""
    if "time_point_hours" not in adata.obs.columns:
        if "source_name_ch1" in adata.obs.columns:
            time_pattern = r"(\d+)\s*(?:hours?|hrs?|h)"
            time_points = adata.obs["source_name_ch1"].str.extract(time_pattern, expand=False)
            adata.obs["time_point_hours"] = pd.to_numeric(time_points, errors="coerce")
        else:
            logger.warning("Cannot parse time_point_hours: source_name_ch1 not found")
            adata.obs["time_point_hours"] = pd.NA


def fill_and_transform_infl_categories(adatas: dict[str, ad.AnnData]) -> None:
    """Fill and transform inflammation categories for all adatas.

    Args:
        adatas: Dictionary of AnnData objects to process.
    """
    for adata in adatas.values():
        if "infl_acute" not in adata.obs.columns:
            continue

        # Set inflammation category columns to boolean
        adata.obs["infl_acute"] = adata.obs["infl_acute"].fillna(False).astype(bool)
        adata.obs["infl_subacute"] = adata.obs["infl_subacute"].fillna(False).astype(bool)
        adata.obs["infl_chronic"] = adata.obs["infl_chronic"].fillna(False).astype(bool)


def preprocess_human_burn(
    adata: ad.AnnData, human_acute_cutoff: int, human_subacute_cutoff: int
) -> None:
    """Preprocess human burn dataset.

    Args:
        adata: AnnData object for human_burn.
        human_acute_cutoff: Acute cutoff in hours.
        human_subacute_cutoff: Subacute cutoff in hours.
    """
    # Parse group and time_point_hours from source_name_ch1 if not present
    _parse_group_from_source(adata)
    _parse_time_point_from_source(adata)

    # Set age if present, otherwise skip
    if "age" in adata.obs.columns:
        adata.obs["age"] = pd.to_numeric(adata.obs["age"], errors="coerce")

    adata.obs.loc[adata.obs["group"] == "control", "time_point_hours"] = pd.NA
    adata.obs["takao_inflamed"] = adata.obs["group"] == "inflammation"
    adata.obs["takao_control"] = adata.obs["group"] == "control"
    adata.obs["takao_status"] = pd.NA
    adata.obs.loc[adata.obs["takao_inflamed"], "takao_status"] = "takao_inflamed"
    adata.obs.loc[adata.obs["takao_control"], "takao_status"] = "takao_control"
    adata.obs["takao_status"] = adata.obs["takao_status"].astype("category")

    adata.obs["infl_acute"] = (adata.obs["group"] != "control") & (
        adata.obs["time_point_hours"] < human_acute_cutoff
    )
    adata.obs["infl_subacute"] = (
        (adata.obs["group"] != "control")
        & (adata.obs["time_point_hours"] >= human_acute_cutoff)
        & (adata.obs["time_point_hours"] < human_subacute_cutoff)
    )
    adata.obs["infl_chronic"] = (adata.obs["group"] != "control") & (
        adata.obs["time_point_hours"] >= human_subacute_cutoff
    )


def preprocess_human_trauma(
    adata: ad.AnnData, human_acute_cutoff: int, human_subacute_cutoff: int
) -> None:
    """Preprocess human trauma dataset.

    Args:
        adata: AnnData object for human_trauma.
        human_acute_cutoff: Acute cutoff in hours.
        human_subacute_cutoff: Subacute cutoff in hours.
    """
    # Parse group and time_point_hours from source_name_ch1 if not present
    _parse_group_from_source(adata)
    _parse_time_point_from_source(adata)

    # Set age if present, otherwise skip
    if "age" in adata.obs.columns:
        adata.obs["age"] = pd.to_numeric(adata.obs["age"], errors="coerce")

    adata.obs["takao_inflamed"] = (adata.obs["group"] == "inflammation") & (
        adata.obs["time_point_hours"] > (14 * 24)
    )
    adata.obs["takao_control"] = adata.obs["group"] == "control"
    adata.obs["takao_status"] = pd.NA
    adata.obs.loc[adata.obs["takao_inflamed"], "takao_status"] = "takao_inflamed"
    adata.obs.loc[adata.obs["takao_control"], "takao_status"] = "takao_control"
    adata.obs.loc[adata.obs["group"] == "control", "time_point_hours"] = pd.NA
    adata.obs["takao_status"] = adata.obs["takao_status"].astype("category")

    adata.obs["infl_acute"] = (adata.obs["group"] != "control") & (
        adata.obs["time_point_hours"] <= human_acute_cutoff
    )
    adata.obs["infl_subacute"] = (
        (adata.obs["group"] != "control")
        & (adata.obs["time_point_hours"] > human_acute_cutoff)
        & (adata.obs["time_point_hours"] <= human_subacute_cutoff)
    )
    adata.obs["infl_chronic"] = (adata.obs["group"] != "control") & (
        adata.obs["time_point_hours"] > human_subacute_cutoff
    )


def preprocess_human_sepsis(adata: ad.AnnData) -> None:
    """Preprocess human sepsis dataset.

    Args:
        adata: AnnData object for human_sepsis.
    """
    # Parse group from source_name_ch1 if not present
    _parse_group_from_source(adata)

    adata.obs["takao_inflamed"] = adata.obs["group"] == "inflammation"
    adata.obs["takao_control"] = adata.obs["group"] == "control"
    adata.obs["takao_status"] = pd.NA
    adata.obs.loc[adata.obs["takao_inflamed"], "takao_status"] = "takao_inflamed"
    adata.obs.loc[adata.obs["takao_control"], "takao_status"] = "takao_control"
    adata.obs["takao_status"] = adata.obs["takao_status"].astype("category")


def preprocess_mouse_burn(
    adata: ad.AnnData, mouse_acute_cutoff: int, mouse_subacute_cutoff: int
) -> None:
    """Preprocess mouse burn dataset.

    Args:
        adata: AnnData object for mouse_burn.
        mouse_acute_cutoff: Acute cutoff in hours.
        mouse_subacute_cutoff: Subacute cutoff in hours.
    """
    # Parse group and time_point_hours from source_name_ch1 if not present
    _parse_group_from_source(adata)
    _parse_time_point_from_source(adata)

    adata.obs["takao_inflamed"] = (adata.obs["group"] == "inflammation") & (
        adata.obs["time_point_hours"] >= 150
    )
    adata.obs["takao_control"] = adata.obs["group"] == "control"
    adata.obs["takao_status"] = pd.NA
    adata.obs.loc[adata.obs["takao_inflamed"], "takao_status"] = "takao_inflamed"
    adata.obs.loc[adata.obs["takao_control"], "takao_status"] = "takao_control"
    adata.obs.loc[adata.obs["group"] == "control", "time_point_hours"] = pd.NA
    adata.obs["takao_status"] = adata.obs["takao_status"].astype("category")

    adata.obs["infl_acute"] = (adata.obs["group"] != "control") & (
        adata.obs["time_point_hours"] < mouse_acute_cutoff
    )
    adata.obs["infl_subacute"] = (
        (adata.obs["group"] != "control")
        & (adata.obs["time_point_hours"] >= mouse_acute_cutoff)
        & (adata.obs["time_point_hours"] < mouse_subacute_cutoff)
    )
    adata.obs["infl_chronic"] = (adata.obs["group"] != "control") & (
        adata.obs["time_point_hours"] >= mouse_subacute_cutoff
    )


def preprocess_mouse_trauma(
    adata: ad.AnnData, mouse_acute_cutoff: int, mouse_subacute_cutoff: int
) -> None:
    """Preprocess mouse trauma dataset.

    Args:
        adata: AnnData object for mouse_trauma.
        mouse_acute_cutoff: Acute cutoff in hours.
        mouse_subacute_cutoff: Subacute cutoff in hours.
    """
    # Parse group and time_point_hours from source_name_ch1 if not present
    _parse_group_from_source(adata)
    _parse_time_point_from_source(adata)

    adata.obs["takao_inflamed"] = (adata.obs["group"] == "inflammation") & (
        adata.obs["time_point_hours"] > 72
    )
    adata.obs["takao_control"] = adata.obs["group"] == "control"
    adata.obs["takao_status"] = pd.NA
    adata.obs.loc[adata.obs["takao_inflamed"], "takao_status"] = "takao_inflamed"
    adata.obs.loc[adata.obs["takao_control"], "takao_status"] = "takao_control"
    adata.obs.loc[adata.obs["group"] == "control", "time_point_hours"] = pd.NA
    adata.obs["takao_status"] = adata.obs["takao_status"].astype("category")

    adata.obs["infl_acute"] = (adata.obs["group"] != "control") & (
        adata.obs["time_point_hours"] < mouse_acute_cutoff
    )
    adata.obs["infl_subacute"] = (
        (adata.obs["group"] != "control")
        & (adata.obs["time_point_hours"] >= mouse_acute_cutoff)
        & (adata.obs["time_point_hours"] < mouse_subacute_cutoff)
    )
    adata.obs["infl_chronic"] = (adata.obs["group"] != "control") & (
        adata.obs["time_point_hours"] >= mouse_subacute_cutoff
    )


def preprocess_mouse_sepsis(
    adata: ad.AnnData, mouse_acute_cutoff: int, mouse_subacute_cutoff: int
) -> None:
    """Preprocess mouse sepsis dataset.

    Args:
        adata: AnnData object for mouse_sepsis.
        mouse_acute_cutoff: Acute cutoff in hours.
        mouse_subacute_cutoff: Subacute cutoff in hours.
    """
    # Parse group from source_name_ch1 if not present
    _parse_group_from_source(adata)

    # Handle time_point column - convert to time_point_hours if present
    if "time_point" in adata.obs.columns:
        adata.obs["time_point_hours"] = pd.to_numeric(adata.obs["time_point"], errors="coerce")
        adata.obs.loc[adata.obs["group"] == "control", "time_point"] = pd.NA
        adata.obs = adata.obs.drop(columns=["time_point"])
    elif "time_point_hours" not in adata.obs.columns:
        _parse_time_point_from_source(adata)

    # Check for strain column, otherwise skip strain filtering
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

    adata.obs["infl_acute"] = (adata.obs["group"] != "control") & (
        adata.obs["time_point_hours"] < mouse_acute_cutoff
    )
    adata.obs["infl_subacute"] = (
        (adata.obs["group"] != "control")
        & (adata.obs["time_point_hours"] >= mouse_acute_cutoff)
        & (adata.obs["time_point_hours"] < mouse_subacute_cutoff)
    )
    adata.obs["infl_chronic"] = (adata.obs["group"] != "control") & (
        adata.obs["time_point_hours"] >= mouse_subacute_cutoff
    )


def preprocess_mouse_infection(
    adata: ad.AnnData, mouse_acute_cutoff: int, mouse_subacute_cutoff: int
) -> None:
    """Preprocess mouse infection dataset.

    Args:
        adata: AnnData object for mouse_infection.
        mouse_acute_cutoff: Acute cutoff in hours.
        mouse_subacute_cutoff: Subacute cutoff in hours.
    """
    # Parse group and time_point_hours from source_name_ch1 if not present
    _parse_group_from_source(adata)
    _parse_time_point_from_source(adata)

    # Check for infection_status_detail column
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

    adata.obs["infl_acute"] = (adata.obs["group"] != "control") & (
        adata.obs["time_point_hours"] < mouse_acute_cutoff
    )
    adata.obs["infl_subacute"] = (
        (adata.obs["group"] != "control")
        & (adata.obs["time_point_hours"] >= mouse_acute_cutoff)
        & (adata.obs["time_point_hours"] < mouse_subacute_cutoff)
    )
    adata.obs["infl_chronic"] = (adata.obs["group"] != "control") & (
        adata.obs["time_point_hours"] >= mouse_subacute_cutoff
    )


def preprocess_all_datasets(adatas: dict[str, ad.AnnData]) -> None:
    """Preprocess all datasets with timepoint categorization.

    Args:
        adatas: Dictionary of AnnData objects to process.
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
        logger.info("Preprocessing human_burn")
        preprocess_human_burn(adatas["human_burn"], human_acute, human_subacute)

    if "human_trauma" in adatas:
        logger.info("Preprocessing human_trauma")
        preprocess_human_trauma(adatas["human_trauma"], human_acute, human_subacute)

    if "human_sepsis" in adatas:
        logger.info("Preprocessing human_sepsis")
        preprocess_human_sepsis(adatas["human_sepsis"])

    # Mouse datasets
    if "mouse_burn" in adatas:
        logger.info("Preprocessing mouse_burn")
        preprocess_mouse_burn(adatas["mouse_burn"], mouse_acute, mouse_subacute)

    if "mouse_trauma" in adatas:
        logger.info("Preprocessing mouse_trauma")
        preprocess_mouse_trauma(adatas["mouse_trauma"], mouse_acute, mouse_subacute)

    if "mouse_sepsis" in adatas:
        logger.info("Preprocessing mouse_sepsis")
        preprocess_mouse_sepsis(adatas["mouse_sepsis"], mouse_acute, mouse_subacute)

    if "mouse_infection" in adatas:
        logger.info("Preprocessing mouse_infection")
        preprocess_mouse_infection(adatas["mouse_infection"], mouse_acute, mouse_subacute)

    # Fill and transform inflammation categories
    fill_and_transform_infl_categories(adatas)
    logger.info("Preprocessing complete")
