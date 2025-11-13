"""Unified data processing pipeline.

This module orchestrates the complete data processing pipeline from raw CEL files
to cleaned, ortholog-mapped AnnData objects ready for analysis.
"""

from pathlib import Path

from loguru import logger

from inflamm_debate_fm.config import (
    METADATA_DIR,
    RMA_PROCESSED_DIR,
    get_config,
)
from inflamm_debate_fm.data.clean import preprocess_all_datasets
from inflamm_debate_fm.data.constants import GSE_IDS
from inflamm_debate_fm.data.download_raw import download_all_raw_data
from inflamm_debate_fm.data.extract_metadata import extract_all_metadata
from inflamm_debate_fm.data.load import load_adatas
from inflamm_debate_fm.data.orthologs import process_orthologs
from inflamm_debate_fm.data.rma import convert_to_anndata, run_rma_preprocessing


def run_pipeline():
    """Run complete data processing pipeline.

    Pipeline steps:
    1. Download raw data (CEL files, orthology files, BrainArray packages)
    2. Extract metadata from GEO datasets
    3. RMA preprocessing using R script (if needed)
    4. Convert to AnnData format
    5. Map orthologs (mouse -> human, human -> Ensembl)
    6. Clean metadata and add derived columns
    7. Save cleaned AnnData objects
    """
    config = get_config()
    geo_download_dir = Path(config["paths"]["geo_download_dir"])
    geo_download_dir.mkdir(parents=True, exist_ok=True)

    expected_gse_ids = set(GSE_IDS.values())
    logger.info(f"Processing {len(expected_gse_ids)} datasets: {sorted(expected_gse_ids)}")

    # Step 1: Download raw data
    logger.info("Step 1: Downloading raw data...")
    download_all_raw_data()

    # Step 2: Extract metadata
    logger.info("Step 2: Extracting metadata...")
    extract_all_metadata(geo_download_dir, METADATA_DIR)

    # Step 3: RMA preprocessing
    missing_rma = [
        gse_id for gse_id in expected_gse_ids if not (RMA_PROCESSED_DIR / gse_id).exists()
    ]
    if missing_rma:
        logger.info(f"Step 3: Running RMA preprocessing for {len(missing_rma)} datasets...")
        run_rma_preprocessing()

    # Step 4: Convert to AnnData (always regenerate to ensure metadata is current)
    logger.info("Step 4: Converting to AnnData format...")
    convert_to_anndata()

    # Step 5: Ortholog mapping (process_orthologs skips existing files)
    logger.info("Step 5: Mapping orthologs...")
    ann_data_dir = Path(config["paths"]["ann_data_dir"])
    ann_data_dir.mkdir(parents=True, exist_ok=True)
    process_orthologs()

    # Step 6: Clean and save
    anndata_cleaned_dir = Path(config["paths"]["anndata_cleaned_dir"])
    anndata_cleaned_dir.mkdir(parents=True, exist_ok=True)

    expected_datasets = [
        "human_burn",
        "human_trauma",
        "human_sepsis",
        "mouse_burn",
        "mouse_trauma",
        "mouse_sepsis",
        "mouse_infection",
    ]

    missing_cleaned = [
        dataset
        for dataset in expected_datasets
        if not (anndata_cleaned_dir / f"{dataset}.h5ad").exists()
    ]

    if missing_cleaned:
        logger.info(f"Step 6: Cleaning and preprocessing {len(missing_cleaned)} datasets...")
        embeddings_dir = Path(config["paths"]["embeddings_dir"])
        adatas = load_adatas(ann_data_dir, embeddings_dir, load_embeddings=False)
        preprocess_all_datasets(adatas)

        for dataset_name in missing_cleaned:
            if dataset_name in adatas:
                output_path = anndata_cleaned_dir / f"{dataset_name}.h5ad"
                adatas[dataset_name].write_h5ad(output_path)
                logger.success(
                    f"Saved {dataset_name}: {adatas[dataset_name].shape[0]} samples × {adatas[dataset_name].shape[1]} genes"
                )
    else:
        logger.info("Step 6: All cleaned datasets already exist")

    logger.success("Pipeline complete!")
