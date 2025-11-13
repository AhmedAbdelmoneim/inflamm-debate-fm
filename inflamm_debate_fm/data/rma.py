"""RMA preprocessing of raw CEL files using BrainArray CDF packages.

This module handles RMA preprocessing via an R script and converts the results to AnnData format.
"""

import os
import subprocess

from loguru import logger
import pandas as pd

from inflamm_debate_fm.config import (
    ANNDATA_RAW_DIR,
    DATA_ROOT,
    METADATA_DIR,
    PROJ_ROOT,
    RMA_PROCESSED_DIR,
)


def run_rma_preprocessing():
    """Run RMA preprocessing using R script."""
    r_script_path = PROJ_ROOT / "scripts" / "preprocess_rma_brainarray.R"
    if not r_script_path.exists():
        raise FileNotFoundError(f"R script not found: {r_script_path}")

    env = os.environ.copy()
    env["INFLAMM_DEBATE_FM_DATA_ROOT"] = str(DATA_ROOT)
    logger.info("Running RMA preprocessing...")
    subprocess.run(
        ["Rscript", str(r_script_path)],
        cwd=str(PROJ_ROOT),
        env=env,
        check=True,
    )


def convert_to_anndata():
    """Convert processed expression matrices to AnnData format."""
    import anndata as ad

    ANNDATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    gse_to_datasets = {
        "GSE37069": "human_burn",
        "GSE36809": "human_trauma",
        "GSE28750": "human_sepsis",
        "GSE19668": "mouse_sepsis",
        "GSE20524": "mouse_infection",
    }

    for dataset_dir in RMA_PROCESSED_DIR.iterdir():
        if not dataset_dir.is_dir():
            continue
        gse_id = dataset_dir.name
        expr_file = dataset_dir / f"{gse_id}_gene_matrix.csv"
        if not expr_file.exists():
            continue

        metadata_file = METADATA_DIR / f"{gse_id}.csv"
        if not metadata_file.exists():
            logger.warning(f"Metadata file not found for {gse_id}: {metadata_file}")
            continue

        expr_df = pd.read_csv(expr_file, index_col=0).astype(float)
        meta_df = pd.read_csv(metadata_file, dtype=str, index_col="geo_accession")

        # Align expression and metadata
        expr_df.columns = expr_df.columns.str.extract(r"(GSM\d+)", expand=False)
        expr_df.index = expr_df.index.str.replace(r"_at$", "", regex=True)
        meta_aligned = meta_df.reindex(expr_df.columns)
        keep = ~meta_aligned.index.isna()
        expr_df = expr_df.loc[:, keep]
        meta_aligned = meta_aligned.loc[keep]

        if gse_id == "GSE7404":
            # Split based on description column
            if "description" in meta_aligned.columns:
                burn_mask = meta_aligned["description"].str.contains(
                    "burn injury model", case=False, na=False
                )
                trauma_mask = meta_aligned["description"].str.contains(
                    "trauma hemorrage injury model", case=False, na=False
                )
                if burn_mask.any():
                    ad.AnnData(
                        X=expr_df.loc[:, burn_mask].T.values,
                        obs=meta_aligned.loc[burn_mask].astype(str),
                        var=pd.DataFrame(index=expr_df.index),
                    ).write(ANNDATA_RAW_DIR / "GSE7404_mouse_burn_gene_rma_brainarray.h5ad")
                    logger.success(f"Saved mouse_burn: {burn_mask.sum()} samples")
                if trauma_mask.any():
                    ad.AnnData(
                        X=expr_df.loc[:, trauma_mask].T.values,
                        obs=meta_aligned.loc[trauma_mask].astype(str),
                        var=pd.DataFrame(index=expr_df.index),
                    ).write(ANNDATA_RAW_DIR / "GSE7404_mouse_trauma_gene_rma_brainarray.h5ad")
                    logger.success(f"Saved mouse_trauma: {trauma_mask.sum()} samples")
        elif gse_id in gse_to_datasets:
            dataset_name = gse_to_datasets[gse_id]
            ad.AnnData(
                X=expr_df.T.values,
                obs=meta_aligned.astype(str),
                var=pd.DataFrame(index=expr_df.index),
            ).write(ANNDATA_RAW_DIR / f"{gse_id}_{dataset_name}_gene_rma_brainarray.h5ad")
            logger.success(f"Saved {dataset_name}: {expr_df.shape[1]} samples")
