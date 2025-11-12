"""Gene Set Enrichment Analysis (GSEA) functions."""

from pathlib import Path

import gseapy as gp
from loguru import logger
import pandas as pd

from inflamm_debate_fm.config import get_config


def run_prerank_from_coef_df(
    df_coef: pd.DataFrame,
    outdir: str | Path,
    score_col: str = "coef_norm",
    min_size: int = 15,
    max_size: int = 500,
    permutation_num: int = 1000,
) -> dict:
    """Run prerank GSEA from coefficient DataFrame.

    Args:
        df_coef: DataFrame with genes as index and coefficients in score_col.
        outdir: Output directory for GSEA results.
        score_col: Column name containing ranked scores.
        min_size: Minimum gene set size.
        max_size: Maximum gene set size.
        permutation_num: Number of permutations for GSEA.

    Returns:
        Dictionary mapping gene set names to GSEA results.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    config = get_config()
    analysis_config = config.get("analysis", {})
    min_size = analysis_config.get("gsea_min_size", min_size)
    max_size = analysis_config.get("gsea_max_size", max_size)
    permutation_num = analysis_config.get("gsea_permutation_num", permutation_num)

    # Create ranked list
    rnk = df_coef[score_col].sort_values(ascending=False)
    rnk_path = outdir / "ranked_list.rnk"
    rnk.to_csv(rnk_path, sep="\t", header=False)

    gene_sets = ["MSigDB_Hallmark_2020", "GO_Biological_Process_2025", "Reactome_Pathways_2024"]

    results = {}
    for gs in gene_sets:
        try:
            logger.info(f"Running GSEA for {gs}...")
            prer = gp.prerank(
                rnk=str(rnk_path),
                gene_sets=gs,
                permutation_num=permutation_num,
                outdir=str(outdir / gs.replace("/", "_")),
                seed=42,
                min_size=min_size,
                max_size=max_size,
                no_plot=True,
            )
            results[gs] = prer
        except Exception as e:
            logger.error(f"prerank failed for {gs}: {e}")

    return results
