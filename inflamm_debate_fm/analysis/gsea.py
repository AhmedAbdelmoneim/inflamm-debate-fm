"""Gene Set Enrichment Analysis (GSEA) functions."""

from pathlib import Path

import gseapy as gp
from loguru import logger
import pandas as pd
from tqdm import tqdm

from inflamm_debate_fm.config import get_config
from inflamm_debate_fm.modeling.coefficients import (
    build_coef_matrix,
    load_all_coefficients,
    normalize_coefficients,
)


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


def post_analysis_from_coeffs(
    coeff_dir: str | Path,
    outdir: str | Path,
) -> dict:
    """Run post-analysis pipeline on coefficients: normalization, GSEA, and summary.

    This function:
    1. Loads all coefficient files from a directory
    2. Normalizes coefficients within each setup/direction
    3. Builds coefficient matrices
    4. Runs GSEA analysis for each setup and direction
    5. Returns summary results

    Args:
        coeff_dir: Directory containing coefficient CSV files.
        outdir: Output directory for GSEA results.

    Returns:
        Dictionary with normalized coefficients, matrices, and GSEA summary.
    """
    coeff_dir = Path(coeff_dir)
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading coefficients from {coeff_dir}...")
    all_coefs = load_all_coefficients(coeff_dir)

    logger.info("Normalizing coefficients...")
    norm_coefs = normalize_coefficients(all_coefs)

    logger.info("Building coefficient matrices...")
    mat_h2m = build_coef_matrix(norm_coefs, direction="HumanToMouse")
    mat_m2h = build_coef_matrix(norm_coefs, direction="MouseToHuman")

    # Get common genes across all setups
    common_genes = set(mat_h2m.index) & set(mat_m2h.index)
    logger.info(f"Found {len(common_genes)} common genes")

    # GSEA for each setup and direction
    logger.info("Running GSEA analysis...")
    gsea_summary = {}
    setups = list(norm_coefs.keys())

    for setup in tqdm(setups, desc="GSEA across setups"):
        gsea_summary[setup] = {}
        for direction in norm_coefs[setup]:
            # Prepare coefficient DataFrame for GSEA
            df = norm_coefs[setup][direction][["coefficient", "coef_norm"]].copy()
            df.index.name = "gene"

            # Create output directory for this setup/direction
            setup_outdir = outdir / f"gsea_{setup}_{direction}_coeffs_{direction}"
            setup_outdir.mkdir(parents=True, exist_ok=True)

            # Run GSEA
            try:
                prerank_results = run_prerank_from_coef_df(
                    df_coef=df,
                    outdir=setup_outdir,
                    score_col="coef_norm",
                )

                # Extract results summary
                results_summary = {}
                for gs, prer in prerank_results.items():
                    try:
                        res_df = prer.res2d.reset_index().rename(
                            columns={
                                "Term": "Term",
                                "NES": "NES",
                                "FDR q-val": "FDR",
                                "Lead_genes": "Genes",
                            }
                        )
                    except Exception:
                        res_df = prer.res2d
                    results_summary[gs] = res_df

                gsea_summary[setup][direction] = results_summary
                logger.info(f"Completed GSEA for {setup} ({direction})")

            except Exception as e:
                logger.error(f"Failed to run GSEA for {setup} ({direction}): {e}")
                gsea_summary[setup][direction] = {}

    logger.success(f"GSEA analysis complete. Results saved to {outdir}")

    return {
        "normalized_coeffs": norm_coefs,
        "matrices": {"HumanToMouse": mat_h2m, "MouseToHuman": mat_m2h},
        "gsea_summary": gsea_summary,
        "common_genes": common_genes,
        "outdir": outdir,
    }
