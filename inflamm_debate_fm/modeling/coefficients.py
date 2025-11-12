"""Coefficient extraction and analysis for gene expression models."""

from collections import defaultdict
from pathlib import Path
import re
from typing import Dict, List, Tuple

import anndata as ad
from loguru import logger
import numpy as np
import pandas as pd
from sklearn.base import clone

from inflamm_debate_fm.modeling.pipelines import get_linear_pipeline_raw_only


def evaluate_linear_models(
    human_adata: ad.AnnData,
    mouse_adata: ad.AnnData,
    setups: List[Tuple[str, callable]],
    output_dir: str | Path = "model_coefficients",
) -> Tuple[Dict, Dict]:
    """Train and evaluate linear models on Raw gene expression data.

    Save standardized coefficients for downstream gene functional analysis.

    Args:
        human_adata: AnnData object for human data.
        mouse_adata: AnnData object for mouse data.
        setups: List of tuples (setup_name, transform_func) returning X_raw, X_emb, y.
        output_dir: Directory to save coefficient files.

    Returns:
        Tuple of (all_results, all_roc_data) dictionaries.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    all_roc_data = {}

    pipeline = get_linear_pipeline_raw_only()
    pipe = pipeline

    for setup_name, transform_func in setups:
        # Prepare data
        human_X, _, human_y = transform_func(human_adata)[:3]
        mouse_X, _, mouse_y = transform_func(mouse_adata)[:3]

        # Ensure data are aligned and numeric
        gene_names = human_adata.var_names
        assert human_X.shape[1] == mouse_X.shape[1], "Feature mismatch!"

        # Human → Mouse
        logger.info(f"Training on Human, testing on Mouse: {setup_name}")
        pipe.fit(human_X, human_y)
        y_pred = pipe.predict_proba(mouse_X)[:, 1]
        from sklearn.metrics import roc_auc_score, roc_curve

        auroc = roc_auc_score(mouse_y, y_pred)
        all_results[f"{setup_name} (Human→Mouse)"] = auroc
        fpr, tpr, _ = roc_curve(mouse_y, y_pred)
        all_roc_data[f"{setup_name} (Human→Mouse)"] = (fpr, tpr)

        # Extract coefficients
        clf = pipe.named_steps["clf"]
        coefs = pd.DataFrame(
            {
                "gene": gene_names,
                "coef": clf.coef_.flatten(),
                "direction": np.sign(clf.coef_.flatten()),
            }
        )
        coefs.to_csv(
            output_dir / f"{setup_name.replace(' ', '_')}_HumanToMouse_coeffs.csv", index=False
        )

        # Mouse → Human
        logger.info(f"Training on Mouse, testing on Human: {setup_name}")
        pipe = clone(pipeline)
        pipe.fit(mouse_X, mouse_y)
        y_pred = pipe.predict_proba(human_X)[:, 1]
        auroc = roc_auc_score(human_y, y_pred)
        all_results[f"{setup_name} (Mouse→Human)"] = auroc
        fpr, tpr, _ = roc_curve(human_y, y_pred)
        all_roc_data[f"{setup_name} (Mouse→Human)"] = (fpr, tpr)

        # Extract coefficients
        clf = pipe.named_steps["clf"]
        coefs = pd.DataFrame(
            {
                "gene": gene_names,
                "coef": clf.coef_.flatten(),
                "direction": np.sign(clf.coef_.flatten()),
            }
        )
        coefs.to_csv(
            output_dir / f"{setup_name.replace(' ', '_')}_MouseToHuman_coeffs.csv", index=False
        )

    return all_results, all_roc_data


def load_all_coefficients(
    coeff_dir: str | Path = "model_coefficients", gene_col: str = "gene", coef_col: str = "coef"
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Load coefficient CSVs from directory.

    Args:
        coeff_dir: Directory containing coefficient CSV files.
        gene_col: Name of gene column.
        coef_col: Name of coefficient column.

    Returns:
        Dictionary mapping setup -> direction -> DataFrame with coefficients.
    """
    coeff_dir = Path(coeff_dir)
    if not coeff_dir.exists():
        raise FileNotFoundError(f"Coefficient directory not found: {coeff_dir}")

    files = list(coeff_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSVs found in {coeff_dir}")

    all_coefs = defaultdict(dict)

    for f in files:
        fname = f.stem
        # Parse direction
        if "_HumanToMouse" in fname:
            direction = "HumanToMouse"
            setup = re.sub(r"(_HumanToMouse)*$", "", fname)
            setup = setup.replace(".", "")
        elif "_MouseToHuman" in fname:
            direction = "MouseToHuman"
            setup = re.sub(r"(_MouseToHuman)*$", "", fname)
            setup = setup.replace(".", "")
        else:
            logger.warning(f"Could not parse direction from {fname}, skipping")
            continue

        df = pd.read_csv(f)
        # Flexible coefficient column detection
        if coef_col not in df.columns:
            for alt in ["coef", "coefficient", "weight"]:
                if alt in df.columns:
                    coef_col = alt
                    break
            else:
                raise ValueError(f"{f} missing coefficient column")

        df2 = df[[gene_col, coef_col]].copy()
        df2.columns = ["gene", "coefficient"]
        df2 = df2.dropna(subset=["gene"]).drop_duplicates(subset=["gene"])
        df2 = df2.set_index("gene").sort_index()
        all_coefs[setup][direction] = df2

    return dict(all_coefs)


def normalize_coefficients(
    all_coefs: Dict[str, Dict[str, pd.DataFrame]],
) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Normalize coefficients within each setup/direction.

    Args:
        all_coefs: Dictionary from load_all_coefficients.

    Returns:
        Dictionary with normalized coefficients added as 'coef_norm' column.
    """
    norm_coefs = {}
    for setup, dirs in all_coefs.items():
        norm_coefs[setup] = {}
        for d, df in dirs.items():
            coefs = df["coefficient"].astype(float).copy()

            mu = coefs.mean()
            sigma = coefs.std(ddof=0) if coefs.std(ddof=0) > 0 else 1.0
            norm = (coefs - mu) / sigma

            df2 = df.copy()
            df2["coef_norm"] = norm
            norm_coefs[setup][d] = df2

    return norm_coefs


def build_coef_matrix(
    norm_coefs: Dict[str, Dict[str, pd.DataFrame]], direction: str
) -> pd.DataFrame:
    """Build coefficient matrix across setups for a given direction.

    Args:
        norm_coefs: Dictionary from normalize_coefficients.
        direction: Direction to extract ('HumanToMouse' or 'MouseToHuman').

    Returns:
        DataFrame with genes as rows and setups as columns.
    """
    setups_with_dir = [s for s in norm_coefs if direction in norm_coefs[s]]
    if not setups_with_dir:
        raise ValueError(f"No setups contain direction '{direction}'")

    mat = pd.DataFrame(index=norm_coefs[setups_with_dir[0]][direction].index)
    for s in setups_with_dir:
        mat[s] = norm_coefs[s][direction]["coef_norm"]

    return mat
