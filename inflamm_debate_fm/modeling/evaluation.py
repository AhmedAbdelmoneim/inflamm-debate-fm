"""Evaluation functions for model performance."""

from collections.abc import Callable
from pathlib import Path

import anndata as ad
import matplotlib.pyplot as plt
import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_score
from sklearn.utils import resample
from tqdm import tqdm
import wandb

from inflamm_debate_fm.config import get_config
from inflamm_debate_fm.modeling.pipelines import get_linear_pipelines, get_nonlinear_pipelines


def _safe_wandb_log(wandb_run, data: dict, context: str = "") -> None:
    """Safely log data to wandb, catching network/certificate errors.

    Args:
        wandb_run: Wandb run object (can be None).
        data: Dictionary of data to log.
        context: Optional context string for error messages.
    """
    if wandb_run is None:
        return

    try:
        wandb_run.log(data)
    except Exception as e:
        context_str = f" ({context})" if context else ""
        tqdm.write(f"Warning: Failed to log to wandb{context_str}: {e}")


def _log_roc_curves(
    roc_data: list[tuple],
    wandb_run,
    key: str,
    title: str | None = None,
) -> None:
    """Log ROC curves to wandb.

    Args:
        roc_data: List of (fpr, tpr) tuples.
        wandb_run: Wandb run object.
        key: Key for logging.
        title: Optional title for the plot.
    """
    if not roc_data or wandb_run is None:
        return

    try:
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 6))

        # Plot all ROC curves
        for fpr, tpr in roc_data:
            ax.plot(fpr, tpr, alpha=0.5, linewidth=1)

        # Plot diagonal line
        ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5, label="Random")

        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(title or f"ROC Curves: {key}")
        ax.legend()
        ax.grid(True, alpha=0.3)

        _safe_wandb_log(wandb_run, {f"{key}/roc_curves": wandb.Image(fig)}, context="ROC curves")
        plt.close(fig)
    except Exception as e:
        tqdm.write(f"Warning: Failed to create/log ROC curves to wandb: {e}")
        plt.close("all")  # Close any open figures


def _log_auroc_comparison(
    results_dict: dict,
    wandb_run,
    key: str,
    title: str | None = None,
) -> None:
    """Log AUROC comparison chart to wandb.

    Args:
        results_dict: Dictionary with setup names as keys and (mean, std) tuples as values.
        wandb_run: Wandb run object.
        key: Key for logging.
        title: Optional title for the plot.
    """
    if not results_dict or wandb_run is None:
        return

    try:
        # Prepare data
        setups = []
        means = []
        stds = []

        for setup_key, value in results_dict.items():
            if isinstance(value, tuple) and len(value) == 2:
                if not np.isnan(value[0]):
                    setups.append(setup_key.replace("::", " - "))
                    means.append(value[0])
                    stds.append(value[1])

        if not setups:
            return

        # Create figure
        fig, ax = plt.subplots(figsize=(12, max(6, len(setups) * 0.5)))

        y_pos = np.arange(len(setups))
        ax.barh(y_pos, means, xerr=stds, capsize=5, alpha=0.7)
        ax.axvline(0.5, color="red", linestyle="--", linewidth=1, alpha=0.5, label="Random (0.5)")

        ax.set_yticks(y_pos)
        ax.set_yticklabels(setups)
        ax.set_xlabel("AUROC")
        ax.set_xlim(0, 1.05)
        ax.set_title(title or f"AUROC Comparison: {key}")
        ax.legend()
        ax.grid(True, alpha=0.3, axis="x")

        _safe_wandb_log(
            wandb_run, {f"{key}/auroc_comparison": wandb.Image(fig)}, context="AUROC comparison"
        )
        plt.close(fig)
    except Exception as e:
        tqdm.write(f"Warning: Failed to create/log AUROC comparison to wandb: {e}")
        plt.close("all")  # Close any open figures


def _extract_model_weights(pipe, data_type: str, feature_names: np.ndarray | None = None):
    """Extract model weights/coefficients for interpretability.

    Args:
        pipe: Fitted sklearn pipeline.
        data_type: Type of data ("Raw" or embedding key).
        feature_names: Names of features (genes for Raw, None for embeddings).

    Returns:
        Dictionary with weights or None if not a linear model.
    """
    if "clf" not in pipe.named_steps:
        return None

    clf = pipe.named_steps["clf"]

    # Check if it's a linear model with coefficients
    if hasattr(clf, "coef_"):
        coefs = clf.coef_.flatten()
        intercept = clf.intercept_[0] if hasattr(clf, "intercept_") else None

        result = {"coefficients": coefs, "intercept": intercept}

        if feature_names is not None and len(feature_names) == len(coefs):
            result["feature_names"] = feature_names

        return result

    return None


def evaluate_within_species(
    adata: ad.AnnData,
    setups: list[tuple[str, Callable]],
    species_name: str,
    lodo_group_key: str = "dataset",
    n_cv_folds: int = 10,
    embedding_keys: list[str] | None = None,
    save_weights: bool = True,
    weights_output_dir: Path | str | None = None,
    use_wandb: bool = False,
    wandb_run=None,
) -> tuple[dict, dict, dict]:
    """Evaluate model performance within a single species.

    Args:
        adata: AnnData object for the species.
        setups: List of tuples (setup_name, transform_func) returning X_raw, X_embeddings_dict, y, groups.
        species_name: Name of the species (e.g., 'Human', 'Mouse').
        lodo_group_key: Column name in adata.obs for LODO groups.
        n_cv_folds: Number of folds for StratifiedKFold CV.
        embedding_keys: List of embedding keys to use. If None, uses all available.
        save_weights: Whether to save model weights for interpretability.
        weights_output_dir: Directory to save model weights. If None, doesn't save.
        use_wandb: Whether to log results to wandb.
        wandb_run: Wandb run object for logging.

    Returns:
        Tuple of (all_results, all_roc_data, all_weights) dictionaries.
    """
    config = get_config()
    n_cv_folds = config.get("model", {}).get("cv_folds", n_cv_folds)

    if weights_output_dir is not None:
        weights_output_dir = Path(weights_output_dir)
        weights_output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize results structure
    all_results = {
        "CrossValidation": {"Linear": {}, "Nonlinear": {}},
        "LODO": {"Linear": {}, "Nonlinear": {}},
    }
    all_roc_data = {
        "CrossValidation": {"Linear": {}, "Nonlinear": {}},
        "LODO": {"Linear": {}, "Nonlinear": {}},
    }
    all_weights = {
        "CrossValidation": {"Linear": {}, "Nonlinear": {}},
        "LODO": {"Linear": {}, "Nonlinear": {}},
    }

    for setup_name, transform_func in tqdm(setups, desc=f"Processing {species_name} setups"):
        tqdm.write(f"\n--- {species_name} - Setup: {setup_name} ---")
        try:
            X_raw, X_embeddings_dict, y, groups = transform_func(
                adata, embedding_keys=embedding_keys
            )
            if len(np.unique(y)) < 2:
                tqdm.write(f"Skipping setup '{setup_name}': Only one class present.")
                continue
        except Exception as e:
            tqdm.write(f"Error transforming data for setup '{setup_name}': {e}")
            continue

        # Prepare all data types: Raw + all embeddings
        data_sources = [("Raw", X_raw, adata.var_names if X_raw is not None else None)]

        # Add all embeddings
        if embedding_keys is None:
            embedding_keys = list(X_embeddings_dict.keys())

        for emb_key in embedding_keys:
            if emb_key in X_embeddings_dict:
                # Clean key name for storage (remove X_ prefix)
                clean_key = emb_key.replace("X_", "") if emb_key.startswith("X_") else emb_key
                data_sources.append((clean_key, X_embeddings_dict[emb_key], None))

        for model_type, pipeline_func in zip(
            ["Linear", "Nonlinear"], [get_linear_pipelines, get_nonlinear_pipelines]
        ):
            pipelines = pipeline_func()

            for data_type, X, feature_names in data_sources:
                if X is None:
                    continue

                # Use appropriate pipeline (Raw vs Embedding)
                pipeline_key = "Raw" if data_type == "Raw" else "Embedding"
                pipe_template = pipelines[pipeline_key]
                full_key = f"{setup_name}::{data_type}"

                # Stratified K-Fold CV
                tqdm.write(f"  Running {n_cv_folds}-Fold CV ({model_type}, {data_type})...")
                try:
                    random_seed = config.get("model", {}).get("random_seed", 42)
                    cv = StratifiedKFold(
                        n_splits=n_cv_folds, shuffle=True, random_state=random_seed
                    )
                    pipe_cv = clone(pipe_template)
                    cv_scores = cross_val_score(pipe_cv, X, y, cv=cv, scoring="roc_auc", n_jobs=-1)
                    mean_score = np.mean(cv_scores)
                    std_score = np.std(cv_scores)
                    all_results["CrossValidation"][model_type][full_key] = (mean_score, std_score)

                    # Log to wandb
                    if use_wandb and wandb_run:
                        _safe_wandb_log(
                            wandb_run,
                            {
                                f"{species_name}/{setup_name}/{model_type}/{data_type}/cv_auroc_mean": mean_score,
                                f"{species_name}/{setup_name}/{model_type}/{data_type}/cv_auroc_std": std_score,
                            },
                            context=f"CV {model_type} {data_type}",
                        )

                    # Store ROC for all folds and weights
                    fold_roc_data = []
                    fold_weights = []
                    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
                        if len(np.unique(y[test_idx])) < 2:
                            continue
                        pipe_fold = clone(pipe_template).fit(X[train_idx], y[train_idx])
                        y_pred_fold = pipe_fold.predict_proba(X[test_idx])[:, 1]
                        fpr, tpr, _ = roc_curve(y[test_idx], y_pred_fold)
                        fold_roc_data.append((fpr, tpr))

                        # Extract weights for linear models
                        if model_type == "Linear" and save_weights:
                            weights = _extract_model_weights(pipe_fold, data_type, feature_names)
                            if weights is not None:
                                fold_weights.append(weights)

                    all_roc_data["CrossValidation"][model_type][full_key] = fold_roc_data
                    if fold_weights:
                        all_weights["CrossValidation"][model_type][full_key] = fold_weights

                    # Log ROC curves to wandb (only for final summary, not during parallel execution)
                    if use_wandb and wandb_run and fold_roc_data:
                        # Only log if we have reasonable number of curves (not too many)
                        if len(fold_roc_data) <= 20:  # Limit to avoid too many images
                            _log_roc_curves(
                                fold_roc_data,
                                wandb_run,
                                f"{species_name}/{setup_name}/{model_type}/{data_type}/cv",
                                f"CV ROC: {setup_name} ({model_type}, {data_type})",
                            )

                except Exception as e:
                    tqdm.write(f"  CV Error: {e}")
                    all_results["CrossValidation"][model_type][full_key] = (np.nan, np.nan)
                    all_roc_data["CrossValidation"][model_type][full_key] = []
                    all_weights["CrossValidation"][model_type][full_key] = []

                # Leave-One-Group-Out CV
                tqdm.write(f"  Running LODO CV ({model_type}, {data_type})...")
                lodo_aurocs = []
                temp_roc_data_lodo = []
                temp_weights_lodo = []
                logo = LeaveOneGroupOut()

                for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups)):
                    if len(np.unique(y[test_idx])) < 2:
                        continue
                    try:
                        pipe = clone(pipe_template).fit(X[train_idx], y[train_idx])
                        y_pred = pipe.predict_proba(X[test_idx])[:, 1]
                        lodo_aurocs.append(roc_auc_score(y[test_idx], y_pred))
                        fpr, tpr, _ = roc_curve(y[test_idx], y_pred)
                        temp_roc_data_lodo.append((fpr, tpr))

                        # Extract weights for linear models
                        if model_type == "Linear" and save_weights:
                            weights = _extract_model_weights(pipe, data_type, feature_names)
                            if weights is not None:
                                temp_weights_lodo.append(weights)
                    except Exception:
                        continue

                if lodo_aurocs:
                    mean_lodo = np.mean(lodo_aurocs)
                    std_lodo = np.std(lodo_aurocs)
                    all_results["LODO"][model_type][full_key] = (mean_lodo, std_lodo)
                    all_roc_data["LODO"][model_type][full_key] = temp_roc_data_lodo
                    if temp_weights_lodo:
                        all_weights["LODO"][model_type][full_key] = temp_weights_lodo

                    # Log to wandb
                    if use_wandb and wandb_run:
                        _safe_wandb_log(
                            wandb_run,
                            {
                                f"{species_name}/{setup_name}/{model_type}/{data_type}/lodo_auroc_mean": mean_lodo,
                                f"{species_name}/{setup_name}/{model_type}/{data_type}/lodo_auroc_std": std_lodo,
                            },
                            context=f"LODO {model_type} {data_type}",
                        )
                        # Log ROC curves (LODO typically has few curves, so safe to log)
                        if temp_roc_data_lodo and len(temp_roc_data_lodo) <= 20:
                            _log_roc_curves(
                                temp_roc_data_lodo,
                                wandb_run,
                                f"{species_name}/{setup_name}/{model_type}/{data_type}/lodo",
                                f"LODO ROC: {setup_name} ({model_type}, {data_type})",
                            )
                else:
                    all_results["LODO"][model_type][full_key] = (np.nan, np.nan)
                    all_roc_data["LODO"][model_type][full_key] = []
                    all_weights["LODO"][model_type][full_key] = []

    # Log summary AUROC comparison charts
    if use_wandb and wandb_run:
        for val_type in ["CrossValidation", "LODO"]:
            for model_type in ["Linear", "Nonlinear"]:
                for data_type in ["Raw", "Embedding"]:
                    # Get all results for this combination
                    results_subset = {}
                    for key, value in all_results[val_type][model_type][data_type].items():
                        if isinstance(value, tuple) and len(value) == 2 and not np.isnan(value[0]):
                            results_subset[key] = value

                    # Log AUROC comparison charts (summary only, not too frequent)
                    if results_subset and len(results_subset) > 0:
                        _log_auroc_comparison(
                            results_subset,
                            wandb_run,
                            f"{species_name}/{val_type}/{model_type}/{data_type}",
                            f"{val_type} AUROC: {species_name} ({model_type}, {data_type})",
                        )

    return all_results, all_roc_data, all_weights


def evaluate_cross_species(
    human_adata: ad.AnnData,
    mouse_adata: ad.AnnData,
    setups: list[tuple[str, Callable]],
    n_bootstraps: int = 20,
    bootstrap_start: int | None = None,
    bootstrap_end: int | None = None,
    embedding_keys: list[str] | None = None,
    save_weights: bool = True,
    weights_output_dir: Path | str | None = None,
    use_wandb: bool = False,
    wandb_run=None,
) -> tuple[dict, dict, dict]:
    """Evaluate cross-species model performance with bootstrapping.

    Args:
        human_adata: AnnData object for human data.
        mouse_adata: AnnData object for mouse data.
        setups: List of tuples (setup_name, transform_func) returning X_raw, X_embeddings_dict, y.
        n_bootstraps: Number of bootstrap iterations.
        bootstrap_start: Start index for bootstrap range (for parallelization). If None, starts at 0.
        bootstrap_end: End index for bootstrap range (for parallelization). If None, uses n_bootstraps.
        embedding_keys: List of embedding keys to use. If None, uses all available.
        save_weights: Whether to save model weights for interpretability.
        weights_output_dir: Directory to save model weights. If None, doesn't save.
        use_wandb: Whether to log results to wandb.
        wandb_run: Wandb run object for logging.

    Returns:
        Tuple of (all_results, all_roc_data, all_weights) dictionaries.
    """
    config = get_config()
    # Use n_bootstraps from parameter (CLI argument), not config
    # Config value is just a default, but CLI argument takes precedence
    random_seed = config.get("model", {}).get("random_seed", 42)

    # Handle bootstrap range for parallelization
    if bootstrap_start is None:
        bootstrap_start = 0
    if bootstrap_end is None:
        bootstrap_end = n_bootstraps

    # Validate bootstrap range
    if bootstrap_start < 0:
        raise ValueError(f"bootstrap_start must be >= 0, got {bootstrap_start}")
    if bootstrap_end > n_bootstraps:
        raise ValueError(
            f"bootstrap_end ({bootstrap_end}) cannot exceed n_bootstraps ({n_bootstraps})"
        )
    if bootstrap_start >= bootstrap_end:
        raise ValueError(
            f"bootstrap_start ({bootstrap_start}) must be < bootstrap_end ({bootstrap_end})"
        )

    bootstrap_range = range(bootstrap_start, bootstrap_end)
    actual_n_bootstraps = len(bootstrap_range)

    tqdm.write(
        f"Running bootstraps {bootstrap_start} to {bootstrap_end - 1} (total: {actual_n_bootstraps})"
    )

    if weights_output_dir is not None:
        weights_output_dir = Path(weights_output_dir)
        weights_output_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}
    all_roc_data = {}
    all_weights = {}

    pipelines = get_linear_pipelines()

    for setup_name, transform_func in tqdm(setups, desc="Processing cross-species setups"):
        tqdm.write(f"\n--- Setup: {setup_name} ---")
        try:
            human_X_raw, human_X_emb, human_y, _ = transform_func(
                human_adata, embedding_keys=embedding_keys
            )
            mouse_X_raw, mouse_X_emb, mouse_y, _ = transform_func(
                mouse_adata, embedding_keys=embedding_keys
            )

            if len(np.unique(human_y)) < 2 or len(np.unique(mouse_y)) < 2:
                tqdm.write(f"Skipping setup '{setup_name}': Only one class present.")
                continue
        except Exception as e:
            tqdm.write(f"Error transforming data for setup '{setup_name}': {e}")
            continue

        # Prepare all data types: Raw + all embeddings
        data_sources = [
            (
                "Raw",
                (human_X_raw, mouse_X_raw),
                human_adata.var_names if human_X_raw is not None else None,
            )
        ]

        # Add all embeddings
        if embedding_keys is None:
            embedding_keys = list(human_X_emb.keys())

        for emb_key in embedding_keys:
            if emb_key in human_X_emb and emb_key in mouse_X_emb:
                clean_key = emb_key.replace("X_", "") if emb_key.startswith("X_") else emb_key
                data_sources.append(
                    (clean_key, (human_X_emb[emb_key], mouse_X_emb[emb_key]), None)
                )

        for data_type, (human_X_data, mouse_X_data), feature_names in data_sources:
            if human_X_data is None or mouse_X_data is None:
                continue

            pipeline_key = "Raw" if data_type == "Raw" else "Embedding"
            pipe_template = pipelines[pipeline_key]

            # Human -> Mouse with bootstrapping
            tqdm.write(
                f"  Training on Human, testing on Mouse ({data_type}) with {actual_n_bootstraps} bootstraps ({bootstrap_start}-{bootstrap_end - 1})..."
            )
            bootstrap_aurocs = []
            bootstrap_roc_data = []
            bootstrap_weights = []

            np.random.seed(random_seed)
            for bs_idx in bootstrap_range:
                # Bootstrap sample from human training data
                bs_indices = resample(
                    np.arange(len(human_X_data)), random_state=random_seed + bs_idx, replace=True
                )
                human_X_bs = human_X_data[bs_indices]
                human_y_bs = human_y[bs_indices]

                try:
                    pipe = clone(pipe_template).fit(human_X_bs, human_y_bs)
                    y_pred = pipe.predict_proba(mouse_X_data)[:, 1]
                    auroc = roc_auc_score(mouse_y, y_pred)
                    bootstrap_aurocs.append(auroc)
                    fpr, tpr, _ = roc_curve(mouse_y, y_pred)
                    bootstrap_roc_data.append((fpr, tpr))

                    # Extract weights for linear models
                    if save_weights:
                        weights = _extract_model_weights(pipe, data_type, feature_names)
                        if weights is not None:
                            bootstrap_weights.append(weights)
                except Exception as e:
                    tqdm.write(f"    Bootstrap {bs_idx} error: {e}")
                    continue

            if bootstrap_aurocs:
                key = f"{setup_name} (Human→Mouse)::{data_type}"
                all_results[key] = {
                    "mean": np.mean(bootstrap_aurocs),
                    "std": np.std(bootstrap_aurocs),
                    "values": bootstrap_aurocs,
                }
                all_roc_data[key] = bootstrap_roc_data
                if bootstrap_weights:
                    all_weights[key] = bootstrap_weights

                # Log to wandb (skip ROC curves for parallel jobs to reduce upload load)
                if use_wandb and wandb_run:
                    _safe_wandb_log(
                        wandb_run,
                        {
                            f"cross_species/{key}/auroc_mean": np.mean(bootstrap_aurocs),
                            f"cross_species/{key}/auroc_std": np.std(bootstrap_aurocs),
                        },
                        context=f"Cross-species Human→Mouse {key}",
                    )
                    # Only log ROC curves if not a parallel job (to reduce upload load)
                    if bootstrap_roc_data and bootstrap_start is None and bootstrap_end is None:
                        _log_roc_curves(
                            bootstrap_roc_data,
                            wandb_run,
                            f"cross_species/{key}",
                            f"Cross-species ROC: {key}",
                        )

            # Mouse -> Human with bootstrapping
            tqdm.write(
                f"  Training on Mouse, testing on Human ({data_type}) with {actual_n_bootstraps} bootstraps ({bootstrap_start}-{bootstrap_end - 1})..."
            )
            bootstrap_aurocs = []
            bootstrap_roc_data = []
            bootstrap_weights = []

            for bs_idx in bootstrap_range:
                # Bootstrap sample from mouse training data
                bs_indices = resample(
                    np.arange(len(mouse_X_data)),
                    random_state=random_seed + bs_idx + n_bootstraps,
                    replace=True,
                )
                mouse_X_bs = mouse_X_data[bs_indices]
                mouse_y_bs = mouse_y[bs_indices]

                try:
                    pipe = clone(pipe_template).fit(mouse_X_bs, mouse_y_bs)
                    y_pred = pipe.predict_proba(human_X_data)[:, 1]
                    auroc = roc_auc_score(human_y, y_pred)
                    bootstrap_aurocs.append(auroc)
                    fpr, tpr, _ = roc_curve(human_y, y_pred)
                    bootstrap_roc_data.append((fpr, tpr))

                    # Extract weights for linear models
                    if save_weights:
                        weights = _extract_model_weights(pipe, data_type, feature_names)
                        if weights is not None:
                            bootstrap_weights.append(weights)
                except Exception as e:
                    tqdm.write(f"    Bootstrap {bs_idx} error: {e}")
                    continue

            if bootstrap_aurocs:
                key = f"{setup_name} (Mouse→Human)::{data_type}"
                all_results[key] = {
                    "mean": np.mean(bootstrap_aurocs),
                    "std": np.std(bootstrap_aurocs),
                    "values": bootstrap_aurocs,
                }
                all_roc_data[key] = bootstrap_roc_data
                if bootstrap_weights:
                    all_weights[key] = bootstrap_weights

                # Log to wandb (skip ROC curves for parallel jobs to reduce upload load)
                if use_wandb and wandb_run:
                    _safe_wandb_log(
                        wandb_run,
                        {
                            f"cross_species/{key}/auroc_mean": np.mean(bootstrap_aurocs),
                            f"cross_species/{key}/auroc_std": np.std(bootstrap_aurocs),
                        },
                        context=f"Cross-species Mouse→Human {key}",
                    )
                    # Only log ROC curves if not a parallel job (to reduce upload load)
                    if bootstrap_roc_data and bootstrap_start is None and bootstrap_end is None:
                        _log_roc_curves(
                            bootstrap_roc_data,
                            wandb_run,
                            f"cross_species/{key}",
                            f"Cross-species ROC: {key}",
                        )

    # Log summary AUROC comparison charts for cross-species
    if use_wandb and wandb_run:
        # Group by data type
        for data_type in set(k.split("::")[-1] for k in all_results.keys() if "::" in k):
            results_subset = {
                k: v
                for k, v in all_results.items()
                if k.endswith(f"::{data_type}") and isinstance(v, dict) and "mean" in v
            }
            # Only log comparison charts if not a parallel job (to reduce upload load)
            if results_subset and bootstrap_start is None and bootstrap_end is None:
                _log_auroc_comparison(
                    {k: (v["mean"], v["std"]) for k, v in results_subset.items()},
                    wandb_run,
                    f"cross_species/{data_type}",
                    f"Cross-species AUROC: {data_type}",
                )

    return all_results, all_roc_data, all_weights
