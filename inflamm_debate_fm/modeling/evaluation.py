"""Evaluation functions for model performance."""

from collections.abc import Callable

import anndata as ad
import numpy as np
from sklearn.base import clone
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import LeaveOneGroupOut, StratifiedKFold, cross_val_score
from tqdm import tqdm

from inflamm_debate_fm.config import get_config
from inflamm_debate_fm.modeling.pipelines import get_linear_pipelines, get_nonlinear_pipelines


def evaluate_within_species(
    adata: ad.AnnData,
    setups: list[tuple[str, Callable]],
    species_name: str,
    lodo_group_key: str = "dataset",
    n_cv_folds: int = 10,
    use_wandb: bool = False,
    wandb_run=None,
) -> tuple[dict, dict]:
    """Evaluate model performance within a single species.

    Args:
        adata: AnnData object for the species.
        setups: List of tuples (setup_name, transform_func) returning X_raw, X_emb, y, groups.
        species_name: Name of the species (e.g., 'Human', 'Mouse').
        lodo_group_key: Column name in adata.obs for LODO groups.
        n_cv_folds: Number of folds for StratifiedKFold CV.
        use_wandb: Whether to log results to wandb.
        wandb_run: Wandb run object for logging.

    Returns:
        Tuple of (all_results, all_roc_data) dictionaries.
    """
    config = get_config()
    n_cv_folds = config.get("model", {}).get("cv_folds", n_cv_folds)

    all_results = {
        "CrossValidation": {
            "Linear": {"Raw": {}, "Embedding": {}},
            "Nonlinear": {"Raw": {}, "Embedding": {}},
        },
        "LODO": {
            "Linear": {"Raw": {}, "Embedding": {}},
            "Nonlinear": {"Raw": {}, "Embedding": {}},
        },
    }
    all_roc_data = {
        "CrossValidation": {
            "Linear": {"Raw": {}, "Embedding": {}},
            "Nonlinear": {"Raw": {}, "Embedding": {}},
        },
        "LODO": {
            "Linear": {"Raw": {}, "Embedding": {}},
            "Nonlinear": {"Raw": {}, "Embedding": {}},
        },
    }

    for setup_name, transform_func in tqdm(setups, desc=f"Processing {species_name} setups"):
        tqdm.write(f"\n--- {species_name} - Setup: {setup_name} ---")
        try:
            X_raw, X_emb, y, groups = transform_func(adata)
            if len(np.unique(y)) < 2:
                tqdm.write(f"Skipping setup '{setup_name}': Only one class present.")
                continue
        except Exception as e:
            tqdm.write(f"Error transforming data for setup '{setup_name}': {e}")
            continue

        for model_type, pipeline_func in zip(
            ["Linear", "Nonlinear"], [get_linear_pipelines, get_nonlinear_pipelines]
        ):
            pipelines = pipeline_func()
            for data_type, X in [("Raw", X_raw), ("Embedding", X_emb)]:
                if X is None:
                    continue

                pipe_template = pipelines[data_type]
                full_key = setup_name

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
                    all_results["CrossValidation"][model_type][data_type][full_key] = (
                        mean_score,
                        std_score,
                    )

                    # Log to wandb
                    if use_wandb and wandb_run:
                        wandb_run.log(
                            {
                                f"{species_name}/{setup_name}/{model_type}/{data_type}/cv_auroc_mean": mean_score,
                                f"{species_name}/{setup_name}/{model_type}/{data_type}/cv_auroc_std": std_score,
                            }
                        )

                    # Store ROC for all folds
                    fold_roc_data = []
                    for train_idx, test_idx in cv.split(X, y):
                        if len(np.unique(y[test_idx])) < 2:
                            continue
                        pipe_fold = clone(pipe_template).fit(X[train_idx], y[train_idx])
                        y_pred_fold = pipe_fold.predict_proba(X[test_idx])[:, 1]
                        fpr, tpr, _ = roc_curve(y[test_idx], y_pred_fold)
                        fold_roc_data.append((fpr, tpr))
                    all_roc_data["CrossValidation"][model_type][data_type][full_key] = (
                        fold_roc_data
                    )

                except Exception as e:
                    tqdm.write(f"  CV Error: {e}")
                    all_results["CrossValidation"][model_type][data_type][full_key] = (
                        np.nan,
                        np.nan,
                    )
                    all_roc_data["CrossValidation"][model_type][data_type][full_key] = []

                # Leave-One-Group-Out CV
                tqdm.write(f"  Running LODO CV ({model_type}, {data_type})...")
                lodo_aurocs = []
                temp_roc_data_lodo = []
                logo = LeaveOneGroupOut()

                for train_idx, test_idx in logo.split(X, y, groups):
                    if len(np.unique(y[test_idx])) < 2:
                        continue
                    try:
                        pipe = clone(pipe_template).fit(X[train_idx], y[train_idx])
                        y_pred = pipe.predict_proba(X[test_idx])[:, 1]
                        lodo_aurocs.append(roc_auc_score(y[test_idx], y_pred))
                        fpr, tpr, _ = roc_curve(y[test_idx], y_pred)
                        temp_roc_data_lodo.append((fpr, tpr))
                    except Exception:
                        continue

                if lodo_aurocs:
                    mean_lodo = np.mean(lodo_aurocs)
                    std_lodo = np.std(lodo_aurocs)
                    all_results["LODO"][model_type][data_type][full_key] = (mean_lodo, std_lodo)
                    all_roc_data["LODO"][model_type][data_type][full_key] = temp_roc_data_lodo

                    # Log to wandb
                    if use_wandb and wandb_run:
                        wandb_run.log(
                            {
                                f"{species_name}/{setup_name}/{model_type}/{data_type}/lodo_auroc_mean": mean_lodo,
                                f"{species_name}/{setup_name}/{model_type}/{data_type}/lodo_auroc_std": std_lodo,
                            }
                        )
                else:
                    all_results["LODO"][model_type][data_type][full_key] = (np.nan, np.nan)
                    all_roc_data["LODO"][model_type][data_type][full_key] = []

    return all_results, all_roc_data


def evaluate_cross_species(
    human_adata: ad.AnnData,
    mouse_adata: ad.AnnData,
    setups: list[tuple[str, Callable]],
    use_wandb: bool = False,
    wandb_run=None,
) -> tuple[dict, dict]:
    """Evaluate cross-species model performance.

    Args:
        human_adata: AnnData object for human data.
        mouse_adata: AnnData object for mouse data.
        setups: List of tuples (setup_name, transform_func) returning X_raw, X_emb, y.
        use_wandb: Whether to log results to wandb.
        wandb_run: Wandb run object for logging.

    Returns:
        Tuple of (all_results, all_roc_data) dictionaries.
    """
    all_results = {}
    all_roc_data = {}

    pipelines = get_linear_pipelines()

    for setup_name, transform_func in tqdm(setups, desc="Processing cross-species setups"):
        tqdm.write(f"\n--- Setup: {setup_name} ---")
        try:
            human_X, human_X_emb, human_y = transform_func(human_adata)[:3]
            mouse_X, mouse_X_emb, mouse_y = transform_func(mouse_adata)[:3]

            if len(np.unique(human_y)) < 2 or len(np.unique(mouse_y)) < 2:
                tqdm.write(f"Skipping setup '{setup_name}': Only one class present.")
                continue
        except Exception as e:
            tqdm.write(f"Error transforming data for setup '{setup_name}': {e}")
            continue

        for data_type, (human_X_data, mouse_X_data) in [
            ("Raw", (human_X, mouse_X)),
            ("Embedding", (human_X_emb, mouse_X_emb)),
        ]:
            if human_X_data is None or mouse_X_data is None:
                continue

            pipe = clone(pipelines[data_type])

            # Human -> Mouse
            tqdm.write(f"  Training on Human, testing on Mouse ({data_type})...")
            try:
                pipe.fit(human_X_data, human_y)
                y_pred = pipe.predict_proba(mouse_X_data)[:, 1]
                auroc = roc_auc_score(mouse_y, y_pred)
                fpr, tpr, _ = roc_curve(mouse_y, y_pred)

                key = f"{setup_name} (Human→Mouse)"
                all_results[key] = auroc
                all_roc_data[key] = (fpr, tpr)

                # Log to wandb
                if use_wandb and wandb_run:
                    wandb_run.log({f"cross_species/{key}/{data_type}/auroc": auroc})

            except Exception as e:
                tqdm.write(f"  Error (Human→Mouse): {e}")

            # Mouse -> Human
            tqdm.write(f"  Training on Mouse, testing on Human ({data_type})...")
            try:
                pipe = clone(pipelines[data_type])
                pipe.fit(mouse_X_data, mouse_y)
                y_pred = pipe.predict_proba(human_X_data)[:, 1]
                auroc = roc_auc_score(human_y, y_pred)
                fpr, tpr, _ = roc_curve(human_y, y_pred)

                key = f"{setup_name} (Mouse→Human)"
                all_results[key] = auroc
                all_roc_data[key] = (fpr, tpr)

                # Log to wandb
                if use_wandb and wandb_run:
                    wandb_run.log({f"cross_species/{key}/{data_type}/auroc": auroc})

            except Exception as e:
                tqdm.write(f"  Error (Mouse→Human): {e}")

    return all_results, all_roc_data
