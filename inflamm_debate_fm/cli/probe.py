"""Probing experiment commands."""

from pathlib import Path
import pickle

import anndata as ad
from loguru import logger
import typer

from inflamm_debate_fm.cli.utils import get_setup_transforms
from inflamm_debate_fm.config import DATA_ROOT, get_config
from inflamm_debate_fm.data.load import combine_adatas
from inflamm_debate_fm.embeddings.multi_model import (
    add_multi_model_embeddings_to_adata,
    detect_available_models,
)
from inflamm_debate_fm.modeling.evaluation import evaluate_cross_species, evaluate_within_species
from inflamm_debate_fm.utils.wandb_utils import init_wandb

app = typer.Typer(help="Probing experiment commands")


@app.callback(invoke_without_command=True)
def probe_callback(
    ctx: typer.Context,
    n_cv_folds: int = typer.Option(
        10, "--n-cv-folds", help="Number of CV folds for within-species"
    ),
    n_bootstraps: int = typer.Option(
        20, "--n-bootstraps", help="Number of bootstrap iterations for cross-species"
    ),
    bootstrap_start: int | None = typer.Option(
        None, "--bootstrap-start", help="Start index for bootstrap range (for parallelization)"
    ),
    bootstrap_end: int | None = typer.Option(
        None, "--bootstrap-end", help="End index for bootstrap range (for parallelization)"
    ),
    embedding_types: str = typer.Option(
        "all", "--embedding-types", help="Comma-separated list of embedding types, or 'all'"
    ),
    load_multi_model_embeddings: bool = typer.Option(
        True,
        "--load-multi-model-embeddings/--no-load-multi-model-embeddings",
        help="Load multi-model embeddings if not present",
    ),
    device: str = typer.Option(
        "cpu", "--device", help="Device for loading embeddings ('cpu' or 'cuda')"
    ),
    batch_size: int = typer.Option(4, "--batch-size", help="Batch size for embedding generation"),
    save_weights: bool = typer.Option(
        True, "--save-weights/--no-save-weights", help="Save model weights for interpretability"
    ),
    output_dir: str | None = typer.Option(None, "--output-dir", help="Output directory"),
    use_wandb: bool = typer.Option(False, "--use-wandb", help="Log to Weights & Biases"),
) -> None:
    """Run both within-species and cross-species probing experiments.

    If no subcommand is specified, runs all experiments.
    """
    if ctx.invoked_subcommand is None:
        # Run the main probe function
        probe(
            n_cv_folds=n_cv_folds,
            n_bootstraps=n_bootstraps,
            bootstrap_start=bootstrap_start,
            bootstrap_end=bootstrap_end,
            embedding_types=embedding_types,
            load_multi_model_embeddings=load_multi_model_embeddings,
            device=device,
            batch_size=batch_size,
            save_weights=save_weights,
            output_dir=output_dir,
            use_wandb=use_wandb,
        )


def _detect_embedding_keys(adata) -> list[str]:
    """Detect available embedding keys in AnnData obsm.

    Returns:
        List of embedding keys (e.g., ['X_zero_shot', 'X_human', 'X_mouse']).
    """
    return [key for key in adata.obsm.keys() if key.startswith("X_")]


def _load_and_prepare_data(
    load_multi_model_embeddings: bool,
    device: str,
    batch_size: int,
    embedding_types: str,
) -> tuple[dict, list[str]]:
    """Load and prepare AnnData objects with embeddings.

    Returns:
        Tuple of (combined_adatas dict, embedding_keys list).
    """
    config = get_config()
    cleaned_data_dir = DATA_ROOT / config["paths"]["anndata_cleaned_dir"]

    logger.info("Loading cleaned anndata files...")
    # Load individual cleaned datasets
    adatas = {}
    for f in sorted(cleaned_data_dir.glob("*.h5ad")):
        name = f.stem
        adatas[name] = ad.read_h5ad(f)

    if len(adatas) == 0:
        raise ValueError(
            f"No AnnData files found in {cleaned_data_dir}. Please run data preprocessing first."
        )

    logger.info(f"Found {len(adatas)} cleaned datasets: {list(adatas.keys())}")

    # Combine by species
    logger.info("Combining datasets by species...")
    try:
        combined_adatas = {
            "human": combine_adatas(adatas, "human"),
            "mouse": combine_adatas(adatas, "mouse"),
        }
    except ValueError as e:
        raise ValueError(
            f"Failed to combine datasets by species: {e}. "
            "Make sure you have both human and mouse datasets."
        ) from e

    logger.info(f"Combined human: {combined_adatas['human'].shape}")
    logger.info(f"Combined mouse: {combined_adatas['mouse'].shape}")

    # Check for existing embeddings
    existing_embeddings = {}
    for species in combined_adatas:
        existing_embeddings[species] = set(_detect_embedding_keys(combined_adatas[species]))
        logger.info(
            f"Found {len(existing_embeddings[species])} existing embeddings in {species}: {sorted(existing_embeddings[species])}"
        )

    # Only generate missing embeddings if requested
    if load_multi_model_embeddings:
        available_models = detect_available_models()
        if available_models:
            logger.info(
                f"Checking for missing embeddings from {len(available_models)} available models..."
            )
            for species in combined_adatas:
                # Check which embeddings are missing
                expected_keys = {f"X_{model_name}" for model_name in available_models.keys()}
                missing_keys = expected_keys - existing_embeddings[species]

                if missing_keys:
                    logger.info(
                        f"Generating {len(missing_keys)} missing embeddings for {species}: {sorted(missing_keys)}"
                    )
                    combined_adatas[species] = add_multi_model_embeddings_to_adata(
                        combined_adatas[species],
                        device=device,
                        batch_size=batch_size,
                        models=available_models,
                    )
                else:
                    logger.info(
                        f"All embeddings already present in {species}, skipping generation"
                    )
        else:
            logger.warning("No models found for multi-model embedding extraction")

    # Detect embedding keys (use intersection of all species)
    all_keys = []
    for adata in combined_adatas.values():
        all_keys.append(set(_detect_embedding_keys(adata)))

    if all_keys:
        embedding_keys = list(set.intersection(*all_keys))
    else:
        embedding_keys = []

    logger.info(f"Found common embedding keys: {embedding_keys}")

    # Filter embedding types if specified
    if embedding_types != "all":
        requested_keys = [k.strip() for k in embedding_types.split(",")]
        requested_keys = [k if k.startswith("X_") else f"X_{k}" for k in requested_keys]
        embedding_keys = [k for k in embedding_keys if k in requested_keys]
        logger.info(f"Using embedding keys: {embedding_keys}")

    return combined_adatas, embedding_keys


def probe(
    n_cv_folds: int = 10,
    n_bootstraps: int = 20,
    bootstrap_start: int | None = None,
    bootstrap_end: int | None = None,
    embedding_types: str = "all",
    load_multi_model_embeddings: bool = True,
    device: str = "cpu",
    batch_size: int = 4,
    save_weights: bool = True,
    output_dir: str | None = None,
    use_wandb: bool = False,
) -> None:
    """Run both within-species and cross-species probing experiments.

    Args:
        n_cv_folds: Number of CV folds for within-species experiments.
        n_bootstraps: Number of bootstrap iterations for cross-species experiments.
        embedding_types: Comma-separated list of embedding types to use, or 'all' for all available.
        load_multi_model_embeddings: Whether to load multi-model embeddings if not present.
        device: Device for loading embeddings ('cpu' or 'cuda').
        batch_size: Batch size for embedding generation.
        save_weights: Whether to save model weights for interpretability.
        output_dir: Output directory. If None, uses default from config.
    """
    config = get_config()
    if output_dir is None:
        base_output_dir = DATA_ROOT / "probing_results"
    else:
        base_output_dir = Path(output_dir)

    # Initialize wandb if requested
    wandb_run = None
    if use_wandb:
        try:
            wandb_run = init_wandb(
                project=config.get("wandb", {}).get("project", "inflamm-debate-fm"),
                tags=config.get("wandb", {}).get("tags", []) + ["probing"],
                config={
                    "n_cv_folds": n_cv_folds,
                    "n_bootstraps": n_bootstraps,
                    "embedding_types": embedding_types,
                    "save_weights": save_weights,
                },
            )
            if wandb_run:
                wandb_run.name = "probing_all_experiments"
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            use_wandb = False

    # Load and prepare data once
    combined_adatas, embedding_keys = _load_and_prepare_data(
        load_multi_model_embeddings=load_multi_model_embeddings,
        device=device,
        batch_size=batch_size,
        embedding_types=embedding_types,
    )

    setups = get_setup_transforms()

    # Run within-species experiments for both species
    logger.info("=" * 80)
    logger.info("Running within-species probing experiments")
    logger.info("=" * 80)

    for species in ["human", "mouse"]:
        logger.info(f"\n{'=' * 80}")
        logger.info(f"Within-species: {species.capitalize()}")
        logger.info(f"{'=' * 80}")

        species_output_dir = base_output_dir / "within_species"
        species_output_dir.mkdir(parents=True, exist_ok=True)

        weights_dir = species_output_dir / "model_weights"
        if save_weights:
            weights_dir.mkdir(parents=True, exist_ok=True)

        adata = combined_adatas[species]

        # Get species-specific embedding keys
        species_embedding_keys = _detect_embedding_keys(adata)
        if embedding_types != "all":
            requested_keys = [k.strip() for k in embedding_types.split(",")]
            requested_keys = [k if k.startswith("X_") else f"X_{k}" for k in requested_keys]
            species_embedding_keys = [k for k in species_embedding_keys if k in requested_keys]

        all_results, all_roc_data, all_weights = evaluate_within_species(
            adata=adata,
            setups=setups,
            species_name=species.capitalize(),
            n_cv_folds=n_cv_folds,
            embedding_keys=species_embedding_keys,
            save_weights=save_weights,
            weights_output_dir=weights_dir if save_weights else None,
            use_wandb=use_wandb,
            wandb_run=wandb_run,
        )

        # Save results
        results_path = species_output_dir / f"{species}_results.pkl"
        with results_path.open("wb") as f:
            pickle.dump(
                {
                    "results": all_results,
                    "roc_data": all_roc_data,
                    "weights": all_weights,
                    "embedding_keys": species_embedding_keys,
                    "species": species,
                    "n_cv_folds": n_cv_folds,
                },
                f,
            )
        logger.success(f"Saved results to {results_path}")

        # Save summary CSV
        summary_path = species_output_dir / f"{species}_summary.csv"
        _save_summary_csv(all_results, summary_path)
        logger.success(f"Saved summary to {summary_path}")

    # Run cross-species experiments
    logger.info("\n" + "=" * 80)
    logger.info("Running cross-species probing experiments")
    logger.info("=" * 80)

    cross_species_output_dir = base_output_dir / "cross_species"
    cross_species_output_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = cross_species_output_dir / "model_weights"
    if save_weights:
        weights_dir.mkdir(parents=True, exist_ok=True)

    human_adata = combined_adatas["human"]
    mouse_adata = combined_adatas["mouse"]

    all_results, all_roc_data, all_weights = evaluate_cross_species(
        human_adata=human_adata,
        mouse_adata=mouse_adata,
        setups=setups,
        n_bootstraps=n_bootstraps,
        bootstrap_start=bootstrap_start,
        bootstrap_end=bootstrap_end,
        embedding_keys=embedding_keys,
        save_weights=save_weights,
        weights_output_dir=weights_dir if save_weights else None,
        use_wandb=use_wandb,
        wandb_run=wandb_run,
    )

    # Save results with bootstrap range in filename if using parallelization
    if bootstrap_start is not None and bootstrap_end is not None:
        results_filename = f"cross_species_results_bs_{bootstrap_start}_{bootstrap_end}.pkl"
    else:
        results_filename = "cross_species_results.pkl"

    results_path = cross_species_output_dir / results_filename
    with results_path.open("wb") as f:
        pickle.dump(
            {
                "results": all_results,
                "roc_data": all_roc_data,
                "weights": all_weights,
                "embedding_keys": embedding_keys,
                "n_bootstraps": n_bootstraps,
            },
            f,
        )
    logger.success(f"Saved results to {results_path}")

    # Save summary CSV (only if not using parallelization, to avoid overwrites)
    if bootstrap_start is None and bootstrap_end is None:
        summary_path = cross_species_output_dir / "cross_species_summary.csv"
        _save_summary_csv(all_results, summary_path)
        logger.success(f"Saved summary to {summary_path}")
    else:
        logger.info(
            "Skipping summary CSV for parallel job (will be generated after combining all results)"
        )

    logger.info("\n" + "=" * 80)
    logger.success("All probing experiments completed!")
    logger.info("=" * 80)

    if use_wandb and wandb_run:
        wandb_run.finish()
        logger.info("Wandb run completed")


@app.command("within-species")
def probe_within_species(
    species: str,
    n_cv_folds: int = 10,
    embedding_types: str = "all",
    load_multi_model_embeddings: bool = True,
    device: str = "cpu",
    batch_size: int = 4,
    save_weights: bool = True,
    output_dir: str | None = None,
    use_wandb: bool = typer.Option(False, "--use-wandb", help="Log to Weights & Biases"),
) -> None:
    """Run within-species probing experiments.

    Args:
        species: Species name ('human' or 'mouse').
        n_cv_folds: Number of CV folds.
        embedding_types: Comma-separated list of embedding types to use, or 'all' for all available.
        load_multi_model_embeddings: Whether to load multi-model embeddings if not present.
        device: Device for loading embeddings ('cpu' or 'cuda').
        batch_size: Batch size for embedding generation.
        save_weights: Whether to save model weights for interpretability.
        output_dir: Output directory. If None, uses default from config.
    """
    config = get_config()
    if output_dir is None:
        output_dir = DATA_ROOT / "probing_results" / "within_species"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = output_dir / "model_weights"
    if save_weights:
        weights_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb if requested
    wandb_run = None
    if use_wandb:
        try:
            wandb_run = init_wandb(
                project=config.get("wandb", {}).get("project", "inflamm-debate-fm"),
                tags=config.get("wandb", {}).get("tags", [])
                + ["probing", "within-species", species],
                config={
                    "n_cv_folds": n_cv_folds,
                    "embedding_types": embedding_types,
                    "save_weights": save_weights,
                    "species": species,
                },
            )
            if wandb_run:
                wandb_run.name = f"probing_within_species_{species}"
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            use_wandb = False

    logger.info(f"Loading {species} combined data...")
    combined_adatas, embedding_keys = _load_and_prepare_data(
        load_multi_model_embeddings=load_multi_model_embeddings,
        device=device,
        batch_size=batch_size,
        embedding_types=embedding_types,
    )

    species_lower = species.lower()
    if species_lower not in combined_adatas:
        raise ValueError(
            f"Species '{species}' not found in combined data. "
            f"Available species: {list(combined_adatas.keys())}"
        )
    adata = combined_adatas[species_lower]

    # Get species-specific embedding keys
    species_embedding_keys = _detect_embedding_keys(adata)
    if embedding_types != "all":
        requested_keys = [k.strip() for k in embedding_types.split(",")]
        requested_keys = [k if k.startswith("X_") else f"X_{k}" for k in requested_keys]
        species_embedding_keys = [k for k in species_embedding_keys if k in requested_keys]
    else:
        species_embedding_keys = embedding_keys

    setups = get_setup_transforms()
    logger.info(f"Running within-species probing for {species} with {len(setups)} setups...")

    all_results, all_roc_data, all_weights = evaluate_within_species(
        adata=adata,
        setups=setups,
        species_name=species.capitalize(),
        n_cv_folds=n_cv_folds,
        embedding_keys=species_embedding_keys,
        save_weights=save_weights,
        weights_output_dir=weights_dir if save_weights else None,
        use_wandb=use_wandb,
        wandb_run=wandb_run,
    )

    # Save results
    results_path = output_dir / f"{species}_results.pkl"
    with results_path.open("wb") as f:
        pickle.dump(
            {
                "results": all_results,
                "roc_data": all_roc_data,
                "weights": all_weights,
                "embedding_keys": species_embedding_keys,
                "species": species,
                "n_cv_folds": n_cv_folds,
            },
            f,
        )
    logger.success(f"Saved results to {results_path}")

    # Save summary CSV
    summary_path = output_dir / f"{species}_summary.csv"
    _save_summary_csv(all_results, summary_path)
    logger.success(f"Saved summary to {summary_path}")

    if use_wandb and wandb_run:
        wandb_run.finish()
        logger.info("Wandb run completed")


@app.command("cross-species")
def probe_cross_species(
    n_bootstraps: int = 20,
    bootstrap_start: int | None = typer.Option(
        None, "--bootstrap-start", help="Start index for bootstrap range (for parallelization)"
    ),
    bootstrap_end: int | None = typer.Option(
        None, "--bootstrap-end", help="End index for bootstrap range (for parallelization)"
    ),
    embedding_types: str = "all",
    load_multi_model_embeddings: bool = True,
    device: str = "cpu",
    batch_size: int = 4,
    save_weights: bool = True,
    output_dir: str | None = None,
    use_wandb: bool = typer.Option(False, "--use-wandb", help="Log to Weights & Biases"),
) -> None:
    """Run cross-species probing experiments with bootstrapping.

    Args:
        n_bootstraps: Number of bootstrap iterations.
        embedding_types: Comma-separated list of embedding types to use, or 'all' for all available.
        load_multi_model_embeddings: Whether to load multi-model embeddings if not present.
        device: Device for loading embeddings ('cpu' or 'cuda').
        batch_size: Batch size for embedding generation.
        save_weights: Whether to save model weights for interpretability.
        output_dir: Output directory. If None, uses default from config.
    """
    config = get_config()
    if output_dir is None:
        output_dir = DATA_ROOT / "probing_results" / "cross_species"
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    weights_dir = output_dir / "model_weights"
    if save_weights:
        weights_dir.mkdir(parents=True, exist_ok=True)

    # Initialize wandb if requested
    wandb_run = None
    if use_wandb:
        try:
            wandb_run = init_wandb(
                project=config.get("wandb", {}).get("project", "inflamm-debate-fm"),
                tags=config.get("wandb", {}).get("tags", []) + ["probing", "cross-species"],
                config={
                    "n_bootstraps": n_bootstraps,
                    "embedding_types": embedding_types,
                    "save_weights": save_weights,
                },
            )
            if wandb_run:
                wandb_run.name = "probing_cross_species"
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
            use_wandb = False

    logger.info("Loading combined data...")
    combined_adatas, embedding_keys = _load_and_prepare_data(
        load_multi_model_embeddings=load_multi_model_embeddings,
        device=device,
        batch_size=batch_size,
        embedding_types=embedding_types,
    )
    human_adata = combined_adatas["human"]
    mouse_adata = combined_adatas["mouse"]

    setups = get_setup_transforms()
    logger.info(
        f"Running cross-species probing with {len(setups)} setups and {n_bootstraps} bootstraps..."
    )

    all_results, all_roc_data, all_weights = evaluate_cross_species(
        human_adata=human_adata,
        mouse_adata=mouse_adata,
        setups=setups,
        n_bootstraps=n_bootstraps,
        bootstrap_start=bootstrap_start,
        bootstrap_end=bootstrap_end,
        embedding_keys=embedding_keys,
        save_weights=save_weights,
        weights_output_dir=weights_dir if save_weights else None,
        use_wandb=use_wandb,
        wandb_run=wandb_run,
    )

    # Save results with bootstrap range in filename if using parallelization
    if bootstrap_start is not None and bootstrap_end is not None:
        results_filename = f"cross_species_results_bs_{bootstrap_start}_{bootstrap_end}.pkl"
    else:
        results_filename = "cross_species_results.pkl"

    results_path = output_dir / results_filename
    with results_path.open("wb") as f:
        pickle.dump(
            {
                "results": all_results,
                "roc_data": all_roc_data,
                "weights": all_weights,
                "embedding_keys": embedding_keys,
                "n_bootstraps": n_bootstraps,
            },
            f,
        )
    logger.success(f"Saved results to {results_path}")

    # Save summary CSV
    summary_path = output_dir / "cross_species_summary.csv"
    _save_summary_csv(all_results, summary_path)
    logger.success(f"Saved summary to {summary_path}")

    if use_wandb and wandb_run:
        wandb_run.finish()
        logger.info("Wandb run completed")


def _save_summary_csv(results: dict, output_path: Path) -> None:
    """Save results summary as CSV."""
    import pandas as pd

    rows = []
    for key, value in results.items():
        if isinstance(value, dict) and "mean" in value:
            # Cross-species bootstrap results
            rows.append(
                {
                    "setup": key,
                    "auroc_mean": value["mean"],
                    "auroc_std": value["std"],
                }
            )
        elif isinstance(value, tuple) and len(value) == 2:
            # Within-species CV/LODO results
            rows.append(
                {
                    "setup": key,
                    "auroc_mean": value[0],
                    "auroc_std": value[1],
                }
            )
        else:
            # Nested structure (CrossValidation/LODO)
            for val_type, model_results in value.items():
                if isinstance(model_results, dict):
                    for model_type, data_results in model_results.items():
                        if isinstance(data_results, dict):
                            for data_key, data_value in data_results.items():
                                if isinstance(data_value, tuple) and len(data_value) == 2:
                                    rows.append(
                                        {
                                            "validation_type": val_type,
                                            "model_type": model_type,
                                            "setup": data_key,
                                            "auroc_mean": data_value[0],
                                            "auroc_std": data_value[1],
                                        }
                                    )

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
    else:
        logger.warning(f"No results to save to {output_path}")
