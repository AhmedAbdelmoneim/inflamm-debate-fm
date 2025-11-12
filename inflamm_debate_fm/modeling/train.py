"""Model training and evaluation entry point."""

from pathlib import Path
import pickle
from typing import List, Optional

from loguru import logger
import typer

from inflamm_debate_fm.config import DATA_DIR
from inflamm_debate_fm.config.config import get_config
from inflamm_debate_fm.data.load import load_combined_adatas
from inflamm_debate_fm.data.transforms import (
    transform_adata_to_X_y_acute,
    transform_adata_to_X_y_acute_and_subacute,
    transform_adata_to_X_y_acute_subacute_to_chronic,
    transform_adata_to_X_y_acute_to_chronic,
    transform_adata_to_X_y_all,
    transform_adata_to_X_y_chronic,
    transform_adata_to_X_y_subacute,
    transform_adata_to_X_y_takao,
)
from inflamm_debate_fm.modeling.evaluation import (
    evaluate_cross_species,
    evaluate_within_species,
)
from inflamm_debate_fm.utils.wandb_utils import init_wandb

app = typer.Typer()


def get_setup_transforms():
    """Get all setup transform functions."""
    return [
        ("All Inflammation Samples vs. Control", transform_adata_to_X_y_all),
        ("Takao Subset for Inflammation vs. Control", transform_adata_to_X_y_takao),
        ("Acute Inflammation vs. Control", transform_adata_to_X_y_acute),
        ("Subacute Inflammation vs. Control", transform_adata_to_X_y_subacute),
        ("Acute and Subacute Inflammation vs. Control", transform_adata_to_X_y_acute_and_subacute),
        ("Chronic Inflammation vs. Control", transform_adata_to_X_y_chronic),
        ("Acute Inflammation vs. Chronic Inflammation", transform_adata_to_X_y_acute_to_chronic),
        (
            "Acute/Subacute Inflammation vs. Chronic Inflammation",
            transform_adata_to_X_y_acute_subacute_to_chronic,
        ),
    ]


@app.command()
def within_species(
    species: str = typer.Option(..., help="Species: 'human' or 'mouse'"),
    combined_data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    n_cv_folds: int = 10,
    use_wandb: bool = True,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_tags: Optional[List[str]] = None,
):
    """Run within-species probing experiments.

    Args:
        species: Species to probe ('human' or 'mouse').
        combined_data_dir: Directory containing combined AnnData files.
        output_dir: Directory to save results.
        n_cv_folds: Number of CV folds.
        use_wandb: Whether to log to wandb.
        wandb_project: Wandb project name.
        wandb_entity: Wandb entity name.
        wandb_tags: Wandb tags.
    """
    config = get_config()

    if combined_data_dir is None:
        combined_data_dir = DATA_DIR / config["paths"]["combined_data_dir"]
    else:
        combined_data_dir = Path(combined_data_dir)

    if output_dir is None:
        output_dir = DATA_DIR / "within_species_results"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load combined data
    logger.info(f"Loading {species} combined data...")
    combined_adatas = load_combined_adatas(combined_data_dir=combined_data_dir)
    adata = combined_adatas[species]

    # Get setups
    setups = get_setup_transforms()

    # Initialize wandb if requested
    wandb_run = None
    if use_wandb:
        wandb_config = {
            "species": species,
            "n_cv_folds": n_cv_folds,
            "experiment_type": "within_species",
        }
        wandb_run = init_wandb(
            project=wandb_project,
            entity=wandb_entity,
            tags=wandb_tags or ["within_species", species],
            config=wandb_config,
        )

    # Run evaluation
    logger.info(f"Running within-species probing for {species}...")
    all_results, all_roc_data = evaluate_within_species(
        adata=adata,
        setups=setups,
        species_name=species.capitalize(),
        n_cv_folds=n_cv_folds,
        use_wandb=use_wandb,
        wandb_run=wandb_run,
    )

    # Save results
    results_path = output_dir / f"{species}_results.pkl"
    with results_path.open("wb") as f:
        pickle.dump({"results": all_results, "roc_data": all_roc_data}, f)
    logger.success(f"Saved results to {results_path}")

    if wandb_run:
        wandb_run.finish()


@app.command()
def cross_species(
    combined_data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    use_wandb: bool = True,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_tags: Optional[List[str]] = None,
):
    """Run cross-species probing experiments.

    Args:
        combined_data_dir: Directory containing combined AnnData files.
        output_dir: Directory to save results.
        use_wandb: Whether to log to wandb.
        wandb_project: Wandb project name.
        wandb_entity: Wandb entity name.
        wandb_tags: Wandb tags.
    """
    config = get_config()

    if combined_data_dir is None:
        combined_data_dir = DATA_DIR / config["paths"]["combined_data_dir"]
    else:
        combined_data_dir = Path(combined_data_dir)

    if output_dir is None:
        output_dir = DATA_DIR / "cross_species_results"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load combined data
    logger.info("Loading combined human and mouse data...")
    combined_adatas = load_combined_adatas(combined_data_dir=combined_data_dir)
    human_adata = combined_adatas["human"]
    mouse_adata = combined_adatas["mouse"]

    # Get setups
    setups = get_setup_transforms()

    # Initialize wandb if requested
    wandb_run = None
    if use_wandb:
        wandb_config = {"experiment_type": "cross_species"}
        wandb_run = init_wandb(
            project=wandb_project,
            entity=wandb_entity,
            tags=wandb_tags or ["cross_species"],
            config=wandb_config,
        )

    # Run evaluation
    logger.info("Running cross-species probing...")
    all_results, all_roc_data = evaluate_cross_species(
        human_adata=human_adata,
        mouse_adata=mouse_adata,
        setups=setups,
        use_wandb=use_wandb,
        wandb_run=wandb_run,
    )

    # Save results
    results_path = output_dir / "cross_species_results.pkl"
    with results_path.open("wb") as f:
        pickle.dump({"results": all_results, "roc_data": all_roc_data}, f)
    logger.success(f"Saved results to {results_path}")

    if wandb_run:
        wandb_run.finish()


if __name__ == "__main__":
    app()
