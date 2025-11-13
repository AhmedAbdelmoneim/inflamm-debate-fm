"""Probing experiment commands."""

import pickle

from loguru import logger
import typer

from inflamm_debate_fm.cli.utils import get_setup_transforms
from inflamm_debate_fm.config import DATA_ROOT, get_config
from inflamm_debate_fm.data.load import load_combined_adatas
from inflamm_debate_fm.modeling.evaluation import evaluate_cross_species, evaluate_within_species

app = typer.Typer(help="Probing experiment commands")


@app.command("within-species")
def probe_within_species(species: str, n_cv_folds: int = 10) -> None:
    """Run within-species probing experiments."""
    config = get_config()
    combined_data_dir = DATA_ROOT / config["paths"]["combined_data_dir"]
    output_dir = DATA_ROOT / "within_species_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading {species} combined data...")
    combined_adatas = load_combined_adatas(combined_data_dir)
    adata = combined_adatas[species]

    setups = get_setup_transforms()
    logger.info(f"Running within-species probing for {species}...")
    all_results, all_roc_data = evaluate_within_species(
        adata=adata, setups=setups, species_name=species.capitalize(), n_cv_folds=n_cv_folds
    )

    results_path = output_dir / f"{species}_results.pkl"
    with results_path.open("wb") as f:
        pickle.dump({"results": all_results, "roc_data": all_roc_data}, f)
    logger.success(f"Saved results to {results_path}")


@app.command("cross-species")
def probe_cross_species(n_cv_folds: int = 10) -> None:
    """Run cross-species probing experiments."""
    config = get_config()
    combined_data_dir = DATA_ROOT / config["paths"]["combined_data_dir"]
    output_dir = DATA_ROOT / "cross_species_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading combined data...")
    combined_adatas = load_combined_adatas(combined_data_dir)

    setups = get_setup_transforms()
    logger.info("Running cross-species probing...")
    all_results, all_roc_data = evaluate_cross_species(
        human_adata=combined_adatas["human"],
        mouse_adata=combined_adatas["mouse"],
        setups=setups,
        n_cv_folds=n_cv_folds,
    )

    results_path = output_dir / "cross_species_results.pkl"
    with results_path.open("wb") as f:
        pickle.dump({"results": all_results, "roc_data": all_roc_data}, f)
    logger.success(f"Saved results to {results_path}")
