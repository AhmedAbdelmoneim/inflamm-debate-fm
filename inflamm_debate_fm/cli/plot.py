"""Plotting commands."""

import pickle

from loguru import logger
import typer

from inflamm_debate_fm.cli.utils import get_setup_transforms
from inflamm_debate_fm.config import DATA_ROOT, FIGURES_DIR
from inflamm_debate_fm.plots import (
    plot_all,
    plot_auroc_bar_clean_sns_top_legend,
    plot_roc_facet_clean,
)

app = typer.Typer(help="Plotting commands")


@app.command("within-species")
def plot_within_species(species: str) -> None:
    """Generate plots for within-species results."""
    results_dir = DATA_ROOT / "within_species_results"
    output_dir = FIGURES_DIR / "within_species" / species
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / f"{species}_results.pkl"
    with results_path.open("rb") as f:
        data = pickle.load(f)
    all_results = data["results"]
    all_roc_data = data["roc_data"]

    logger.info(f"Generating plots for {species}...")
    for model_type in ["Linear", "Nonlinear"]:
        plot_all(
            species=species.capitalize(),
            model=model_type,
            all_results=all_results,
            all_roc_data=all_roc_data,
            output_dir=output_dir,
        )
    logger.success(f"Saved plots to {output_dir}")


@app.command("cross-species")
def plot_cross_species() -> None:
    """Generate plots for cross-species results."""
    results_dir = DATA_ROOT / "cross_species_results"
    output_dir = FIGURES_DIR / "cross_species"
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = results_dir / "cross_species_results.pkl"
    with results_path.open("rb") as f:
        data = pickle.load(f)
    all_results = data["results"]
    all_roc_data = data["roc_data"]

    setups = get_setup_transforms()
    setup_order = [s[0] for s in setups]

    logger.info("Generating cross-species plots...")
    plot_auroc_bar_clean_sns_top_legend(
        all_results=all_results,
        setup_order=setup_order,
        output_path=output_dir / "cross_species_auroc_bar.png",
    )
    plot_roc_facet_clean(
        all_roc_data=all_roc_data,
        model_type="Linear",
        setup_order=setup_order,
        output_path=output_dir / "cross_species_roc_curves_linear.png",
    )
    plot_roc_facet_clean(
        all_roc_data=all_roc_data,
        model_type="Nonlinear",
        setup_order=setup_order,
        output_path=output_dir / "cross_species_roc_curves_nonlinear.png",
    )
    logger.success(f"Saved plots to {output_dir}")
