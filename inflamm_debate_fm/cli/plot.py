"""Plotting commands."""

from pathlib import Path
import pickle

from loguru import logger
import typer

from inflamm_debate_fm.cli.utils import get_setup_transforms
from inflamm_debate_fm.config import DATA_DIR, FIGURES_DIR
from inflamm_debate_fm.plots import (
    plot_all,
    plot_auroc_bar_clean_sns_top_legend,
    plot_roc_facet_clean,
)

app = typer.Typer(help="Plotting commands")


@app.command("within-species")
def plot_within_species(
    species: str = typer.Option(..., help="Species: 'human' or 'mouse'"),
    results_dir: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """Generate plots for within-species results.

    Args:
        species: Species to plot ('human' or 'mouse').
        results_dir: Directory containing results.
        output_dir: Directory to save plots.
    """
    if results_dir is None:
        results_dir = DATA_DIR / "within_species_results"
    else:
        results_dir = Path(results_dir)

    if output_dir is None:
        output_dir = FIGURES_DIR / "within_species" / species
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results_path = results_dir / f"{species}_results.pkl"
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with results_path.open("rb") as f:
        data = pickle.load(f)
        all_results = data["results"]
        all_roc_data = data["roc_data"]

    # Generate plots
    logger.info(f"Generating plots for {species}...")
    for model_type in ["Linear", "Nonlinear"]:
        plot_all(
            species=species.capitalize(),
            model=model_type,
            combined_results={"results": all_results, "roc_data": all_roc_data},
        )

    logger.success(f"Saved plots to {output_dir}")


@app.command("cross-species")
def plot_cross_species(
    results_dir: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """Generate plots for cross-species results.

    Args:
        results_dir: Directory containing results.
        output_dir: Directory to save plots.
    """
    if results_dir is None:
        results_dir = DATA_DIR / "cross_species_results"
    else:
        results_dir = Path(results_dir)

    if output_dir is None:
        output_dir = FIGURES_DIR / "cross_species"
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results_path = results_dir / "cross_species_results.pkl"
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    with results_path.open("rb") as f:
        data = pickle.load(f)
        all_results = data["results"]
        all_roc_data = data["roc_data"]

    # Get setup order
    setups = get_setup_transforms()
    setup_order = [s[0] for s in setups]

    # Generate plots
    logger.info("Generating cross-species plots...")
    # Plot AUROC bar
    plot_auroc_bar_clean_sns_top_legend(all_results=all_results, setup_order=setup_order)
    # Plot ROC curves
    plot_roc_facet_clean(all_roc_data=all_roc_data, model_type="Linear", setup_order=setup_order)

    logger.success(f"Saved plots to {output_dir}")
