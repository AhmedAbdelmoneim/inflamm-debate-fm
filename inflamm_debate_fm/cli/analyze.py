"""Analysis commands."""

from pathlib import Path

from loguru import logger
import typer

from inflamm_debate_fm.cli.utils import get_setup_transforms
from inflamm_debate_fm.config import DATA_DIR, get_config
from inflamm_debate_fm.data.load import load_combined_adatas
from inflamm_debate_fm.modeling.coefficients import (
    evaluate_linear_models,
    post_analysis_from_coeffs,
)
from inflamm_debate_fm.utils.wandb_utils import init_wandb

app = typer.Typer(help="Analysis commands")


@app.command("coefficients")
def analyze_coefficients(
    combined_data_dir: Path | None = None,
    output_dir: Path | None = None,
    use_wandb: bool = False,
) -> None:
    """Extract and analyze model coefficients.

    Args:
        combined_data_dir: Directory containing combined AnnData files.
        output_dir: Directory to save coefficients.
        use_wandb: Whether to log to wandb.
    """
    config = get_config()

    if combined_data_dir is None:
        combined_data_dir = DATA_DIR / config["paths"]["combined_data_dir"]
    else:
        combined_data_dir = Path(combined_data_dir)

    if output_dir is None:
        output_dir = DATA_DIR / config["paths"]["model_coefficients_dir"]
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
        wandb_run = init_wandb(
            project=None,
            entity=None,
            tags=["coefficients", "analysis"],
            config={"experiment_type": "coefficients"},
        )

    # Run evaluation
    logger.info("Extracting coefficients...")
    all_results, all_roc_data = evaluate_linear_models(
        human_adata=human_adata,
        mouse_adata=mouse_adata,
        setups=setups,
        output_dir=output_dir,
    )

    logger.success(f"Saved coefficients to {output_dir}")

    if wandb_run:
        wandb_run.finish()


@app.command("gsea")
def analyze_gsea(
    coeff_dir: Path | None = None,
    output_dir: Path | None = None,
) -> None:
    """Run GSEA analysis on coefficients.

    Args:
        coeff_dir: Directory containing coefficient files.
        output_dir: Directory to save GSEA results.
    """
    config = get_config()

    if coeff_dir is None:
        coeff_dir = DATA_DIR / config["paths"]["model_coefficients_dir"]
    else:
        coeff_dir = Path(coeff_dir)

    if output_dir is None:
        output_dir = DATA_DIR / config["paths"]["gsea_results_dir"]
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running GSEA analysis...")
    post_analysis_from_coeffs(coeff_dir=coeff_dir, outdir=output_dir)
    logger.success(f"Saved GSEA results to {output_dir}")
