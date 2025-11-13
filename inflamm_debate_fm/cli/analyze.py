"""Analysis commands."""

from loguru import logger
import typer

from inflamm_debate_fm.analysis.gsea import post_analysis_from_coeffs
from inflamm_debate_fm.cli.utils import get_setup_transforms
from inflamm_debate_fm.config import DATA_ROOT, get_config
from inflamm_debate_fm.data.load import load_combined_adatas
from inflamm_debate_fm.modeling.coefficients import evaluate_linear_models

app = typer.Typer(help="Analysis commands")


@app.command("coefficients")
def analyze_coefficients() -> None:
    """Extract and analyze model coefficients."""
    config = get_config()
    combined_data_dir = DATA_ROOT / config["paths"]["combined_data_dir"]
    output_dir = DATA_ROOT / config["paths"]["model_coefficients_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading combined data...")
    combined_adatas = load_combined_adatas(combined_data_dir)

    setups = get_setup_transforms()
    logger.info("Evaluating linear models...")
    evaluate_linear_models(
        human_adata=combined_adatas["human"],
        mouse_adata=combined_adatas["mouse"],
        setups=setups,
        output_dir=output_dir,
    )
    logger.success(f"Saved coefficients to {output_dir}")


@app.command("gsea")
def analyze_gsea() -> None:
    """Run GSEA analysis on model coefficients."""
    config = get_config()
    coeff_dir = DATA_ROOT / config["paths"]["model_coefficients_dir"]
    output_dir = DATA_ROOT / config["paths"]["gsea_results_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running GSEA analysis...")
    post_analysis_from_coeffs(coeff_dir, output_dir)
    logger.success(f"Saved GSEA results to {output_dir}")
