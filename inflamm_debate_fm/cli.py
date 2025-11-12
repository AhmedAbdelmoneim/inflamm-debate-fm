"""Command-line interface for inflamm-debate-fm."""

from pathlib import Path
import pickle
from typing import List, Optional

import anndata as ad
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
from inflamm_debate_fm.embeddings.generate import generate_embeddings
from inflamm_debate_fm.modeling.coefficients import (
    evaluate_linear_models,
    post_analysis_from_coeffs,
)
from inflamm_debate_fm.modeling.evaluation import (
    evaluate_cross_species,
    evaluate_within_species,
)
from inflamm_debate_fm.plots import (
    plot_all,
    plot_auroc_bar_clean_sns_top_legend,
    plot_roc_facet_clean,
)
from inflamm_debate_fm.utils.wandb_utils import init_wandb

app = typer.Typer(help="inflamm-debate-fm CLI")

# Subcommands
preprocess_app = typer.Typer(help="Data preprocessing commands")
embed_app = typer.Typer(help="Embedding generation commands")
probe_app = typer.Typer(help="Probing experiment commands")
analyze_app = typer.Typer(help="Analysis commands")
plot_app = typer.Typer(help="Plotting commands")

app.add_typer(preprocess_app, name="preprocess")
app.add_typer(embed_app, name="embed")
app.add_typer(probe_app, name="probe")
app.add_typer(analyze_app, name="analyze")
app.add_typer(plot_app, name="plot")


# Preprocess commands
@preprocess_app.command("all")
def preprocess_all(
    ann_data_dir: Optional[Path] = None,
    embeddings_dir: Optional[Path] = None,
    combined_data_dir: Optional[Path] = None,
    load_embeddings: bool = True,
    skip_preprocessing: bool = False,
    skip_combining: bool = False,
):
    """Run the complete data preprocessing pipeline."""
    from inflamm_debate_fm.dataset import main as preprocess_main

    preprocess_main(
        ann_data_dir=ann_data_dir,
        embeddings_dir=embeddings_dir,
        combined_data_dir=combined_data_dir,
        load_embeddings=load_embeddings,
        skip_preprocessing=skip_preprocessing,
        skip_combining=skip_combining,
    )


# Embed commands
@embed_app.command("generate")
def embed_generate(
    dataset_name: str,
    ann_data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    model_name: str = "bulkformer",
    flavor: str = "default",
    batch_size: int = 32,
    device: str = "cpu",
):
    """Generate embeddings for a dataset.

    Args:
        dataset_name: Name of the dataset to process.
        ann_data_dir: Directory containing AnnData files.
        output_dir: Directory to save embeddings.
        model_name: Name of the embedding model.
        flavor: Embedding flavor identifier.
        batch_size: Batch size for embedding generation.
        device: Device to use ('cpu' or 'cuda').
    """
    config = get_config()

    if ann_data_dir is None:
        ann_data_dir = DATA_DIR / config["paths"]["ann_data_dir"]
    else:
        ann_data_dir = Path(ann_data_dir)

    if output_dir is None:
        output_dir = DATA_DIR / config["paths"]["embeddings_dir"]
    else:
        output_dir = Path(output_dir)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Load AnnData
    adata_path = ann_data_dir / f"{dataset_name}_orthologs.h5ad"
    if not adata_path.exists():
        raise FileNotFoundError(f"AnnData file not found: {adata_path}")

    adata = ad.read_h5ad(adata_path)
    logger.info(f"Loaded {dataset_name}: {adata.shape}")

    # Generate embeddings
    logger.info(f"Generating embeddings for {dataset_name}...")
    embeddings = generate_embeddings(
        adata=adata,
        model_name=model_name,
        flavor=flavor,
        batch_size=batch_size,
        device=device,
        output_dir=output_dir,
    )

    # Save embeddings
    output_path = output_dir / f"{dataset_name}_transcriptome_embeddings.npy"
    import numpy as np

    np.save(output_path, embeddings)
    logger.success(f"Saved embeddings to {output_path}")


# Probe commands
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


@probe_app.command("within-species")
def probe_within_species(
    species: str = typer.Option(..., help="Species: 'human' or 'mouse'"),
    combined_data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    n_cv_folds: int = 10,
    use_wandb: bool = False,
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


@probe_app.command("cross-species")
def probe_cross_species(
    combined_data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    use_wandb: bool = False,
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


# Analyze commands
@analyze_app.command("coefficients")
def analyze_coefficients(
    combined_data_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    use_wandb: bool = False,
):
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


@analyze_app.command("gsea")
def analyze_gsea(
    coeff_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
):
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


# Plot commands
@plot_app.command("within-species")
def plot_within_species(
    species: str = typer.Option(..., help="Species: 'human' or 'mouse'"),
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
):
    """Generate plots for within-species results.

    Args:
        species: Species to plot ('human' or 'mouse').
        results_dir: Directory containing results.
        output_dir: Directory to save plots.
    """
    from inflamm_debate_fm.config import FIGURES_DIR

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


@plot_app.command("cross-species")
def plot_cross_species(
    results_dir: Optional[Path] = None,
    output_dir: Optional[Path] = None,
):
    """Generate plots for cross-species results.

    Args:
        results_dir: Directory containing results.
        output_dir: Directory to save plots.
    """
    from inflamm_debate_fm.config import FIGURES_DIR

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


if __name__ == "__main__":
    app()
