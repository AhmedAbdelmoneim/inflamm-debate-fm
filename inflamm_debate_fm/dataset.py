"""Data preprocessing pipeline orchestration."""

from pathlib import Path

from loguru import logger
import typer

from inflamm_debate_fm.config import DATA_DIR, get_config
from inflamm_debate_fm.data.load import combine_adatas, load_adatas
from inflamm_debate_fm.data.preprocessing import preprocess_all_datasets

app = typer.Typer()


@app.command()
def main(
    ann_data_dir: Path | None = None,
    embeddings_dir: Path | None = None,
    combined_data_dir: Path | None = None,
    load_embeddings: bool = True,
    skip_preprocessing: bool = False,
    skip_combining: bool = False,
):
    """Orchestrate the data preprocessing pipeline.

    This function:
    1. Loads AnnData files from the orthologs directory
    2. Optionally loads and adds embeddings
    3. Runs preprocessing (timepoint categorization, inflammation categories)
    4. Combines human and mouse datasets separately
    5. Saves combined datasets

    Args:
        ann_data_dir: Directory containing AnnData files. If None, uses config default.
        embeddings_dir: Directory containing embedding files. If None, uses config default.
        combined_data_dir: Directory to save combined datasets. If None, uses config default.
        load_embeddings: Whether to load and add embeddings to AnnData objects.
        skip_preprocessing: Skip preprocessing step if datasets are already preprocessed.
        skip_combining: Skip combining step if combined datasets already exist.
    """
    config = get_config()

    # Set up directories
    if ann_data_dir is None:
        ann_data_dir = DATA_DIR / config["paths"]["ann_data_dir"]
    else:
        ann_data_dir = Path(ann_data_dir)

    if embeddings_dir is None:
        embeddings_dir = DATA_DIR / config["paths"]["embeddings_dir"]
    else:
        embeddings_dir = Path(embeddings_dir)

    if combined_data_dir is None:
        combined_data_dir = DATA_DIR / config["paths"]["combined_data_dir"]
    else:
        combined_data_dir = Path(combined_data_dir)

    # Create output directory if it doesn't exist
    combined_data_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 80)
    logger.info("Data Preprocessing Pipeline")
    logger.info("=" * 80)
    logger.info(f"AnnData directory: {ann_data_dir}")
    logger.info(f"Embeddings directory: {embeddings_dir}")
    logger.info(f"Combined data directory: {combined_data_dir}")
    logger.info(f"Load embeddings: {load_embeddings}")
    logger.info("=" * 80)

    # Step 1: Load AnnData files
    logger.info("\n[Step 1] Loading AnnData files...")
    adatas = load_adatas(
        ann_data_dir=ann_data_dir,
        embeddings_dir=embeddings_dir,
        load_embeddings=load_embeddings,
    )
    logger.success(f"Loaded {len(adatas)} datasets")

    # Step 2: Preprocessing
    if not skip_preprocessing:
        logger.info("\n[Step 2] Preprocessing datasets...")
        preprocess_all_datasets(adatas)
        logger.success("Preprocessing complete")
    else:
        logger.info("\n[Step 2] Skipping preprocessing (--skip-preprocessing)")

    # Step 3: Combine datasets by species
    if not skip_combining:
        logger.info("\n[Step 3] Combining datasets by species...")

        # Combine human datasets
        logger.info("Combining human datasets...")
        human_combined = combine_adatas(adatas, "human")
        human_path = combined_data_dir / "human_combined.h5ad"
        human_combined.write_h5ad(human_path)
        logger.success(f"Saved human combined dataset to {human_path}")

        # Combine mouse datasets
        logger.info("Combining mouse datasets...")
        mouse_combined = combine_adatas(adatas, "mouse")
        mouse_path = combined_data_dir / "mouse_combined.h5ad"
        mouse_combined.write_h5ad(mouse_path)
        logger.success(f"Saved mouse combined dataset to {mouse_path}")

        logger.success("Combining complete")
    else:
        logger.info("\n[Step 3] Skipping combining (--skip-combining)")

    logger.info("\n" + "=" * 80)
    logger.success("Data preprocessing pipeline complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    app()
