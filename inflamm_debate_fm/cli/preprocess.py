"""Data preprocessing commands."""

from pathlib import Path

from loguru import logger
import typer

from inflamm_debate_fm.config import DATA_DIR, get_config
from inflamm_debate_fm.data.geo_preprocessing import GSE_IDS, process_gse_dataset

app = typer.Typer(help="Data preprocessing commands")


@app.command("gse")
def preprocess_gse(
    dataset_name: str = typer.Option(..., help="Dataset name (e.g., human_burn)"),
    gse_id: str | None = typer.Option(None, help="GEO Series ID (e.g., GSE37069)"),
    geo_download_dir: Path | None = typer.Option(None, help="GEO download directory"),
    output_dir: Path | None = typer.Option(None, help="Output directory for processed data"),
    species: str = typer.Option("human", help="Species (human or mouse)"),
    overwrite: bool = typer.Option(False, help="Overwrite existing files"),
) -> None:
    """Download and preprocess a GEO dataset into AnnData format.

    Examples:
        # Preprocess human burn dataset
        python -m inflamm_debate_fm.cli preprocess gse --dataset-name human_burn --gse-id GSE37069

        # Preprocess mouse burn dataset
        python -m inflamm_debate_fm.cli preprocess gse --dataset-name mouse_burn --gse-id GSE7404 --species mouse
    """
    config = get_config()

    if gse_id is None:
        # Try to find GSE ID from dataset name
        dataset_key = dataset_name.replace("_", "").title()
        if dataset_key in GSE_IDS:
            gse_id = GSE_IDS[dataset_key]
            logger.info(f"Using GSE ID {gse_id} for dataset {dataset_name}")
        else:
            raise ValueError(
                f"GSE ID not found for dataset {dataset_name}. "
                f"Please specify --gse-id. Available datasets: {list(GSE_IDS.keys())}"
            )

    if geo_download_dir is None:
        geo_download_dir = DATA_DIR / config["paths"]["geo_download_dir"]
    else:
        geo_download_dir = Path(geo_download_dir)

    if output_dir is None:
        output_dir = DATA_DIR / config["paths"]["raw_data_dir"]
    else:
        output_dir = Path(output_dir)

    # Download and process GSE dataset
    process_gse_dataset(
        gse_id=gse_id,
        dataset_name=dataset_name,
        geo_download_dir=geo_download_dir,
        output_dir=output_dir,
        species=species,
    )


@app.command("all")
def preprocess_all(
    ann_data_dir: Path | None = None,
    embeddings_dir: Path | None = None,
    combined_data_dir: Path | None = None,
    load_embeddings: bool = True,
    skip_preprocessing: bool = False,
    skip_combining: bool = False,
) -> None:
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
