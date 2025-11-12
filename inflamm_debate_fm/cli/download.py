"""Download commands for data and models."""

from pathlib import Path

from loguru import logger
import typer

from inflamm_debate_fm.config import (
    BULKFORMER_DATA_DIR,
    BULKFORMER_MODEL_DIR,
    DATA_DIR,
    get_config,
)
from inflamm_debate_fm.data.geo_preprocessing import GSE_IDS
from inflamm_debate_fm.utils.download import download_bulkformer_models, download_geo_dataset

app = typer.Typer(help="Download commands for data and models")


@app.command("models")
def download_models(
    model_dir: Path | None = typer.Option(None, help="Directory to save model files"),
    data_dir: Path | None = typer.Option(None, help="Directory to save data files"),
    overwrite: bool = typer.Option(False, help="Overwrite existing files"),
    record_id: str | None = typer.Option(None, help="Zenodo record ID"),
) -> None:
    """Download BulkFormer model files from Zenodo.

    Examples:
        # Download models to default location
        python -m inflamm_debate_fm.cli download models

        # Download models to custom location
        python -m inflamm_debate_fm.cli download models --model-dir /path/to/models --data-dir /path/to/data
    """
    if model_dir is None:
        model_dir = BULKFORMER_MODEL_DIR
    else:
        model_dir = Path(model_dir)

    if data_dir is None:
        data_dir = BULKFORMER_DATA_DIR
    else:
        data_dir = Path(data_dir)

    download_bulkformer_models(
        model_dir=model_dir,
        data_dir=data_dir,
        overwrite=overwrite,
        record_id=record_id,
    )


@app.command("geo")
def download_geo(
    gse_id: str = typer.Option(..., help="GEO Series ID (e.g., GSE37069)"),
    dest_dir: Path | None = typer.Option(None, help="Directory to download GEO data"),
    overwrite: bool = typer.Option(False, help="Overwrite existing files"),
) -> None:
    """Download a GEO dataset using GEOparse.

    Examples:
        # Download human burn dataset
        python -m inflamm_debate_fm.cli download geo --gse-id GSE37069

        # Download to custom directory
        python -m inflamm_debate_fm.cli download geo --gse-id GSE37069 --dest-dir /path/to/geo
    """
    config = get_config()

    if dest_dir is None:
        dest_dir = DATA_DIR / config["paths"]["geo_download_dir"]
    else:
        dest_dir = Path(dest_dir)

    download_geo_dataset(gse_id=gse_id, dest_dir=dest_dir, overwrite=overwrite)


@app.command("all")
def download_all(
    overwrite: bool = typer.Option(False, help="Overwrite existing files"),
) -> None:
    """Download all required data and models.

    This includes:
    - BulkFormer model files from Zenodo
    - All GEO datasets for inflammation studies

    Examples:
        # Download everything
        python -m inflamm_debate_fm.cli download all

        # Download with overwrite
        python -m inflamm_debate_fm.cli download all --overwrite
    """
    logger.info("Downloading all required data and models...")

    # Download BulkFormer models
    download_bulkformer_models(
        model_dir=BULKFORMER_MODEL_DIR,
        data_dir=BULKFORMER_DATA_DIR,
        overwrite=overwrite,
    )

    # Download GEO datasets
    config = get_config()
    geo_download_dir = DATA_DIR / config["paths"]["geo_download_dir"]

    for dataset_name, gse_id in GSE_IDS.items():
        logger.info(f"Downloading {dataset_name} ({gse_id})...")
        try:
            download_geo_dataset(gse_id=gse_id, dest_dir=geo_download_dir, overwrite=overwrite)
        except Exception as e:
            logger.warning(f"Failed to download {gse_id}: {e}")

    logger.success("All downloads completed!")
