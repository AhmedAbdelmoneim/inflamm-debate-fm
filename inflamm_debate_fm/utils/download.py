"""Download utilities for data and models."""

from pathlib import Path

from loguru import logger
import requests
from tqdm import tqdm

# Zenodo record for BulkFormer model
ZENODO_RECORD_ID = "15559368"
ZENODO_BASE_URL = f"https://zenodo.org/record/{ZENODO_RECORD_ID}/files"

# Files to download from Zenodo
ZENODO_FILES = {
    "model": [
        "Bulkformer_ckpt_epoch_29.pt",
        "checkpoint.pt",
        "G_gtex.pt",
        "G_gtex_weight.pt",
        "esm2_feature_concat.pt",
        "high_var_gene_list.pt",
    ],
    "data": [
        "G_gtex.pt",
        "G_gtex_weight.pt",
        "esm2_feature_concat.pt",
        "high_var_gene_list.pt",
        "bulkformer_gene_info.csv",
        "gene_length_df.csv",
        "demo.csv",
    ],
}


def download_file(url: str, dest_path: Path, chunk_size: int = 8192) -> None:
    """Download a file from a URL with progress bar.

    Parameters
    ----------
    url : str
        URL to download from
    dest_path : Path
        Destination path for the file
    chunk_size : int
        Chunk size for streaming download
    """
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Downloading {url} to {dest_path}...")

    response = requests.get(url, stream=True)
    response.raise_for_status()

    total_size = int(response.headers.get("content-length", 0))

    with (
        open(dest_path, "wb") as f,
        tqdm(
            desc=dest_path.name,
            total=total_size,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar,
    ):
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    logger.success(f"Downloaded {dest_path.name}")


def download_from_zenodo(
    file_name: str,
    dest_dir: Path,
    record_id: str | None = None,
    overwrite: bool = False,
) -> Path:
    """Download a file from Zenodo.

    Parameters
    ----------
    file_name : str
        Name of the file to download
    dest_dir : Path
        Destination directory
    record_id : str, optional
        Zenodo record ID. If None, uses default.
    overwrite : bool
        Whether to overwrite existing files

    Returns
    -------
    dest_path : Path
        Path to downloaded file
    """
    if record_id is None:
        record_id = ZENODO_RECORD_ID

    url = f"https://zenodo.org/record/{record_id}/files/{file_name}?download=1"
    dest_path = dest_dir / file_name

    if dest_path.exists() and not overwrite:
        logger.info(f"File already exists: {dest_path}. Skipping download.")
        return dest_path

    try:
        download_file(url, dest_path)
    except requests.exceptions.HTTPError as e:
        logger.error(f"Failed to download {file_name}: {e}")
        raise

    return dest_path


def download_bulkformer_models(
    model_dir: Path,
    data_dir: Path,
    overwrite: bool = False,
    record_id: str | None = None,
) -> None:
    """Download BulkFormer model files from Zenodo.

    Parameters
    ----------
    model_dir : Path
        Directory to save model files
    data_dir : Path
        Directory to save data files
    overwrite : bool
        Whether to overwrite existing files
    record_id : str, optional
        Zenodo record ID. If None, uses default.
    """
    logger.info("Downloading BulkFormer model files from Zenodo...")

    model_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    # Download model files
    for file_name in ZENODO_FILES["model"]:
        try:
            download_from_zenodo(file_name, model_dir, record_id=record_id, overwrite=overwrite)
        except requests.exceptions.HTTPError:
            logger.warning(f"Failed to download {file_name}, skipping...")

    # Download data files
    for file_name in ZENODO_FILES["data"]:
        try:
            download_from_zenodo(file_name, data_dir, record_id=record_id, overwrite=overwrite)
        except requests.exceptions.HTTPError:
            logger.warning(f"Failed to download {file_name}, skipping...")

    logger.success("BulkFormer model files downloaded successfully")


def download_geo_dataset(
    gse_id: str,
    dest_dir: Path,
    overwrite: bool = False,
) -> None:
    """Download a GEO dataset using GEOparse.

    Parameters
    ----------
    gse_id : str
        GEO Series ID (e.g., "GSE37069")
    dest_dir : Path
        Directory to download and cache GEO data
    overwrite : bool
        Whether to re-download if already exists
    """
    import GEOparse

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    cache_file = dest_dir / f"{gse_id}_family.soft.gz"
    if cache_file.exists() and not overwrite:
        logger.info(f"GEO dataset {gse_id} already downloaded. Use --overwrite to re-download.")
        return

    logger.info(f"Downloading GEO dataset {gse_id}...")
    try:
        gse = GEOparse.get_GEO(geo=gse_id, destdir=str(dest_dir))
        logger.success(f"Downloaded GEO dataset {gse_id}: {len(gse.gsms)} samples")
    except Exception as e:
        logger.error(f"Failed to download {gse_id}: {e}")
        raise
