"""Download raw data files."""

from pathlib import Path
import tarfile
import urllib.request

from loguru import logger
import requests

from inflamm_debate_fm.config import ORTHOLOGY_DIR, PLATFORMS_DIR, RAW_CEL_DIR
from inflamm_debate_fm.data.constants import (
    BRAINARRAY_PACKAGES,
    GEO_FTP_BASE_URL,
    GSE_IDS,
    MGI_FILES,
)


def download_file(url: str, dest_path: Path) -> None:
    """Download file."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
        logger.info(f"File {dest_path.name} already exists, skipping download.")
        return
    try:
        if url.startswith("ftp://"):
            urllib.request.urlretrieve(url, dest_path)
        else:
            response = requests.get(url, stream=True, timeout=60)
            response.raise_for_status()
            with open(dest_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
        logger.info(f"Downloaded {dest_path.name}")
    except Exception as e:
        logger.warning(f"Failed to download {url}: {e}")


def download_all_raw_data():
    """Download all raw data files."""
    logger.info("Downloading orthology files...")
    for filename, url in MGI_FILES.items():
        download_file(url, ORTHOLOGY_DIR / filename)

    logger.info("Downloading BrainArray packages...")
    for pkg_name, url in BRAINARRAY_PACKAGES.items():
        download_file(url, PLATFORMS_DIR / f"{pkg_name}_25.0.0.tar.gz")

    logger.info("Downloading CEL files...")

    for gse_id in GSE_IDS.values():
        gse_dir = RAW_CEL_DIR / gse_id
        gse_dir.mkdir(parents=True, exist_ok=True)

        if any(gse_dir.glob("*.CEL")) or any(gse_dir.glob("*.CEL.gz")):
            logger.info(f"Skipping {gse_id} (CEL files already present).")
            continue

        series_dir = f"{gse_id[:-3]}nnn"
        tar_path = gse_dir / f"{gse_id}_RAW.tar"
        ftp_url = f"{GEO_FTP_BASE_URL}/{series_dir}/{gse_id}/suppl/{gse_id}_RAW.tar"

        try:
            download_file(ftp_url, tar_path)
            # Extract all CEL files
            with tarfile.open(tar_path, "r") as tar:
                tar.extractall(gse_dir)
            tar_path.unlink(missing_ok=True)
            logger.info(f"Extracted CEL files for {gse_id}")
            continue  # Done with this GSE
        except Exception as e:
            logger.warning(f"Failed to download {gse_id}_RAW.tar ({e}); using GEOparse fallback.")

    logger.success("Downloads complete!")
