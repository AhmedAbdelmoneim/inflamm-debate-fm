"""Download raw data files."""

from pathlib import Path
import urllib.request

from loguru import logger
import requests

from inflamm_debate_fm.config import ORTHOLOGY_DIR, PLATFORMS_DIR, RAW_CEL_DIR
from inflamm_debate_fm.data.constants import GSE_IDS

MGI_FILES = {
    "HMD_HumanPhenotype.rpt": "https://www.informatics.jax.org/downloads/reports/HMD_HumanPhenotype.rpt",
    "MGI_EntrezGene.rpt": "https://www.informatics.jax.org/downloads/reports/MGI_EntrezGene.rpt",
}

BRAINARRAY_PACKAGES = {
    "hgu133plus2hsensgcdf": "https://brainarray.mbni.med.umich.edu/Brainarray/Database/CustomCDF/25.0.0/entrezg.download/hgu133plus2hsensgcdf_25.0.0.tar.gz",
    "mouse4302mmensgcdf": "https://brainarray.mbni.med.umich.edu/Brainarray/Database/CustomCDF/25.0.0/entrezg.download/mouse4302mmensgcdf_25.0.0.tar.gz",
    "mouse430a2mmensgcdf": "https://brainarray.mbni.med.umich.edu/Brainarray/Database/CustomCDF/25.0.0/entrezg.download/mouse430a2mmensgcdf_25.0.0.tar.gz",
}


def download_file(url: str, dest_path: Path) -> None:
    """Download file."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if dest_path.exists():
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
            continue

        gse_num = gse_id.replace("GSE", "")
        series_dir = f"{gse_num[:-3]}nnn"
        tar_path = gse_dir / f"{gse_id}_RAW.tar"
        ftp_url = (
            f"ftp://ftp.ncbi.nlm.nih.gov/geo/series/{series_dir}/{gse_id}/suppl/{gse_id}_RAW.tar"
        )

        try:
            urllib.request.urlretrieve(ftp_url, tar_path)
            logger.info(f"Downloaded {gse_id}_RAW.tar")
        except Exception:
            import GEOparse

            gse = GEOparse.get_GEO(geo=gse_id, destdir=str(RAW_CEL_DIR.parent))
            for gsm_name, gsm in gse.gsms.items():
                if "supplementary_file" in gsm.metadata:
                    for supp_file in gsm.metadata["supplementary_file"]:
                        if supp_file.endswith((".CEL.gz", ".CEL")):
                            cel_path = gse_dir / Path(supp_file).name
                            if not cel_path.exists():
                                if supp_file.startswith("ftp://"):
                                    urllib.request.urlretrieve(supp_file, cel_path)
                                else:
                                    response = requests.get(supp_file, timeout=120, stream=True)
                                    response.raise_for_status()
                                    with open(cel_path, "wb") as f:
                                        for chunk in response.iter_content(chunk_size=8192):
                                            f.write(chunk)
                                logger.info(f"Downloaded {cel_path.name}")

    logger.success("Downloads complete!")
