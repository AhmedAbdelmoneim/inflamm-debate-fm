"""Setup script for BulkFormer model and data files."""

from pathlib import Path
import subprocess
import sys

from loguru import logger

from inflamm_debate_fm.config import MODELS_ROOT

BULKFORMER_REPO_URL = "https://github.com/KangBoming/BulkFormer.git"
ZENODO_RECORD = "15559368"
BULKFORMER_DIR = MODELS_ROOT / "BulkFormer"
MODEL_DIR = BULKFORMER_DIR / "model"
DATA_DIR = BULKFORMER_DIR / "data"

# Required files from Zenodo
REQUIRED_MODEL_FILES = [
    "Bulkformer_ckpt_epoch_29.pt",
    "G_gtex.pt",
    "G_gtex_weight.pt",
    "esm2_feature_concat.pt",
    "high_var_gene_list.pt",
]

REQUIRED_DATA_FILES = [
    "bulkformer_gene_info.csv",
    "gene_length_df.csv",
]


def clone_repo() -> None:
    """Clone BulkFormer repository if it doesn't exist."""
    if BULKFORMER_DIR.exists() and (BULKFORMER_DIR / ".git").exists():
        logger.info(f"BulkFormer repository already exists at {BULKFORMER_DIR}")
        return

    logger.info(f"Cloning BulkFormer repository to {BULKFORMER_DIR}...")
    BULKFORMER_DIR.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "clone", BULKFORMER_REPO_URL, str(BULKFORMER_DIR)],
        check=True,
    )
    logger.success("Repository cloned successfully")


def check_files() -> tuple[list[str], list[str]]:
    """Check which files are missing."""
    missing_model = []
    missing_data = []

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename in REQUIRED_MODEL_FILES:
        filepath = MODEL_DIR / filename
        if not filepath.exists():
            missing_model.append(filename)
        else:
            logger.debug(f"Found model file: {filename}")

    for filename in REQUIRED_DATA_FILES:
        filepath = DATA_DIR / filename
        if not filepath.exists():
            missing_data.append(filename)
        else:
            logger.debug(f"Found data file: {filename}")

    return missing_model, missing_data


def download_from_zenodo(filenames: list[str], target_dir: Path) -> None:
    """Download files from Zenodo record."""
    if not filenames:
        return

    logger.info(f"Downloading {len(filenames)} file(s) from Zenodo record {ZENODO_RECORD}...")
    logger.warning(
        f"Please download the following files manually from "
        f"https://zenodo.org/records/{ZENODO_RECORD}:\n"
        + "\n".join(f"  - {f}" for f in filenames)
        + f"\n\nPlace them in: {target_dir}"
    )

    # Note: We could use zenodo_get or similar tool here, but manual download is more reliable
    # for large files. The user can download via browser or use wget/curl if preferred.


def setup_bulkformer() -> None:
    """Set up BulkFormer repository and model files."""
    logger.info("Setting up BulkFormer...")

    # Clone repository
    clone_repo()

    # Check for missing files
    missing_model, missing_data = check_files()

    if missing_model or missing_data:
        logger.warning("Some required files are missing:")
        if missing_model:
            logger.warning(f"  Model files: {', '.join(missing_model)}")
        if missing_data:
            logger.warning(f"  Data files: {', '.join(missing_data)}")

        # Download missing files
        if missing_model:
            download_from_zenodo(missing_model, MODEL_DIR)
        if missing_data:
            download_from_zenodo(missing_data, DATA_DIR)

        logger.error("Please download missing files and run setup again")
        sys.exit(1)
    else:
        logger.success("All required files are present")


if __name__ == "__main__":
    setup_bulkformer()
