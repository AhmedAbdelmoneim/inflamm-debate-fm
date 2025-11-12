import os
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

# Load environment variables from .env file if it exists
load_dotenv()

# Paths
PROJ_ROOT = Path(__file__).resolve().parents[1]
logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")

# Data directory can be configured via environment variable
# This allows storing large data files in a different location (e.g., on HPC)
DATA_ROOT = Path(os.environ.get("INFLAMM_DEBATE_FM_DATA_ROOT", PROJ_ROOT / "data"))
logger.info(f"DATA_ROOT path is: {DATA_ROOT}")

# Internal data structure (relative to DATA_ROOT)
DATA_DIR = DATA_ROOT
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
GEO_DOWNLOAD_DIR = DATA_DIR / "raw" / "geo_downloads"

# Models directory (can also be configured)
MODELS_ROOT = Path(os.environ.get("INFLAMM_DEBATE_FM_MODELS_ROOT", PROJ_ROOT / "models"))
logger.info(f"MODELS_ROOT path is: {MODELS_ROOT}")

MODELS_DIR = MODELS_ROOT
BULKFORMER_MODEL_DIR = MODELS_DIR / "BulkFormer" / "model"
BULKFORMER_DATA_DIR = MODELS_DIR / "BulkFormer" / "data"

# Reports directory (usually in project root)
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# If tqdm is installed, configure loguru with tqdm.write
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass
