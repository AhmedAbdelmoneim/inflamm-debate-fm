"""Configuration loading and management."""

import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from loguru import logger

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Python < 3.11

# Compute PROJ_ROOT relative to this file
# config/config.py is at inflamm_debate_fm/config/config.py
# PROJ_ROOT is two levels up from config/config.py
PROJ_ROOT = Path(__file__).resolve().parents[2]

# Load environment variables from .env file if it exists
# Load from project root directory
env_path = PROJ_ROOT / ".env"
if env_path.exists():
    load_dotenv(dotenv_path=env_path, override=False)
    logger.debug(f"Loaded environment variables from {env_path}")
else:
    # Also try loading from current directory (backward compatibility)
    # Use override=False to respect existing environment variables
    load_dotenv(override=False)

# Use environment variables if set, otherwise use defaults
# DATA_ROOT and MODELS_ROOT are configurable via environment variables
# This allows storing large data files in a different location (e.g., on HPC)
DATA_ROOT = Path(os.environ.get("INFLAMM_DEBATE_FM_DATA_ROOT", PROJ_ROOT / "data"))
MODELS_ROOT = Path(os.environ.get("INFLAMM_DEBATE_FM_MODELS_ROOT", PROJ_ROOT / "models"))

logger.info(f"PROJ_ROOT path is: {PROJ_ROOT}")
logger.info(f"DATA_ROOT path is: {DATA_ROOT}")
logger.info(f"MODELS_ROOT path is: {MODELS_ROOT}")

# DATA_DIR is the same as DATA_ROOT (for backward compatibility)
DATA_DIR = DATA_ROOT
MODELS_DIR = MODELS_ROOT

# Internal data structure (relative to DATA_ROOT)
RAW_DATA_DIR = DATA_DIR / "raw"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
GEO_DOWNLOAD_DIR = DATA_DIR / "raw" / "geo_downloads"

# BulkFormer directories (relative to MODELS_ROOT)
BULKFORMER_MODEL_DIR = MODELS_DIR / "BulkFormer" / "model"
BULKFORMER_DATA_DIR = MODELS_DIR / "BulkFormer" / "data"

# Other directories (usually in project root)
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

# Configure loguru to work with tqdm if available
# https://github.com/Delgan/loguru/issues/135
try:
    from tqdm import tqdm

    logger.remove(0)
    logger.add(lambda msg: tqdm.write(msg, end=""), colorize=True)
except ModuleNotFoundError:
    pass

# Path to default config file
DEFAULT_CONFIG_PATH = Path(__file__).parent / "default.toml"

# Global config cache
_config_cache: dict[str, Any] | None = None


def load_config(config_path: Path | str | None = None) -> dict[str, Any]:
    """Load configuration from TOML file.

    Args:
        config_path: Path to config file. If None, uses default.toml.

    Returns:
        Configuration dictionary.

    Raises:
        FileNotFoundError: If config file does not exist.
    """
    global _config_cache

    if config_path is None:
        config_path = DEFAULT_CONFIG_PATH
    else:
        config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Cache config if loading default
    if config_path == DEFAULT_CONFIG_PATH and _config_cache is not None:
        return _config_cache

    logger.info(f"Loading config from {config_path}")
    with config_path.open("rb") as f:
        config = tomllib.load(f)

    # Override paths with environment variables if set
    if "paths" in config:
        for key in config["paths"]:
            env_key = f"INFLAMM_DEBATE_FM_{key.upper()}"
            if env_key in os.environ:
                config["paths"][key] = os.environ[env_key]
                logger.info(f"Overriding {key} from environment: {config['paths'][key]}")

    # Resolve relative paths
    if "paths" in config:
        for key, value in config["paths"].items():
            if isinstance(value, str) and not Path(value).is_absolute():
                if "model_coefficients" in key or "gsea" in key or "post_analysis" in key:
                    config["paths"][key] = str(DATA_DIR / value)
                elif "embeddings" in key or "ann_data" in key or "combined" in key:
                    config["paths"][key] = str(DATA_DIR / value)
                else:
                    config["paths"][key] = str(PROJ_ROOT / value)

    # Cache default config
    if config_path == DEFAULT_CONFIG_PATH:
        _config_cache = config

    return config


def get_config() -> dict[str, Any]:
    """Get cached configuration or load default.

    Returns:
        Configuration dictionary. If cache is empty, loads default config.
    """
    if _config_cache is None:
        return load_config()
    return _config_cache
