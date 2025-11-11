"""Configuration loading and management."""

import os
from pathlib import Path
from typing import Any

from loguru import logger

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Python < 3.11

# Compute PROJ_ROOT relative to this file
# config/config.py is at inflamm_debate_fm/config/config.py
# PROJ_ROOT is two levels up from config/config.py
PROJ_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJ_ROOT / "data"
MODELS_DIR = PROJ_ROOT / "models"
REPORTS_DIR = PROJ_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

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
        Configuration dictionary.
    """
    if _config_cache is None:
        return load_config()
    return _config_cache
