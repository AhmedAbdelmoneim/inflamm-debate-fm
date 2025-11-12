"""Configuration module for inflamm-debate-fm."""

from inflamm_debate_fm.config.config import (
    BULKFORMER_DATA_DIR,
    BULKFORMER_MODEL_DIR,
    DATA_DIR,
    DATA_ROOT,
    EXTERNAL_DATA_DIR,
    FIGURES_DIR,
    GEO_DOWNLOAD_DIR,
    INTERIM_DATA_DIR,
    MODELS_DIR,
    MODELS_ROOT,
    PROCESSED_DATA_DIR,
    PROJ_ROOT,
    RAW_DATA_DIR,
    REPORTS_DIR,
    get_config,
    load_config,
)

# Export all configuration variables and functions
__all__ = [
    "BULKFORMER_DATA_DIR",
    "BULKFORMER_MODEL_DIR",
    "DATA_DIR",
    "DATA_ROOT",
    "EXTERNAL_DATA_DIR",
    "FIGURES_DIR",
    "GEO_DOWNLOAD_DIR",
    "INTERIM_DATA_DIR",
    "MODELS_DIR",
    "MODELS_ROOT",
    "PROCESSED_DATA_DIR",
    "PROJ_ROOT",
    "RAW_DATA_DIR",
    "REPORTS_DIR",
    "get_config",
    "load_config",
]
