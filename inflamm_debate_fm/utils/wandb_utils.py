"""Wandb utility functions."""

import os
from typing import Any

from loguru import logger
import wandb

from inflamm_debate_fm.config import get_config


def init_wandb(
    project: str | None = None,
    entity: str | None = None,
    tags: list[str] | None = None,
    config: dict[str, Any] | None = None,
    mode: str = "online",
) -> wandb.Run:
    """Initialize wandb run.

    Args:
        project: Wandb project name. If None, uses config default.
        entity: Wandb entity name. If None, uses config or environment variable.
        tags: List of tags for the run.
        config: Additional config to log.
        mode: Wandb mode ('online', 'offline', 'disabled').

    Returns:
        Wandb run object.
    """
    wandb_config = get_config().get("wandb", {})
    project = project or wandb_config.get("project", "inflamm-debate-fm")
    entity = entity or wandb_config.get("entity") or os.getenv("WANDB_ENTITY")
    tags = tags or wandb_config.get("tags", [])

    # Merge config
    if config is None:
        config = {}
    config.update(get_config())

    logger.info(f"Initializing wandb: project={project}, entity={entity}, mode={mode}")

    run = wandb.init(project=project, entity=entity, tags=tags, config=config, mode=mode)
    return run


def log_results(results: dict[str, Any], prefix: str = "", step: int | None = None) -> None:
    """Log results to wandb.

    Args:
        results: Dictionary of results to log.
        prefix: Prefix to add to all keys.
        step: Step number for logging.
    """
    if not wandb.run:
        logger.warning("Wandb run not initialized. Skipping logging.")
        return

    logged_results = {}
    for key, value in results.items():
        if prefix:
            logged_key = f"{prefix}/{key}"
        else:
            logged_key = key

        # Handle nested dictionaries
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                logged_results[f"{logged_key}/{sub_key}"] = sub_value
        else:
            logged_results[logged_key] = value

    wandb.log(logged_results, step=step)
    logger.info(f"Logged {len(logged_results)} metrics to wandb")
