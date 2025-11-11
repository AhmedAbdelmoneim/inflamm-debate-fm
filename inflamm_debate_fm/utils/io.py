"""I/O utilities for saving and loading results."""

from pathlib import Path
import pickle
from typing import Any

from loguru import logger


def save_results(results: Any, output_path: str | Path) -> None:
    """Save results to pickle file.

    Args:
        results: Results object to save.
        output_path: Path to save results.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info(f"Saving results to {output_path}")
    with output_path.open("wb") as f:
        pickle.dump(results, f)


def load_results(input_path: str | Path) -> Any:
    """Load results from pickle file.

    Args:
        input_path: Path to load results from.

    Returns:
        Loaded results object.
    """
    input_path = Path(input_path)
    if not input_path.exists():
        raise FileNotFoundError(f"Results file not found: {input_path}")

    logger.info(f"Loading results from {input_path}")
    with input_path.open("rb") as f:
        return pickle.load(f)
