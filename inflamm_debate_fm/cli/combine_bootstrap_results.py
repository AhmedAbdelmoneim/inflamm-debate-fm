"""Script to combine bootstrap results from multiple parallel jobs."""

from collections import defaultdict
from pathlib import Path
import pickle

from loguru import logger
import numpy as np


def combine_bootstrap_results(
    result_files: list[Path],
    output_path: Path,
) -> None:
    """Combine bootstrap results from multiple parallel jobs.

    Args:
        result_files: List of paths to partial result pickle files.
        output_path: Path to save combined results.
    """
    logger.info(f"Combining {len(result_files)} result files...")

    # Load all results
    all_combined_results = defaultdict(lambda: {"values": [], "roc_data": [], "weights": []})

    for result_file in result_files:
        if not result_file.exists():
            logger.warning(f"Result file not found: {result_file}")
            continue

        logger.info(f"Loading {result_file}...")
        with result_file.open("rb") as f:
            data = pickle.load(f)

        results = data.get("results", {})
        roc_data = data.get("roc_data", {})
        weights = data.get("weights", {})

        # Combine results
        for key, value in results.items():
            if isinstance(value, dict) and "values" in value:
                # Bootstrap results
                all_combined_results[key]["values"].extend(value["values"])
            elif isinstance(value, tuple) and len(value) == 2:
                # CV/LODO results - keep as is (don't combine)
                if key not in all_combined_results:
                    all_combined_results[key] = {"mean": value[0], "std": value[1]}

        # Combine ROC data
        for key, roc_list in roc_data.items():
            if isinstance(roc_list, list):
                all_combined_results[key]["roc_data"].extend(roc_list)

        # Combine weights
        for key, weight_list in weights.items():
            if isinstance(weight_list, list):
                all_combined_results[key]["weights"].extend(weight_list)

    # Recalculate statistics for bootstrap results
    final_results = {}
    final_roc_data = {}
    final_weights = {}

    for key, combined in all_combined_results.items():
        if "values" in combined and combined["values"]:
            # Bootstrap results - recalculate mean/std
            values = combined["values"]
            final_results[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "values": values,
            }
            if combined["roc_data"]:
                final_roc_data[key] = combined["roc_data"]
            if combined["weights"]:
                final_weights[key] = combined["weights"]
        elif "mean" in combined:
            # CV/LODO results - keep as is
            final_results[key] = (combined["mean"], combined["std"])

    # Save combined results
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as f:
        pickle.dump(
            {
                "results": final_results,
                "roc_data": final_roc_data,
                "weights": final_weights,
            },
            f,
        )

    logger.success(f"Combined results saved to {output_path}")
    logger.info(f"Total keys: {len(final_results)}")
    logger.info(
        f"Bootstrap keys: {sum(1 for v in final_results.values() if isinstance(v, dict) and 'values' in v)}"
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 3:
        print(
            "Usage: combine_bootstrap_results.py <output_path> <result_file1> [result_file2] ..."
        )
        sys.exit(1)

    output_path = Path(sys.argv[1])
    result_files = [Path(f) for f in sys.argv[2:]]

    combine_bootstrap_results(result_files, output_path)
