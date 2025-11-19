"""Script to combine bootstrap results from multiple parallel jobs."""

from collections import defaultdict
from pathlib import Path
import pickle

from loguru import logger
import numpy as np
import pandas as pd


def combine_bootstrap_results(
    result_files: list[Path],
    output_path: Path,
    save_summary_csv: bool = True,
) -> None:
    """Combine bootstrap results from multiple parallel jobs.

    Args:
        result_files: List of paths to partial result pickle files.
        output_path: Path to save combined results.
        save_summary_csv: Whether to save a summary CSV file.
    """
    if len(result_files) == 0:
        raise ValueError("No result files provided to combine")

    logger.info(f"Combining {len(result_files)} result files...")

    # Load all results
    all_combined_results = defaultdict(lambda: {"values": [], "roc_data": [], "weights": []})
    loaded_count = 0

    for result_file in result_files:
        if not result_file.exists():
            logger.warning(f"Result file not found: {result_file}")
            continue

        loaded_count += 1

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

    if loaded_count == 0:
        raise ValueError("No valid result files found to combine")

    logger.info(f"Successfully loaded {loaded_count} result files")

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

    # Save summary CSV
    if save_summary_csv:
        summary_path = output_path.parent / f"{output_path.stem}_summary.csv"
        _save_summary_csv(final_results, summary_path)
        logger.success(f"Saved summary CSV to {summary_path}")


def _save_summary_csv(results: dict, output_path: Path) -> None:
    """Save results summary as CSV."""
    rows = []
    for key, value in results.items():
        if isinstance(value, dict) and "mean" in value:
            # Bootstrap results
            rows.append(
                {
                    "setup": key,
                    "auroc_mean": value["mean"],
                    "auroc_std": value["std"],
                }
            )
        elif isinstance(value, tuple) and len(value) == 2:
            # CV/LODO results
            rows.append(
                {
                    "setup": key,
                    "auroc_mean": value[0],
                    "auroc_std": value[1],
                }
            )

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(output_path, index=False)
    else:
        logger.warning(f"No results to save to {output_path}")


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
