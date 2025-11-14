"""Plotting functions for within-species and cross-species results."""

from pathlib import Path

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_auroc_summary(
    all_results: dict,
    model_type: str = "Linear",
    setup_order: list[str] | None = None,
    output_path: Path | None = None,
) -> None:
    """
    Barplot summary of AUROC for CV and LODO across setups and data types.

    Args:
        all_results: Dictionary with structure:
            {
                "CrossValidation": {
                    "Linear": {"Raw": {setup: (mean, std), ...}, "Embedding": {...}},
                    "Nonlinear": {...}
                },
                "LODO": {...}
            }
        model_type: Model type ("Linear" or "Nonlinear").
        setup_order: List of setup names in desired order.
        output_path: Optional path to save the figure.
    """
    data_types = ["Raw", "Embedding"]
    val_types = ["CrossValidation", "LODO"]
    colors = sns.color_palette("Set2", n_colors=2)
    hatches = ["//", ""]

    if setup_order is None:
        # Extract setups from all_results
        example_dict = all_results["CrossValidation"][model_type]["Raw"]
        setup_order = sorted(example_dict.keys())

    fig, ax = plt.subplots(figsize=(14, 8))
    width = 0.2
    x = np.arange(len(setup_order))

    for i, val_type in enumerate(val_types):
        for j, data_type in enumerate(data_types):
            heights, errs = [], []
            for setup in setup_order:
                mean_std = all_results[val_type][model_type][data_type].get(
                    setup, (np.nan, np.nan)
                )
                heights.append(mean_std[0])
                errs.append(mean_std[1])
            ax.bar(
                x + (i * len(data_types) + j) * width,
                heights,
                width=width,
                yerr=errs,
                capsize=3,
                label=f"{val_type}-{data_type}",
                color=colors[j],
                hatch=hatches[i],
            )

    ax.set_xticks(x + width * 1.5)
    ax.set_xticklabels(setup_order, rotation=0, ha="right", fontsize=8)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1.19)
    ax.axhline(0.5, color="k", linestyle="--", alpha=0.3, linewidth=0.8)

    for i, tick in enumerate(ax.xaxis.get_major_ticks()):
        if i % 2 != 0:  # Select every other tick (starting from the second)
            tick.set_pad(15)  # Increase padding to move it down

    for label in ax.get_xticklabels():
        label.set_horizontalalignment("center")

    # Move legend outside
    ax.legend(loc="upper right", fontsize=8)

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)  # Close figure to free memory
    else:
        plt.show()


def plot_all(
    species: str,
    model: str,
    all_results: dict,
    all_roc_data: dict,
    setup_order: list[str] | None = None,
    output_dir: Path | None = None,
) -> None:
    """
    Generate all plots for within-species results.

    Args:
        species: Species name ("Human" or "Mouse").
        model: Model type ("Linear" or "Nonlinear").
        all_results: Dictionary with structure:
            {
                "CrossValidation": {
                    "Linear": {"Raw": {setup: (mean, std), ...}, "Embedding": {...}},
                    "Nonlinear": {...}
                },
                "LODO": {...}
            }
        all_roc_data: Dictionary with ROC curve data.
        setup_order: List of setup names in desired order.
        output_dir: Optional directory to save figures.
    """
    if setup_order is None:
        setup_order = [
            "All Inflammation Samples vs. Control",
            "Takao Subset for Inflammation vs. Control",
            "Acute Inflammation vs. Control",
            "Subacute Inflammation vs. Control",
            "Chronic Inflammation vs. Control",
            "Acute Inflammation vs. Chronic Inflammation",
        ]

    # Figure 1: AUROC summary
    output_path = None
    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{species}_{model}_auroc_summary.png"

    plot_auroc_summary(
        all_results=all_results,
        model_type=model,
        setup_order=setup_order,
        output_path=output_path,
    )


def plot_roc_facet_clean(
    all_roc_data: dict,
    model_type: str = "Linear",
    setup_order: list[str] | None = None,
    output_path: Path | None = None,
) -> None:
    """
    Clean ROC facet grid: Linear and Nonlinear separately, four lines per subplot.

    - Only leftmost column and bottom row show axis labels and ticks
    - Only bottom-right subplot shows legend
    - Optional setup_order to enforce subplot order

    Args:
        all_roc_data: Dictionary with structure:
            {
                "Linear": {
                    "Raw": {setup: [(fpr, tpr, auroc), ...], ...},
                    "Embedding": {...}
                },
                "Nonlinear": {...}
            }
        model_type: Model type ("Linear" or "Nonlinear").
        setup_order: List of setup names in desired order.
        output_path: Optional path to save the figure.
    """
    colors = {"Raw": "C0", "Embedding": "C1"}
    linestyles = {"Human→Mouse": "-", "Mouse→Human": "--"}

    # Extract setups
    example_dict = all_roc_data[model_type]["Raw"]
    setups = sorted({key.split(" (")[0] for key in example_dict.keys()})
    if setup_order is not None:
        setups = [s for s in setup_order if s in setups]

    n = len(setups)
    cols = 3
    rows = (n + cols - 1) // cols
    fig = plt.figure(figsize=(cols * 5, rows * 4))
    gs = gridspec.GridSpec(rows, cols, figure=fig)

    for i, setup in enumerate(setups):
        ax = fig.add_subplot(gs[i])
        for data_type in ["Raw", "Embedding"]:
            for direction in ["Human→Mouse", "Mouse→Human"]:
                key = f"{setup} ({direction})"
                if key in all_roc_data[model_type][data_type]:
                    fpr, tpr, auroc = all_roc_data[model_type][data_type][key]
                    ax.plot(
                        fpr,
                        tpr,
                        color=colors[data_type],
                        linestyle=linestyles[direction],
                        lw=2,
                        label=f"{data_type} {direction} AUROC={auroc:.2f}",
                    )

        ax.plot([0, 1], [0, 1], "--", color="black", lw=1, alpha=0.7)
        ax.set_title(setup)

        # Only show x-labels for bottom row
        if i // cols == rows - 1:
            ax.set_xlabel("FPR")
        else:
            ax.set_xticklabels([])
            ax.set_xlabel("")

        # Only show y-labels for leftmost column
        if i % cols == 0:
            ax.set_ylabel("TPR")
        else:
            ax.set_yticklabels([])
            ax.set_ylabel("")

        # Show legend only on bottom-right subplot
        if i == n - 1:
            ax.legend(fontsize=8)

    plt.tight_layout()

    if output_path is not None:
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.close(fig)  # Close figure to free memory
    else:
        plt.show()


def plot_auroc_bar_clean_sns_top_legend(
    all_results: dict,
    setup_order: list[str] | None = None,
    output_path: Path | None = None,
) -> None:
    """
    AUROC barplot for cross-species setups.

    - Groups: Raw/Embedding side by side for each direction
    - Colors by Raw vs Embedding using sns.Set2
    - Leftmost column shows Y-axis label 'AUROC'
    - Legend on top-left
    - Horizontal line at 0.5 for random chance

    Args:
        all_results: Dictionary with structure:
            {
                "Linear": {
                    "Raw": {setup: auroc_value, ...},
                    "Embedding": {...}
                },
                "Nonlinear": {...}
            }
        setup_order: List of setup names in desired order.
        output_path: Optional path to save the figure.
    """
    palette = sns.color_palette("Set2", 2)  # Raw, Embedding
    directions = ["Human→Mouse", "Mouse→Human"]
    data_types = ["Raw", "Embedding"]

    for model_type in ["Linear", "Nonlinear"]:
        example_dict = all_results[model_type]["Raw"]
        setups = sorted({key.split(" (")[0] for key in example_dict.keys()})
        if setup_order is not None:
            setups = [s for s in setup_order if s in setups]

        n = len(setups)
        cols = 3
        rows = (n + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(cols * 5, rows * 4))
        axes = axes.flatten()

        for i, setup in enumerate(setups):
            ax = axes[i]
            # Bar positions
            x = np.arange(len(directions))  # 0=H→M, 1=M→H
            width = 0.35
            for j, data_type in enumerate(data_types):
                heights = [
                    all_results[model_type][data_type].get(f"{setup} ({dir_})", np.nan)
                    for dir_ in directions
                ]
                ax.bar(
                    x + (j - 0.5) * width,
                    heights,
                    width=width,
                    color=palette[j],
                    edgecolor="black",
                    label=data_type if i == 0 else "",
                )

            # Horizontal line at 0.5
            ax.axhline(0.5, color="black", linestyle="--", linewidth=1, alpha=0.7)

            ax.set_xticks(x)
            ax.set_xticklabels(directions)
            ax.set_ylim(0, 1.05)
            ax.set_title(setup)

            # Y-axis label only on leftmost column
            if i % cols == 0:
                ax.set_ylabel("AUROC")
            else:
                ax.set_yticklabels([])

        # Remove extra axes
        for j in range(len(setups), len(axes)):
            fig.delaxes(axes[j])

        axes[0].legend()

        plt.tight_layout(rect=[0, 0, 1, 0.95])

        if output_path is not None:
            # Save with model_type in filename
            output_path_str = str(output_path)
            if output_path_str.endswith(".png"):
                output_path_model = Path(output_path_str.replace(".png", f"_{model_type}.png"))
            else:
                output_path_model = Path(f"{output_path_str}_{model_type}.png")
            plt.savefig(output_path_model, dpi=300, bbox_inches="tight")
            plt.close(fig)  # Close figure to free memory
        else:
            plt.show()
