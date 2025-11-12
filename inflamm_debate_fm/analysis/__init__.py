"""Analysis modules for inflammation research."""

from inflamm_debate_fm.analysis.dimensionality import (
    intrinsic_dimensionality,
    participation_ratio,
)
from inflamm_debate_fm.analysis.gsea import run_prerank_from_coef_df
from inflamm_debate_fm.analysis.inflammation_vector import (
    bootstrap_vector,
    calculate_inflammation_vector,
)

__all__ = [
    "calculate_inflammation_vector",
    "bootstrap_vector",
    "run_prerank_from_coef_df",
    "intrinsic_dimensionality",
    "participation_ratio",
]
