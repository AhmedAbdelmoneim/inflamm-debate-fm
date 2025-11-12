"""Flyte workflows for inflamm-debate-fm."""

from inflamm_debate_fm.workflows.workflows import (
    analysis_workflow,
    embedding_workflow,
    preprocessing_workflow,
    probing_workflow,
)

__all__ = [
    "preprocessing_workflow",
    "embedding_workflow",
    "probing_workflow",
    "analysis_workflow",
]
