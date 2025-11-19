"""Embedding generation modules."""

from inflamm_debate_fm.embeddings.load import load_embedding, load_embeddings_for_adatas
from inflamm_debate_fm.embeddings.multi_model import (
    add_multi_model_embeddings_to_adata,
    detect_available_models,
    extract_multi_model_embeddings,
)

__all__ = [
    "load_embedding",
    "load_embeddings_for_adatas",
    "detect_available_models",
    "extract_multi_model_embeddings",
    "add_multi_model_embeddings_to_adata",
]
