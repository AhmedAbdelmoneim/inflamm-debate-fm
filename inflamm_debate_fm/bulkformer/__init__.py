"""BulkFormer embedding generation module."""

from inflamm_debate_fm.bulkformer.embed import (
    extract_embeddings,
    extract_embeddings_from_adata,
    load_bulkformer_model,
)
from inflamm_debate_fm.bulkformer.pipeline import generate_all_embeddings
from inflamm_debate_fm.bulkformer.setup import setup_bulkformer

__all__ = [
    "extract_embeddings",
    "extract_embeddings_from_adata",
    "generate_all_embeddings",
    "load_bulkformer_model",
    "setup_bulkformer",
]
