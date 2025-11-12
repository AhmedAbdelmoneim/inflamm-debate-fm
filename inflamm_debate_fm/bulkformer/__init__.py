"""BulkFormer embedding generation module."""

from pathlib import Path

# Import BulkFormer utilities from models directory
# This allows us to use the existing BulkFormer code
import sys

BULKFORMER_BASE = Path(__file__).resolve().parents[2] / "models" / "BulkFormer"
if BULKFORMER_BASE.exists():
    sys.path.insert(0, str(BULKFORMER_BASE))

__all__ = ["generate_bulkformer_embeddings", "load_bulkformer_model"]
