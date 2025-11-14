"""BulkFormer embedding extraction functions."""

from collections import OrderedDict
import sys

import anndata as ad
from loguru import logger
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.typing import SparseTensor
from tqdm import tqdm

from inflamm_debate_fm.config import BULKFORMER_DATA_DIR, BULKFORMER_MODEL_DIR, MODELS_ROOT

# Add BulkFormer to path for imports
# Use MODELS_ROOT from config to handle environment variable overrides
BULKFORMER_BASE = MODELS_ROOT / "BulkFormer"
if BULKFORMER_BASE.exists():
    sys.path.insert(0, str(BULKFORMER_BASE))
    logger.debug(f"Added BulkFormer to Python path: {BULKFORMER_BASE}")
else:
    logger.warning(
        f"BulkFormer directory not found at {BULKFORMER_BASE}. "
        "Run 'make bulkformer-setup' to clone the repository."
    )

try:
    from model.config import model_params
    from utils.BulkFormer import BulkFormer
except ImportError as e:
    logger.error(
        f"Failed to import BulkFormer: {e}\n"
        f"Make sure BulkFormer repository is cloned at {BULKFORMER_BASE}\n"
        "Run 'make bulkformer-setup' to set up BulkFormer."
    )
    raise


def load_bulkformer_model(device: str = "cpu") -> torch.nn.Module:
    """Load BulkFormer model.

    Args:
        device: Device to load model on ('cpu' or 'cuda').

    Returns:
        Loaded BulkFormer model in eval mode.
    """
    device = torch.device(device)
    logger.info(f"Loading BulkFormer model from {BULKFORMER_MODEL_DIR} on {device}")

    model_graph_path = BULKFORMER_MODEL_DIR / "G_gtex.pt"
    model_graph_weights_path = BULKFORMER_MODEL_DIR / "G_gtex_weight.pt"
    model_gene_embedding_path = BULKFORMER_MODEL_DIR / "esm2_feature_concat.pt"
    model_checkpoint_path = BULKFORMER_MODEL_DIR / "checkpoint.pt"

    # Try alternative checkpoint name
    if not model_checkpoint_path.exists():
        model_checkpoint_path = BULKFORMER_MODEL_DIR / "Bulkformer_ckpt_epoch_29.pt"
        if not model_checkpoint_path.exists():
            raise FileNotFoundError(f"Model checkpoint not found in {BULKFORMER_MODEL_DIR}")

    graph = torch.load(model_graph_path, map_location="cpu", weights_only=False)
    weights = torch.load(model_graph_weights_path, map_location="cpu", weights_only=False)
    graph = SparseTensor(row=graph[1], col=graph[0], value=weights).t().to(device)

    gene_emb = torch.load(model_gene_embedding_path, map_location="cpu", weights_only=False)

    model_config = model_params.copy()
    model_config["graph"] = graph
    model_config["gene_emb"] = gene_emb

    model = BulkFormer(**model_config).to(device)

    ckpt_model = torch.load(model_checkpoint_path, map_location=device, weights_only=False)
    new_state_dict = OrderedDict()
    for key, value in ckpt_model.items():
        new_key = key[7:] if key.startswith("module.") else key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.eval()
    return model


def load_bulkformer_gene_data() -> dict:
    """Load BulkFormer gene data files.

    Returns:
        Dictionary with 'bulkformer_gene_list' and 'gene_length_dict'.
    """
    gene_length_file = BULKFORMER_DATA_DIR / "gene_length_df.csv"
    gene_info_file = BULKFORMER_DATA_DIR / "bulkformer_gene_info.csv"

    gene_length_df = pd.read_csv(gene_length_file)
    gene_length_dict = gene_length_df.set_index("ensg_id")["length"].to_dict()
    bulkformer_gene_info = pd.read_csv(gene_info_file)
    bulkformer_gene_list = bulkformer_gene_info["ensg_id"].to_list()

    return {
        "gene_length_dict": gene_length_dict,
        "bulkformer_gene_list": bulkformer_gene_list,
    }


def align_genes_to_bulkformer(
    expr_df: pd.DataFrame, bulkformer_gene_list: list
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Align gene expression matrix to BulkFormer gene list.

    Args:
        expr_df: Expression DataFrame with genes as columns.
        bulkformer_gene_list: List of Ensembl gene IDs to align to.

    Returns:
        Tuple of (aligned DataFrame, var DataFrame with mask column).
    """
    to_fill_columns = list(set(bulkformer_gene_list) - set(expr_df.columns))

    padding_df = pd.DataFrame(
        np.full((expr_df.shape[0], len(to_fill_columns)), -10),
        columns=to_fill_columns,
        index=expr_df.index,
    )

    aligned_df = pd.concat([expr_df, padding_df], axis=1)[bulkformer_gene_list]

    var = pd.DataFrame(index=aligned_df.columns)
    var["mask"] = [1 if i in to_fill_columns else 0 for i in list(var.index)]

    return aligned_df, var


def adata_to_dataframe(adata: ad.AnnData) -> pd.DataFrame:
    """Convert AnnData to DataFrame with Ensembl IDs as columns.

    Args:
        adata: AnnData object with 'ensembl_id' in var.

    Returns:
        DataFrame with samples as rows and genes (ensembl_id) as columns.
    """
    if "ensembl_id" not in adata.var.columns:
        adata.var["ensembl_id"] = adata.var.index

    df = pd.DataFrame(
        adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
        index=adata.obs_names,
        columns=adata.var["ensembl_id"],
    )

    return df.loc[:, df.columns.notna()]


def extract_embeddings(
    model: torch.nn.Module,
    expr_array: np.ndarray,
    device: str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """Extract gene-level embeddings from BulkFormer model.

    This function extracts embeddings without filtering or aggregation.
    Returns gene-level embeddings for all genes in the model.

    Args:
        model: Loaded BulkFormer model.
        expr_array: Expression array of shape [N_samples, N_genes].
        device: Device to run inference on.
        batch_size: Batch size for inference.

    Returns:
        Embeddings array of shape [N_samples, N_genes, embedding_dim].
    """
    device = torch.device(device)
    model.eval()
    expr_tensor = torch.tensor(expr_array, dtype=torch.float32, device=device)
    dataset = TensorDataset(expr_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_emb_list = []
    with torch.no_grad():
        for (X,) in tqdm(dataloader, total=len(dataloader), desc="Extracting embeddings"):
            X = X.to(device)
            output, emb = model(X, [2])
            # Extract embeddings from layer 2 (gene-level)
            emb = emb[2].detach().cpu().numpy()
            all_emb_list.append(emb)

    return np.vstack(all_emb_list)


def extract_embeddings_from_adata(
    adata: ad.AnnData,
    model: torch.nn.Module | None = None,
    device: str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """Extract BulkFormer embeddings from an AnnData object.

    Args:
        adata: AnnData object with 'ensembl_id' in var.
        model: Pre-loaded BulkFormer model. If None, will load it.
        device: Device to run inference on.
        batch_size: Batch size for inference.

    Returns:
        Embeddings array of shape [N_samples, N_genes, embedding_dim].
    """
    if model is None:
        model = load_bulkformer_model(device=device)

    gene_data = load_bulkformer_gene_data()

    if "ensembl_id" not in adata.var.columns:
        adata.var["ensembl_id"] = adata.var.index

    expr_df = adata_to_dataframe(adata)
    aligned_df, var = align_genes_to_bulkformer(expr_df, gene_data["bulkformer_gene_list"])

    logger.info(f"Aligned expression matrix: {expr_df.shape} -> {aligned_df.shape}")

    embeddings = extract_embeddings(
        model=model,
        expr_array=aligned_df.values,
        device=device,
        batch_size=batch_size,
    )

    logger.success(f"Generated BulkFormer embeddings: {embeddings.shape}")
    return embeddings
