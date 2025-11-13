"""BulkFormer embedding generation functions."""

from collections import OrderedDict
from pathlib import Path
import sys

import anndata as ad
from loguru import logger
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.typing import SparseTensor
from tqdm import tqdm

from inflamm_debate_fm.config import BULKFORMER_DATA_DIR, BULKFORMER_MODEL_DIR

BULKFORMER_BASE = Path(__file__).resolve().parents[2] / "models" / "BulkFormer"
if BULKFORMER_BASE.exists():
    sys.path.insert(0, str(BULKFORMER_BASE))

try:
    from model.config import model_params
    from utils.BulkFormer import BulkFormer
except ImportError as e:
    logger.error(f"Failed to import BulkFormer: {e}")
    raise


def load_bulkformer_model(device: str = "cpu"):
    """Load BulkFormer model."""
    device = torch.device(device)
    logger.info(f"Loading BulkFormer model from {BULKFORMER_MODEL_DIR} on {device}")

    model_graph_path = BULKFORMER_MODEL_DIR / "G_gtex.pt"
    model_graph_weights_path = BULKFORMER_MODEL_DIR / "G_gtex_weight.pt"
    model_gene_embedding_path = BULKFORMER_MODEL_DIR / "esm2_feature_concat.pt"
    model_checkpoint_path = BULKFORMER_MODEL_DIR / "checkpoint.pt"

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


def load_bulkformer_gene_data():
    """Load BulkFormer gene data files."""
    gene_length_file = BULKFORMER_DATA_DIR / "gene_length_df.csv"
    gene_info_file = BULKFORMER_DATA_DIR / "bulkformer_gene_info.csv"
    high_var_genes_file = BULKFORMER_MODEL_DIR / "high_var_gene_list.pt"

    gene_length_df = pd.read_csv(gene_length_file)
    gene_length_dict = gene_length_df.set_index("ensg_id")["length"].to_dict()
    bulkformer_gene_info = pd.read_csv(gene_info_file)
    bulkformer_gene_list = bulkformer_gene_info["ensg_id"].to_list()
    high_var_gene_idx = torch.load(high_var_genes_file, map_location="cpu", weights_only=False)

    return {
        "gene_length_dict": gene_length_dict,
        "bulkformer_gene_list": bulkformer_gene_list,
        "high_var_gene_idx": high_var_gene_idx,
    }


def align_genes_to_bulkformer(expr_df: pd.DataFrame, bulkformer_gene_list: list) -> pd.DataFrame:
    """Align gene expression matrix to BulkFormer gene list."""
    to_fill_columns = list(set(bulkformer_gene_list) - set(expr_df.columns))
    padding_df = pd.DataFrame(
        np.full((expr_df.shape[0], len(to_fill_columns)), -10),
        columns=to_fill_columns,
        index=expr_df.index,
    )
    aligned_df = pd.concat([expr_df, padding_df], axis=1)[bulkformer_gene_list]
    return aligned_df


def adata_to_dataframe(adata: ad.AnnData) -> pd.DataFrame:
    """Convert AnnData to DataFrame with Ensembl IDs as columns."""
    if "ensembl_id" not in adata.var.columns:
        adata.var["ensembl_id"] = adata.var.index
    df = pd.DataFrame(
        adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
        index=adata.obs_names,
        columns=adata.var["ensembl_id"],
    )
    return df.loc[:, df.columns.notna()]


def extract_bulkformer_features(
    model: torch.nn.Module,
    expr_array: np.ndarray,
    high_var_gene_idx: np.ndarray,
    aggregate_type: str = "max",
    device: str = "cpu",
    batch_size: int = 256,
) -> np.ndarray:
    """Extract features from BulkFormer model."""
    device = torch.device(device)
    model.eval()
    expr_tensor = torch.tensor(expr_array, dtype=torch.float32, device=device)
    dataset = TensorDataset(expr_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_emb_list = []
    with torch.no_grad():
        for (X,) in tqdm(dataloader, total=len(dataloader), desc="Extracting features"):
            X = X.to(device)
            output, emb = model(X, [2])
            emb = emb[2].detach().cpu().numpy()
            emb_valid = emb[:, high_var_gene_idx, :]

            if aggregate_type == "max":
                final_emb = np.max(emb_valid, axis=1)
            elif aggregate_type == "mean":
                final_emb = np.mean(emb_valid, axis=1)
            elif aggregate_type == "median":
                final_emb = np.median(emb_valid, axis=1)
            elif aggregate_type == "all":
                final_emb = (
                    np.max(emb_valid, axis=1)
                    + np.mean(emb_valid, axis=1)
                    + np.median(emb_valid, axis=1)
                )
            else:
                raise ValueError(f"Unknown aggregate_type: {aggregate_type}")
            all_emb_list.append(final_emb)

    return np.vstack(all_emb_list)


def generate_bulkformer_embeddings(
    adata: ad.AnnData,
    device: str = "cpu",
    batch_size: int = 256,
    aggregate_type: str = "max",
) -> np.ndarray:
    """Generate BulkFormer embeddings for an AnnData object."""
    logger.info(f"Generating BulkFormer embeddings for {adata.shape[0]} samples")

    model = load_bulkformer_model(device=device)
    gene_data = load_bulkformer_gene_data()

    if "ensembl_id" not in adata.var.columns:
        adata.var["ensembl_id"] = adata.var.index

    expr_df = adata_to_dataframe(adata)
    aligned_df = align_genes_to_bulkformer(expr_df, gene_data["bulkformer_gene_list"])

    logger.info(f"Aligned expression matrix: {expr_df.shape} -> {aligned_df.shape}")

    embeddings = extract_bulkformer_features(
        model=model,
        expr_array=aligned_df.values,
        high_var_gene_idx=gene_data["high_var_gene_idx"],
        aggregate_type=aggregate_type,
        device=device,
        batch_size=batch_size,
    )

    logger.success(f"Generated BulkFormer embeddings: {embeddings.shape}")
    return embeddings
