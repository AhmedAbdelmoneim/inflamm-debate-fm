"""BulkFormer embedding generation functions."""

from collections import OrderedDict
from pathlib import Path
import sys
from typing import Optional

import anndata as ad
from loguru import logger
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.typing import SparseTensor
from tqdm import tqdm

from inflamm_debate_fm.config import BULKFORMER_DATA_DIR, BULKFORMER_MODEL_DIR

# Add BulkFormer directory to path
BULKFORMER_BASE = Path(__file__).resolve().parents[2] / "models" / "BulkFormer"
if BULKFORMER_BASE.exists():
    sys.path.insert(0, str(BULKFORMER_BASE))

try:
    from model.config import model_params
    from utils.BulkFormer import BulkFormer
except ImportError as e:
    logger.error(f"Failed to import BulkFormer: {e}")
    logger.error(f"BulkFormer directory not found at {BULKFORMER_BASE}")
    raise


def load_bulkformer_model(
    model_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    device: str = "cpu",
) -> torch.nn.Module:
    """Load BulkFormer model.

    Parameters
    ----------
    model_dir : Path, optional
        Directory containing model files. If None, uses config default.
    data_dir : Path, optional
        Directory containing data files. If None, uses config default.
    device : str
        Device to load model on ('cpu' or 'cuda').

    Returns
    -------
    model : torch.nn.Module
        Loaded BulkFormer model
    """
    if model_dir is None:
        model_dir = BULKFORMER_MODEL_DIR
    else:
        model_dir = Path(model_dir)

    if data_dir is None:
        data_dir = BULKFORMER_DATA_DIR
    else:
        data_dir = Path(data_dir)

    device = torch.device(device)
    logger.info(f"Loading BulkFormer model from {model_dir} on {device}")

    # Load model files
    model_graph_path = model_dir / "G_gtex.pt"
    model_graph_weights_path = model_dir / "G_gtex_weight.pt"
    model_gene_embedding_path = model_dir / "esm2_feature_concat.pt"
    model_checkpoint_path = model_dir / "checkpoint.pt"

    if not model_checkpoint_path.exists():
        # Try alternative checkpoint name
        model_checkpoint_path = model_dir / "Bulkformer_ckpt_epoch_29.pt"
        if not model_checkpoint_path.exists():
            raise FileNotFoundError(
                f"Model checkpoint not found in {model_dir}. "
                f"Please download models using: python -m inflamm_debate_fm.cli download models"
            )

    # Load graph
    graph = torch.load(model_graph_path, map_location="cpu", weights_only=False)
    weights = torch.load(model_graph_weights_path, map_location="cpu", weights_only=False)
    graph = SparseTensor(row=graph[1], col=graph[0], value=weights).t().to(device)

    # Load gene embeddings
    gene_emb = torch.load(model_gene_embedding_path, map_location="cpu", weights_only=False)

    # Update model params
    model_config = model_params.copy()
    model_config["graph"] = graph
    model_config["gene_emb"] = gene_emb

    # Create model
    model = BulkFormer(**model_config).to(device)

    # Load checkpoint
    ckpt_model = torch.load(model_checkpoint_path, map_location=device, weights_only=False)

    # Handle module prefix
    new_state_dict = OrderedDict()
    for key, value in ckpt_model.items():
        new_key = key[7:] if key.startswith("module.") else key
        new_state_dict[new_key] = value

    model.load_state_dict(new_state_dict)
    model.eval()

    logger.success("BulkFormer model loaded successfully")
    return model, model_config


def load_bulkformer_gene_data(
    data_dir: Optional[Path] = None,
    model_dir: Optional[Path] = None,
) -> dict:
    """Load BulkFormer gene data files.

    Parameters
    ----------
    data_dir : Path, optional
        Directory containing data files. If None, uses config default.
    model_dir : Path, optional
        Directory containing model files. If None, uses config default.

    Returns
    -------
    gene_data : dict
        Dictionary containing:
        - gene_length_dict: dict mapping Ensembl IDs to gene lengths
        - bulkformer_gene_list: list of Ensembl IDs
        - high_var_gene_idx: indices of highly variable genes
    """
    if data_dir is None:
        data_dir = BULKFORMER_DATA_DIR
    else:
        data_dir = Path(data_dir)

    if model_dir is None:
        model_dir = BULKFORMER_MODEL_DIR
    else:
        model_dir = Path(model_dir)

    logger.info(f"Loading BulkFormer gene data from {data_dir}")

    # Load gene length file
    gene_length_file = data_dir / "gene_length_df.csv"
    if not gene_length_file.exists():
        raise FileNotFoundError(
            f"Gene length file not found: {gene_length_file}. "
            f"Please download data using: python -m inflamm_debate_fm.cli download models"
        )

    gene_length_df = pd.read_csv(gene_length_file)
    gene_length_dict = gene_length_df.set_index("ensg_id")["length"].to_dict()

    # Load gene info file
    gene_info_file = data_dir / "bulkformer_gene_info.csv"
    if not gene_info_file.exists():
        raise FileNotFoundError(
            f"Gene info file not found: {gene_info_file}. "
            f"Please download data using: python -m inflamm_debate_fm.cli download models"
        )

    bulkformer_gene_info = pd.read_csv(gene_info_file)
    bulkformer_gene_list = bulkformer_gene_info["ensg_id"].to_list()

    # Load high variance genes
    high_var_genes_file = model_dir / "high_var_gene_list.pt"
    if not high_var_genes_file.exists():
        raise FileNotFoundError(
            f"High variance genes file not found: {high_var_genes_file}. "
            f"Please download models using: python -m inflamm_debate_fm.cli download models"
        )

    high_var_gene_idx = torch.load(high_var_genes_file, map_location="cpu", weights_only=False)

    return {
        "gene_length_dict": gene_length_dict,
        "bulkformer_gene_list": bulkformer_gene_list,
        "high_var_gene_idx": high_var_gene_idx,
    }


def align_genes_to_bulkformer(expr_df: pd.DataFrame, bulkformer_gene_list: list) -> tuple:
    """Align gene expression matrix to BulkFormer gene list.

    Parameters
    ----------
    expr_df : pd.DataFrame
        Expression DataFrame with genes as columns (Ensembl IDs), samples as rows.
    bulkformer_gene_list : list
        List of Ensembl IDs required by BulkFormer.

    Returns
    -------
    aligned_df : pd.DataFrame
        Aligned expression DataFrame.
    to_fill_columns : list
        List of genes that were filled with placeholder values.
    var : pd.DataFrame
        DataFrame with mask indicating which genes were imputed.
    """
    to_fill_columns = list(set(bulkformer_gene_list) - set(expr_df.columns))

    # Create padding for missing genes
    padding_df = pd.DataFrame(
        np.full((expr_df.shape[0], len(to_fill_columns)), -10),
        columns=to_fill_columns,
        index=expr_df.index,
    )

    # Concatenate and reorder
    aligned_df = pd.DataFrame(
        np.concatenate([df.values for df in [expr_df, padding_df]], axis=1),
        index=expr_df.index,
        columns=list(expr_df.columns) + list(padding_df.columns),
    )
    aligned_df = aligned_df[bulkformer_gene_list]

    # Create mask
    var = pd.DataFrame(index=aligned_df.columns)
    var["mask"] = [1 if i in to_fill_columns else 0 for i in list(var.index)]

    return aligned_df, to_fill_columns, var


def adata_to_dataframe(adata: ad.AnnData) -> pd.DataFrame:
    """Convert AnnData to DataFrame with Ensembl IDs as columns.

    Parameters
    ----------
    adata : AnnData
        AnnData object with ensembl_id in var.

    Returns
    -------
    df : pd.DataFrame
        DataFrame with samples as rows, genes (Ensembl IDs) as columns.
    """
    if "ensembl_id" not in adata.var.columns:
        raise ValueError("adata.var must contain 'ensembl_id' column")

    # Create DataFrame from adata.X
    df = pd.DataFrame(
        adata.X.toarray() if hasattr(adata.X, "toarray") else adata.X,
        index=adata.obs_names,
        columns=adata.var["ensembl_id"],
    )

    # Drop columns with NA ensembl_id
    df = df.loc[:, df.columns.notna()]

    return df


def extract_bulkformer_features(
    model: torch.nn.Module,
    expr_array: np.ndarray,
    high_var_gene_idx: np.ndarray,
    feature_type: str = "transcriptome_level",
    aggregate_type: str = "max",
    device: str = "cpu",
    batch_size: int = 256,
    esm2_emb: Optional[torch.Tensor] = None,
    valid_gene_idx: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Extract features from BulkFormer model.

    Parameters
    ----------
    model : torch.nn.Module
        BulkFormer model.
    expr_array : np.ndarray
        Expression array of shape [N_samples, N_genes].
    high_var_gene_idx : np.ndarray
        Indices of highly variable genes.
    feature_type : str
        Type of feature to extract ('transcriptome_level' or 'gene_level').
    aggregate_type : str
        Aggregation method for transcriptome-level features ('max', 'mean', 'median', 'all').
    device : str
        Device to use ('cpu' or 'cuda').
    batch_size : int
        Batch size for inference.
    esm2_emb : torch.Tensor, optional
        ESM2 embeddings for gene-level features.
    valid_gene_idx : np.ndarray, optional
        Indices of valid genes for gene-level features.

    Returns
    -------
    features : np.ndarray
        Extracted features.
    """
    device = torch.device(device)
    model.eval()

    expr_tensor = torch.tensor(expr_array, dtype=torch.float32, device=device)
    dataset = TensorDataset(expr_tensor)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    all_emb_list = []

    with torch.no_grad():
        if feature_type == "transcriptome_level":
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
                    max_emb = np.max(emb_valid, axis=1)
                    mean_emb = np.mean(emb_valid, axis=1)
                    median_emb = np.median(emb_valid, axis=1)
                    final_emb = max_emb + mean_emb + median_emb
                else:
                    raise ValueError(f"Unknown aggregate_type: {aggregate_type}")

                all_emb_list.append(final_emb)

            result_emb = np.vstack(all_emb_list)

        elif feature_type == "gene_level":
            if esm2_emb is None:
                raise ValueError("esm2_emb is required for gene-level features")
            if valid_gene_idx is None:
                raise ValueError("valid_gene_idx is required for gene-level features")

            for (X,) in tqdm(dataloader, total=len(dataloader), desc="Extracting features"):
                X = X.to(device)
                output, emb = model(X, [2])
                emb = emb[2].detach().cpu().numpy()
                emb_valid = emb[:, valid_gene_idx, :]
                all_emb_list.append(emb_valid)

            all_emb = np.vstack(all_emb_list)
            all_emb_tensor = torch.tensor(all_emb, device="cpu", dtype=torch.float32)
            esm2_emb_selected = esm2_emb[valid_gene_idx]
            esm2_emb_expanded = esm2_emb_selected.unsqueeze(0).expand(
                all_emb_tensor.shape[0], -1, -1
            )
            esm2_emb_expanded = esm2_emb_expanded.to("cpu")

            result_emb = torch.cat([all_emb_tensor, esm2_emb_expanded], dim=-1).numpy()
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

    return result_emb


def generate_bulkformer_embeddings(
    adata: ad.AnnData,
    model_dir: Optional[Path] = None,
    data_dir: Optional[Path] = None,
    device: str = "cpu",
    batch_size: int = 256,
    aggregate_type: str = "max",
    output_path: Optional[Path] = None,
) -> np.ndarray:
    """Generate BulkFormer embeddings for an AnnData object.

    Parameters
    ----------
    adata : AnnData
        AnnData object with ensembl_id in var.
    model_dir : Path, optional
        Directory containing model files. If None, uses config default.
    data_dir : Path, optional
        Directory containing data files. If None, uses config default.
    device : str
        Device to use ('cpu' or 'cuda').
    batch_size : int
        Batch size for inference.
    aggregate_type : str
        Aggregation method for transcriptome-level features ('max', 'mean', 'median', 'all').
    output_path : Path, optional
        Path to save embeddings. If None, embeddings are not saved.

    Returns
    -------
    embeddings : np.ndarray
        BulkFormer embeddings of shape [N_samples, D].
    """
    logger.info(f"Generating BulkFormer embeddings for {adata.shape[0]} samples")

    # Load model and gene data
    model, model_config = load_bulkformer_model(
        model_dir=model_dir, data_dir=data_dir, device=device
    )
    gene_data = load_bulkformer_gene_data(data_dir=data_dir, model_dir=model_dir)

    # Convert AnnData to DataFrame
    if "ensembl_id" not in adata.var.columns:
        logger.warning("ensembl_id not found in adata.var, using var.index")
        adata.var["ensembl_id"] = adata.var.index

    expr_df = adata_to_dataframe(adata)

    # Align genes to BulkFormer gene list
    aligned_df, to_fill_columns, var = align_genes_to_bulkformer(
        expr_df, gene_data["bulkformer_gene_list"]
    )

    logger.info(
        f"Aligned expression matrix: {expr_df.shape} -> {aligned_df.shape}, "
        f"filled {len(to_fill_columns)} genes"
    )

    # Extract features
    # For transcriptome-level features, we don't need valid_gene_idx
    # Only high_var_gene_idx is used for aggregation
    embeddings = extract_bulkformer_features(
        model=model,
        expr_array=aligned_df.values,
        high_var_gene_idx=gene_data["high_var_gene_idx"],
        feature_type="transcriptome_level",
        aggregate_type=aggregate_type,
        device=device,
        batch_size=batch_size,
        esm2_emb=None,  # Not needed for transcriptome-level
        valid_gene_idx=None,  # Not needed for transcriptome-level
    )

    logger.success(f"Generated BulkFormer embeddings: {embeddings.shape}")

    # Save embeddings if output path is provided
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(output_path, embeddings)
        logger.info(f"Saved embeddings to {output_path}")

    return embeddings
