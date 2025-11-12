"""GEO data preprocessing functions for inflammation datasets."""

from pathlib import Path
from typing import Dict, Optional

import anndata as ad
import GEOparse
from loguru import logger
from mygene import MyGeneInfo
import numpy as np
import pandas as pd

# GSE IDs for inflammation datasets
GSE_IDS = {
    "HumanBurn": "GSE37069",
    "HumanTrauma": "GSE36809",
    "HumanSepsis": "GSE28750",
    "MouseBurn": "GSE7404",
    "MouseTrauma": "GSE7404",
    "MouseSepsis": "GSE19668",
    "MouseInfection": "GSE20524",
}


def process_geo_to_entrez(
    gse_obj: GEOparse.GEO, min_expr: float = 4.0, min_frac: float = 0.2
) -> pd.DataFrame:
    """Process GEO microarray data into a gene x sample matrix using Entrez IDs.

    Parameters
    ----------
    gse_obj : GEOparse.GEO
        Parsed GEO object (from GEOparse.get_GEO).
    min_expr : float, optional
        Minimum log2 expression to consider a gene "expressed" in a sample.
    min_frac : float, optional
        Minimum fraction of samples that must pass `min_expr` for the gene to be kept.

    Returns
    -------
    exprs_gene : pd.DataFrame
        Expression DataFrame with Entrez IDs as index, samples as columns.
    """
    # Get expression matrix (probe x sample)
    exprs_df = gse_obj.pivot_samples("VALUE")
    exprs_df.columns = gse_obj.phenotype_data.index
    exprs_df.index.name = "ID_REF"

    # Load platform annotation
    platform_id = gse_obj.metadata["platform_id"][0]
    gpl_table = gse_obj.gpls[platform_id].table

    if "ENTREZ_GENE_ID" not in gpl_table.columns:
        raise ValueError("GPL table does not contain ENTREZ_GENE_ID column")

    # Build probe -> Entrez map
    probe_to_entrez = gpl_table.set_index("ID")["ENTREZ_GENE_ID"]

    # Merge annotation into expression
    exprs_df = exprs_df.join(probe_to_entrez, how="inner")
    exprs_df = exprs_df[exprs_df["ENTREZ_GENE_ID"].notna()]
    exprs_df["ENTREZ_GENE_ID"] = exprs_df["ENTREZ_GENE_ID"].astype(str)

    # Drop probes mapping to multiple genes (contain '///')
    exprs_df = exprs_df[~exprs_df["ENTREZ_GENE_ID"].str.contains("///")]

    # Remove whitespace
    exprs_df["ENTREZ_GENE_ID"] = exprs_df["ENTREZ_GENE_ID"].str.strip()

    # Collapse redundant probes
    exprs_gene = exprs_df.drop(columns="ENTREZ_GENE_ID").groupby(exprs_df["ENTREZ_GENE_ID"]).mean()

    exprs_gene.index.name = "ENTREZ_GENE_ID"

    # filter lowly expressed genes
    min_samples = int(min_frac * exprs_gene.shape[1])
    exprs_gene = exprs_gene[exprs_gene.gt(min_expr).sum(axis=1) >= min_samples]

    return exprs_gene


def quantile_normalize(X: np.ndarray) -> np.ndarray:
    """Perform quantile normalization on a 2D numpy array (samples x genes).

    Parameters
    ----------
    X : np.ndarray
        Input array of shape (samples, genes)

    Returns
    -------
    X_qn : np.ndarray
        Quantile-normalized array of the same shape
    """
    # Step 1: sort each sample (row) and get sorting indices
    sorted_idx = np.argsort(X, axis=1)
    sorted_X = np.sort(X, axis=1)

    # Step 2: compute rank-wise mean across samples
    mean_sorted = np.mean(sorted_X, axis=0)

    # Step 3: assign averaged ranks back to original positions
    X_qn = np.zeros_like(X)
    for i in range(X.shape[0]):  # for each sample
        X_qn[i, sorted_idx[i, :]] = mean_sorted

    return X_qn


def add_gene_metadata(adata: ad.AnnData, species: str = "human") -> ad.AnnData:
    """Add gene symbol and Ensembl ID to an existing AnnData object.

    Parameters
    ----------
    adata : AnnData
        AnnData object with var_names = Entrez IDs
    species : str, default "human"
        Species for mygene query ("human" or "mouse")

    Returns
    -------
    adata : AnnData
        The same AnnData object with added .var["symbol"] and .var["ensembl_id"]
    """
    mg = MyGeneInfo()
    entrez_ids = adata.var_names.tolist()

    logger.info(f"Querying MyGene.info for {len(entrez_ids)} genes...")
    results = mg.querymany(
        entrez_ids, scopes="entrezgene", fields=["symbol", "ensembl.gene"], species=species
    )
    logger.info("Query complete.")

    symbols = []
    ensembl_ids = []
    for r in results:
        symbol = r.get("symbol", None)
        ensembl = None
        if "ensembl" in r:
            if isinstance(r["ensembl"], list):
                ensembl = r["ensembl"][0].get("gene")
            else:
                ensembl = r["ensembl"].get("gene")
        symbols.append(symbol)
        ensembl_ids.append(ensembl)

    adata.var["symbol"] = symbols
    adata.var["ensembl_id"] = ensembl_ids

    return adata


def create_anndata(
    exprs_df: pd.DataFrame, pheno_df: pd.DataFrame, species: str = "human"
) -> ad.AnnData:
    """Create an AnnData object from expression and phenotype DataFrames.

    Applies normalization and metadata addition.

    Parameters
    ----------
    exprs_df : pd.DataFrame
        Expression DataFrame with genes as index, samples as columns.
    pheno_df : pd.DataFrame
        Phenotype DataFrame with samples as index, phenotypes as columns.
    species : str, default "human"
        Species for gene metadata lookup.

    Returns
    -------
    adata : AnnData
        AnnData object with .X = expression matrix, .obs = phenotype data, .var = gene metadata
    """
    adata = ad.AnnData(X=exprs_df.T, obs=pheno_df)

    # Apply quantile normalization
    adata.X = quantile_normalize(adata.X)
    adata = add_gene_metadata(adata, species=species)

    # assert no NaNs
    if np.isnan(adata.X).any():
        raise ValueError("Data contains NaNs after processing")

    return adata


def download_gse(gse_id: str, dest_dir: Path) -> GEOparse.GEO:
    """Download a GSE dataset from GEO.

    Parameters
    ----------
    gse_id : str
        GEO Series ID (e.g., "GSE37069")
    dest_dir : Path
        Directory to download and cache GEO data

    Returns
    -------
    gse : GEOparse.GEO
        Parsed GEO object
    """
    logger.info(f"Downloading {gse_id}...")
    gse = GEOparse.get_GEO(geo=gse_id, destdir=str(dest_dir))
    logger.info(f"Downloaded {gse_id}: {len(gse.gsms)} samples")
    return gse


def process_gse_dataset(
    gse_id: str,
    dataset_name: str,
    geo_download_dir: Path,
    output_dir: Path,
    species: str = "human",
    phenotype_columns: Optional[Dict[str, str]] = None,
) -> ad.AnnData:
    """Download and process a GSE dataset into an AnnData object.

    Parameters
    ----------
    gse_id : str
        GEO Series ID (e.g., "GSE37069")
    dataset_name : str
        Name for the dataset (e.g., "human_burn")
    geo_download_dir : Path
        Directory to download and cache GEO data
    output_dir : Path
        Directory to save processed AnnData files
    species : str, default "human"
        Species ("human" or "mouse")
    phenotype_columns : dict, optional
        Mapping from GSE phenotype column names to desired column names.
        If None, uses default mapping.

    Returns
    -------
    adata : AnnData
        Processed AnnData object
    """
    # Download GSE
    gse = download_gse(gse_id, geo_download_dir)

    # Process expression data
    exprs_df = process_geo_to_entrez(gse)

    # Prepare phenotype data
    if phenotype_columns is None:
        # Default phenotype column mapping
        phenotype_columns = {
            "source_name_ch1": "group",
            "characteristics_ch1.0.tissue": "tissue",
            "characteristics_ch1.1.Sex": "sex",
            "characteristics_ch1.2.age": "age",
            "characteristics_ch1.3.hours_since_injury": "time_point",
        }

    # Extract phenotype columns
    available_cols = [col for col in phenotype_columns.keys() if col in gse.phenotype_data.columns]
    if not available_cols:
        logger.warning(f"No matching phenotype columns found for {gse_id}")
        pheno_df = pd.DataFrame(index=gse.phenotype_data.index)
    else:
        pheno_df = gse.phenotype_data[available_cols].copy()
        pheno_df.columns = [phenotype_columns[col] for col in available_cols]

        # Clean up 'group' column if present
        if "group" in pheno_df.columns:
            pheno_df["group"] = pheno_df["group"].apply(
                lambda x: "inflammation" if "Subject" in str(x) else "control"
            )

        # Clean up 'time_point' column if present
        if "time_point" in pheno_df.columns:
            pheno_df["time_point"] = (
                pheno_df["time_point"].str.extract(r"(\d+\.?\d*)").astype(float)
            )
            pheno_df["time_point"] = pheno_df["time_point"].fillna(0)

    # Remove rows with any NaN values in exprs_df
    exprs_df = exprs_df.dropna()

    # Create AnnData object
    adata = create_anndata(exprs_df, pheno_df, species=species)

    # Save to file
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{dataset_name}.h5ad"
    adata.write_h5ad(output_path)
    logger.success(f"Saved processed dataset to {output_path}")

    return adata
