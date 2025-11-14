"""Ortholog mapping functions for mouse and human genes."""

import anndata as ad
from loguru import logger
from mygene import MyGeneInfo
import pandas as pd

from inflamm_debate_fm.config import (
    ANNDATA_RAW_DIR,
    DATA_ROOT,
    ORTHOLOGY_DIR,
    get_config,
)


def query_mygene_batch(
    entrez_ids: list[str], species: str = "human", include_symbol: bool = False
) -> dict[str, list | str | dict]:
    """Query MyGene.info for Entrez IDs and return Ensembl ID mapping.

    Args:
        entrez_ids: List of Entrez IDs to query.
        species: Species to query ("human" or "mouse").
        include_symbol: If True, also return symbol in the mapping.

    Returns:
        If include_symbol=False: Dictionary mapping Entrez ID to Ensembl ID(s).
        If include_symbol=True: Dictionary mapping Entrez ID to {'ensembl': ..., 'symbol': ...}.
        Can be a single string or a list of strings if multiple Ensembl IDs exist.
    """
    if not entrez_ids:
        return {}

    logger.info(f"Querying MyGene.info for {len(entrez_ids)} {species} genes...")
    mg = MyGeneInfo()
    fields = "ensembl.gene" if not include_symbol else ["ensembl.gene", "symbol"]
    results = mg.querymany(
        entrez_ids,
        scopes="entrezgene",
        fields=fields,
        species=species,
    )

    mapping = {}
    for r in results:
        entrez = str(r.get("query", ""))
        if "ensembl" in r:
            ensembl_data = r["ensembl"]
            if isinstance(ensembl_data, list):
                # Multiple Ensembl IDs - extract all gene IDs
                ensembl_ids = [e.get("gene") for e in ensembl_data if e.get("gene")]
                if ensembl_ids:
                    ensembl_result = ensembl_ids if len(ensembl_ids) > 1 else ensembl_ids[0]
                    if include_symbol:
                        mapping[entrez] = {
                            "ensembl": ensembl_result,
                            "symbol": r.get("symbol"),
                        }
                    else:
                        mapping[entrez] = ensembl_result
            else:
                # Single Ensembl ID
                if ensembl_data.get("gene"):
                    if include_symbol:
                        mapping[entrez] = {
                            "ensembl": ensembl_data.get("gene"),
                            "symbol": r.get("symbol"),
                        }
                    else:
                        mapping[entrez] = ensembl_data.get("gene")

    return mapping


def add_ensembl_ids_exploded(
    df: pd.DataFrame, human_col: str = "human_entrez", mouse_col: str = "mouse_entrez"
) -> pd.DataFrame:
    """Add Ensembl IDs for human and mouse genes using MyGene, keeping all IDs.

    Rows are exploded so each human-mouse pair may have multiple rows if
    multiple Ensembl IDs exist for either gene.

    Args:
        df: DataFrame with human and mouse Entrez IDs.
        human_col: Column name for human Entrez IDs.
        mouse_col: Column name for mouse Entrez IDs.

    Returns:
        DataFrame with columns ['human_symbol', 'human_entrez', 'mouse_entrez',
                                'human_ensembl', 'mouse_ensembl'].
        Multiple Ensembl IDs are exploded into separate rows.
    """
    df = df.copy()
    mg = MyGeneInfo()

    # Human mapping
    human_ids = df[human_col].astype(str).tolist()
    h_res = mg.querymany(
        human_ids,
        scopes="entrezgene",
        fields="ensembl.gene",
        species="human",
        as_dataframe=True,
    )
    h_map = h_res["ensembl.gene"].to_dict()
    df["human_ensembl"] = df[human_col].astype(str).map(h_map)

    # --- Mouse mapping (exactly as in notebook) ---
    mouse_ids = df[mouse_col].astype(str).tolist()
    m_res = mg.querymany(
        mouse_ids,
        scopes="entrezgene",
        fields="ensembl.gene",
        species="mouse",
        as_dataframe=True,
    )
    m_map = m_res["ensembl.gene"].to_dict()
    df["mouse_ensembl"] = df[mouse_col].astype(str).map(m_map)

    # Ensure list type for explode
    df["human_ensembl"] = df["human_ensembl"].apply(
        lambda x: x if isinstance(x, list) else [x] if pd.notna(x) else []
    )
    df["mouse_ensembl"] = df["mouse_ensembl"].apply(
        lambda x: x if isinstance(x, list) else [x] if pd.notna(x) else []
    )

    # Explode both dimensions
    df = df.explode("human_ensembl").explode("mouse_ensembl").reset_index(drop=True)

    return df


def get_strict_orthologs(orthologs_df: pd.DataFrame) -> pd.DataFrame:
    """Restrict ortholog mapping to strict 1:1 pairs.

    Removes any mouse_ensembl or human_ensembl that appears more than once,
    ensuring a strict one-to-one mapping.

    Args:
        orthologs_df: DataFrame with 'human_ensembl' and 'mouse_ensembl' columns.

    Returns:
        Strict 1:1 ortholog mapping DataFrame.
    """
    df = orthologs_df.dropna(subset=["human_ensembl", "mouse_ensembl"]).copy()

    # Remove duplicates: 1 human -> many mouse OR 1 mouse -> many human
    humans = df["human_ensembl"].value_counts()
    mice = df["mouse_ensembl"].value_counts()

    df = df[~df["human_ensembl"].isin(humans[humans > 1].index)]
    df = df[~df["mouse_ensembl"].isin(mice[mice > 1].index)]

    return df.reset_index(drop=True)


def load_orthology_mapping() -> pd.DataFrame:
    """Load mouse-to-human orthology mapping with Ensembl IDs.

    Returns:
        DataFrame with strict 1:1 ortholog mapping including Ensembl IDs.
    """
    hmd_file = ORTHOLOGY_DIR / "HMD_HumanPhenotype.rpt"
    mgi_file = ORTHOLOGY_DIR / "MGI_EntrezGene.rpt"

    logger.info("Loading orthology mapping...")
    # HMD file: columns 0-3 are human_symbol, human_entrez, mouse_symbol, mouse_mgi
    hmd = pd.read_csv(
        hmd_file,
        sep="\t",
        header=None,
        usecols=[0, 1, 2, 3],
        names=["human_symbol", "human_entrez", "mouse_symbol", "mouse_mgi"],
        dtype=str,
        low_memory=False,
    )
    # MGI file: column 0 = MGI ID, column 1 = symbol, column 2 = type, column 8 = Entrez ID
    # Filter to only type "O" (official) entries which have Entrez IDs
    mgi = pd.read_csv(
        mgi_file,
        sep="\t",
        header=None,
        usecols=[0, 1, 2, 8],
        names=["mgi_id", "mouse_symbol", "type", "mouse_entrez"],
        dtype=str,
        low_memory=False,
    )

    # Clean up data before merge
    hmd = hmd.dropna(subset=["mouse_mgi", "human_entrez"])
    # Filter MGI to only official entries with Entrez IDs
    mgi = mgi[(mgi["type"] == "O") & mgi["mouse_entrez"].notna()].copy()
    mgi = mgi[["mgi_id", "mouse_symbol", "mouse_entrez"]].drop_duplicates(subset="mgi_id")

    # Merge on MGI ID
    mapping = hmd.merge(mgi, left_on="mouse_mgi", right_on="mgi_id", how="inner")

    # Select final columns and clean
    mapping = mapping[["mouse_entrez", "human_entrez", "human_symbol"]].copy()
    mapping = mapping.dropna(subset=["mouse_entrez", "human_entrez"]).drop_duplicates()

    logger.info(f"Loaded {len(mapping)} mouse->human ortholog pairs (Entrez IDs)")

    # Add Ensembl IDs and explode
    logger.info("Adding Ensembl IDs and exploding...")
    mapping = add_ensembl_ids_exploded(mapping)
    logger.info(f"After adding Ensembl IDs: {len(mapping)} pairs")

    # Filter to strict 1:1 orthologs
    logger.info("Filtering to strict 1:1 orthologs...")
    mapping = get_strict_orthologs(mapping)
    logger.info(f"Strict 1:1 orthologs: {len(mapping)} pairs")

    return mapping


def add_symbols_to_human_adata(adata_human: ad.AnnData) -> ad.AnnData:
    """Add gene symbols to human AnnData using MyGene.info.

    Args:
        adata_human: Human AnnData with human Ensembl IDs as var_names.

    Returns:
        Human AnnData with symbols added to .var.
    """
    # Filter out non-Ensembl IDs (e.g., AFFX-*)
    ensembl_mask = adata_human.var.index.str.startswith("ENSG")
    adata_human = adata_human[:, ensembl_mask].copy()

    if adata_human.shape[1] == 0:
        return adata_human

    # Query MyGene.info for symbols
    mg = MyGeneInfo()
    ensembl_ids = adata_human.var.index.tolist()
    logger.info(f"Querying MyGene.info for {len(ensembl_ids)} human genes...")

    results = mg.querymany(
        ensembl_ids,
        scopes="ensembl.gene",
        fields="symbol",
        species="human",
        as_dataframe=True,
    )

    # Create mapping from Ensembl ID to symbol
    symbol_map = results["symbol"].to_dict()
    adata_human.var["ensembl"] = adata_human.var.index
    adata_human.var["symbol"] = adata_human.var.index.map(symbol_map)

    # Fill missing symbols with Ensembl IDs
    missing_symbol_mask = adata_human.var["symbol"].isna()
    adata_human.var.loc[missing_symbol_mask, "symbol"] = adata_human.var.index[missing_symbol_mask]

    return adata_human


def humanize_mouse_adata(adata_mouse: ad.AnnData, strict_map: pd.DataFrame) -> ad.AnnData:
    """Map mouse AnnData into human gene space using strict 1:1 orthologs.

    Args:
        adata_mouse: Mouse AnnData with mouse Ensembl IDs as var_names.
        strict_map: DataFrame with ['mouse_ensembl', 'human_ensembl', 'human_symbol'].

    Returns:
        Mouse data in human gene space (strict orthologs only).
    """
    # Keep only mouse genes in mapping
    keep_mouse = adata_mouse.var.index.isin(strict_map["mouse_ensembl"])
    adata_mouse = adata_mouse[:, keep_mouse].copy()

    # Build mapping dicts
    ens_map = dict(zip(strict_map["mouse_ensembl"], strict_map["human_ensembl"]))
    sym_map = dict(zip(strict_map["human_ensembl"], strict_map["human_symbol"]))

    # Rename to human Ensembl IDs
    adata_mouse.var["ensembl"] = adata_mouse.var.index.map(ens_map)
    adata_mouse.var_names = adata_mouse.var["ensembl"]

    # Add human symbols
    adata_mouse.var["symbol"] = adata_mouse.var_names.map(sym_map)

    # Fill missing symbols with Ensembl IDs
    # Convert Index to Series for fillna compatibility
    missing_mask = adata_mouse.var["symbol"].isna()
    adata_mouse.var.loc[missing_mask, "symbol"] = adata_mouse.var_names[missing_mask]

    # Drop duplicates
    adata_mouse = adata_mouse[:, ~adata_mouse.var_names.duplicated()].copy()

    return adata_mouse


def process_orthologs():
    """Process ortholog mapping for all AnnData files.

    Human datasets are kept as-is, mouse datasets are mapped to human gene space.
    """
    import anndata as ad

    # Step 1: Load orthology mapping with Ensembl IDs (strict 1:1)
    orthology_mapping = load_orthology_mapping()

    # Step 2: Process each dataset
    ann_data_dir = DATA_ROOT / get_config()["paths"]["ann_data_dir"]
    ann_data_dir.mkdir(parents=True, exist_ok=True)

    for adata_file in sorted(ANNDATA_RAW_DIR.glob("*.h5ad")):
        filename = adata_file.stem.replace("_gene_rma_brainarray", "")
        parts = filename.split("_")
        if len(parts) < 3:
            continue
        dataset_name = f"{parts[1]}_{parts[2]}"
        species = "human" if dataset_name.startswith("human_") else "mouse"

        output_file = ann_data_dir / f"{dataset_name}_orthologs.h5ad"
        if output_file.exists():
            logger.info(f"Skipping {dataset_name} (already mapped)")
            continue

        logger.info(f"Mapping {dataset_name}...")
        adata = ad.read_h5ad(adata_file)

        if species == "mouse":
            # Map mouse Ensembl -> human Ensembl using strict orthologs
            # var_names are already mouse Ensembl IDs
            adata_mapped = humanize_mouse_adata(adata, orthology_mapping)
        else:
            # Human datasets: add symbols to .var
            # var_names are already human Ensembl IDs
            adata_mapped = add_symbols_to_human_adata(adata)

        if adata_mapped.shape[1] == 0:
            logger.warning(f"No genes mapped for {dataset_name}, skipping...")
            continue

        adata_mapped.write_h5ad(output_file)
        logger.success(
            f"Mapped {dataset_name}: {len(adata.var_names)} -> {len(adata_mapped.var_names)} genes"
        )
