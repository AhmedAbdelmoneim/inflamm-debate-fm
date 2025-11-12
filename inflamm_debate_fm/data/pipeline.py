"""Unified data processing pipeline from raw CEL files to processed ortholog-mapped AnnData."""

import anndata as ad
from loguru import logger
from mygene import MyGeneInfo
import pandas as pd

from inflamm_debate_fm.config import (
    ANNDATA_RAW_DIR,
    DATA_ROOT,
    METADATA_DIR,
    ORTHOLOGY_DIR,
    RMA_PROCESSED_DIR,
    get_config,
)
from inflamm_debate_fm.data.constants import GSE_IDS
from inflamm_debate_fm.data.download_raw import download_all_raw_data
from inflamm_debate_fm.data.extract_metadata import extract_all_metadata
from inflamm_debate_fm.data.load import combine_adatas, load_adatas
from inflamm_debate_fm.data.preprocess_raw import convert_to_anndata, run_rma_preprocessing
from inflamm_debate_fm.data.preprocessing import preprocess_all_datasets


def query_mygene_batch(entrez_ids: list[str]) -> dict[str, dict]:
    """Query MyGene.info for Entrez IDs and return mapping.

    Args:
        entrez_ids: List of Entrez IDs to query.

    Returns:
        Dictionary mapping Entrez ID to {'ensembl': str, 'symbol': str}.
    """
    if not entrez_ids:
        return {}

    logger.info(f"Querying MyGene.info for {len(entrez_ids)} genes...")
    mg = MyGeneInfo()
    results = mg.querymany(
        entrez_ids,
        scopes="entrezgene",
        fields=["ensembl.gene", "symbol"],
        species="human",
    )

    mapping = {}
    for r in results:
        entrez = str(r.get("query", ""))
        ensembl = None
        if "ensembl" in r:
            if isinstance(r["ensembl"], list):
                ensembl = r["ensembl"][0].get("gene") if r["ensembl"] else None
            else:
                ensembl = r["ensembl"].get("gene")
        if ensembl:
            mapping[entrez] = {"ensembl": ensembl, "symbol": r.get("symbol")}

    return mapping


def load_orthology_mapping() -> pd.DataFrame:
    """Load mouse-to-human orthology mapping."""
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

    logger.info(f"Loaded {len(mapping)} mouse->human ortholog pairs")
    return mapping


def process_orthologs():
    """Process ortholog mapping for all AnnData files."""
    # Step 1: Load orthology mapping (mouse Entrez -> human Entrez)
    orthology_mapping = load_orthology_mapping()

    # Step 2: Collect all human Entrez IDs we need to map
    # - From orthology mapping (for mouse datasets)
    # - From human datasets directly
    human_entrez_ids = set(orthology_mapping["human_entrez"].unique().tolist())

    # Collect Entrez IDs from all human datasets
    for adata_file in sorted(ANNDATA_RAW_DIR.glob("*_human_*.h5ad")):
        adata = ad.read_h5ad(adata_file)
        human_entrez_ids.update(adata.var_names.astype(str).tolist())

    # Step 3: Query MyGene.info once for all human Entrez IDs
    human_entrez_to_ensembl = query_mygene_batch(list(human_entrez_ids))

    # Step 4: Add Ensembl IDs to orthology mapping
    orthology_mapping["human_ensembl"] = (
        orthology_mapping["human_entrez"]
        .astype(str)
        .map(lambda x: human_entrez_to_ensembl.get(x, {}).get("ensembl"))
    )
    orthology_mapping["human_symbol"] = orthology_mapping.apply(
        lambda row: (
            row["human_symbol"]
            if pd.notna(row["human_symbol"])
            else human_entrez_to_ensembl.get(str(row["human_entrez"]), {}).get("symbol")
        ),
        axis=1,
    )

    # Filter to only entries with Ensembl IDs
    orthology_mapping = orthology_mapping[
        orthology_mapping["human_ensembl"].notna()
    ].drop_duplicates()

    # Step 5: Process each dataset
    ann_data_dir = DATA_ROOT / get_config()["paths"]["ann_data_dir"]
    ann_data_dir.mkdir(parents=True, exist_ok=True)

    for adata_file in sorted(ANNDATA_RAW_DIR.glob("*.h5ad")):
        filename = adata_file.stem.replace("_gene_rma_brainarray", "")
        parts = filename.split("_")
        if len(parts) < 3:
            continue
        dataset_name = f"{parts[1]}_{parts[2]}"
        species = "human" if dataset_name.startswith("human_") else "mouse"
        logger.info(f"Mapping {dataset_name}...")
        adata = ad.read_h5ad(adata_file)

        if species == "mouse":
            # Map mouse Entrez -> human Ensembl using orthology mapping
            mouse_entrez = adata.var_names.astype(str)
            mapped = orthology_mapping[orthology_mapping["mouse_entrez"].isin(mouse_entrez)].copy()
            if len(mapped) == 0:
                logger.warning(f"No ortholog mappings found for {dataset_name}")
                continue

            keep_genes = mouse_entrez.isin(mapped["mouse_entrez"].values)
            adata_filtered = adata[:, keep_genes].copy()
            mouse_to_human = dict(zip(mapped["mouse_entrez"], mapped["human_ensembl"]))
            adata_filtered.var_names = adata_filtered.var_names.astype(str).map(mouse_to_human)

            expr_df = pd.DataFrame(
                adata_filtered.X.T,
                index=adata_filtered.var_names,
                columns=adata_filtered.obs_names,
            )
            expr_aggregated = expr_df.groupby(expr_df.index).mean()
            adata_mapped = ad.AnnData(
                X=expr_aggregated.T.values,
                obs=adata_filtered.obs.copy(),
                var=pd.DataFrame(index=expr_aggregated.index),
            )
            symbol_map = (
                mapped[["human_ensembl", "human_symbol"]]
                .drop_duplicates(subset="human_ensembl")
                .set_index("human_ensembl")["human_symbol"]
                .to_dict()
            )
            adata_mapped.var["symbol"] = adata_mapped.var_names.map(symbol_map)
            adata_mapped.var["human_ensembl"] = adata_mapped.var_names
        else:
            # Map human Entrez -> human Ensembl using cached mapping
            human_entrez = adata.var_names.astype(str)
            entrez_to_ensembl = {
                entrez: info["ensembl"]
                for entrez, info in human_entrez_to_ensembl.items()
                if entrez in human_entrez.values
            }
            if not entrez_to_ensembl:
                logger.warning(f"No Ensembl mappings found for {dataset_name}")
                continue

            keep_genes = human_entrez.isin(entrez_to_ensembl.keys())
            adata_filtered = adata[:, keep_genes].copy()
            adata_filtered.var_names = adata_filtered.var_names.astype(str).map(entrez_to_ensembl)

            expr_df = pd.DataFrame(
                adata_filtered.X.T,
                index=adata_filtered.var_names,
                columns=adata_filtered.obs_names,
            )
            expr_aggregated = expr_df.groupby(expr_df.index).mean()
            adata_mapped = ad.AnnData(
                X=expr_aggregated.T.values,
                obs=adata_filtered.obs.copy(),
                var=pd.DataFrame(index=expr_aggregated.index),
            )
            # Map symbols: create mapping from Ensembl -> symbol
            ensembl_to_symbol = {}
            for entrez, ensembl in entrez_to_ensembl.items():
                if ensembl not in ensembl_to_symbol:
                    symbol = human_entrez_to_ensembl.get(entrez, {}).get("symbol")
                    ensembl_to_symbol[ensembl] = symbol
            adata_mapped.var["symbol"] = adata_mapped.var_names.map(ensembl_to_symbol)
            adata_mapped.var["human_ensembl"] = adata_mapped.var_names

        adata_mapped.write_h5ad(ann_data_dir / f"{dataset_name}_orthologs.h5ad")
        logger.success(
            f"Mapped {dataset_name}: {len(adata.var_names)} -> {len(adata_mapped.var_names)} genes"
        )


def run_pipeline():
    """Run complete data processing pipeline."""
    config = get_config()
    geo_download_dir = DATA_ROOT / config["paths"]["geo_download_dir"]
    geo_download_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running data processing pipeline...")

    # Get unique GSE IDs from constants
    expected_gse_ids = set(GSE_IDS.values())
    logger.info(f"Expected GSE IDs: {sorted(expected_gse_ids)}")

    # Step 1: Download - check if all GSE directories exist
    download_all_raw_data()

    # Step 2: Extract metadata - check if all metadata files exist
    extract_all_metadata(geo_download_dir, METADATA_DIR)

    # Step 3: RMA preprocessing - check if all processed directories exist
    missing_rma = [
        gse_id for gse_id in expected_gse_ids if not (RMA_PROCESSED_DIR / gse_id).exists()
    ]
    if missing_rma:
        logger.info(f"Running RMA preprocessing for: {missing_rma}")
        run_rma_preprocessing()
        convert_to_anndata()

    # Step 4: Ortholog mapping - check if all expected ortholog files exist
    ann_data_dir = DATA_ROOT / config["paths"]["ann_data_dir"]
    expected_datasets = {
        "GSE37069": "human_burn",
        "GSE36809": "human_trauma",
        "GSE28750": "human_sepsis",
        "GSE19668": "mouse_sepsis",
        "GSE20524": "mouse_infection",
        "GSE7404": ["mouse_burn", "mouse_trauma"],
    }
    missing_orthologs = []
    for gse_id, datasets in expected_datasets.items():
        if isinstance(datasets, list):
            for dataset in datasets:
                if not (ann_data_dir / f"{dataset}_orthologs.h5ad").exists():
                    missing_orthologs.append(f"{dataset}_orthologs.h5ad")
        else:
            if not (ann_data_dir / f"{datasets}_orthologs.h5ad").exists():
                missing_orthologs.append(f"{datasets}_orthologs.h5ad")

    if missing_orthologs:
        logger.info(f"Mapping orthologs for: {missing_orthologs}")
        process_orthologs()

    # Step 5: Preprocessing and combining
    logger.info("Preprocessing and combining...")
    ann_data_dir = DATA_ROOT / config["paths"]["ann_data_dir"]
    embeddings_dir = DATA_ROOT / config["paths"]["embeddings_dir"]
    combined_data_dir = DATA_ROOT / config["paths"]["combined_data_dir"]
    combined_data_dir.mkdir(parents=True, exist_ok=True)

    adatas = load_adatas(ann_data_dir, embeddings_dir, load_embeddings=True)
    preprocess_all_datasets(adatas)

    # Combine datasets for each species if available
    human_datasets = [k for k in adatas.keys() if k.startswith("human_")]
    mouse_datasets = [k for k in adatas.keys() if k.startswith("mouse_")]

    if human_datasets:
        combine_adatas(adatas, "human").write_h5ad(combined_data_dir / "human_combined.h5ad")
        logger.success(f"Combined {len(human_datasets)} human dataset(s)")

    if mouse_datasets:
        combine_adatas(adatas, "mouse").write_h5ad(combined_data_dir / "mouse_combined.h5ad")
        logger.success(f"Combined {len(mouse_datasets)} mouse dataset(s)")

    logger.success("Pipeline complete!")
