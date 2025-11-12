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
    RAW_CEL_DIR,
    RMA_PROCESSED_DIR,
    get_config,
)
from inflamm_debate_fm.data.download_raw import download_all_raw_data
from inflamm_debate_fm.data.extract_metadata import extract_all_metadata
from inflamm_debate_fm.data.load import combine_adatas, load_adatas
from inflamm_debate_fm.data.preprocess_raw import convert_to_anndata, run_rma_preprocessing
from inflamm_debate_fm.data.preprocessing import preprocess_all_datasets


def load_orthology_mapping() -> pd.DataFrame:
    """Load mouse-to-human orthology mapping."""
    hmd_file = ORTHOLOGY_DIR / "HMD_HumanPhenotype.rpt"
    mgi_file = ORTHOLOGY_DIR / "MGI_EntrezGene.rpt"

    logger.info("Loading orthology mapping...")
    hmd = pd.read_csv(
        hmd_file,
        sep="\t",
        header=None,
        names=["human_symbol", "human_entrez", "mouse_symbol", "mouse_mgi", "phenotype_id"],
        dtype=str,
        low_memory=False,
    )
    mgi = pd.read_csv(
        mgi_file,
        sep="\t",
        header=None,
        usecols=[0, 1, 3],
        names=["mgi_id", "mouse_symbol", "mouse_entrez"],
        dtype=str,
        low_memory=False,
    )

    mapping = hmd.merge(mgi, left_on="mouse_mgi", right_on="mgi_id", how="inner")
    mapping = (
        mapping[["mouse_entrez", "human_entrez", "human_symbol"]]
        .dropna(subset=["mouse_entrez", "human_entrez"])
        .drop_duplicates()
    )

    logger.info(f"Querying MyGene.info for {len(mapping['human_entrez'].unique())} human genes...")
    mg = MyGeneInfo()
    results = mg.querymany(
        mapping["human_entrez"].unique().tolist(),
        scopes="entrezgene",
        fields=["ensembl.gene", "symbol"],
        species="human",
    )

    human_map = {}
    for r in results:
        entrez = str(r.get("query", ""))
        ensembl = None
        if "ensembl" in r:
            ensembl = (
                r["ensembl"][0].get("gene")
                if isinstance(r["ensembl"], list)
                else r["ensembl"].get("gene")
            )
        if ensembl:
            human_map[entrez] = {"ensembl": ensembl, "symbol": r.get("symbol")}

    mapping["human_ensembl"] = (
        mapping["human_entrez"].astype(str).map(lambda x: human_map.get(x, {}).get("ensembl"))
    )
    mapping["human_symbol"] = mapping.apply(
        lambda row: row["human_symbol"]
        if pd.notna(row["human_symbol"])
        else human_map.get(str(row["human_entrez"]), {}).get("symbol"),
        axis=1,
    )

    return mapping[mapping["human_ensembl"].notna()].drop_duplicates()[
        ["mouse_entrez", "human_entrez", "human_ensembl", "human_symbol"]
    ]


def map_orthologs(
    adata: ad.AnnData, species: str, orthology_mapping: pd.DataFrame | None = None
) -> ad.AnnData:
    """Map genes to human Ensembl IDs."""
    if species == "mouse":
        if orthology_mapping is None:
            orthology_mapping = load_orthology_mapping()
        mouse_entrez = adata.var_names.astype(str)
        mapped = orthology_mapping[orthology_mapping["mouse_entrez"].isin(mouse_entrez)].copy()
        if len(mapped) == 0:
            return adata
        keep_genes = mouse_entrez.isin(mapped["mouse_entrez"].values)
        adata_filtered = adata[:, keep_genes].copy()
        mouse_to_human = dict(zip(mapped["mouse_entrez"], mapped["human_ensembl"]))
        adata_filtered.var_names = adata_filtered.var_names.astype(str).map(mouse_to_human)
        expr_df = pd.DataFrame(
            adata_filtered.X.T, index=adata_filtered.var_names, columns=adata_filtered.obs_names
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
        return adata_mapped
    else:
        logger.info(f"Querying MyGene.info for {len(adata.var_names)} human genes...")
        mg = MyGeneInfo()
        results = mg.querymany(
            adata.var_names.astype(str).tolist(),
            scopes="entrezgene",
            fields=["ensembl.gene", "symbol"],
            species="human",
        )
        entrez_to_ensembl = {}
        entrez_to_symbol = {}
        for r in results:
            entrez = str(r.get("query", ""))
            if "ensembl" in r:
                ensembl = (
                    r["ensembl"][0].get("gene")
                    if isinstance(r["ensembl"], list)
                    else r.get("ensembl", {}).get("gene")
                )
                if ensembl:
                    entrez_to_ensembl[entrez] = ensembl
                    entrez_to_symbol[entrez] = r.get("symbol")
        keep_genes = adata.var_names.astype(str).isin(entrez_to_ensembl.keys())
        adata_filtered = adata[:, keep_genes].copy()
        adata_filtered.var_names = adata_filtered.var_names.astype(str).map(entrez_to_ensembl)
        expr_df = pd.DataFrame(
            adata_filtered.X.T, index=adata_filtered.var_names, columns=adata_filtered.obs_names
        )
        expr_aggregated = expr_df.groupby(expr_df.index).mean()
        adata_mapped = ad.AnnData(
            X=expr_aggregated.T.values,
            obs=adata_filtered.obs.copy(),
            var=pd.DataFrame(index=expr_aggregated.index),
        )
        ensembl_to_symbol = {}
        for entrez, ensembl in entrez_to_ensembl.items():
            if ensembl not in ensembl_to_symbol:
                ensembl_to_symbol[ensembl] = entrez_to_symbol.get(entrez)
        adata_mapped.var["symbol"] = adata_mapped.var_names.map(ensembl_to_symbol)
        adata_mapped.var["human_ensembl"] = adata_mapped.var_names
        return adata_mapped


def process_orthologs():
    """Process ortholog mapping for all AnnData files."""
    orthology_mapping = load_orthology_mapping()
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
        adata_mapped = map_orthologs(adata, species, orthology_mapping)
        adata_mapped.write_h5ad(ann_data_dir / f"{dataset_name}_orthologs.h5ad")


def run_pipeline():
    """Run complete data processing pipeline."""
    config = get_config()
    geo_download_dir = DATA_ROOT / config["paths"]["geo_download_dir"]
    geo_download_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Running data processing pipeline...")

    # Step 1: Download
    if not any(RAW_CEL_DIR.glob("GSE*")):
        logger.info("Downloading raw data...")
        download_all_raw_data()

    # Step 2: Extract metadata
    if not any(METADATA_DIR.glob("*.csv")):
        logger.info("Extracting metadata...")
        extract_all_metadata(geo_download_dir, METADATA_DIR)

    # Step 3: RMA preprocessing
    if not any(RMA_PROCESSED_DIR.glob("GSE*")):
        logger.info("Running RMA preprocessing...")
        run_rma_preprocessing()
        convert_to_anndata()

    # Step 4: Ortholog mapping
    ann_data_dir = DATA_ROOT / config["paths"]["ann_data_dir"]
    if not any(ann_data_dir.glob("*_orthologs.h5ad")):
        logger.info("Mapping orthologs...")
        process_orthologs()

    # Step 5: Preprocessing and combining
    logger.info("Preprocessing and combining...")
    ann_data_dir = DATA_ROOT / config["paths"]["ann_data_dir"]
    embeddings_dir = DATA_ROOT / config["paths"]["embeddings_dir"]
    combined_data_dir = DATA_ROOT / config["paths"]["combined_data_dir"]
    combined_data_dir.mkdir(parents=True, exist_ok=True)

    adatas = load_adatas(ann_data_dir, embeddings_dir, load_embeddings=True)
    preprocess_all_datasets(adatas)
    combine_adatas(adatas, "human").write_h5ad(combined_data_dir / "human_combined.h5ad")
    combine_adatas(adatas, "mouse").write_h5ad(combined_data_dir / "mouse_combined.h5ad")

    logger.success("Pipeline complete!")
