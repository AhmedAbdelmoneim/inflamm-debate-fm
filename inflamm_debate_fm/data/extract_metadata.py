"""Extract metadata from GEO datasets."""

from pathlib import Path

import GEOparse
from loguru import logger
import pandas as pd

from inflamm_debate_fm.data.constants import GSE_IDS


def extract_all_metadata(geo_download_dir: Path, output_dir: Path):
    """Extract metadata from all GEO datasets."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for gse_id in GSE_IDS.values():
        output_file = output_dir / f"{gse_id}.csv"
        if output_file.exists():
            logger.info(f"Metadata for {gse_id} already exists, skipping extraction.")
            continue
        logger.info(f"Extracting metadata for {gse_id}...")
        try:
            gse = GEOparse.get_GEO(geo=gse_id, destdir=str(geo_download_dir))
            metadata_rows = []
            for gsm_name, gsm in gse.gsms.items():
                row = {"geo_accession": gsm_name}
                if "source_name_ch1" in gsm.metadata:
                    row["source_name_ch1"] = "; ".join(gsm.metadata["source_name_ch1"])
                for key, value in gsm.metadata.items():
                    if key.startswith("characteristics_ch1."):
                        row[key.replace("characteristics_ch1.", "")] = (
                            "; ".join(value) if isinstance(value, list) else str(value)
                        )
                if "platform_id" in gsm.metadata:
                    row["platform_id"] = "; ".join(gsm.metadata["platform_id"])
                metadata_rows.append(row)
            metadata_df = pd.DataFrame(metadata_rows)
            if "geo_accession" in metadata_df.columns:
                metadata_df = metadata_df.set_index("geo_accession")
            metadata_df.to_csv(output_file)
        except Exception as e:
            logger.warning(f"Failed to extract metadata for {gse_id}: {e}")
