"""Extract metadata from GEO datasets.

This module extracts all metadata columns from GEO samples, including parsing
characteristics_ch1 into separate columns when present.
"""

from pathlib import Path

import GEOparse
from loguru import logger
import pandas as pd

from inflamm_debate_fm.data.constants import GSE_IDS


def _parse_characteristics(characteristics_str: str) -> dict[str, str]:
    """Parse characteristics_ch1 string into dictionary.

    Example: "tissue: White Blood Cells; Sex: F; age: 30" ->
    {"tissue:ch1": "White Blood Cells", "Sex:ch1": "F", "age:ch1": "30"}
    """
    if not characteristics_str or pd.isna(characteristics_str):
        return {}

    result = {}
    for part in str(characteristics_str).split(";"):
        part = part.strip()
        if ":" in part:
            key, value = part.split(":", 1)
            key = key.strip()
            value = value.strip()
            # Add :ch1 suffix to match expected column names
            if not key.endswith(":ch1"):
                key = f"{key}:ch1"
            result[key] = value
    return result


def extract_all_metadata(geo_download_dir: Path, output_dir: Path):
    """Extract metadata from all GEO datasets.

    Extracts all metadata columns from GEO samples and parses characteristics_ch1
    into separate columns when present.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    for gse_id in set(GSE_IDS.values()):
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
                # Extract all metadata columns
                for key, value in gsm.metadata.items():
                    # Join list values with "; " separator
                    if isinstance(value, list):
                        row[key] = "; ".join(str(v) for v in value)
                    else:
                        row[key] = str(value) if value is not None else ""

                # Parse characteristics_ch1 into separate columns
                if "characteristics_ch1" in row:
                    characteristics = _parse_characteristics(row["characteristics_ch1"])
                    # Add parsed characteristics to row (may overwrite existing columns)
                    for key, value in characteristics.items():
                        row[key] = value

                metadata_rows.append(row)

            metadata_df = pd.DataFrame(metadata_rows)
            if "geo_accession" in metadata_df.columns:
                metadata_df = metadata_df.set_index("geo_accession")
            metadata_df.to_csv(output_file)
            logger.success(f"Extracted {len(metadata_df.columns)} metadata columns for {gse_id}")
        except Exception as e:
            logger.warning(f"Failed to extract metadata for {gse_id}: {e}")
