# Pipeline Order: From Raw Data to Processed Ortholog-Mapped AnnData

This document outlines the complete pipeline order to go from raw CEL files to final processed ortholog-mapped AnnData files.

## Overview

The pipeline processes microarray data (CEL files) from GEO datasets, performs RMA preprocessing, converts to AnnData format, maps orthologs, and prepares the data for analysis.

## Step-by-Step Pipeline

### 1. Download Raw Data
**Command:** `make download-data` or `python -m inflamm_debate_fm.cli download data`

**What it does:**
- Downloads CEL files from GEO for all GSE datasets
- Downloads mouse orthology files from MGI (HMD_HumanPhenotype.rpt, MGI_EntrezGene.rpt)
- Downloads BrainArray CDF packages (hgu133plus2hsensgcdf, mouse4302mmensgcdf, mouse430a2mmensgcdf)

**Output locations:**
- `{DATA_ROOT}/raw/raw_cel/{GSE_ID}/` - CEL files
- `{DATA_ROOT}/external/mouse_orthology/` - Orthology files
- `{DATA_ROOT}/external/platforms/` - BrainArray CDF packages

**Note:** If downloads fail (e.g., network timeouts), the process continues and logs warnings. Some files may need to be downloaded manually.

---

### 2. Extract Metadata from GEO Datasets
**Command:** `make extract-metadata` or `python -m inflamm_debate_fm.cli preprocess metadata`

**What it does:**
- Downloads GEO SOFT files for each GSE dataset (if not already downloaded)
- Extracts metadata (sample information, phenotypes, etc.) from GEO datasets
- Saves metadata as CSV files

**Output location:**
- `{DATA_ROOT}/raw/metadata/{GSE_ID}.csv` - Metadata CSV files

**Prerequisites:** Step 1 (GEO downloads can happen here if not done earlier)

---

### 3. RMA Preprocessing (R Script)
**Command:** `make preprocess-data` or `python -m inflamm_debate_fm.cli preprocess raw`

**What it does:**
- Runs R script (`scripts/preprocess_rma_brainarray.R`) that:
  - Reads CEL files for each GSE dataset
  - Uses BrainArray CDF packages for gene-level summarization
  - Performs RMA normalization (background correction + quantile normalization + summarization + log2 transformation)
  - Outputs gene expression matrices (genes × samples)

**Output location:**
- `{DATA_ROOT}/interim/rma_brainarray_processed/{GSE_ID}/{GSE_ID}_gene_matrix.csv` - Expression matrices

**Prerequisites:** 
- Step 1 (CEL files and BrainArray CDF packages must be downloaded)
- Step 2 (Metadata CSV files must exist)

**Note:** The R script:
- Requires R and Bioconductor packages (affy, readr, tibble)
- Automatically installs missing packages via BiocManager
- Installs BrainArray CDF packages from local tar files
- Uses `INFLAMM_DEBATE_FM_DATA_ROOT` environment variable (set automatically by Python)

---

### 4. Convert to AnnData Format
**Command:** `python -m inflamm_debate_fm.cli preprocess raw` (same as step 3, but continues to AnnData conversion)

**What it does:**
- Loads expression matrices from step 3
- Loads metadata CSV files
- Creates AnnData objects with:
  - `.X` - Expression matrix (samples × genes)
  - `.obs` - Sample metadata
  - `.var` - Gene metadata (gene IDs)
- Cleans gene IDs (removes "_at" suffix from Affymetrix probes)
- Aligns metadata to expression matrix

**Output location:**
- `{DATA_ROOT}/interim/anndata/{GSE_ID}_gene_rma_brainarray.h5ad` - AnnData files

**Prerequisites:** Step 3 (RMA preprocessing must complete)

---

### 5. Orthology Mapping (Notebook-based)
**Command:** Currently done in Jupyter notebook (`notebooks/1.0-data-preprocessing.ipynb`)

**What it does:**
- Loads AnnData files from step 4
- For mouse datasets:
  - Uses mouse orthology files (HMD_HumanPhenotype.rpt, MGI_EntrezGene.rpt) to map mouse genes to human orthologs
  - Maps mouse Entrez IDs → Human Ensembl IDs
  - Filters to keep only ortholog-mapped genes
- For human datasets:
  - Adds Ensembl IDs using MyGene.info
  - Keeps all genes (no filtering)
- Re-indexes AnnData objects to use human Ensembl IDs (for mouse) or Ensembl IDs (for human)
- Adds gene symbols using MyGene.info

**Output location:**
- `{DATA_ROOT}/processed/anndata_orthologs/{dataset_name}_orthologs.h5ad` - Ortholog-mapped AnnData files

**Prerequisites:** 
- Step 4 (AnnData files must exist)
- Step 1 (Orthology files must be downloaded)

**Note:** This step is currently implemented in a Jupyter notebook and may need to be converted to a CLI command or Python script for automation.

---

### 6. Add Embeddings (Optional)
**Command:** `make embed DATASET=<dataset_name>` or `python -m inflamm_debate_fm.cli embed generate <dataset_name>`

**What it does:**
- Generates transcriptome embeddings using BulkFormer model
- Adds embeddings to AnnData objects in `.obsm["X_bulkformer"]`

**Output location:**
- `{DATA_ROOT}/processed/bulkformer_embeddings/{dataset_name}_transcriptome_embeddings.npy` - Embedding files

**Prerequisites:** 
- Step 5 (Ortholog-mapped AnnData files must exist)
- BulkFormer models must be downloaded (`python -m inflamm_debate_fm.cli download models`)

---

### 7. Preprocessing and Combining
**Command:** `make data` or `python -m inflamm_debate_fm.cli preprocess all`

**What it does:**
1. **Load AnnData files:**
   - Loads all ortholog-mapped AnnData files from `processed/anndata_orthologs/`
   - Optionally loads and adds embeddings

2. **Preprocessing:**
   - Adds timepoint categorization (acute, subacute, chronic) based on config cutoffs
   - Adds inflammation categories (`infl_acute`, `infl_subacute`, `infl_chronic`)
   - Dataset-specific preprocessing (e.g., human_burn, mouse_burn, etc.)

3. **Combining by species:**
   - Combines all human datasets into `human_combined.h5ad`
   - Combines all mouse datasets into `mouse_combined.h5ad`
   - Uses outer join to combine datasets (keeps all genes)

**Output locations:**
- `{DATA_ROOT}/processed/anndata_combined/human_combined.h5ad` - Combined human dataset
- `{DATA_ROOT}/processed/anndata_combined/mouse_combined.h5ad` - Combined mouse dataset

**Prerequisites:** 
- Step 5 (Ortholog-mapped AnnData files must exist)
- Step 6 (Optional, but embeddings are typically added before combining)

---

## Complete Pipeline Commands

### Full Automated Pipeline (Once orthology mapping is scripted):

```bash
# 1. Download raw data
make download-data

# 2. Extract metadata
make extract-metadata

# 3. RMA preprocessing + AnnData conversion
make preprocess-data

# 4. Orthology mapping (currently manual via notebook)
# TODO: Convert to CLI command
# python -m inflamm_debate_fm.cli preprocess orthologs

# 5. Generate embeddings (optional, but recommended)
make embed DATASET=human_burn
make embed DATASET=human_trauma
make embed DATASET=human_sepsis
make embed DATASET=mouse_burn
make embed DATASET=mouse_trauma
make embed DATASET=mouse_sepsis
make embed DATASET=mouse_infection

# 6. Preprocessing and combining
make data
```

### Current Manual Steps:

1. **Orthology Mapping:** Run `notebooks/1.0-data-preprocessing.ipynb` to map orthologs
2. **Embedding Generation:** Run embedding generation for each dataset
3. **Final Combining:** Run `make data` to combine and preprocess

---

## Directory Structure

```
{DATA_ROOT}/
├── raw/
│   ├── raw_cel/
│   │   └── {GSE_ID}/
│   │       └── *.CEL / *.CEL.gz
│   └── metadata/
│       └── {GSE_ID}.csv
├── external/
│   ├── mouse_orthology/
│   │   ├── HMD_HumanPhenotype.rpt
│   │   └── MGI_EntrezGene.rpt
│   └── platforms/
│       ├── hgu133plus2hsensgcdf_25.0.0.tar.gz
│       ├── mouse4302mmensgcdf_25.0.0.tar.gz
│       └── mouse430a2mmensgcdf_25.0.0.tar.gz
├── interim/
│   ├── rma_brainarray_processed/
│   │   └── {GSE_ID}/
│   │       └── {GSE_ID}_gene_matrix.csv
│   └── anndata/
│       └── {GSE_ID}_gene_rma_brainarray.h5ad
└── processed/
    ├── anndata_orthologs/
    │   └── {dataset_name}_orthologs.h5ad
    ├── bulkformer_embeddings/
    │   └── {dataset_name}_transcriptome_embeddings.npy
    └── anndata_combined/
        ├── human_combined.h5ad
        └── mouse_combined.h5ad
```

---

## Key Points

1. **Order Matters:** Steps must be run in sequence as each step depends on the previous one's output.

2. **Orthology Mapping:** Currently implemented in a Jupyter notebook. This should be converted to a CLI command for full automation.

3. **Embeddings:** Optional but recommended. Embeddings can be added at any point after orthology mapping and before combining.

4. **DATA_ROOT:** All paths are relative to `DATA_ROOT`, which can be set via `INFLAMM_DEBATE_FM_DATA_ROOT` environment variable. Defaults to `{PROJ_ROOT}/data`.

5. **R Script:** The R script (`preprocess_rma_brainarray.R`) now reads `DATA_ROOT` from environment variables, making it compatible with custom data directories.

6. **Error Handling:** The download step continues even if some files fail to download (e.g., network timeouts). Check logs for warnings about failed downloads.

---

## Troubleshooting

1. **Missing CEL files:** Re-run `make download-data` or download manually from GEO
2. **Missing BrainArray packages:** Check if tar files exist in `{DATA_ROOT}/external/platforms/`. If missing, re-run download or download manually.
3. **R script errors:** Ensure R and required Bioconductor packages are installed. The script will attempt to install missing packages automatically.
4. **Orthology mapping errors:** Ensure orthology files are downloaded and MyGene.info is accessible (requires internet connection).

---

## Next Steps

After completing the pipeline, you can:
1. Run probing experiments: `make probe-within SPECIES=human` or `make probe-cross`
2. Analyze coefficients: `make analyze-coeffs`
3. Run GSEA analysis: `make analyze-gsea`
4. Generate plots: `make plot-within SPECIES=human` or `make plot-cross`

