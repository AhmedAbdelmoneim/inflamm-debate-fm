"""Constants for data processing."""

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

# MGI orthology file URLs
MGI_BASE_URL = "https://www.informatics.jax.org/downloads/reports"
MGI_FILES = {
    "HMD_HumanPhenotype.rpt": f"{MGI_BASE_URL}/HMD_HumanPhenotype.rpt",
    "MGI_EntrezGene.rpt": f"{MGI_BASE_URL}/MGI_EntrezGene.rpt",
}

# BrainArray CDF package URLs
BRAINARRAY_BASE_URL = (
    "https://brainarray.mbni.med.umich.edu/Brainarray/Database/CustomCDF/25.0.0/entrezg.download"
)
BRAINARRAY_PACKAGES = {
    "hgu133plus2hsensgcdf": f"{BRAINARRAY_BASE_URL}/hgu133plus2hsensgcdf_25.0.0.tar.gz",
    "mouse4302mmensgcdf": f"{BRAINARRAY_BASE_URL}/mouse4302mmensgcdf_25.0.0.tar.gz",
    "mouse430a2mmensgcdf": f"{BRAINARRAY_BASE_URL}/mouse430a2mmensgcdf_25.0.0.tar.gz",
}

# GEO FTP base URL pattern for RAW.tar files
# Format: ftp://ftp.ncbi.nlm.nih.gov/geo/series/{series_dir}/{gse_id}/suppl/{gse_id}_RAW.tar
# where series_dir = "{gse_num[:-3]}nnn" (e.g., "GSE12345" -> "12nnn")
GEO_FTP_BASE_URL = "ftp://ftp.ncbi.nlm.nih.gov/geo/series"
