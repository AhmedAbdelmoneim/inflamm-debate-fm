#!/usr/bin/env Rscript

suppressPackageStartupMessages({
  library(readr)
  library(tibble)
})

# --- Helper: install Bioconductor/BrainArray packages if missing ---
ensure_pkg <- function(pkg) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    if (!requireNamespace("BiocManager", quietly = TRUE))
      install.packages("BiocManager")
    message("Installing missing package: ", pkg)
    BiocManager::install(pkg, ask = FALSE, update = FALSE)
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

install_cdf <- function(pkg, tar_path) {
  if (!requireNamespace(pkg, quietly = TRUE)) {
    message("Installing BrainArray CDF package: ", pkg)
    install.packages(tar_path, repos = NULL, type = "source")
  }
  suppressPackageStartupMessages(library(pkg, character.only = TRUE))
}

# --- Lookup table: GPL → BrainArray CDF package & tarball ---
# Will be updated with absolute paths after root is determined
platform_map_template <- list(
  "GPL1261" = list(pkg = "mouse4302mmensgcdf",
                   tar_rel = "data/external/platforms/mouse4302mmensgcdf_25.0.0.tar.gz"),
  "GPL8321" = list(pkg = "mouse430a2mmensgcdf",
                   tar_rel = "data/external/platforms/mouse430a2mmensgcdf_25.0.0.tar.gz"),
  "GPL570"  = list(pkg = "hgu133plus2hsensgcdf",
                   tar_rel = "data/external/platforms/hgu133plus2hsensgcdf_25.0.0.tar.gz")
)

# --- Paths ---
# Get DATA_ROOT from environment variable (set by Python config)
# If not set, use project root/data as fallback (for backward compatibility)
data_root <- Sys.getenv("INFLAMM_DEBATE_FM_DATA_ROOT", unset = NA)
if (is.na(data_root)) {
  # Fallback: assume script is run from project root
  project_root <- getwd()
  data_root <- file.path(project_root, "data")
  message("INFLAMM_DEBATE_FM_DATA_ROOT not set, using: ", data_root)
} else {
  message("Using DATA_ROOT from environment: ", data_root)
}

# Convert to absolute path
data_root <- normalizePath(data_root, mustWork = FALSE)

# Build paths relative to DATA_ROOT
raw_dir <- file.path(data_root, "raw", "raw_cel")
meta_dir <- file.path(data_root, "raw", "metadata")
platforms_base <- file.path(data_root, "external", "platforms")
out_root <- file.path(data_root, "interim", "rma_brainarray_processed")
dir.create(out_root, showWarnings = FALSE, recursive = TRUE)

# Ensure raw_dir exists
if (!dir.exists(raw_dir)) {
  stop("Raw CEL directory not found: ", raw_dir)
}

# Ensure metadata directory exists
if (!dir.exists(meta_dir)) {
  stop("Metadata directory not found: ", meta_dir)
}

# Ensure platforms directory exists
if (!dir.exists(platforms_base)) {
  stop("Platforms directory not found: ", platforms_base)
}

# Build platform_map with absolute paths relative to DATA_ROOT
platform_map <- lapply(platform_map_template, function(plat) {
  # Extract relative path from template (e.g., "data/external/platforms/...")
  # Remove "data/" prefix and use DATA_ROOT directly
  tar_rel <- sub("^data/", "", plat$tar_rel)
  tar_path <- file.path(data_root, tar_rel)
  list(pkg = plat$pkg, tar = normalizePath(tar_path, mustWork = FALSE))
})

datasets <- list.dirs(raw_dir, full.names = TRUE, recursive = FALSE)

process_one <- function(ds_path) {
  ds_name <- basename(ds_path)
  message("\n=== Processing ", ds_name, " ===")

  # CEL files (prefer .CEL, fallback .CEL.gz)
  cel_files <- list.files(ds_path, pattern = "\\.CEL$", full.names = TRUE, ignore.case = TRUE)
  if (length(cel_files) == 0) {
    cel_files <- list.files(ds_path, pattern = "\\.CEL\\.gz$", full.names = TRUE, ignore.case = TRUE)
  }
  if (length(cel_files) == 0) {
    warning("No CEL files found in ", ds_path)
    return(NULL)
  }

  # Guess GSE accession from dir name
  gse <- gsub("^.*(GSE[0-9]+).*$", "\\1", ds_name)

  # Load metadata for GPL
  meta_csv <- file.path(meta_dir, paste0(gse, ".csv"))
  if (!file.exists(meta_csv)) {
    warning("No metadata CSV for ", gse, ". Skipping.")
    return(NULL)
  }
  meta <- suppressMessages(read_csv(meta_csv, show_col_types = FALSE))
  gpl_col <- intersect(c("platform_id", "platform", "GPL", "gpl"), names(meta))
  if (length(gpl_col) == 0) {
    warning("No platform column in metadata for ", gse, ". Skipping.")
    return(NULL)
  }
  gpl <- unique(na.omit(meta[[gpl_col[1]]]))[1]

  if (!(gpl %in% names(platform_map))) {
    warning("No BrainArray mapping configured for ", gpl, ". Skipping.")
    return(NULL)
  }
  plat <- platform_map[[gpl]]
  message("  Detected GPL: ", gpl, " → using BrainArray package ", plat$pkg)

  # Ensure affy + BrainArray CDF package
  ensure_pkg("affy")
  
  # Build full path to tar file (already absolute from platform_map)
  tar_path <- plat$tar
  if (!file.exists(tar_path)) {
    warning("BrainArray CDF package not found: ", tar_path, ". Skipping.")
    return(NULL)
  }
  
  install_cdf(plat$pkg, tar_path)

  # --- Use affy::ReadAffy with custom CDF ---
  library(affy)
  raw <- ReadAffy(filenames = cel_files, cdfname = plat$pkg)
  eset <- rma(raw)   # bg correction + quantile norm + summarization + log2
  exprs_mat <- exprs(eset)

  # --- Save ---
  out_dir <- file.path(out_root, ds_name)
  dir.create(out_dir, showWarnings = FALSE, recursive = TRUE)
  out_file <- file.path(out_dir, paste0(ds_name, "_gene_matrix.csv"))
  write_csv(as.data.frame(exprs_mat) |> rownames_to_column("GENE_ID"), out_file)

  message("  Wrote: ", out_file)
}

for (d in datasets) {
  try(process_one(d), silent = FALSE)
}
message("\nAll datasets processed with BrainArray RMA.")

