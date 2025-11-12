# HPC Job Scripts

This directory contains pre-configured SLURM job scripts for common pipeline tasks.

## Setup

Before using these scripts, you must:

1. Edit each script to set your Compute Canada account:
   ```bash
   #SBATCH --account=def-<account>  # Replace <account> with your account name
   ```

2. Ensure you've run `./setup_hpc.sh` to build the Apptainer container and create the wrapper script.

## Available Job Scripts

### `preprocess.sh`
Runs the complete data processing pipeline (download, preprocess, map orthologs, combine).

This pipeline:
1. Downloads raw data (CEL files, orthology files, BrainArray packages)
2. Extracts metadata from GEO datasets
3. Runs RMA preprocessing using R script
4. Converts to AnnData format
5. Maps orthologs (mouse → human, human → Ensembl)
6. Preprocesses and combines datasets

**Note:** The pipeline auto-detects existing files and skips completed steps, so it's safe to re-run.

**Usage:**
```bash
sbatch jobs/preprocess.sh
```

**Resources:** 8 CPUs, 32GB RAM, 24 hours

**Note:** This job does NOT require GPU - it's CPU-only data processing. The longer time limit (24 hours) accounts for:
- Network downloads (can be slow)
- RMA preprocessing with R (computationally intensive)
- Ortholog mapping with MyGene.info API calls

---

### `embed.sh`
Generates embeddings for a dataset (requires GPU).

**Usage:**
```bash
sbatch jobs/embed.sh <dataset_name>
```

**Example:**
```bash
sbatch jobs/embed.sh human_burn
```

**Resources:** 8 CPUs, 32GB RAM, 1 GPU, 8 hours

**GPU Support:**
- Automatically requests GPU via `--gres=gpu:1`
- Passes `--device cuda` to the CLI
- The `run_job.sh` script automatically detects GPU allocation and adds `--nv` flag to Apptainer
- GPU information is displayed in job logs

**Note:** To request a specific GPU type on Compute Canada, edit the script and change:
```bash
#SBATCH --gres=gpu:1
```
to:
```bash
#SBATCH --gres=gpu:gpu_type:1
# For example: --gres=gpu:v100:1 or --gres=gpu:a100:1
```

---

### `probe_within.sh`
Runs within-species probing experiments.

**Usage:**
```bash
sbatch jobs/probe_within.sh <species>
```

**Example:**
```bash
sbatch jobs/probe_within.sh human
sbatch jobs/probe_within.sh mouse
```

**Resources:** 4 CPUs, 16GB RAM, 4 hours

---

### `probe_cross.sh`
Runs cross-species probing experiments.

**Usage:**
```bash
sbatch jobs/probe_cross.sh
```

**Resources:** 4 CPUs, 16GB RAM, 4 hours

---

### `analyze_coeffs.sh`
Extracts and analyzes model coefficients.

**Usage:**
```bash
sbatch jobs/analyze_coeffs.sh
```

**Resources:** 4 CPUs, 16GB RAM, 2 hours

---

### `analyze_gsea.sh`
Runs Gene Set Enrichment Analysis on coefficients.

**Usage:**
```bash
sbatch jobs/analyze_gsea.sh
```

**Resources:** 4 CPUs, 16GB RAM, 2 hours

---

## Customizing Resources

You can override resource requirements when submitting:

```bash
# Request more memory
sbatch --mem=64G jobs/preprocess.sh

# Request more time
sbatch --time=08:00:00 jobs/embed.sh human_burn

# Request more CPUs
sbatch --cpus-per-task=16 jobs/probe_within.sh human
```

## GPU Support

GPU support is automatically enabled when:
1. The job requests a GPU via `--gres=gpu:1` (or similar) in the SLURM script
2. The command includes `--device cuda` flag

The scripts automatically:
- Detect GPU allocation via SLURM environment variables (`CUDA_VISIBLE_DEVICES`, `SLURM_GPUS_ON_NODE`)
- Add the `--nv` flag to Apptainer for GPU access
- Display GPU information in job logs

**Example:**
```bash
# The embed.sh script is already configured for GPU
sbatch jobs/embed.sh human_burn

# Using the flexible script with GPU
sbatch --gres=gpu:1 run_job.sh embed generate human_burn --device cuda
```

## Using the Flexible Script

For more control or custom commands, use the main `run_job.sh` script:

```bash
# From the project root
sbatch run_job.sh <any-cli-command>
```

See the main README.md for more examples.

