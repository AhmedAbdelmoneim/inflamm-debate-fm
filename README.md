# inflamm-debate-fm

Revisiting the mouse-human inflammation response debate using foundation models

## Overview

This project analyzes inflammation responses across human and mouse datasets using foundation models. It provides a modular pipeline for data preprocessing, embedding generation, probing experiments, and analysis.

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd inflamm-debate-fm
```

2. Install dependencies using `uv`:
```bash
# Install uv if not already installed
curl -LsSf https://astral.sh/uv/install.sh | sh

# Sync dependencies (installs PyTorch 2.8.* with CUDA 12.9 and PyG optional dependencies)
make requirements
```

**Note:** This project uses Python 3.12.7 and PyTorch 2.8.* with CUDA 12.9 support. The `make requirements` command will:
- Install PyTorch 2.8.* with CUDA 12.9 from PyTorch's wheel repository
- Install torch-geometric and other dependencies
- Install PyG optional dependencies (pyg_lib, torch_scatter, torch_sparse, torch_cluster, torch_spline_conv) from the PyG wheel repository

3. Activate the virtual environment:
```bash
source .venv/bin/activate  # Unix/macOS
```

4. (Optional) Configure environment variables in `.env`:
```bash
# Copy example if available
cp .env.example .env

# Edit .env to set paths (e.g., for HPC with separate storage)
# INFLAMM_DEBATE_FM_DATA_ROOT=/path/to/large/data/directory
# INFLAMM_DEBATE_FM_MODELS_ROOT=/path/to/models/directory
```

## Quick Start

### Using Make Commands

The easiest way to run the pipeline is using Make commands:

```bash
# See all available commands
make help

# Install dependencies
make requirements

# Set up BulkFormer model (clone repo and check for model files)
make bulkformer-setup

# Process data (download, preprocess, map orthologs, combine)
make data

# Generate embeddings for a dataset
make embed DATASET=human_burn

# Generate embeddings for all configurations (human-only, mouse-only, human-ortholog-filtered)
make embed-all DEVICE=cuda USE_WANDB=true

# Generate multi-model embeddings (zero-shot + fine-tuned) with mean-pooling
make embed-multi-model DATASET=human_burn DEVICE=cuda
# Or process all datasets:
make embed-multi-model DEVICE=cuda

# Run within-species probing
make probe-within SPECIES=human

# Run cross-species probing
make probe-cross

# Analyze coefficients
make analyze-coeffs

# Run GSEA analysis
make analyze-gsea

# Fine-tune models with LoRA
make finetune SPECIES=human
make finetune SPECIES=mouse
make finetune SPECIES=combined

# Generate plots
make plot-within SPECIES=human
make plot-cross

# Development commands
make lint      # Check code style
make format    # Format code
make test      # Run tests
make clean     # Remove compiled Python files
```

### Pipeline Workflow

1. **BulkFormer Setup**: `make bulkformer-setup` (clone repo and verify model files)
2. **Data Preprocessing**: `make data`
3. **Embedding Generation**: 
   - Single dataset: `make embed DATASET=<dataset_name>`
   - All configurations: `make embed-all DEVICE=cuda` (generates human-only, mouse-only, and human-ortholog-filtered embeddings)
   - Multi-model embeddings: `make embed-multi-model [DATASET=<dataset_name|all>] DEVICE=cuda` (zero-shot + fine-tuned with mean-pooling, defaults to all datasets)
4. **Probing Experiments**: `make probe-within SPECIES=human` or `make probe-cross`
5. **Fine-tuning (LoRA)**: `make finetune SPECIES=human` (see Fine-tuning section below)
6. **Coefficient Analysis**: `make analyze-coeffs`
7. **GSEA Analysis**: `make analyze-gsea`
8. **Visualization**: `make plot-within SPECIES=human` or `make plot-cross`

## Configuration

Configuration is managed through `inflamm_debate_fm/config/default.toml`. Key settings:

- **`[time_cutoffs]`**: Time cutoffs for inflammation classification
- **`[model]`**: Model hyperparameters (CV folds, random seed)
- **`[paths]`**: Directory paths (can be overridden via environment variables)
- **`[wandb]`**: Wandb configuration for experiment tracking (optional)

Environment variables can be set in `.env` or exported:
- `INFLAMM_DEBATE_FM_DATA_ROOT`: Root directory for data files
- `INFLAMM_DEBATE_FM_MODELS_ROOT`: Root directory for model files
- `WANDB_ENTITY` and `WANDB_API_KEY`: For experiment tracking (optional)

## HPC Usage (Compute Canada)

1. Load modules and run setup:
```bash
module load apptainer
./hpc/setup_hpc.sh
```

2. Check CUDA versions (optional):
```bash
# List available CUDA module versions
module avail cuda

# Check GPU and driver information
nvidia-smi

# Check CUDA version if nvcc is available
nvcc --version
```

3. Submit jobs using the flexible script:
```bash
# CPU jobs
sbatch hpc/run_job.sh preprocess data

# GPU jobs (CUDA is in the container, host CUDA module usually not needed)
sbatch --gpus=h100:1 hpc/run_job.sh embed generate human_burn --device cuda
# For large datasets, use chunk-size to avoid OOM (processes samples in chunks)
sbatch --gpus=h100:1 --time=24:00:00 --mem=64G hpc/run_job.sh embed all-configs --device cuda --batch-size 16 --chunk-size 2000 --use-wandb
# Multi-model embeddings (zero-shot + fine-tuned) with mean-pooling
sbatch --gpus=h100:1 --time=12:00:00 --mem=32G hpc/run_job.sh embed multi-model human_burn --device cuda --batch-size 4
# Or process all datasets:
sbatch --gpus=h100:1 --time=24:00:00 --mem=64G hpc/run_job.sh embed multi-model all --device cuda --batch-size 4

# If you need a specific CUDA module version on the host:
CUDA_VERSION=12.9 sbatch --gpus=h100:1 hpc/run_job.sh embed generate human_burn --device cuda

# Other commands
sbatch hpc/run_job.sh probe within-species --species human
```

**Note:** The container includes PyTorch with CUDA 12.9 support. The host CUDA module is typically not required, but you can load a specific version using the `CUDA_VERSION` environment variable if needed.

## Fine-tuning with LoRA

Fine-tune BulkFormer models using LoRA (Low-Rank Adaptation) for inflammation classification. This uses a subset of the data (32+32 samples by default) and tracks which samples were used to exclude them from downstream evaluation. A new **universal** mode adds a cross-species InfoNCE objective that L2-normalizes embeddings and pulls human/mouse inflammation samples together while pushing all other cross-species pairs apart.

### Data Validation

The fine-tuning process:
- Uses 32 inflammation + 32 control samples by default (configurable)
- Tracks which samples were used for fine-tuning
- Saves metadata to exclude these samples from downstream evaluation
- Reports the percentage of available data used

### Training Models

Fine-tune four models:
1. `human` — human-only classification objective
2. `mouse` — mouse-only classification objective
3. `combined` — human+mouse classification objective
4. `universal` — cross-species contrastive objective (aligns human & mouse inflammation embeddings via InfoNCE)

```bash
# Train on human data only
python -m inflamm_debate_fm.cli finetune train --species human

# Train on mouse data only
python -m inflamm_debate_fm.cli finetune train --species mouse

# Train on combined human+mouse data
python -m inflamm_debate_fm.cli finetune train --species combined

# Train universal cross-species model (InfoNCE + classification)
python -m inflamm_debate_fm.cli finetune train \
    --species universal \
    --contrastive-weight 1.0 \
    --contrastive-temperature 0.07 \
    --use-wandb
```

### Custom Training Options

```bash
# Custom sample sizes and training parameters
python -m inflamm_debate_fm.cli finetune train \
    --species human \
    --n-inflammation 32 \
    --n-control 32 \
    --epochs 20 \
    --batch-size 16 \
    --lr 5e-5 \
    --weight-decay 0.01 \
    --use-wandb
```

For the universal contrastive mode, tune the shared embedding alignment via `--contrastive-weight` and `--contrastive-temperature`.

### Output Structure

Each fine-tuning run creates:
- `checkpoint_best/`: Best checkpoint (lowest loss)
- `checkpoint_final/`: Final checkpoint after all epochs
- `finetuning_samples_{species}.csv`: CSV file listing which samples were used
- `finetuning_summary_{species}.json`: Summary with percentage of data used

### HPC Usage

```bash
# Submit fine-tuning job on HPC
sbatch --gpus=h100:1 --time=12:00:00 --mem=32G \
    hpc/run_job.sh finetune train --species human --use-wandb

# Train all four models (including universal contrastive mode)
for species in human mouse combined universal; do
    sbatch --gpus=h100:1 --time=12:00:00 --mem=32G \
        hpc/run_job.sh finetune train --species $species --use-wandb
done
```

### Using Fine-tuned Models

Fine-tuned models can be loaded and used alongside the zero-shot model for probing experiments. The sample metadata files (`finetuning_samples_{species}.csv`) should be used to exclude fine-tuning samples from evaluation datasets.

### Multi-Model Embeddings

After fine-tuning, generate mean-pooled sample-level embeddings from all available models (zero-shot + fine-tuned):

```bash
# Generate multi-model embeddings for a single dataset
python -m inflamm_debate_fm.cli embed multi-model human_burn --device cuda --batch-size 4

# Or process all datasets
python -m inflamm_debate_fm.cli embed multi-model all --device cuda --batch-size 4

# Or using Make (defaults to all datasets if DATASET not specified)
make embed-multi-model DATASET=human_burn DEVICE=cuda
make embed-multi-model DEVICE=cuda  # Processes all datasets
```

This command:
- Automatically detects available models (zero-shot + fine-tuned)
- Generates mean-pooled sample-level embeddings from each model
- Saves embeddings to the AnnData file in `obsm` with keys:
  - `X_zero_shot`: Zero-shot model embeddings
  - `X_human`: Human fine-tuned model embeddings (if available)
  - `X_mouse`: Mouse fine-tuned model embeddings (if available)
  - `X_combined`: Combined fine-tuned model embeddings (if available)
  - `X_universal`: Cross-species contrastive embeddings (if available)

The embeddings are ready for probing analysis. Uses `batch_size=4` by default for CUDA memory optimization.

## Additional Resources

- **CLI Usage**: Run `python -m inflamm_debate_fm.cli --help` for detailed CLI options
- **Configuration**: See `inflamm_debate_fm/config/default.toml` for all configuration options

## Contributing

1. Install development dependencies: `make requirements`
2. Run linting: `make lint`
3. Format code: `make format`
4. Run tests: `make test`
