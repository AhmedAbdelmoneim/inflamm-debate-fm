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

# Sync dependencies
make requirements
```

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

# Process data (download, preprocess, map orthologs, combine)
make data

# Generate embeddings for a dataset
make embed DATASET=human_burn

# Run within-species probing
make probe-within SPECIES=human

# Run cross-species probing
make probe-cross

# Analyze coefficients
make analyze-coeffs

# Run GSEA analysis
make analyze-gsea

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

1. **Data Preprocessing**: `make data`
2. **Embedding Generation**: `make embed DATASET=<dataset_name>`
3. **Probing Experiments**: `make probe-within SPECIES=human` or `make probe-cross`
4. **Coefficient Analysis**: `make analyze-coeffs`
5. **GSEA Analysis**: `make analyze-gsea`
6. **Visualization**: `make plot-within SPECIES=human` or `make plot-cross`

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
./setup_hpc.sh
```

2. Submit jobs using the flexible script:
```bash
sbatch run_job.sh preprocess all
sbatch --gres=gpu:1 run_job.sh embed generate human_burn --device cuda
sbatch run_job.sh probe within-species --species human
```

See `jobs/README.md` for pre-configured job scripts.

## Additional Resources

- **CLI Usage**: Run `python -m inflamm_debate_fm.cli --help` for detailed CLI options
- **Configuration**: See `inflamm_debate_fm/config/default.toml` for all configuration options
- **HPC Jobs**: See `jobs/README.md` for HPC job submission details

## Contributing

1. Install development dependencies: `make requirements`
2. Run linting: `make lint`
3. Format code: `make format`
4. Run tests: `make test`
