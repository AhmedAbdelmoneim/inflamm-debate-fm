# inflamm-debate-fm

Revisiting the mouse-human inflammation response debate using foundation models

## Overview

This project analyzes inflammation responses across human and mouse datasets using foundation models. It provides a modular pipeline for data preprocessing, embedding generation, probing experiments, and analysis.

## Project Organization

```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands
├── README.md          <- The top-level README for developers using this project
├── Singularity.def    <- Singularity container definition for HPC
├── setup_hpc.sh       <- Setup script for Compute Canada HPC
├── pyproject.toml     <- Project configuration file with package metadata
├── data
│   ├── external       <- Data from third party sources
│   ├── interim        <- Intermediate data that has been transformed
│   ├── processed      <- The final, canonical data sets for modeling
│   │   ├── anndata_orthologs      <- AnnData files with ortholog mapping
│   │   ├── bulkformer_embeddings  <- Pre-computed transcriptome embeddings
│   │   └── anndata_combined       <- Combined human and mouse datasets
│   └── raw            <- The original, immutable data dump
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks for exploratory analysis
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
└── inflamm_debate_fm  <- Source code for use in this project
    │
    ├── __init__.py
    │
    ├── cli.py                  <- Main command-line interface
    ├── dataset.py              <- Data preprocessing orchestration
    ├── plots.py                <- Visualization functions
    │
    ├── config/                 <- Configuration management
    │   ├── __init__.py
    │   ├── config.py           <- Configuration loading utilities
    │   └── default.toml        <- Default configuration file
    │
    ├── data/                   <- Data loading and preprocessing
    │   ├── __init__.py
    │   ├── load.py             <- Data loading functions
    │   ├── preprocessing.py    <- Data preprocessing functions
    │   └── transforms.py       <- Data transformation functions
    │
    ├── embeddings/             <- Embedding generation and loading
    │   ├── __init__.py
    │   ├── generate.py         <- Embedding generation functions
    │   └── load.py             <- Embedding loading functions
    │
    ├── modeling/               <- Modeling and evaluation
    │   ├── __init__.py
    │   ├── pipelines.py        <- Model pipeline definitions
    │   ├── evaluation.py       <- Evaluation functions (CV, LODO, cross-species)
    │   ├── coefficients.py     <- Coefficient extraction and analysis
    │   ├── train.py            <- Training entry point
    │   └── predict.py          <- Prediction entry point
    │
    ├── analysis/               <- Analysis functions
    │   ├── __init__.py
    │   ├── inflammation_vector.py <- Inflammation vector calculation
    │   ├── dimensionality.py   <- Dimensionality analysis
    │   └── gsea.py             <- GSEA analysis
    │
    └── utils/                  <- Utility functions
        ├── __init__.py
        ├── gene_utils.py       <- Gene utility functions
        ├── io.py               <- I/O utility functions
        └── wandb_utils.py      <- Wandb integration utilities
```

## Installation

### Local Development

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
# or
uv sync
```

3. Activate the virtual environment:
```bash
source .venv/bin/activate  # Unix/macOS
# or
.venv\Scripts\activate  # Windows
```

### Compute Canada HPC Setup

1. Load required modules:
```bash
module load singularity
module load python/3.13
```

2. Run the setup script:
```bash
./setup_hpc.sh
```

This will:
- Build the Singularity container image
- Create necessary data directories
- Set up environment variables
- Create a wrapper script for running commands

3. Use the wrapper script to run commands:
```bash
./run_singularity.sh python -m inflamm_debate_fm.cli <command>
```

## Configuration

Configuration is managed through TOML files. The default configuration is in `inflamm_debate_fm/config/default.toml`.

### Key Configuration Sections

- **`[time_cutoffs]`**: Time cutoffs for acute, subacute, and chronic inflammation classification
- **`[model]`**: Model hyperparameters (random seed, CV folds, bootstraps)
- **`[model.linear]`**: Linear model (Logistic Regression) parameters
- **`[model.nonlinear]`**: Nonlinear model (SVM) parameters
- **`[embedding]`**: Embedding model configuration
- **`[analysis]`**: Analysis parameters (GSEA FDR threshold, top k genes, figure settings)
- **`[paths]`**: Directory paths for data and outputs
- **`[wandb]`**: Wandb configuration for experiment tracking

### Environment Variable Overrides

You can override configuration paths using environment variables:
```bash
export INFLAMM_DEBATE_FM_ANN_DATA_DIR="/path/to/anndata"
export INFLAMM_DEBATE_FM_EMBEDDINGS_DIR="/path/to/embeddings"
```

### Wandb Configuration

Set your Wandb entity (optional):
```bash
export WANDB_ENTITY="your-entity-name"
export WANDB_API_KEY="your-api-key"
```

## Usage

### Command-Line Interface

The project provides a comprehensive CLI through `inflamm_debate_fm.cli`:

```bash
# Preprocess data
python -m inflamm_debate_fm.cli preprocess all

# Generate embeddings
python -m inflamm_debate_fm.cli embed generate <dataset_name> --model-name bulkformer

# Run within-species probing
python -m inflamm_debate_fm.cli probe within-species --species human --use-wandb

# Run cross-species probing
python -m inflamm_debate_fm.cli probe cross-species --use-wandb

# Analyze coefficients
python -m inflamm_debate_fm.cli analyze coefficients

# Run GSEA analysis
python -m inflamm_debate_fm.cli analyze gsea

# Generate plots
python -m inflamm_debate_fm.cli plot within-species --species human
python -m inflamm_debate_fm.cli plot cross-species
```

### Makefile Commands

For convenience, use the Makefile commands:

```bash
# Preprocess data
make data

# Generate embeddings
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

# Run full pipeline
make pipeline

# Run tests
make test
```

### Python API

You can also use the Python API directly:

```python
from inflamm_debate_fm.data.load import load_combined_adatas
from inflamm_debate_fm.modeling.evaluation import evaluate_within_species
from inflamm_debate_fm.data.transforms import transform_adata_to_X_y_all

# Load data
combined_adatas = load_combined_adatas()
human_adata = combined_adatas["human"]

# Define setups
setups = [
    ("All Inflammation vs. Control", transform_adata_to_X_y_all),
    # ... other setups
]

# Run evaluation
results, roc_data = evaluate_within_species(
    adata=human_adata,
    setups=setups,
    species_name="Human",
    n_cv_folds=10,
    use_wandb=True,
)
```

## Workflow

1. **Data Preprocessing**: Load and preprocess AnnData files, combine datasets by species
2. **Embedding Generation**: Generate transcriptome embeddings (if not pre-computed)
3. **Probing Experiments**: Run within-species and cross-species probing experiments
4. **Coefficient Analysis**: Extract and analyze model coefficients
5. **GSEA Analysis**: Run Gene Set Enrichment Analysis on ranked coefficients
6. **Visualization**: Generate plots for results

## HPC Usage (Compute Canada)

### Building the Container

The Singularity container can be built on a login node or compute node with Singularity access:

```bash
module load singularity
singularity build inflamm-debate-fm.sif Singularity.def
```

### Running Jobs

Create a SLURM job script:

```bash
#!/bin/bash
#SBATCH --job-name=inflamm-probe
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1  # If GPU is needed

module load singularity

# Set environment variables
export SINGULARITY_IMAGE="/path/to/inflamm-debate-fm.sif"
export DATA_DIR="/path/to/data"
export WANDB_API_KEY="your-api-key"

# Run command
singularity exec \
    --bind $DATA_DIR:/opt/inflamm-debate-fm/data \
    --bind $SCRATCH:/scratch \
    $SINGULARITY_IMAGE \
    python -m inflamm_debate_fm.cli probe within-species --species human --use-wandb
```

### Using the Wrapper Script

The setup script creates a `run_singularity.sh` wrapper that handles bind mounts:

```bash
./run_singularity.sh python -m inflamm_debate_fm.cli probe within-species --species human
```

## Experiment Tracking

The project integrates with Weights & Biases (Wandb) for experiment tracking. To use Wandb:

1. Set your Wandb API key:
```bash
export WANDB_API_KEY="your-api-key"
```

2. Enable Wandb logging in commands:
```bash
python -m inflamm_debate_fm.cli probe within-species --species human --use-wandb
```

3. Configure Wandb in `inflamm_debate_fm/config/default.toml`:
```toml
[wandb]
project = "inflamm-debate-fm"
entity = "your-entity"  # or set via WANDB_ENTITY environment variable
tags = ["research", "inflammation"]
```

## Data Format

The project expects AnnData files with the following structure:
- **`.X`**: Gene expression matrix (cells × genes)
- **`.obs`**: Cell metadata including:
  - `inflammation_category`: Inflammation category (control, acute, subacute, chronic)
  - `timepoint_hours`: Time since injury in hours
  - `dataset`: Dataset identifier
  - `species`: Species (human or mouse)
- **`.var`**: Gene metadata including:
  - `gene_symbol`: Gene symbol
  - `ensembl_id`: Ensembl gene ID
- **`.obsm`**: Optional embedding matrices (e.g., `X_bulkformer`)

## Contributing

1. Install development dependencies
2. Run linting: `make lint`
3. Format code: `make format`
4. Run tests: `make test`

## License

[Add your license here]

## Citation

[Add citation information here]

--------
