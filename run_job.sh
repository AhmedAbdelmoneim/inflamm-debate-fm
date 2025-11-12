#!/bin/bash
#SBATCH --job-name=inflamm-fm
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=def-<account>  # Replace with your Compute Canada account
#SBATCH --output=logs/job_%j.out
#SBATCH --error=logs/job_%j.err

# Flexible SLURM job script for inflamm-debate-fm
# Usage examples:
#   sbatch run_job.sh preprocess all
#   sbatch run_job.sh embed generate human_burn --device cuda
#   sbatch run_job.sh probe within-species --species human
#   sbatch run_job.sh probe cross-species
#   sbatch run_job.sh analyze coefficients
#   sbatch run_job.sh analyze gsea
#   sbatch run_job.sh plot within-species --species human
#
# For GPU jobs (e.g., embeddings), add to sbatch command:
#   sbatch --gres=gpu:1 run_job.sh embed generate human_burn --device cuda

set -e

# Load required modules
module load apptainer

# Load CUDA module if GPU is requested (Compute Canada)
# Note: This might not be needed if CUDA is in the container, but doesn't hurt
# Uncomment the next line if needed:
# module load cuda

# Get project root directory
PROJ_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJ_ROOT"

# Set up paths
export APPTAINER_IMAGE="${APPTAINER_IMAGE:-$PROJ_ROOT/inflamm-debate-fm.sif}"
export DATA_DIR="${DATA_DIR:-$PROJ_ROOT/data}"

# Create logs directory if it doesn't exist
mkdir -p logs

# Check if Apptainer image exists
if [ ! -f "$APPTAINER_IMAGE" ]; then
    echo "Error: Apptainer image not found at $APPTAINER_IMAGE"
    echo "Please run ./setup_hpc.sh first to build the container"
    exit 1
fi

# Check if run_apptainer.sh exists
if [ ! -f "$PROJ_ROOT/run_apptainer.sh" ]; then
    echo "Error: run_apptainer.sh not found. Please run ./setup_hpc.sh first"
    exit 1
fi

# Detect if GPU is requested
USE_GPU=0

# Check SLURM GPU allocation (Compute Canada sets CUDA_VISIBLE_DEVICES when --gres=gpu is used)
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [ "${CUDA_VISIBLE_DEVICES:-}" != "NoDevFiles" ]; then
    USE_GPU=1
    echo "GPU detected via CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
# Check SLURM_GPUS_ON_NODE (some SLURM configurations)
elif [ -n "${SLURM_GPUS_ON_NODE:-}" ] && [ "${SLURM_GPUS_ON_NODE:-0}" -gt 0 ]; then
    USE_GPU=1
    echo "GPU detected via SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
# Check SLURM_GPUS (alternative SLURM variable)
elif [ -n "${SLURM_GPUS:-}" ] && [ "${SLURM_GPUS:-0}" -gt 0 ]; then
    USE_GPU=1
    echo "GPU detected via SLURM_GPUS: $SLURM_GPUS"
# Check if --device cuda is in command arguments
elif echo "$@" | grep -q -- "--device cuda"; then
    USE_GPU=1
    echo "GPU requested via --device cuda flag"
fi

# Set environment variable for GPU access
export APPTAINER_USE_GPU=$USE_GPU

# Print job information
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
if [ "$USE_GPU" = "1" ]; then
    echo "GPU: Enabled (--nv flag will be used)"
    if command -v nvidia-smi &> /dev/null; then
        echo "GPU Information:"
        nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
    fi
else
    echo "GPU: Disabled"
fi
echo "========================================="
echo "Command: $@"
echo "========================================="

# Run the command using the wrapper script
# All arguments after the script name are passed to the CLI
"$PROJ_ROOT/run_apptainer.sh" python -m inflamm_debate_fm.cli "$@"

# Print completion information
echo "========================================="
echo "End Time: $(date)"
echo "Job completed successfully"
echo "========================================="

