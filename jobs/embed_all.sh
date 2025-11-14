#!/bin/bash
#SBATCH --job-name=inflamm-embed-all
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --account=def-<account>  # Replace with your Compute Canada account
#SBATCH --output=../logs/embed_all_%j.out
#SBATCH --error=../logs/embed_all_%j.err

# Job script for generating embeddings for all configurations
# This script requests a GPU (--gres=gpu:1) and passes --device cuda to the CLI
#
# Usage: sbatch jobs/embed_all.sh [USE_WANDB=true]
# Example: sbatch jobs/embed_all.sh USE_WANDB=true
#
# Note: To request a specific GPU type on Compute Canada, add:
#   #SBATCH --gres=gpu:gpu_type:1
#   # For example: --gres=gpu:v100:1 or --gres=gpu:a100:1

cd "$(dirname "$0")/.."

USE_WANDB=${USE_WANDB:-false}

if [ "$USE_WANDB" = "true" ]; then
    exec ./run_job.sh embed all-configs --device cuda --use-wandb
else
    exec ./run_job.sh embed all-configs --device cuda
fi

