#!/bin/bash
#SBATCH --job-name=inflamm-embed
#SBATCH --time=08:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --account=def-<account>  # Replace with your Compute Canada account
#SBATCH --output=../logs/embed_%j.out
#SBATCH --error=../logs/embed_%j.err

# Job script for embedding generation with GPU support
# This script requests a GPU (--gres=gpu:1) and passes --device cuda to the CLI
# The run_job.sh script will automatically detect GPU allocation and add --nv flag to Apptainer
#
# Usage: sbatch jobs/embed.sh <dataset_name>
# Example: sbatch jobs/embed.sh human_burn
#
# Note: To request a specific GPU type on Compute Canada, add:
#   #SBATCH --gres=gpu:gpu_type:1
#   # For example: --gres=gpu:v100:1 or --gres=gpu:a100:1

if [ -z "$1" ]; then
    echo "Error: Dataset name required"
    echo "Usage: sbatch jobs/embed.sh <dataset_name>"
    echo "Example: sbatch jobs/embed.sh human_burn"
    exit 1
fi

cd "$(dirname "$0")/.."
exec ./run_job.sh embed generate "$1" --device cuda

