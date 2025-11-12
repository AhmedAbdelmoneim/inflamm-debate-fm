#!/bin/bash
#SBATCH --job-name=inflamm-preprocess
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=def-<account>  # Replace with your Compute Canada account
#SBATCH --output=../logs/preprocess_%j.out
#SBATCH --error=../logs/preprocess_%j.err

# Job script for data preprocessing
# Usage: sbatch jobs/preprocess.sh

cd "$(dirname "$0")/.."
exec ./run_job.sh preprocess all

