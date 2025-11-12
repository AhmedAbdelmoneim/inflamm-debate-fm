#!/bin/bash
#SBATCH --job-name=inflamm-analyze-gsea
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=def-<account>  # Replace with your Compute Canada account
#SBATCH --output=../logs/analyze_gsea_%j.out
#SBATCH --error=../logs/analyze_gsea_%j.err

# Job script for GSEA analysis
# Usage: sbatch jobs/analyze_gsea.sh

cd "$(dirname "$0")/.."
exec ./run_job.sh analyze gsea

