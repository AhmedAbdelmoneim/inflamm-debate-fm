#!/bin/bash
#SBATCH --job-name=inflamm-data
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=def-<account>  # Replace with your Compute Canada account
#SBATCH --output=../logs/data_%j.out
#SBATCH --error=../logs/data_%j.err

# Job script for data processing pipeline
# This script runs the complete data processing pipeline:
# 1. Download raw data (CEL files, orthology, BrainArray packages)
# 2. Extract metadata from GEO datasets
# 3. RMA preprocessing using R script
# 4. Convert to AnnData format
# 5. Map orthologs (mouse → human, human → Ensembl)
# 6. Preprocessing and combining datasets
#
# Note: This job does NOT require GPU - it's CPU-only data processing
# The pipeline auto-detects existing files and skips completed steps
#
# Usage: sbatch jobs/preprocess.sh

cd "$(dirname "$0")/.."
exec ./run_job.sh preprocess data
