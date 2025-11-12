#!/bin/bash
#SBATCH --job-name=inflamm-analyze-coeffs
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=def-<account>  # Replace with your Compute Canada account
#SBATCH --output=../logs/analyze_coeffs_%j.out
#SBATCH --error=../logs/analyze_coeffs_%j.err

# Job script for coefficient analysis
# Usage: sbatch jobs/analyze_coeffs.sh

cd "$(dirname "$0")/.."
exec ./run_job.sh analyze coefficients

