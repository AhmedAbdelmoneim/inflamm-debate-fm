#!/bin/bash
#SBATCH --job-name=inflamm-probe-cross
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=def-<account>  # Replace with your Compute Canada account
#SBATCH --output=../logs/probe_cross_%j.out
#SBATCH --error=../logs/probe_cross_%j.err

# Job script for cross-species probing
# Usage: sbatch jobs/probe_cross.sh

cd "$(dirname "$0")/.."
exec ./run_job.sh probe cross-species

