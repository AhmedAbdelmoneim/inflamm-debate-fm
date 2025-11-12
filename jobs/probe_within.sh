#!/bin/bash
#SBATCH --job-name=inflamm-probe-within
#SBATCH --time=04:00:00
#SBATCH --nodes=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --account=def-<account>  # Replace with your Compute Canada account
#SBATCH --output=../logs/probe_within_%j.out
#SBATCH --error=../logs/probe_within_%j.err

# Job script for within-species probing
# Usage: sbatch jobs/probe_within.sh <species>
# Example: sbatch jobs/probe_within.sh human

if [ -z "$1" ]; then
    echo "Error: Species required (human or mouse)"
    echo "Usage: sbatch jobs/probe_within.sh <species>"
    echo "Example: sbatch jobs/probe_within.sh human"
    exit 1
fi

cd "$(dirname "$0")/.."
exec ./run_job.sh probe within-species --species "$1"

