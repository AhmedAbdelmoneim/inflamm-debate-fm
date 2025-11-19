#!/bin/bash
# Complete probing pipeline: within-species + parallel cross-species bootstrapping
# Usage: ./scripts/run_probing_parallel.sh [options]
#
# Options:
#   --n-bootstraps N      Total number of bootstraps (default: 100)
#   --n-jobs N           Number of parallel jobs for cross-species (default: 10)
#   --n-cv-folds N       Number of CV folds for within-species (default: 10)
#   --use-wandb          Enable wandb logging
#   --skip-within        Skip within-species experiments
#   --skip-cross         Skip cross-species experiments
#
# Example:
#   ./scripts/run_probing_parallel.sh --n-bootstraps 100 --n-jobs 10 --use-wandb

set -e

# Default values
N_BOOTSTRAPS=100
N_JOBS=10
N_CV_FOLDS=10
USE_WANDB=""
SKIP_WITHIN=false
SKIP_CROSS=false
OTHER_ARGS=()

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --n-bootstraps)
            N_BOOTSTRAPS="$2"
            shift 2
            ;;
        --n-jobs)
            N_JOBS="$2"
            shift 2
            ;;
        --n-cv-folds)
            N_CV_FOLDS="$2"
            shift 2
            ;;
        --use-wandb)
            USE_WANDB="--use-wandb"
            shift
            ;;
        --skip-within)
            SKIP_WITHIN=true
            shift
            ;;
        --skip-cross)
            SKIP_CROSS=true
            shift
            ;;
        *)
            OTHER_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "========================================="
echo "Parallel Probing Pipeline"
echo "========================================="
echo "Bootstrap iterations: $N_BOOTSTRAPS"
echo "Parallel jobs: $N_JOBS"
echo "CV folds: $N_CV_FOLDS"
echo "Wandb: ${USE_WANDB:-disabled}"
echo "========================================="
echo ""

# Step 1: Within-species experiments
if [ "$SKIP_WITHIN" = false ]; then
    echo "Step 1: Submitting within-species experiments..."
    
    # Human
    JOB_HUMAN=$(sbatch \
        --job-name=probe-within-human \
        --time=12:00:00 \
        --cpus-per-task=8 \
        --mem=32G \
        --parsable \
        hpc/run_job.sh \
        probe within-species --species human \
        --n-cv-folds $N_CV_FOLDS \
        $USE_WANDB \
        "${OTHER_ARGS[@]}")
    
    echo "  Submitted human job: $JOB_HUMAN"
    
    # Mouse
    JOB_MOUSE=$(sbatch \
        --job-name=probe-within-mouse \
        --time=12:00:00 \
        --cpus-per-task=8 \
        --mem=32G \
        --parsable \
        hpc/run_job.sh \
        probe within-species --species mouse \
        --n-cv-folds $N_CV_FOLDS \
        $USE_WANDB \
        "${OTHER_ARGS[@]}")
    
    echo "  Submitted mouse job: $JOB_MOUSE"
    echo ""
else
    echo "Skipping within-species experiments"
    JOB_HUMAN=""
    JOB_MOUSE=""
fi

# Step 2: Cross-species experiments with parallel bootstraps
if [ "$SKIP_CROSS" = false ]; then
    echo "Step 2: Submitting cross-species parallel bootstrap jobs..."
    
    BOOTSTRAPS_PER_JOB=$((N_BOOTSTRAPS / N_JOBS))
    REMAINDER=$((N_BOOTSTRAPS % N_JOBS))
    
    echo "  Bootstraps per job: $BOOTSTRAPS_PER_JOB (remainder: $REMAINDER)"
    
    CROSS_JOB_IDS=()
    START=0
    
    for i in $(seq 0 $((N_JOBS - 1))); do
        END=$((START + BOOTSTRAPS_PER_JOB))
        
        # Add remainder to last job
        if [ $i -eq $((N_JOBS - 1)) ]; then
            END=$((END + REMAINDER))
        fi
        
        JOB_ID=$(sbatch \
            --job-name=probe-cross-bs-${START}-${END} \
            --time=12:00:00 \
            --cpus-per-task=8 \
            --mem=32G \
            --parsable \
            hpc/run_job.sh \
            probe cross-species \
            --n-bootstraps $N_BOOTSTRAPS \
            --bootstrap-start $START \
            --bootstrap-end $END \
            $USE_WANDB \
            "${OTHER_ARGS[@]}")
        
        CROSS_JOB_IDS+=($JOB_ID)
        echo "  Submitted job $((i + 1))/$N_JOBS: bootstraps $START-$((END - 1)) (job $JOB_ID)"
        
        START=$END
    done
    
    echo ""
    echo "All cross-species jobs submitted: ${CROSS_JOB_IDS[@]}"
else
    echo "Skipping cross-species experiments"
    CROSS_JOB_IDS=()
fi

# Step 3: Generate combine script
echo ""
echo "Step 3: Generating combine script..."
COMBINE_SCRIPT="combine_cross_species_results.sh"

cat > "$COMBINE_SCRIPT" << EOF
#!/bin/bash
# Combine cross-species bootstrap results
# Run this after all jobs complete

set -e

OUTPUT_DIR="data/probing_results/cross_species"
COMBINED_OUTPUT="\$OUTPUT_DIR/cross_species_results_combined.pkl"

# Collect all partial result files
RESULT_FILES=(\$(ls \$OUTPUT_DIR/cross_species_results_bs_*.pkl 2>/dev/null | sort -V))

if [ \${#RESULT_FILES[@]} -eq 0 ]; then
    echo "Error: No partial result files found in \$OUTPUT_DIR"
    echo "Make sure all jobs have completed successfully"
    exit 1
fi

echo "Found \${#RESULT_FILES[@]} result files to combine"
echo ""

# Combine results
./run_apptainer.sh python -m inflamm_debate_fm.cli.combine_bootstrap_results \\
    "\$COMBINED_OUTPUT" \\
    "\${RESULT_FILES[@]}"

echo ""
echo "========================================="
echo "Combined results saved to: \$COMBINED_OUTPUT"
echo "========================================="
EOF

chmod +x "$COMBINE_SCRIPT"

echo "Created combine script: $COMBINE_SCRIPT"
echo ""

# Summary
echo "========================================="
echo "Summary"
echo "========================================="
if [ "$SKIP_WITHIN" = false ]; then
    echo "Within-species jobs:"
    echo "  Human: $JOB_HUMAN"
    echo "  Mouse: $JOB_MOUSE"
fi
if [ "$SKIP_CROSS" = false ]; then
    echo "Cross-species jobs: ${#CROSS_JOB_IDS[@]} jobs"
    echo "  Job IDs: ${CROSS_JOB_IDS[@]}"
fi
echo ""
echo "To check job status:"
echo "  squeue -u \$USER"
echo ""
echo "After all jobs complete, combine cross-species results with:"
echo "  ./$COMBINE_SCRIPT"
echo ""
echo "Or manually:"
echo "  ./run_apptainer.sh python -m inflamm_debate_fm.cli.combine_bootstrap_results \\"
echo "    data/probing_results/cross_species/cross_species_results_combined.pkl \\"
echo "    data/probing_results/cross_species/cross_species_results_bs_*.pkl"
echo "========================================="

