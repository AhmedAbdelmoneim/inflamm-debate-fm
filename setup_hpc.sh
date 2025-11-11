#!/bin/bash
# Setup script for Compute Canada HPC
# This script builds the Singularity container and sets up the environment

set -e

echo "========================================="
echo "inflamm-debate-fm HPC Setup"
echo "========================================="

# Check if Singularity is available
if ! command -v singularity &> /dev/null; then
    echo "Error: Singularity is not available. Please load the Singularity module:"
    echo "  module load singularity"
    exit 1
fi

echo "Singularity version: $(singularity --version)"

# Set project root
PROJ_ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
cd "$PROJ_ROOT"

# Build Singularity image
echo ""
echo "Building Singularity image..."
SINGULARITY_IMAGE="inflamm-debate-fm.sif"
if [ -f "$SINGULARITY_IMAGE" ]; then
    echo "Warning: $SINGULARITY_IMAGE already exists. Removing it..."
    rm -f "$SINGULARITY_IMAGE"
fi

singularity build "$SINGULARITY_IMAGE" Singularity.def

echo ""
echo "Singularity image built successfully: $SINGULARITY_IMAGE"

# Create data directories if they don't exist
echo ""
echo "Creating data directories..."
mkdir -p data/raw
mkdir -p data/interim
mkdir -p data/processed
mkdir -p models
mkdir -p reports/figures

# Set up environment variables
echo ""
echo "Setting up environment..."
export SINGULARITY_IMAGE="$PROJ_ROOT/$SINGULARITY_IMAGE"
export DATA_DIR="$PROJ_ROOT/data"
export MODELS_DIR="$PROJ_ROOT/models"

# Create a wrapper script for running commands
cat > run_singularity.sh << 'EOF'
#!/bin/bash
# Wrapper script to run commands in Singularity container

SINGULARITY_IMAGE="${SINGULARITY_IMAGE:-inflamm-debate-fm.sif}"
PROJ_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Bind mount data and models directories
singularity exec \
    --bind "$PROJ_ROOT/data:/opt/inflamm-debate-fm/data" \
    --bind "$PROJ_ROOT/models:/opt/inflamm-debate-fm/models" \
    --bind "$PROJ_ROOT/reports:/opt/inflamm-debate-fm/reports" \
    --bind "$PROJ_ROOT/notebooks:/opt/inflamm-debate-fm/notebooks" \
    "$SINGULARITY_IMAGE" \
    "$@"
EOF

chmod +x run_singularity.sh

echo ""
echo "========================================="
echo "Setup complete!"
echo "========================================="
echo ""
echo "To run commands in the Singularity container:"
echo "  ./run_singularity.sh python -m inflamm_debate_fm.cli <command>"
echo ""
echo "Example:"
echo "  ./run_singularity.sh python -m inflamm_debate_fm.cli preprocess all"
echo "  ./run_singularity.sh python -m inflamm_debate_fm.cli probe within-species --species human"
echo ""
echo "Or use the Makefile commands which will use the Singularity container automatically."
echo ""

