#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = inflamm-debate-fm
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install Python dependencies
.PHONY: requirements
requirements:
	@# Install PyTorch 2.8.* with CUDA 12.9 first (before uv sync to ensure CUDA version)
	@echo "Installing PyTorch 2.8.* with CUDA 12.9..."
	@uv pip install --index-url https://download.pytorch.org/whl/cu129 \
		"torch>=2.8.0,<2.9.0" || echo "Warning: PyTorch installation failed."
	@# Sync other dependencies (torch should already be installed, so this won't reinstall it)
	uv sync
	@# Ensure torch is still the CUDA version (in case uv sync changed it)
	@uv pip install --index-url https://download.pytorch.org/whl/cu129 \
		--upgrade-package torch "torch>=2.8.0,<2.9.0" || true
	@# Install PyG optional dependencies from wheel find-links (requires torch to be installed first)
	@echo "Installing PyG optional dependencies from wheel repository..."
	@uv pip install --find-links https://data.pyg.org/whl/torch-2.8.0+cu129.html \
		pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv || \
		echo "Warning: Some PyG optional dependencies failed to install."

## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format

## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	uv venv --python 3.12.7 --clear
	@echo ">>> New uv virtual environment created. Activate with:"
	@echo ">>> Windows: .\\\\.venv\\\\Scripts\\\\activate"
	@echo ">>> Unix/macOS: source ./.venv/bin/activate"

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Process data (download, preprocess, map orthologs, combine)
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) -m inflamm_debate_fm.cli preprocess data

## Set up BulkFormer model (clone repo and check for model files)
.PHONY: bulkformer-setup
bulkformer-setup: requirements
	$(PYTHON_INTERPRETER) -m inflamm_debate_fm.bulkformer.setup

## Generate embeddings for a dataset
.PHONY: embed
embed: requirements
	@echo "Usage: make embed DATASET=<dataset_name> [DEVICE=<cpu|cuda>] [BATCH_SIZE=<16>]"
	@echo "Note: For GPU (especially MIG instances), use smaller BATCH_SIZE (e.g., 8-16) to avoid OOM errors"
	@if [ -z "$(DATASET)" ]; then \
		echo "Error: DATASET variable is required"; \
		echo "Example: make embed DATASET=human_burn DEVICE=cuda BATCH_SIZE=16"; \
		exit 1; \
	fi
	@# Set PyTorch CUDA allocator config to reduce fragmentation
	@if [ "$(or $(DEVICE),cpu)" = "cuda" ]; then \
		PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(PYTHON_INTERPRETER) -m inflamm_debate_fm.cli embed generate $(DATASET) \
			--device $(or $(DEVICE),cpu) \
			--batch-size $(or $(BATCH_SIZE),16); \
	else \
		$(PYTHON_INTERPRETER) -m inflamm_debate_fm.cli embed generate $(DATASET) \
			--device $(or $(DEVICE),cpu) \
			--batch-size $(or $(BATCH_SIZE),256); \
	fi

## Generate embeddings for all configurations (human-only, mouse-only, human-ortholog-filtered)
.PHONY: embed-all
embed-all: requirements
	@echo "Usage: make embed-all [DEVICE=<cpu|cuda>] [BATCH_SIZE=<16>] [USE_WANDB=<true|false>]"
	@echo "Note: For GPU (especially MIG instances), use smaller BATCH_SIZE (e.g., 8-16) to avoid OOM errors"
	@# Set PyTorch CUDA allocator config to reduce fragmentation
	@if [ "$(USE_WANDB)" = "true" ]; then \
		if [ "$(or $(DEVICE),cpu)" = "cuda" ]; then \
			PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(PYTHON_INTERPRETER) -m inflamm_debate_fm.cli embed all-configs \
				--device cuda \
				--batch-size $(or $(BATCH_SIZE),16) \
				--use-wandb; \
		else \
			$(PYTHON_INTERPRETER) -m inflamm_debate_fm.cli embed all-configs \
				--device cpu \
				--batch-size $(or $(BATCH_SIZE),256) \
				--use-wandb; \
		fi; \
	else \
		if [ "$(or $(DEVICE),cpu)" = "cuda" ]; then \
			PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $(PYTHON_INTERPRETER) -m inflamm_debate_fm.cli embed all-configs \
				--device cuda \
				--batch-size $(or $(BATCH_SIZE),16); \
		else \
			$(PYTHON_INTERPRETER) -m inflamm_debate_fm.cli embed all-configs \
				--device cpu \
				--batch-size $(or $(BATCH_SIZE),256); \
		fi; \
	fi

## Run within-species probing experiments
.PHONY: probe-within
probe-within: requirements
	@echo "Usage: make probe-within SPECIES=<human|mouse>"
	@if [ -z "$(SPECIES)" ]; then \
		echo "Error: SPECIES variable is required (human or mouse)"; \
		echo "Example: make probe-within SPECIES=human"; \
		exit 1; \
	fi
	$(PYTHON_INTERPRETER) -m inflamm_debate_fm.cli probe within-species --species $(SPECIES)

## Run cross-species probing experiments
.PHONY: probe-cross
probe-cross: requirements
	$(PYTHON_INTERPRETER) -m inflamm_debate_fm.cli probe cross-species

## Fine-tune model with LoRA
.PHONY: finetune
finetune: requirements
	@echo "Usage: make finetune SPECIES=<human|mouse|combined> [EPOCHS=<10>] [BATCH_SIZE=<8>] [USE_WANDB=<true|false>]"
	@if [ -z "$(SPECIES)" ]; then \
		echo "Error: SPECIES variable is required"; \
		echo "Example: make finetune SPECIES=human USE_WANDB=true"; \
		exit 1; \
	fi
	@if [ "$(USE_WANDB)" = "true" ]; then \
		$(PYTHON_INTERPRETER) -m inflamm_debate_fm.cli finetune train \
			--species $(SPECIES) \
			--epochs $(or $(EPOCHS),10) \
			--batch-size $(or $(BATCH_SIZE),8) \
			--use-wandb; \
	else \
		$(PYTHON_INTERPRETER) -m inflamm_debate_fm.cli finetune train \
			--species $(SPECIES) \
			--epochs $(or $(EPOCHS),10) \
			--batch-size $(or $(BATCH_SIZE),8); \
	fi

## Analyze coefficients
.PHONY: analyze-coeffs
analyze-coeffs: requirements
	$(PYTHON_INTERPRETER) -m inflamm_debate_fm.cli analyze coefficients

## Run GSEA analysis
.PHONY: analyze-gsea
analyze-gsea: requirements
	$(PYTHON_INTERPRETER) -m inflamm_debate_fm.cli analyze gsea

## Generate plots for within-species results
.PHONY: plot-within
plot-within: requirements
	@echo "Usage: make plot-within SPECIES=<human|mouse>"
	@if [ -z "$(SPECIES)" ]; then \
		echo "Error: SPECIES variable is required (human or mouse)"; \
		echo "Example: make plot-within SPECIES=human"; \
		exit 1; \
	fi
	$(PYTHON_INTERPRETER) -m inflamm_debate_fm.cli plot within-species --species $(SPECIES)

## Generate plots for cross-species results
.PHONY: plot-cross
plot-cross: requirements
	$(PYTHON_INTERPRETER) -m inflamm_debate_fm.cli plot cross-species

## Run tests
.PHONY: test
test: requirements
	$(PYTHON_INTERPRETER) -m pytest tests/ -v

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
