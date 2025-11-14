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
	uv sync

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
	uv venv --python $(PYTHON_VERSION)
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
	@echo "Usage: make embed DATASET=<dataset_name> [DEVICE=<cpu|cuda>] [BATCH_SIZE=<256>]"
	@if [ -z "$(DATASET)" ]; then \
		echo "Error: DATASET variable is required"; \
		echo "Example: make embed DATASET=human_burn"; \
		exit 1; \
	fi
	$(PYTHON_INTERPRETER) -m inflamm_debate_fm.cli embed generate $(DATASET) \
		--device $(or $(DEVICE),cpu) \
		--batch-size $(or $(BATCH_SIZE),256)

## Generate embeddings for all configurations (human-only, mouse-only, human-ortholog-filtered)
.PHONY: embed-all
embed-all: requirements
	@echo "Usage: make embed-all [DEVICE=<cpu|cuda>] [BATCH_SIZE=<256>] [USE_WANDB=<true|false>]"
	@if [ "$(USE_WANDB)" = "true" ]; then \
		$(PYTHON_INTERPRETER) -m inflamm_debate_fm.cli embed all-configs \
			--device $(or $(DEVICE),cpu) \
			--batch-size $(or $(BATCH_SIZE),256) \
			--use-wandb; \
	else \
		$(PYTHON_INTERPRETER) -m inflamm_debate_fm.cli embed all-configs \
			--device $(or $(DEVICE),cpu) \
			--batch-size $(or $(BATCH_SIZE),256); \
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
