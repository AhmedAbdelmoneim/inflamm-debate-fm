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


## Make dataset (preprocess data)
.PHONY: data
data: requirements
	$(PYTHON_INTERPRETER) -m inflamm_debate_fm.cli preprocess all

## Generate embeddings for a dataset
.PHONY: embed
embed: requirements
	@echo "Usage: make embed DATASET=<dataset_name>"
	@if [ -z "$(DATASET)" ]; then \
		echo "Error: DATASET variable is required"; \
		echo "Example: make embed DATASET=human_burn"; \
		exit 1; \
	fi
	$(PYTHON_INTERPRETER) -m inflamm_debate_fm.cli embed generate $(DATASET)

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

## Run full pipeline (preprocess -> probe -> analyze -> plot)
.PHONY: pipeline
pipeline: requirements
	@echo "Running full pipeline..."
	$(PYTHON_INTERPRETER) -m inflamm_debate_fm.cli preprocess all
	$(PYTHON_INTERPRETER) -m inflamm_debate_fm.cli probe within-species --species human
	$(PYTHON_INTERPRETER) -m inflamm_debate_fm.cli probe within-species --species mouse
	$(PYTHON_INTERPRETER) -m inflamm_debate_fm.cli probe cross-species
	$(PYTHON_INTERPRETER) -m inflamm_debate_fm.cli analyze coefficients
	$(PYTHON_INTERPRETER) -m inflamm_debate_fm.cli analyze gsea
	$(PYTHON_INTERPRETER) -m inflamm_debate_fm.cli plot within-species --species human
	$(PYTHON_INTERPRETER) -m inflamm_debate_fm.cli plot within-species --species mouse
	$(PYTHON_INTERPRETER) -m inflamm_debate_fm.cli plot cross-species
	@echo "Pipeline complete!"


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
