
# programs
UV ?= uv
RUFF ?= ruff

.DEFAULT_GOAL := help

.PHONY: help
help:
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

.PHONY: install
install: ## install dependencies (uv sync --extra dev)
	$(UV) sync --extra dev

.PHONY: main
main: ## run main.py
	$(UV) run python main.py

.PHONY: check
check: ## check with ruff
	$(UV) run $(RUFF) check

.PHONY: format
format: ## format python
	$(UV) run $(RUFF) format

.PHONY: lint
lint: ## run all linters (ruff check + format check)
	$(UV) run $(RUFF) check
	$(UV) run $(RUFF) format --check

.PHONY: test
test: ## run tests
	$(UV) run pytest

.PHONY: test-cov
test-cov: ## run tests with coverage (80% minimum)
	$(UV) run pytest --cov --cov-report=term-missing --cov-fail-under=80

.PHONY: generate-ref
generate-ref: ## regenerate R reference data (requires R + mgcv)
	$(UV) run python scripts/generate_reference_data.py

.PHONY: pre-commit
pre-commit: ## run pre-commit on all files
	$(UV) run pre-commit run --all-files

.PHONY: pre-commit-install
pre-commit-install: ## install pre-commit hooks
	$(UV) run pre-commit install
