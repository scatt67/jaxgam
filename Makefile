
# programs
UV ?= uv
RUFF ?= ruff
VULTURE ?= vulture
DOCKER_IMAGE ?= pymgcv-test
DOCKER_TAG ?= latest
COLIMA_CPU ?= 4
COLIMA_MEMORY ?= 16

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
lint: ## run all linters (ruff check + format check + vulture)
	$(UV) run $(RUFF) check
	$(UV) run $(RUFF) format --check
	$(UV) run $(VULTURE) pymgcv --min-confidence 80

.PHONY: test
test: docker-build ## run full test suite in Docker (includes R tests)
	docker run --rm $(DOCKER_IMAGE):$(DOCKER_TAG)

.PHONY: test-local
test-local: ## run tests locally (R tests auto-skip if R unavailable)
	$(UV) run pytest

.PHONY: test-cov
test-cov: docker-build ## run tests with coverage in Docker
	docker run --rm $(DOCKER_IMAGE):$(DOCKER_TAG) \
		uv run pytest --cov --cov-report=term-missing --cov-fail-under=80

.PHONY: colima-start
colima-start: ## start colima VM (no-op if already running)
	@colima status 2>/dev/null || colima start --cpu $(COLIMA_CPU) --memory $(COLIMA_MEMORY)

.PHONY: docker-build
docker-build: colima-start ## build the test Docker image
	docker build -t $(DOCKER_IMAGE):$(DOCKER_TAG) .

.PHONY: pre-commit
pre-commit: ## run pre-commit on all files
	$(UV) run pre-commit run --all-files

.PHONY: pre-commit-install
pre-commit-install: ## install pre-commit hooks
	$(UV) run pre-commit install
