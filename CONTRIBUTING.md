# Contributing to pymgcv

## Getting Started

```bash
git clone https://github.com/<org>/pymgcv.git
cd pymgcv
uv sync --extra dev
make pre-commit-install
```

## Development Workflow

### Running Tests

```bash
# Local tests (R comparison tests auto-skip without correct R version)
make test-local

# Full test suite in Docker (includes R comparison tests, requires colima/Docker)
make test

# Tests with coverage
make test-cov
```

### Linting

```bash
make lint        # ruff check + format check + vulture
make format      # auto-format with ruff
```

### Pre-commit Hooks

Pre-commit hooks run linting on every commit. Install them once:

```bash
make pre-commit-install
```

## Docker Test Environment

R comparison tests require pinned **R 4.5.2** + **mgcv 1.9-3** for reproducible results. Running locally with a different R version will cause R tests to skip with a version mismatch message.

The Docker environment provides the exact pinned versions:

```bash
# Install colima + docker CLI (macOS)
brew install colima docker

# Run full suite (starts colima automatically)
make test
```

## Architecture

Read [AGENTS.md](AGENTS.md) for the full architecture guide. Key points:

- **Phase 1 (NumPy)** -- formula parsing, basis/penalty construction. No JAX imports allowed.
- **Phase 2 (JAX)** -- PIRLS inner loop + REML/ML outer Newton loop. Must be JIT-compatible.
- **Phase 3 (NumPy)** -- predict, summary, plot on CPU.

Do not mix phases. Phase 1 modules must never import JAX.

## Testing Rules

- Every new module gets a corresponding test file.
- Use tolerance classes from `tests/tolerances.py`: `STRICT`, `MODERATE`, `LOOSE`.
- R comparison tests use `tests/r_bridge.py`. Results **must** match R at `STRICT` or `MODERATE` tolerance.
- All new modules must have > 80% test coverage.
- Hard-gate invariants (objective monotonicity, H symmetry/PSD, penalty PSD, no NaN) block the build on failure.

## PR Conventions

- One logical change per commit.
- Every PR must include tests.
- PR title format: `[phase] component: description` -- e.g., `[phase1] smooths/tprs: implement thin plate basis construction`.
- If a change touches Phase 2 code, include a JIT compilation test.

## What Is NOT in v1.0

Do not implement sparse solvers, `bam()`, extended families (NB, Tweedie, etc.), random effects, P-splines, multi-GPU, `gamm()`, or GCV/UBRE. See [AGENTS.md](AGENTS.md) for the full list.

## Getting Help

Open an issue on GitHub for bugs or feature requests.
