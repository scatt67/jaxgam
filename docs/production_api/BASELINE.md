# Production API Redesign Baseline

Captured for Commit A on 2026-07-28 before production API implementation
changes.

| Metric | Value |
|---|---:|
| Collected test count | 1196 |
| Source-level test count | 809 |
| Coverage | 92.93% |
| `make test-cov` pytest time | 10:54.98 |
| `make test-cov` wall-clock | 11:35.84 |

## Retained Memory

The baseline uses fixed smoothing parameters so the four fits measure retained
result state without adding optimizer runtime:

- Tensor: `y ~ te(x1, x2, k=10)`, `sp=[1.0, 1.0]`
- GP: `y ~ s(x, bs='gp', k=40)`, `sp=[1.0]`
- Data: deterministic seed `20260728`, Gaussian response, `n` in `{300, 5000}`

The throwaway walker recursively traversed dataclasses, object attributes, and
containers from the `GAMResults` root. It counted every distinct
`numpy.ndarray` once by `id()` and summed `.nbytes`. The line items below are
also identity-deduplicated and partition the total; they are ownership/state
categories rather than a shape-based estimate.

### Totals

| Model | `n` | Distinct bytes | Decimal MB |
|---|---:|---:|---:|
| Tensor | 300 | 785,808 | 0.786 |
| Tensor | 5000 | 5,523,408 | 5.523 |
| GP | 300 | 1,068,840 | 1.069 |
| GP | 5000 | 36,075,240 | 36.075 |

### Tensor Line Items

| Retained state | `n=300` bytes | `n=5000` bytes |
|---|---:|---:|
| `setup.X` | 240,000 | 4,000,000 |
| `setup.y` | 2,400 | 40,000 |
| `setup.weights` | 2,400 | 40,000 |
| `setup.offset` | 0 | 0 |
| `setup.penalties` | 160,016 | 160,016 |
| Per-smooth `_X` | 48,000 | 800,000 |
| Per-smooth `_S` | 1,600 | 1,600 |
| Tensor `_penalties` | **160,000** | **160,000** |
| GP `_E_knot` | 0 | 0 |
| `training_data` | 4,800 | 80,000 |
| `fitted_values` | 2,400 | 40,000 |
| `linear_predictor` | 2,400 | 40,000 |
| Predict transforms | 80,960 | 80,960 |
| Retained diagnostics | 80,832 | 80,832 |
| **Total** | **785,808** | **5,523,408** |

Tensor predict transforms are:

| Transform | `n=300` bytes | `n=5000` bytes |
|---|---:|---:|
| `TermBlock.Z_centering` | 79,200 | 79,200 |
| `_F` | 1,600 | 1,600 |
| `_knots` | 160 | 160 |
| `_Xu`, `_knt`, `_UZ`, `_XP_list`, `_Z_list`, `_shift` | 0 | 0 |

Tensor retained diagnostics are:

| Diagnostic | `n=300` bytes | `n=5000` bytes |
|---|---:|---:|
| `Vp` | 80,000 | 80,000 |
| `coefficients` | 800 | 800 |
| `smoothing_params` | 16 | 16 |
| `edf` | 8 | 8 |
| `edf1` | 8 | 8 |

### GP Line Items

| Retained state | `n=300` bytes | `n=5000` bytes |
|---|---:|---:|
| `setup.X` | 96,000 | 1,600,000 |
| `setup.y` | 2,400 | 40,000 |
| `setup.weights` | 2,400 | 40,000 |
| `setup.offset` | 0 | 0 |
| `setup.penalties` | 12,808 | 12,808 |
| Per-smooth `_X` | 96,000 | 1,600,000 |
| Per-smooth `_S` | 12,800 | 12,800 |
| Tensor `_penalties` | 0 | 0 |
| GP `_E_knot` | **720,000** | **32,000,000** |
| `training_data` | 2,400 | 40,000 |
| `fitted_values` | 2,400 | 40,000 |
| `linear_predictor` | 2,400 | 40,000 |
| Predict transforms | 106,088 | 636,488 |
| Retained diagnostics | 13,144 | 13,144 |
| **Total** | **1,068,840** | **36,075,240** |

GP predict transforms are:

| Transform | `n=300` bytes | `n=5000` bytes |
|---|---:|---:|
| `_knt` | 2,400 | 16,000 |
| `_UZ` | 91,200 | 608,000 |
| `_shift` | 8 | 8 |
| `TermBlock.Z_centering` | 12,480 | 12,480 |
| `_Xu`, `_F`, `_XP_list`, `_Z_list` | 0 | 0 |

GP retained diagnostics are:

| Diagnostic | `n=300` bytes | `n=5000` bytes |
|---|---:|---:|
| `Vp` | 12,800 | 12,800 |
| `coefficients` | 320 | 320 |
| `smoothing_params` | 8 | 8 |
| `edf` | 8 | 8 |
| `edf1` | 8 | 8 |

The `n=5000` `_E_knot` allocation is 32.0 MB, or 88.7% of the complete
36.075 MB GP result. Commit B0 must be measured separately so this dead-store
reduction is not attributed to the later inference result mode. At `n=300`,
all 300 unique covariate rows become knots, so `_E_knot` is `300 x 300`; at
`n=5000`, knot harvesting reaches the `max_knots=2000` cap, so it is
`2000 x 2000`.

## Post-B0 GP Retained Memory

Commit B0 clears `_E_knot` immediately after its setup-only eigendecomposition.
The same recursive, identity-deduplicated walker and the same deterministic GP
models now measure:

| `n` | Pre-B0 bytes | Post-B0 bytes | `_E_knot` bytes | Reduction |
|---:|---:|---:|---:|---:|
| 300 | 1,068,840 | **348,840** (0.349 MB) | 720,000 → **0** | 720,000 (67.4%) |
| 5000 | 36,075,240 | **4,075,240** (4.075 MB) | 32,000,000 → **0** | 32,000,000 (88.7%) |

All other retained-state line items are unchanged. These post-B0 `"full"`
figures are the true baseline for measuring the later
`result="inference"` reduction.

## Commit-E Inference Retained Memory

Commit E adds the prediction-only materialization. Using the same models,
data seed, fixed smoothing parameters, and recursive identity-deduplicated
walker as the baseline gives:

| Model | `n` | Post-B0 full bytes | Inference bytes | Reduction | Reduction % |
|---|---:|---:|---:|---:|---:|
| Tensor | 300 | 785,808 | **161,792** | 624,016 | 79.4% |
| Tensor | 5000 | 5,523,408 | **161,792** | 5,361,616 | 97.1% |
| GP | 300 | 348,840 | **119,232** | 229,608 | 65.8% |
| GP | 5000 | 4,075,240 | **649,632** | 3,425,608 | 84.1% |

The deltas are fully attributable to the banned state removed with `setup`
and the in-sample caches:

| Dropped state | Tensor `n=300` | Tensor `n=5000` | GP `n=300` | GP `n=5000` |
|---|---:|---:|---:|---:|
| `setup.X` | 240,000 | 4,000,000 | 96,000 | 1,600,000 |
| `setup.y` + `setup.weights` | 4,800 | 80,000 | 4,800 | 80,000 |
| `setup.penalties` | 160,016 | 160,016 | 12,808 | 12,808 |
| Per-smooth `_X` | 48,000 | 800,000 | 96,000 | 1,600,000 |
| Per-smooth `_S` | 1,600 | 1,600 | 12,800 | 12,800 |
| Tensor `_penalties` | 160,000 | 160,000 | 0 | 0 |
| `training_data` | 4,800 | 80,000 | 2,400 | 40,000 |
| `fitted_values` | 2,400 | 40,000 | 2,400 | 40,000 |
| `linear_predictor` | 2,400 | 40,000 | 2,400 | 40,000 |
| **Total reduction** | **624,016** | **5,361,616** | **229,608** | **3,425,608** |

The lean totals are exactly the retained prediction transforms plus the core
diagnostic arrays: 80,960 + 80,832 bytes for tensor; 106,088 + 13,144 bytes
for GP at `n=300`; and 636,488 + 13,144 bytes for GP at `n=5000`. GP
`_E_knot` is already zero after Commit B0 and is deliberately not credited to
the result mode.

## Commit-F Final Measurements

The closing measurement on 2026-08-01 reran the full Docker coverage suite,
test collection, source-level test count, and the Commit-E retained-memory
walker. The memory values reproduced the Commit-E table above exactly.

| Metric | Commit A | Commit F | Delta |
|---|---:|---:|---:|
| Collected test count | 1196 | **1212** | **+16** |
| Source-level test count | 809 | **825** | **+16** |
| Coverage | 92.93% | **93.14%** | **+0.21 percentage points** |
| `make test-cov` pytest time | 10:54.98 | **10:49.66** | **-5.32 s** |
| `make test-cov` wall-clock | 11:35.84 | **10:59.25** | **-36.59 s** |

The test and timing deltas are informational; the hard gates are the complete
pass, coverage above 80%, byte-identical prediction behavior, and direct mgcv
parity. Timing varies with Docker cache and host load.

### Final retained-memory deltas

| Model | `n` | Honest full baseline | Final inference | Reduction | Reduction % |
|---|---:|---:|---:|---:|---:|
| Tensor | 300 | 785,808 | **161,792** | 624,016 | 79.4% |
| Tensor | 5000 | 5,523,408 | **161,792** | 5,361,616 | 97.1% |
| GP | 300 | 348,840 | **119,232** | 229,608 | 65.8% |
| GP | 5000 | 4,075,240 | **649,632** | 3,425,608 | 84.1% |

The GP comparison uses the post-B0 full result as its baseline. The separate
B0 dead-store reductions remain 720,000 bytes (`n=300`) and 32,000,000 bytes
(`n=5000`) and are not attributed to inference mode.

## Commands

```sh
uv run pytest --collect-only -q tests
grep -rc "def test_" tests | grep -v ":0$" | awk -F: '{s+=$2} END {print s}'
make install
/usr/bin/time -p make test-cov
uv run python /tmp/jaxgam_commit_a_memory.py
uv run python /tmp/jaxgam_commit_b0_memory.py
uv run python /tmp/jaxgam_commit_e_memory.py
uv run python /tmp/jaxgam_commit_e_memory.py  # Commit F rerun; walker unchanged
```

The memory script was temporary and is not part of the repository.

## Validation

`make test-cov` passed:

```text
Required test coverage of 80% reached. Total coverage: 92.93%
1196 passed, 295 warnings in 654.98s (0:10:54)
real 695.84
```

`make install` resolved and installed `cloudpickle==3.1.2` from the
`[project.optional-dependencies].dev` extra.

The final Commit-F run also passed:

```text
Required test coverage of 80% reached. Total coverage: 93.14%
1212 passed, 311 warnings in 649.66s (0:10:49)
real 659.25
```
