# Clean Unit Tests Baseline

Captured on 2026-05-11 from a clean working tree before Phase 1 cleanup.

| Metric | Baseline |
|---|---:|
| Collected tests | 2,151 |
| Source-level tests | 1,083 |
| Coverage | 93.09% |
| `make test-local` wall-clock | 13:18.47 |

## Commands

```sh
uv run pytest --collect-only -q tests | tail -3
grep -rc "def test_" tests | grep -v ":0$" | awk -F: '{s+=$2} END {print s}'
uv run pytest --cov=jaxgam --cov-report=term tests/ 2>&1 | tail -20
time make test-local 2>&1 | tail -5
```

## Captured Output

Collected tests:

```text
tests/test_validation_matrix.py::TestHardGateInvariants::test_model_matrix_rank[re_mixed-nb]

2151 tests collected in 7.12s
```

Source-level tests:

```text
1083
```

Coverage:

```text
TOTAL                                   4095    283    93%
Required test coverage of 80.0% reached. Total coverage: 93.09%
========= 2071 passed, 80 skipped, 613 warnings in 1298.31s (0:21:38) ==========
```

Local test timing:

```text
========== 2071 passed, 80 skipped, 613 warnings in 796.39s (0:13:16) ==========
make test-local 2>&1  766.98s user 56.29s system 103% cpu 13:18.47 total
tail -5  0.00s user 0.01s system 0% cpu 13:18.47 total
```
