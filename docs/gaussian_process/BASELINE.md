# Gaussian Process Implementation Baseline

Captured for Commit A on 2026-05-23 before GP implementation changes.

| Metric | Value |
|---|---:|
| Collected test count | 1043 |
| Source-level test count | 728 |
| Coverage | 92.28% |
| `make test-cov` wall-clock | 8:07.63 |
| TPRS collected test count | 53 |

## Commands

```sh
uv run pytest --collect-only -q tests | tail -3
grep -rc "def test_" tests | grep -v ":0$" | awk -F: '{s+=$2} END {print s}'
time make test-cov 2>&1 | tail -25
uv run pytest --collect-only -q tests/test_smooths/test_tprs.py | tail -1
```

## Validation

`make test-cov` passed:

```text
1043 passed, 235 warnings in 469.65s (0:07:49)
Required test coverage of 80% reached. Total coverage: 92.28%
make test-cov 2>&1  0.23s user 0.21s system 0% cpu 8:07.63 total
```
