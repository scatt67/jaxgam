# PR12 TPRS source parity: findings recorded for later work (2026-10-24)

The PR12 fourth TPRS correction pass made the thin-plate construction
source-exact (shared dense/operator `tprs_setup` port, exact `Rlanczos`
against the pinned OpenBLAS 0.3.26, smoothCon's `ti` margin stage). Two
findings from that work are outside its scope. They are recorded here, not
fixed.

## 1. Dense public fixed-sp Gaussian fit is not STRICT against R

With a raw basis that matches pinned `smoothCon`, the dense public route
`GAM(formula, sp=...).fit(data)` does not reproduce R's fixed-sp fitted values
at STRICT. The bounded operator fixed-sp fitter does reproduce them on the
same models.

Exact case (regular layout of `tests/test_smooths/test_tprs_source_parity.py`):

- data: `x = np.linspace(-1.3, 1.7, 61)`,
  `z = np.sin(np.linspace(-0.8, 2.2, 61)) + np.linspace(0, 0.1, 61)`,
  `y = np.cos(1.7 * x) + 0.2 * z`;
- model: `y ~ s(x, bs='tp', k=4)`, Gaussian identity, `sp = [0.7]`;
- reference: `RBridge.fit_fixed_sp("y~s(x,bs='tp',k=4)", data, "gaussian",
  [0.7])` (`gam(..., sp=0.7, method="REML",
  control=gam.control(epsilon=1e-10, maxit=200))`).

| Quantity | Dense public fit | Operator fixed-sp fit |
| --- | --- | --- |
| `fitted_values` max abs | 1.37e-11 (5 of 61 beyond STRICT, 8.1x the bound) | 2.2e-15 (macOS), 4.4e-16 (Linux) |
| `deviance` abs | 2.75e-10 | 3.0e-14 |
| raw basis vs `smoothCon` | 3.6e-15 (macOS), 5.6e-16 (Linux) | same basis |

The values are identical at HEAD `5b69880` (before the PR12 pass), so they do
not come from the basis. The gap grows with the conditioning of the model
matrix. For `ti(x,z,bs='tp',k=8)` on the irregular layout, the dense fitted
values differ by 7.95e-8 (all 61 beyond STRICT), although on Linux that basis
is bitwise equal to R's and cond(X) is about 3e8. That pattern is consistent
with a normal-equations solve, whose error scales with cond(X)^2, rather than
R's QR-based fit. The operator fitter matches R to 7.6e-12 on that case.

## 2. Zero-copy rpy2 views in `tests/r_bridge.py`

rpy2 exposes R numeric vectors through buffers that NumPy wraps without
copying. `np.asarray(r_object)` (also with `dtype=np.float64` on an R double
vector) therefore returns a view of R-managed memory. R's garbage collector
may reuse that memory after any later R allocation once the R object is no
longer referenced. Results then silently change after the bridge method has
returned: during the PR12 audit a stored `smoothCon` `X` read 2.8e+281 after
later calls.

- PR12's first fix (`3622a84`) copied only the `smoothCon` extractor
  fields (`np.array`). It is superseded by the bridge-wide copy fix
  (`8dbcb9e`, `[tests] r_bridge: copy every R result into NumPy-owned
  memory`): one `_r_array` helper copies every R-to-NumPy conversion in the
  bridge, and regression tests check ownership and values after forced R
  `gc()` and fresh allocations. That fix is stacked below PR9, so the
  composed PR12 line drops `3622a84`; the bridge methods PR12 adds return
  NumPy-owned copies through the same `_r_array`.
