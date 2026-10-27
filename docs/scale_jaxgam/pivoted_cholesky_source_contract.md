# Source pivoted Cholesky contract

The Gaussian working-model REML proposal (PR11) consumes the pivoted Cholesky
factors that pinned R computes inside `Sl.fitChol` and `ldetSblock`. JaxGAM
does not reimplement that factorization. It calls the LAPACK routine pinned R
uses, DPSTRF, on the host through the declared provider and passes the
factors into the JAX kernel as inputs. Ordinary dense fitting does not use
this path.

## Source and provider

The pinned oracle is R 4.5.2 with mgcv 1.9-3 linked to OpenBLAS 0.3.26. In
mgcv's `R/fast-REML.r`:

- `Sl.fitChol` factors the diagonally preconditioned penalized Hessian
  `t(XXp/d)/d`, with `d = sqrt(diag(XXp))` (and 1 where that diagonal is not
  positive);
- `ldetSblock` factors each coupled penalty block `t(S/d)/d`, with
  `S = sum_i lambda_i rS_i rS_i'` and `d = sqrt(diag(S))`.

Both call `chol(x, pivot=TRUE)`. R's `La_chol` runs upper DPSTRF with
LAPACK's default tolerance, zeroes the lower triangle, and returns the upper
storage with `pivot` and `rank` attributes. The algorithm reference is
Reference LAPACK 3.12.0
[DPSTRF](https://github.com/Reference-LAPACK/lapack/blob/v3.12.0/SRC/dpstrf.f)
and [DPSTF2](https://github.com/Reference-LAPACK/lapack/blob/v3.12.0/SRC/dpstf2.f).

`jaxgam.smooths.source_lapack.source_pivoted_cholesky` binds
`scipy_LAPACKE_dpstrf_work` from the declared `scipy-openblas32==0.3.26.0.4`
wheel, the same LP64 library as the cubic-shrinkage and pivoted-QR entry
points. The module fixes that library at one OpenBLAS thread when it loads,
so the factor is single-thread arithmetic whatever the caller's environment.
It never loads R's LAPACK, a system LAPACK or SciPy's LAPACK for this factor.

## Interface

`source_pivoted_cholesky(matrix, *, tolerance=-1.0,
available_workspace_bytes=None)` returns a frozen `SourcePivotedCholesky` with
three fields:

- `upper`: the upper triangle of DPSTRF's output with the lower triangle zeroed,
  as `La_chol` returns it.
- `pivots`: the pivot order, zero-based, as `int32`.
- `rank`: DPSTRF's completed rank.

The input must be a square `float64` NumPy array. Only its upper triangle is
read; lower values, including NaN, have no effect. The caller's array is not
modified, since the call factors a column-major copy. An empty matrix returns
empty factors and rank zero.

A negative `tolerance` (the default) selects LAPACK's own stop,
`n * DLAMCH('Epsilon') * max(diag(A))`, which is what R uses. Ties choose the
first remaining index, and LAPACK accepts the first positive pivot whatever
the tolerance. When DPSTRF stops early (`info > 0`), the call returns the
completed rank, as R's `chol(pivot=TRUE)` does with only a warning. The
trailing block of `upper` then keeps LAPACK's uncompleted residual. Only the
first `rank` rows are completed factor rows.

## Workspace

`source_pivoted_cholesky_workspace_bound(n)` bounds the bytes the call owns
beyond the caller's matrix:

- the column-major input copy that DPSTRF overwrites and that becomes `upper`;
- the one-based integer pivots;
- LAPACK's `2n` real work array;
- the returned zero-based pivots.

That is `8n^2 + 24n` bytes. With `available_workspace_bytes`, a call whose
bound exceeds the allowance raises `MemoryError` before the provider is
loaded. The working-model controller adds this bound, plus the host copy and
device copy of each factored matrix, to its compiled-workspace plan before
running a proposal.

## Fail-closed behaviour

The call rejects, before calling the provider:

- an input that is not a square `float64` NumPy array;
- a non-finite or non-real `tolerance`;
- a non-finite entry in the upper triangle;
- a dimension beyond the LP64 integer range;
- an invalid workspace allowance.

A negative DPSTRF `info` raises `RuntimeError`, and a non-finite factor raises
`FloatingPointError`. No jitter, eigenvalue clipping or alternative rank
tolerance is added.

## Use in the working model

`jaxgam.fitting.working_model.working_model_factor_inputs` assembles exactly
the preconditioned matrices pinned R factors at one trial, from the same
smoothing parameters the trial uses. `jaxgam.execution.working_model` factors
each one on the host with `source_pivoted_cholesky`. Each factor, with its
diagonal preconditioner, then enters `gaussian_working_model_proposal` as a
kernel input:

- The Hessian factor gives `Sl.fitChol`'s scaled pivoted solve and
  `log|XXp|` from the leading DPSTRF-rank rows, with static-size padding.
  `Sl.fitChol`'s additional `Rrank` truncation is not ported: the working
  model requires a positive-definite penalized system, and the controller
  treats any non-finite proposal, factor or determinant as a failure.
- Each coupled penalty factor gives `ldetSblock`'s
  `2 * sum(log(diag(R)) + log(d[piv]))`. A rank-deficient coupled block makes
  that determinant non-finite, so the proposal fails rather than reporting a
  finite score. Rank-dropping penalties remain unported.

Factors have no derivatives. They are computed outside JAX, so nothing
differentiates through DPSTRF. Every derivative of the criterion is an
analytic trace or implicit-function derivative given the factors, following
`Sl.fitChol` and `ldetSblock`.

## Owning gates

Run with `make test-pivoted-cholesky`, which runs
`tests/test_smooths/test_source_lapack.py -k pivoted_cholesky`.

- **Against SciPy's DPSTRF.** Factors at sizes 1, 3, 8, 64, 65, 96 and 129,
  across LAPACK's 64-column block boundary, must match SciPy's DPSTRF in
  storage, pivots and rank, and must reconstruct the pivoted matrix.
- **Rank stops.** Rank-stop storage under default, zero and explicit
  tolerances, for rank-deficient, indefinite, zero and identity inputs.
- **Validation.** Every rejection above, the non-finite-factor failure, and
  the workspace preflight that runs before any provider call.
- **Against pinned R.**
  - Inputs: R's `chol(pivot=TRUE)` on the five coupled
    `te(x,z,bs="cr",k=3)` penalty profiles from `bam(discrete=30,fit=FALSE)`
    `Sl.setup` roots, smoothing (0.35, 0.7), (1e-6, 1), (1e-10, 100),
    (1e-4, 1e4) and (1e4, 1e-4), plus three generic matrices above the block
    threshold. The comparison is through the typed `RBridge`.
  - Bitwise when R's LAPACK is detected (`openblas_get_config`) as the pinned
    OpenBLAS 0.3.26 build: rank, pivots and upper storage for the coupled
    profiles.
  - At STRICT for the other matrices, and for every matrix when R uses
    another LAPACK.

The working-model tests in `tests/test_fitting/test_working_model.py` and
`tests/test_execution/test_working_model.py` also feed pinned R's exact
trial state into the kernel. With the host DPSTRF factors, they check
`Sl.fitChol`'s proposal and `ldetS`'s derivatives at matched state.

Compiler and backend arithmetic remain part of numerical provenance. These
gates do not assert bit identity across platforms or LAPACK builds beyond the
pinned provider.
