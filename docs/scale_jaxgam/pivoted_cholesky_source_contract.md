# Source pivoted Cholesky contract

This PR11 primitive factors coefficient-sized matrices. It does not change
ordinary dense fitting or implement the working-model controller. The
controller owns source Sl metadata, reparameterization, analytic traces and
implicit coefficient derivatives.

The primary algorithm references are Reference LAPACK 3.12.0
[DPSTRF](https://github.com/Reference-LAPACK/lapack/blob/v3.12.0/SRC/dpstrf.f)
and [DPSTF2](https://github.com/Reference-LAPACK/lapack/blob/v3.12.0/SRC/dpstf2.f).
R's `chol(pivot=TRUE)` calls upper DPSTRF, zeros the lower triangle, and
returns its upper storage with pivot/rank attributes. The pinned oracle is
R 4.5.2 with mgcv 1.9-3. Relevant mgcv source is `R/fast-REML.r`'s
`ldetSblock` and `Sl.fitChol`, `R/misc.r`'s `mchol`, and `R/mgcv.r`'s `mroot`.

`pivoted_cholesky(matrix, tolerance=-1)` returns the immutable JAX tuple
`(upper, pivots, rank)`. Input is square binary64; pivots are zero-based.
Only the input upper triangle is referenced. Lower values, including NaNs,
have no effect. Empty input returns empty factors and rank zero.

The default threshold is `p * DLAMCH('Epsilon') * max(diag(A))`. For binary64,
the LAPACK epsilon here is unit roundoff, `numpy.finfo(float64).eps / 2`.
Using the spacing at one without that division changes source rank decisions.
Tolerance zero supports mroot's requested-rank policy. LAPACK accepts the
first positive pivot regardless of the requested stopping tolerance; the
threshold applies to subsequent pivots. Ties choose the first remaining
index. Rank termination leaves the original source residual upper storage;
only the first `rank` rows are completed factor rows. Callers must truncate
or pad using the declared rank rather than treat the whole returned matrix
as a completed root.

The implementation preserves the source's upper swaps, sequential diagonal
products, ordered transpose DGEMV dot sums and reciprocal DSCAL. It uses the
pinned LAPACK's 64-column block policy; larger matrices receive the ordered
DSYRK trailing update. Its arrays and compiled loop scratch are O(p²), with
no rank-by-p-by-p tensor. Compiled-memory owning gates exercise two sizes
above the block threshold; these measurements concern compiler buffers,
not total process peak RSS.

Nonfinite upper input, nonfinite tolerance or nonfinite factor arithmetic
returns rank -1 and finite zero upper storage. A caller must reject it.
The trailing DSYRK is checked before a subsequent finite rank stop can hide
an overflow in the uncompleted upper residual. The owning overflow fixture
has finite input and finite first-block factor rows, then a nonfinite trailing
off-diagonal and a zero-scale residual pivot.
Finite rank-deficient/indefinite input returns the source completed rank and
residual storage. Full rank is not an independent certification that arbitrary
input was positive semidefinite. No jitter, eigenvalue clipping or alternate
rank tolerance is introduced.

Factor differentiation explicitly raises `NotImplementedError`, including
under JIT. Controllers must use analytic trace/implicit derivatives; a stopped
factor gradient must not stand in for the objective derivative.

The preserved coupled input uses 97 deterministic observations and
`te(x,z,bs='cr',k=3)` with `bam(discrete=30,fit=FALSE)` setup. The owning
oracle rebuilds Sl roots and supplies the exact R preconditioned matrix
`t(St/d)/d` through binary arrays. It checks actual R factors, pivots and
ranks at all five profiles: (.35,.7), (1e-6,1), (1e-10,100), (1e-4,1e4),
(1e4,1e-4). The most conditioned matrix is about 1.45e12.

A separately rounded `WORK += A**2` gave three failing factor cells at the
(1e-10,100) profile: maximum absolute error 2.77526e-11 and relative error
1.38747e-5. On the tested pinned ARM CPU image, a serial source reproduction
using IEEE fused multiply-add for that diagonal recurrence matches the
entire R factor bit for bit; separate rounding does not. The kernel therefore
leaves that source contraction available while materializing the surrounding
vector-operation boundaries. All five live R factor gates pass STRICT after
this correction. Compiler/backend arithmetic remains part of numerical
provenance; these observations do not assert universal bit identity across
platforms. Live oracle gates flag a platform mismatch rather than replace
it with machine-specific constants.

Run the owning gates with `make test-pivoted-cholesky`. They also compare
SciPy DPSTRF at sizes 1/3/8/64/65/96/129, pinned R above the block threshold,
rank70 termination in a 96-dimensional matrix, rank/tie/tolerance boundaries,
finite failure, factor-AD rejection and compiled workspace.
