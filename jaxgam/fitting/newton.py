"""Newton optimizer for smoothing parameter selection.

Minimizes the REML/ML criterion over ``log_lambda`` to find optimal
smoothing parameters. This is the outer loop in the GAM fitting
pipeline: Newton over ``log_lambda`` (outer) wrapping PIRLS over
``beta`` (inner). At each Newton step, PIRLS must re-converge at
the proposed ``log_lambda``.

The algorithm follows R's ``newton()`` (gam.fit3.r lines
1290-1719): eigenvalue-safe Newton direction with step-halving
line search and steepest-descent fallback for indefinite Hessians.

Design doc reference: Section 8.2 (Outer Newton with Damped Hessian)
R source reference: gam.fit3.r lines 1290-1719

Notes
-----
Key R-matching design choices (see docs/experiments.md):

- ``conv_tol = 1e-6`` (R's ``gam.control()$newton$conv.tol``)
- Gradient convergence uses ``5 * conv_tol`` factor (R line 1652)
- PIRLS tolerance tightened to ``conv_tol / 100`` (R line 1308)
- ``score_scale = abs(log(scale)) + abs(score)`` for REML (R line 1648)
- ``uconv.ind`` dimension subsetting with ``|grad| > max(|grad|)*0.001``
  filter (R lines 1430-1432, 1435-1436, 1651)
- Eigenvalue floor ``eps^0.7`` (R line 1450)
- ``lsp_max = 40`` safety net for flat REML surfaces
- ``custom_jvp`` on PIRLS for all families with penalties (Exp 18):
  defines how ``(β*, XtWX, deviance)`` change with ``S_lambda``
  via the implicit function theorem. ``jax.grad`` and
  ``jax.hessian`` of the end-to-end score function automatically
  capture all first- and second-order terms, matching R's
  analytical Hessian from ``gdi.c`` / ``Sl.ift``. Cost: 1 PIRLS
  per Newton iteration (same as R).
- Gaussian uses ``fast.REML.fit``-style step handling (fast-REML.r
  lines 1822-1844): no qerror check, eigenvalue floor ``eps^0.5``,
  no SD fallback, step failure breaks immediately.
- Non-Gaussian uses ``newton()``-style step handling (gam.fit3.r
  lines 1491-1571): qerror check, eigenvalue floor ``eps^0.7``,
  SD fallback after 3 halvings.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from enum import Enum, auto

import jax
import jax.numpy as jnp
import numpy as np
from jax.scipy.linalg import cho_solve

from jaxgam.families.base import ExponentialFamily
from jaxgam.fitting.data import FittingData
from jaxgam.fitting.initialization import initialize_beta
from jaxgam.fitting.pirls import PIRLSResult, pirls_loop
from jaxgam.fitting.reml import (
    JointMLCriterion,
    JointREMLCriterion,
    MLCriterion,
    REMLCriterion,
    _criterion_core,
    _CriterionBase,
    _JointCriterionBase,
    estimate_edf,
    fletcher_scale,
)
from jaxgam.jax_utils import build_S_lambda, cho_factor

# R's default Newton convergence tolerance (gam.control()$newton$conv.tol).
# This is ~67x looser than sqrt(eps) ≈ 1.5e-8, matching R's deliberate
# choice for the Newton optimizer that wraps PIRLS (gam.fit3.r line 1307).
_DEFAULT_CONV_TOL = 1e-6

# R's maxSstep default for steepest-descent step length (gam.fit3.r).
_MAX_SD_STEP = 2.0

# Maximum step-halving iterations for Gaussian (fast-REML.r).
_MAX_HALVINGS_GAUSSIAN = 25

# Maximum step-halving iterations for non-Gaussian (gam.fit3.r).
# Non-Gaussian uses more halvings because of qerror checking.
_MAX_HALVINGS_NON_GAUSSIAN = 30

# Tolerance for detecting active bounds on log smoothing parameters.
_BOUND_TOL = 1e-10

# Gradient filter threshold for uconv.ind (R lines 1430-1431).
_GRAD_FILTER_THRESH = 0.001


# ---------------------------------------------------------------------------
# Module-level differentiable score (JIT-cached across fits)
# ---------------------------------------------------------------------------
#
# ``_diff_score`` is a module-level function with explicit arguments so that
# ``jax.jit(jax.grad(...))`` and ``jax.jit(jax.hessian(...))`` are defined
# once at module load time. The JIT cache is keyed by static args (family,
# tol, method, structure), so after the first fit per combination, subsequent
# fits reuse the compiled XLA code — no per-fit recompilation of
# ``jax.hessian(jacfwd(jacrev(...)))`` which costs ~300ms.
#
# The ``custom_jvp`` on PIRLS is defined as a closure INSIDE ``_diff_score``
# to keep only 2 primals (S_lambda, beta_warm). This is critical: extra
# primals would make ``jax.hessian`` compute unnecessary JVP/VJP passes,
# significantly increasing compile and run time.


def _diff_score(
    # Dynamic args (JAX arrays, differentiated through)
    params: jax.Array,
    beta_warm: jax.Array,
    X: jax.Array,
    y: jax.Array,
    wt: jax.Array,
    offset: jax.Array,
    S_list: tuple[jax.Array, ...],
    singleton_eig_constants: jax.Array,
    multi_block_proj_S: tuple[tuple[jax.Array, ...], ...],
    # Static args (JIT cache keys, not traced)
    family: ExponentialFamily,
    pirls_tol: float,
    is_reml: bool,
    joint_scale: bool,
    n_lambda: int,
    Mp: int,
    singleton_sp_indices: tuple[int, ...],
    singleton_ranks: tuple[int, ...],
    multi_block_sp_indices: tuple[tuple[int, ...], ...],
    multi_block_ranks: tuple[int, ...],
    p: int,
) -> jax.Array:
    """End-to-end differentiable score: PIRLS + criterion.

    All data flows as explicit arguments (no per-fit closures). Static
    arguments (family, pirls_tol, is_reml, etc.) are compile-time constants
    that key the JIT cache. The ``custom_jvp`` on PIRLS is defined inside
    this function as a closure over X, y, wt, offset, family, pirls_tol
    (all concrete or traced at trace time), keeping only S_lambda and
    beta_warm as differentiable primals.

    Parameters
    ----------
    params : jax.Array, shape (m,) or (m+1,)
        Log smoothing parameters, or ``[log_lambda, log_phi]`` for joint.
    beta_warm : jax.Array, shape (p,)
        Warm-start coefficients from previous PIRLS.
    X, y, wt, offset : jax.Array
        Model data on device.
    S_list : tuple[jax.Array, ...]
        Per-penalty matrices.
    singleton_eig_constants, multi_block_proj_S
        Block-structured log|S+| data (dynamic JAX arrays).
    family : ExponentialFamily
        Family (static, JIT cache key).
    pirls_tol : float
        PIRLS convergence tolerance (static).
    is_reml : bool
        Whether to use REML vs ML criterion (static).
    joint_scale : bool
        Whether params includes log_phi (static).
    n_lambda : int
        Number of smoothing parameters (static).
    Mp : int
        Total penalty null space dimension (static).
    singleton_sp_indices, singleton_ranks, multi_block_sp_indices,
    multi_block_ranks : tuple[int, ...]
        Block metadata (static, JIT cache keys).
    p : int
        Number of coefficients (static).

    Returns
    -------
    jax.Array, scalar
        REML or ML criterion score.
    """

    # ---- custom_jvp on PIRLS (closure over traced X, y, wt, offset) ----
    @jax.custom_jvp
    def _pirls_out(S_lambda, beta_warm_inner):
        result = pirls_loop(
            X,
            y,
            beta_warm_inner,
            S_lambda,
            family,
            wt,
            offset,
            tol=pirls_tol,
        )
        return result.coefficients, result.XtWX, result.deviance

    @_pirls_out.defjvp
    def _pirls_jvp(primals, tangents):
        S_lambda, beta_warm_inner = primals
        dS, _ = tangents

        beta, XtWX, dev = _pirls_out(S_lambda, beta_warm_inner)

        # IFT: dβ = -H⁻¹(dS @ β)
        H = XtWX + S_lambda
        L, _ = cho_factor(H)
        dbeta = cho_solve((L, True), -(dS @ beta))

        # Chain: dη → dW → dXtWX
        eta = X @ beta + offset
        deta = X @ dbeta

        def _eta_to_W(e):
            return family.working_weights(family.link.inverse(e), wt)

        _, dW = jax.jvp(_eta_to_W, (eta,), (deta,))
        dXtWX = (X.T * dW) @ X

        # Chain: dη → dμ → ddeviance
        def _eta_to_dev(e):
            return jnp.sum(family.dev_resids(y, family.link.inverse(e), wt))

        _, ddev = jax.jvp(_eta_to_dev, (eta,), (deta,))

        return (beta, XtWX, dev), (dbeta, dXtWX, ddev)

    # ---- End-to-end score ----
    if joint_scale:
        log_lambda = params[:n_lambda]
        phi = jnp.exp(params[n_lambda])
        ls_sat = family.saturated_loglik(y, wt, phi)
    else:
        log_lambda = params
        phi = jnp.array(1.0)
        ls_sat = family.saturated_loglik(y, wt, phi)

    S_lambda = build_S_lambda(log_lambda, S_list, p)
    beta, XtWX, dev = _pirls_out(S_lambda, beta_warm)

    core = _criterion_core(
        log_lambda,
        XtWX,
        beta,
        dev,
        ls_sat,
        S_list,
        phi,
        singleton_sp_indices,
        singleton_ranks,
        singleton_eig_constants,
        multi_block_sp_indices,
        multi_block_ranks,
        multi_block_proj_S,
    )

    if is_reml:
        return core - Mp / 2.0 * jnp.log(2.0 * jnp.pi * phi)
    return core


# Static argument names for JIT caching of _diff_score derivatives.
# These are Python values (not JAX arrays) that key the compilation cache.
_DIFF_STATIC = (
    "family",
    "pirls_tol",
    "is_reml",
    "joint_scale",
    "n_lambda",
    "Mp",
    "singleton_sp_indices",
    "singleton_ranks",
    "multi_block_sp_indices",
    "multi_block_ranks",
    "p",
)

# Module-level JIT'd gradient and Hessian — compiled ONCE per
# (family, tol, method, structure) combination, reused across all
# subsequent fits. This is the key performance optimization: the
# jax.hessian (jacfwd(jacrev)) compilation that costs ~300ms happens
# only on the first fit, not on every fit.
_jit_diff_grad = jax.jit(
    jax.grad(_diff_score, argnums=0),
    static_argnames=_DIFF_STATIC,
)
_jit_diff_hess = jax.jit(
    jax.hessian(_diff_score, argnums=0),
    static_argnames=_DIFF_STATIC,
)


# Fused gradient + Hessian: single XLA program, halves Python↔XLA syncs.
def _diff_grad_hess(
    *args: jax.Array, **kwargs: jax.Array
) -> tuple[jax.Array, jax.Array]:
    """Fused gradient + Hessian of ``_diff_score`` in a single XLA trace."""
    _grad = jax.grad(_diff_score, argnums=0)
    _hess = jax.hessian(_diff_score, argnums=0)
    return _grad(*args, **kwargs), _hess(*args, **kwargs)


_jit_diff_grad_hess = jax.jit(_diff_grad_hess, static_argnames=_DIFF_STATIC)


# ---------------------------------------------------------------------------
# Fused forward pass: PIRLS + criterion score in one XLA dispatch
# ---------------------------------------------------------------------------
#
# ``_fit_and_score_impl`` fuses S_lambda construction, PIRLS (while_loop),
# and criterion score evaluation into a single JIT program. This reduces
# 4 JIT dispatches (S_lambda + PIRLS + edf/scale + score) to 1, saving
# ~300μs per call. During step-halving (1-30 trials per Newton iteration),
# these savings compound significantly.
#
# PIRLSResult is a registered JAX pytree, so it can be returned from JIT.
# The criterion object is NOT created here — only after step acceptance
# (when gradient/Hessian are needed).


def _fit_and_score_impl(
    # Dynamic args (JAX arrays)
    params: jax.Array,
    beta_init: jax.Array,
    X: jax.Array,
    y: jax.Array,
    wt: jax.Array,
    offset: jax.Array,
    S_list: tuple[jax.Array, ...],
    singleton_eig_constants: jax.Array,
    multi_block_proj_S: tuple[tuple[jax.Array, ...], ...],
    # Static args (JIT cache keys)
    family: ExponentialFamily,
    pirls_tol: float,
    is_reml: bool,
    joint_scale: bool,
    n_lambda: int,
    Mp: int,
    singleton_sp_indices: tuple[int, ...],
    singleton_ranks: tuple[int, ...],
    multi_block_sp_indices: tuple[tuple[int, ...], ...],
    multi_block_ranks: tuple[int, ...],
    p: int,
) -> tuple[jax.Array, PIRLSResult]:
    """Fused PIRLS + criterion score in one XLA program.

    Same signature as ``_diff_score`` but forward-pass only (no
    ``custom_jvp``). Returns ``(score, pirls_result)`` where
    ``pirls_result`` is a registered JAX pytree.

    Returns
    -------
    score : jax.Array, scalar
        REML or ML criterion score.
    pirls_result : PIRLSResult
        Converged PIRLS output.
    """
    if joint_scale:
        log_lambda = params[:n_lambda]
        phi = jnp.exp(params[n_lambda])
        ls_sat = family.saturated_loglik(y, wt, phi)
    else:
        log_lambda = params
        phi = jnp.array(1.0)
        ls_sat = family.saturated_loglik(y, wt, phi)

    S_lambda = build_S_lambda(log_lambda, S_list, p)
    pirls_result = pirls_loop(
        X,
        y,
        beta_init,
        S_lambda,
        family,
        wt,
        offset,
        tol=pirls_tol,
    )

    core = _criterion_core(
        log_lambda,
        pirls_result.XtWX,
        pirls_result.coefficients,
        pirls_result.deviance,
        ls_sat,
        S_list,
        phi,
        singleton_sp_indices,
        singleton_ranks,
        singleton_eig_constants,
        multi_block_sp_indices,
        multi_block_ranks,
        multi_block_proj_S,
    )

    score = core - Mp / 2.0 * jnp.log(2.0 * jnp.pi * phi) if is_reml else core

    return score, pirls_result


_jit_fit_and_score = jax.jit(
    _fit_and_score_impl,
    static_argnames=_DIFF_STATIC,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NewtonResult:
    """Result of Newton smoothing parameter optimization.

    Not a JAX pytree -- this is a Python-level post-estimation container.
    Carries everything ``gam()`` (Task 2.6) needs.

    Attributes
    ----------
    log_lambda : jax.Array, shape (m,)
        Optimized log smoothing parameters.
    smoothing_params : jax.Array, shape (m,)
        ``exp(log_lambda)`` -- smoothing parameters on original scale.
    converged : bool
        Whether convergence criteria were met.
    n_iter : int
        Number of Newton iterations taken.
    score : jax.Array, scalar
        Final REML/ML criterion value.
    gradient : jax.Array, shape (m,)
        Final gradient of criterion w.r.t. log_lambda.
    edf : jax.Array, scalar
        Final effective degrees of freedom.
    scale : jax.Array, scalar
        Final estimated dispersion parameter.
    pirls_result : PIRLSResult
        Final PIRLS output (coefficients, mu, etc.).
    convergence_info : str
        ``"full convergence"`` | ``"step failed"`` | ``"iteration limit"``
    """

    log_lambda: jax.Array
    smoothing_params: jax.Array
    converged: bool
    n_iter: int
    score: jax.Array
    gradient: jax.Array
    edf: jax.Array
    scale: jax.Array
    pirls_result: PIRLSResult
    convergence_info: str


# ---------------------------------------------------------------------------
# Safe Newton step (pure function, no state)
# ---------------------------------------------------------------------------


@jax.jit(static_argnames=("max_step", "eig_floor_power"))
def _safe_newton_step(
    gradient: jax.Array,
    hessian: jax.Array,
    max_step: float = 5.0,
    eig_floor_power: float = 0.7,
) -> tuple[jax.Array, jax.Array]:
    """Newton step with eigenvalue safety, component-wise capping, and
    indefiniteness detection.

    Follows R's eigenvalue-safe Newton step:

    1. Eigendecompose Hessian: ``H = V D V^T``
    2. Flip negative eigenvalues to positive: ``D = |D|``
    3. Floor small eigenvalues: ``D = max(D, max(|D|) * eps^power)``
    4. Compute step: ``step = -V @ ((V^T @ g) / D_safe)``
    5. Cap step: ``max(|step|) <= max_step`` (component-wise)

    The eigenvalue floor power differs by code path:
    - ``newton()`` (non-Gaussian): ``eps^0.7`` (gam.fit3.r line 1450)
    - ``fast.REML.fit`` (Gaussian): ``eps^0.5`` (fast-REML.r line 1810)

    Parameters
    ----------
    gradient : jax.Array, shape (m,)
        Gradient of criterion w.r.t. log_lambda.
    hessian : jax.Array, shape (m, m)
        Hessian of criterion w.r.t. log_lambda.
    max_step : float
        Maximum allowed component magnitude in log-lambda space.
    eig_floor_power : float
        Power of machine epsilon for eigenvalue floor. ``0.7`` for
        newton() (non-Gaussian), ``0.5`` for fast.REML.fit (Gaussian).

    Returns
    -------
    step : jax.Array, shape (m,)
        Safe Newton direction.
    is_pdef : jax.Array, scalar bool
        Whether the original Hessian was positive definite.
    """
    eigs, V = jnp.linalg.eigh(hessian)

    # Check positive definiteness before modification
    is_pdef = jnp.all(eigs > 0)

    # Flip negative eigenvalues
    eigs_safe = jnp.abs(eigs)

    # Floor small eigenvalues
    eps = jnp.finfo(jnp.float64).eps
    floor = jnp.max(eigs_safe) * eps**eig_floor_power
    eigs_safe = jnp.maximum(eigs_safe, floor)

    # Newton step: -H^{-1} g = -V diag(1/D) V^T g
    step = -V @ ((V.T @ gradient) / eigs_safe)

    # Cap step — component-wise max
    ms = jnp.max(jnp.abs(step))
    step = jnp.where(ms > max_step, step * max_step / ms, step)

    return step, is_pdef


# ---------------------------------------------------------------------------
# Step-halving outcome
# ---------------------------------------------------------------------------


class _StepOutcome(Enum):
    ACCEPTED = auto()
    STUCK = auto()
    FAILED = auto()


# ---------------------------------------------------------------------------
# Newton optimizer
# ---------------------------------------------------------------------------


class NewtonOptimizer:
    """Newton optimizer for smoothing parameter selection.

    Matches R's ``newton()`` (gam.fit3.r lines 1290-1719).

    Parameters
    ----------
    fd : FittingData
        Phase 1->2 boundary container with model data and penalties.
    method : str
        ``"REML"`` (default) or ``"ML"``.
    max_iter : int
        Maximum Newton iterations.
    tol : float, optional
        Convergence tolerance. Defaults to ``1e-6`` matching R's
        ``gam.control()$newton$conv.tol``.
    max_step : float
        Maximum step component in log-lambda space.
    lsp_max : float or None
        Maximum absolute value for log smoothing parameters. Defaults
        to ``40.0``. While R's Newton has no upper bounds (gam.fit3.r
        line 1323 disabled), our AD gradient has slightly more noise
        than R's analytical IFT gradient, which can cause drift on flat
        REML surfaces. The cap acts as a safety net without affecting
        well-determined directions. ``None`` disables clamping.
    """

    def __init__(
        self,
        fd: FittingData,
        method: str = "REML",
        max_iter: int = 200,
        tol: float | None = None,
        max_step: float = 5.0,
        lsp_max: float = 40.0,
    ) -> None:
        if method not in ("REML", "ML"):
            raise ValueError(f"Unknown method: {method!r}. Use 'REML' or 'ML'.")

        self._fd = fd
        self._method = method
        self._max_iter = max_iter
        self._max_step = max_step
        self._lsp_max = lsp_max
        self._is_gaussian = fd.family.family_name == "gaussian"

        # Eigenvalue floor power: R's fast.REML.fit (Gaussian) uses
        # eps^0.5 (fast-REML.r line 1810), more conservative than
        # newton() which uses eps^0.7 (gam.fit3.r line 1450).
        self._eig_floor_power = 0.5 if self._is_gaussian else 0.7

        # Convergence tolerance: family-dependent, matching R.
        # Gaussian: sqrt(eps) ≈ 1.5e-8 (matching fast.REML.fit).
        # Non-Gaussian: 1e-6 (R's gam.control()$newton$conv.tol).
        # R converges in ~10 outer iterations using the exact analytical
        # Hessian from gdi.c. Our custom_jvp approach (Exp 18) gives
        # the correct analytical Hessian via implicit differentiation,
        # matching R's convergence rate.
        if tol is not None:
            self._tol = tol
        elif fd.family.family_name == "gaussian":
            self._tol = float(jnp.finfo(jnp.float64).eps ** 0.5)
        else:
            self._tol = _DEFAULT_CONV_TOL

        # PIRLS tolerance: R tightens to conv.tol/100 inside Newton
        # (gam.fit3.r line 1308). This produces more accurate PIRLS
        # solutions, which in turn gives more accurate AD gradients.
        self._pirls_tol = min(self._tol / 100.0, 1e-8)
        # Joint optimization: append log_phi to params when scale is unknown
        # (matches R's mgcv.r line 2033: lsp <- c(lsp, log.scale))
        self._joint_scale = not fd.family.scale_known and fd.n_penalties > 0

        # Pre-bind all arguments except (params, beta_init) which vary
        # per Newton iteration. Shared by _jit_fit_and_score (forward pass)
        # and _jit_diff_grad_hess (gradient/Hessian for non-Gaussian).
        # Dynamic args are JAX arrays; static args are Python values that
        # key the JIT cache.
        offset = fd.offset if fd.offset is not None else jnp.zeros(fd.n_obs)
        self._jit_kwargs = {
            "X": fd.X,
            "y": fd.y,
            "wt": fd.wt,
            "offset": offset,
            "S_list": fd.S_list,
            "singleton_eig_constants": fd.singleton_eig_constants,
            "multi_block_proj_S": fd.multi_block_proj_S,
            "family": fd.family,
            "pirls_tol": self._pirls_tol,
            "is_reml": self._method == "REML",
            "joint_scale": self._joint_scale,
            "n_lambda": fd.n_penalties,
            "Mp": fd.total_penalty_null_dim,
            "singleton_sp_indices": fd.singleton_sp_indices,
            "singleton_ranks": fd.singleton_ranks,
            "multi_block_sp_indices": fd.multi_block_sp_indices,
            "multi_block_ranks": fd.multi_block_ranks,
            "p": fd.n_coef,
        }

        # Build custom_jvp-based differentiable score for all families.
        # For non-Gaussian: captures d(beta*)/d(rho), d(XtWX)/d(rho),
        #   d(deviance)/d(rho) via IFT — matches R's gdi.c.
        # For Gaussian: W is constant so dXtWX=0, but d(beta*)/d(rho)
        #   is still needed for the Hessian cross-penalty terms
        #   d²(beta'S beta)/d(rho_i)(rho_j). The AD Hessian misses these
        #   because it treats beta* as constant. For single-smooth (m=1)
        #   this has little effect, but for multi-smooth (m>1) the missing
        #   terms cause 2-14x more iterations than R.
        if fd.n_penalties > 0:
            kw = self._jit_kwargs

            def _grad_hess_fn(params, beta_warm):
                return _jit_diff_grad_hess(params, beta_warm, **kw)

            self._diff_grad_hess = _grad_hess_fn
        else:
            self._diff_grad_hess = None

    def _clamp_params(self, params: jax.Array) -> jax.Array:
        """Optionally clamp log smoothing parameters.

        When ``lsp_max`` is None, no clamping is applied. When set
        (default ``40.0``), clamps to ``[-lsp_max, lsp_max]``.

        For joint optimization, only the log_lambda portion is clamped;
        log_phi is left unclamped.
        """
        if self._lsp_max is None:
            return params
        lsp_max = self._lsp_max
        if self._joint_scale:
            n_lambda = self._fd.n_penalties
            log_lambda = params[:n_lambda]
            log_phi = params[n_lambda:]
            log_lambda = jnp.clip(log_lambda, -lsp_max, lsp_max)
            return jnp.concatenate([log_lambda, log_phi])
        return jnp.clip(params, -lsp_max, lsp_max)

    def _initial_beta(self) -> jax.Array:
        """Compute initial beta via Phase 1 initialization (NumPy -> JAX)."""
        fd = self._fd
        offset_np = np.asarray(fd.offset) if fd.offset is not None else None
        return initialize_beta(
            np.asarray(fd.X),
            np.asarray(fd.y),
            np.asarray(fd.wt),
            fd.family,
            offset_np,
        )

    def _make_criterion(
        self, pirls_result: PIRLSResult
    ) -> _CriterionBase | _JointCriterionBase:
        """Create the appropriate criterion object from converged PIRLS."""
        if self._joint_scale:
            if self._method == "REML":
                return JointREMLCriterion(self._fd, pirls_result)
            return JointMLCriterion(self._fd, pirls_result)
        if self._method == "REML":
            return REMLCriterion(self._fd, pirls_result)
        return MLCriterion(self._fd, pirls_result)

    def _fit_and_score(
        self,
        params: jax.Array,
        beta_init: jax.Array,
    ) -> tuple[PIRLSResult, jax.Array]:
        """Run PIRLS and compute criterion score in a single JIT dispatch.

        Uses ``_jit_fit_and_score`` which fuses S_lambda construction,
        PIRLS (while_loop), and criterion score into one XLA program.
        No criterion object is created — use ``_make_criterion()``
        separately after step acceptance when gradient/Hessian are needed.

        For joint optimization, ``params = [log_lambda..., log_phi]``.
        PIRLS only uses the ``log_lambda`` portion (via ``S_lambda``);
        ``log_phi`` enters only through the criterion scoring.

        Uses tightened PIRLS tolerance (``conv_tol / 100``) matching
        R's gam.fit3.r line 1308.
        """
        score, pirls_result = _jit_fit_and_score(
            params,
            beta_init,
            **self._jit_kwargs,
        )
        return pirls_result, score

    def _step_halve_gaussian(
        self,
        log_lambda: jax.Array,
        step: jax.Array,
        score: float,
        beta_warm: jax.Array,
        score_scale: float,
    ) -> tuple[jax.Array, PIRLSResult, jax.Array, _StepOutcome]:
        """Step-halving for Gaussian (R's ``fast.REML.fit``, fast-REML.r
        lines 1822-1844).

        Simpler than ``newton()``'s step acceptance: no quadratic-error
        check, no steepest-descent fallback, and step failure immediately
        ends the optimization. Just halves until score decreases.

        Returns
        -------
        log_lambda_new, pirls_new, score_new, outcome
        """
        tol = self._tol

        log_lambda_new = self._clamp_params(log_lambda + step)
        pirls_new, score_new = self._fit_and_score(log_lambda_new, beta_warm)

        # Accept immediately if score decreased (R line 1827)
        if jnp.isfinite(score_new) and float(score_new) < score:
            return log_lambda_new, pirls_new, score_new, _StepOutcome.ACCEPTED

        # Step-halving (R lines 1827-1839)
        k = 0
        not_moved = 0
        while float(score_new) >= score:
            # Count steps with no numerically significant change (R line 1831)
            if float(score_new) - score < tol * score_scale:
                not_moved += 1
            else:
                not_moved = 0

            # Break conditions (R line 1832)
            if k == _MAX_HALVINGS_GAUSSIAN or not_moved > 3:
                return log_lambda_new, pirls_new, score_new, _StepOutcome.FAILED
            if bool(jnp.allclose(log_lambda, log_lambda + step)):
                return log_lambda_new, pirls_new, score_new, _StepOutcome.FAILED

            step = step / 2
            k += 1
            log_lambda_new = self._clamp_params(log_lambda + step)
            pirls_new, score_new = self._fit_and_score(log_lambda_new, beta_warm)

        return log_lambda_new, pirls_new, score_new, _StepOutcome.ACCEPTED

    def _step_halve(
        self,
        log_lambda: jax.Array,
        step: jax.Array,
        score: float,
        beta_warm: jax.Array,
        score_scale: float,
        grad: jax.Array,
        hess: jax.Array,
        outer_iter: int,
        is_pdef: bool,
    ) -> tuple[jax.Array, PIRLSResult, jax.Array, _StepOutcome]:
        """Trial step with quadratic-error check and steepest-descent fallback.

        Follows R's ``newton()`` (gam.fit3.r lines 1491-1571) for
        non-Gaussian families:

        1. Try Newton step, check if quadratic model is accurate (qerror < 0.8)
        2. If Hessian is positive definite and step accepted: done
        3. If rejected: halve the step, retry
        4. After 3 halvings (in early iterations): switch to steepest-descent
           step with length min(newton_norm, maxSstep=2)
        5. For indefinite Hessians: also try pure steepest descent and pick best

        No criterion object is created — only the score is computed via
        the fused ``_jit_fit_and_score``. The criterion is created in
        ``run()`` after the step is accepted.

        Returns
        -------
        log_lambda_new, pirls_new, score_new, outcome
        """
        tol = self._tol

        # Compute predicted change from quadratic model
        pred_change = float(jnp.sum(grad * step) + 0.5 * step @ hess @ step)

        log_lambda_new = self._clamp_params(log_lambda + step)
        pirls_new, score_new = self._fit_and_score(log_lambda_new, beta_warm)

        score_change = float(score_new) - score
        qerror = abs(pred_change - score_change) / (
            max(abs(pred_change), abs(score_change)) + score_scale * tol
        )

        # Accept immediately if score decreased, quadratic model is accurate,
        # AND Hessian is positive definite (R line 1499)
        if jnp.isfinite(score_new) and score_change < 0 and is_pdef and qerror < 0.8:
            return log_lambda_new, pirls_new, score_new, _StepOutcome.ACCEPTED

        # Step-halving with steepest-descent fallback
        sd_step = -grad / float(jnp.max(jnp.abs(grad)))  # unit steepest-descent
        sd_used = False
        k = 0
        not_moved = 0
        while k < _MAX_HALVINGS_NON_GAUSSIAN:
            if float(score_new) - score < tol * score_scale:
                not_moved += 1
            else:
                not_moved = 0

            if not_moved > 3:
                break

            # After 3 halvings in early iterations, try steepest descent
            if k == 3 and outer_iter < 10 and not sd_used:
                s_length = min(float(jnp.sqrt(jnp.sum(step**2))), _MAX_SD_STEP)
                sd_norm = float(jnp.sqrt(jnp.sum(sd_step**2)))
                if sd_norm > 0:
                    step = sd_step * (s_length / sd_norm)
                    sd_used = True
            else:
                step = step / 2

            if bool(jnp.allclose(log_lambda, log_lambda + step)):
                break

            k += 1
            log_lambda_new = self._clamp_params(log_lambda + step)
            pirls_new, score_new = self._fit_and_score(log_lambda_new, beta_warm)

            score_change = float(score_new) - score
            # R relaxes qerror threshold after 4+ halvings (gam.fit3.r line 1540)
            qerror_thresh = 0.4 if k > 4 else 0.8
            pred_change = float(jnp.sum(grad * step) + 0.5 * step @ hess @ step)
            qerror = abs(pred_change - score_change) / (
                max(abs(pred_change), abs(score_change)) + score_scale * tol
            )

            if jnp.isfinite(score_new) and score_change < 0 and qerror < qerror_thresh:
                return (
                    log_lambda_new,
                    pirls_new,
                    score_new,
                    _StepOutcome.ACCEPTED,
                )

        if float(score_new) <= score:
            outcome = _StepOutcome.ACCEPTED
        elif float(score_new) - score >= tol * score_scale:
            outcome = _StepOutcome.FAILED
        else:
            outcome = _StepOutcome.STUCK

        return log_lambda_new, pirls_new, score_new, outcome

    def _compute_uconv_ind(
        self,
        grad: jax.Array,
        hess: jax.Array,
        score_scale: float,
    ) -> np.ndarray:
        """Identify unconverged dimensions for Newton step subsetting.

        Matches R's ``uconv.ind`` logic (gam.fit3.r lines 1430-1432, 1651):

        1. Base filter: ``|grad| > score_scale * tol * 0.1``
           OR ``|diag(H)| > score_scale * tol * 0.1``
        2. Refinement: additionally require ``|grad| > max(|grad|) * 0.001``
           to exclude dimensions where the gradient is tiny relative to
           the largest, as these are "likely to be poorly modelled on the
           scale of Newton step" (R comment at line 1427).

        Returns
        -------
        np.ndarray of bool, shape (m,)
            True for dimensions to include in the Newton step.
        """
        tol = self._tol
        grad_np = np.asarray(grad)
        hess_diag = np.asarray(jnp.diag(hess))

        # Base uconv.ind (R line 1651)
        thresh = score_scale * tol * 0.1
        uconv = (np.abs(grad_np) > thresh) | (np.abs(hess_diag) > thresh)

        # Refinement: exclude tiny-gradient dimensions (R lines 1430-1431)
        max_grad = np.max(np.abs(grad_np)) if len(grad_np) > 0 else 0.0
        uconv1 = uconv & (np.abs(grad_np) > max_grad * _GRAD_FILTER_THRESH)

        # Fallback: if nothing left, use full uconv (R line 1431)
        if np.sum(uconv1) == 0:
            uconv1 = uconv

        # Ultimate fallback: at least one dimension (R line 1432)
        if np.sum(uconv1) == 0:
            uconv1 = np.ones_like(uconv, dtype=bool)

        return uconv1

    def _projected_gradient(self, grad: jax.Array, params: jax.Array) -> jax.Array:
        """Projected gradient for bounded log smoothing parameters.

        At a bound, if the gradient points outward (into the constraint),
        the KKT conditions are satisfied and that component is zeroed.
        This prevents the optimizer from cycling at ``lsp_max`` when a
        smoothing parameter is on a flat REML surface.
        """
        if self._lsp_max is None:
            return grad
        lsp_max = self._lsp_max
        if self._joint_scale:
            n_lambda = self._fd.n_penalties
            log_lambda = params[:n_lambda]
            at_upper = jnp.abs(log_lambda - lsp_max) < _BOUND_TOL
            at_lower = jnp.abs(log_lambda + lsp_max) < _BOUND_TOL
            proj_lambda = jnp.where(
                at_upper & (grad[:n_lambda] < 0), 0.0, grad[:n_lambda]
            )
            proj_lambda = jnp.where(at_lower & (proj_lambda > 0), 0.0, proj_lambda)
            return jnp.concatenate([proj_lambda, grad[n_lambda:]])
        at_upper = jnp.abs(params - lsp_max) < _BOUND_TOL
        at_lower = jnp.abs(params + lsp_max) < _BOUND_TOL
        proj = jnp.where(at_upper & (grad < 0), 0.0, grad)
        return jnp.where(at_lower & (proj > 0), 0.0, proj)

    def _check_convergence(
        self,
        criterion: _CriterionBase | _JointCriterionBase,
        params: jax.Array,
        score: float,
        score_old: float,
        is_pdef: bool,
        pirls_result: PIRLSResult,
    ) -> tuple[jax.Array, jax.Array, float, bool]:
        """Compute gradient/Hessian and check convergence at accepted point.

        Uses family-dependent convergence logic matching R's two code paths:

        **Gaussian** (matches ``fast.REML.fit``, fast-REML.r lines 1740-1875):
        - ``score_scale = 1 + abs(score)``
        - Gradient check: ``max(|grad|) < tol * score_scale``
        - Score change: ``abs(delta) < tol * score_scale``
        - ``custom_jvp`` Hessian (matches R's ``Sl.ift`` analytical Hessian)

        **Non-Gaussian** (matches ``newton()``, gam.fit3.r lines 1647-1658):
        - ``score_scale = abs(log(scale)) + abs(score)``
        - Gradient check: ``max(|proj_grad|) < 5 * tol * score_scale``
        - Score change: ``abs(delta) < tol * score_scale``
        - Not converged if Hessian was indefinite (R line 1647)
        - Projected gradient zeros gradient at active ``lsp_max`` bounds
        - ``custom_jvp`` Hessian via implicit differentiation (matches
          R's analytical Hessian from gdi.c which accounts for dbeta*/drho)

        Returns (grad, hess, score_scale, converged).
        """
        tol = self._tol
        is_gaussian = self._fd.family.family_name == "gaussian"

        if self._diff_grad_hess is not None:
            # custom_jvp gradient and Hessian via implicit differentiation.
            # Captures all first- and second-order terms from how PIRLS
            # output (beta*, W, XtWX) changes with rho. For Gaussian,
            # dXtWX=0 (W constant) but d(beta*)/d(rho) contributes to
            # the Hessian cross-penalty terms, critical for multi-smooth.
            # For non-Gaussian, also captures d(XtWX)/d(rho).
            # Matches R's analytical Hessian from gdi.c / fast.REML.fit.
            beta_warm = pirls_result.coefficients
            grad, hess = self._diff_grad_hess(params, beta_warm)
            hess = (hess + hess.T) / 2
        else:
            # Purely parametric (no smoothing parameters): use criterion AD.
            grad, hess = criterion.grad_hess(params)
            hess = (hess + hess.T) / 2

        scale_val = float(criterion.scale)
        if is_gaussian:
            # fast.REML.fit: score.scale <- 1 + abs(score)
            score_scale = 1.0 + abs(score)
        elif self._method in ("REML", "ML"):
            # newton(): score.scale <- abs(log(b$scale.est)) + abs(score)
            score_scale = abs(np.log(max(scale_val, 1e-30))) + abs(score)
        else:
            score_scale = scale_val + abs(score)

        if is_gaussian:
            # fast.REML.fit convergence: simple gradient + score change
            converged = True
            if float(jnp.max(jnp.abs(grad))) > score_scale * tol:
                converged = False
            if abs(score - score_old) > score_scale * tol:
                converged = False
        else:
            # newton() convergence: is_pdef + projected gradient + score change
            converged = bool(is_pdef)
            proj_grad = self._projected_gradient(grad, params)
            if float(jnp.max(jnp.abs(proj_grad))) > score_scale * tol * 5:
                converged = False
            if abs(score - score_old) > score_scale * tol:
                converged = False

        return grad, hess, score_scale, converged

    def _build_result(
        self,
        params: jax.Array,
        score: jax.Array,
        gradient: jax.Array,
        criterion: _CriterionBase | _JointCriterionBase,
        pirls_result: PIRLSResult,
        converged: bool,
        step_failed: bool,
        n_iter: int,
    ) -> NewtonResult:
        """Assemble the final NewtonResult."""
        if converged:
            info = "full convergence"
        elif step_failed:
            info = "step failed"
        else:
            info = "iteration limit"

        if self._joint_scale:
            # Extract log_lambda from joint params; keep Fletcher scale
            # for post-estimation (Vp) matching R's model$scale.
            # The REML-optimized phi is internal to the criterion only.
            log_lambda = params[: self._fd.n_penalties]
            grad_lambda = gradient[: self._fd.n_penalties]
        else:
            log_lambda = params
            grad_lambda = gradient

        return NewtonResult(
            log_lambda=log_lambda,
            smoothing_params=jnp.exp(log_lambda),
            converged=converged,
            n_iter=n_iter,
            score=score,
            gradient=grad_lambda,
            edf=criterion.edf,
            scale=criterion.scale,
            pirls_result=pirls_result,
            convergence_info=info,
        )

    def run(self) -> NewtonResult:
        """Run the Newton optimization loop.

        Returns
        -------
        NewtonResult
            Optimization result with converged smoothing parameters,
            final PIRLS fit, and diagnostics.
        """
        fd = self._fd

        # -- Purely parametric shortcut --
        # Uses separate PIRLS + criterion (not the fused path) because
        # purely parametric unknown-scale families need Fletcher scale
        # as phi, not the joint optimization phi=1.0 assumption.
        if fd.n_penalties == 0:
            log_lambda = jnp.zeros(0)
            S = fd.S_lambda(log_lambda)
            pirls_result = pirls_loop(
                fd.X,
                fd.y,
                self._initial_beta(),
                S,
                fd.family,
                fd.wt,
                fd.offset,
                tol=self._pirls_tol,
            )
            criterion = self._make_criterion(pirls_result)
            score = criterion.score(log_lambda)
            return self._build_result(
                log_lambda,
                score,
                jnp.zeros(0),
                criterion,
                pirls_result,
                converged=True,
                step_failed=False,
                n_iter=0,
            )

        # -- Build initial params vector --
        log_lambda = fd.log_lambda_init.copy()
        if self._joint_scale:
            # Run initial PIRLS to get Fletcher scale for log_phi init
            S_init = fd.S_lambda(log_lambda)
            pirls_init = pirls_loop(
                fd.X,
                fd.y,
                self._initial_beta(),
                S_init,
                fd.family,
                fd.wt,
                fd.offset,
                tol=self._pirls_tol,
            )
            edf_init = estimate_edf(pirls_init.XtWX, pirls_init.L)
            phi_init = fletcher_scale(fd.y, pirls_init.mu, fd.wt, fd.family, edf_init)
            log_phi_init = jnp.log(phi_init)
            params = jnp.concatenate([log_lambda, log_phi_init[None]])
        else:
            params = log_lambda

        # -- Initial fit and score --
        pirls_result, score = self._fit_and_score(params, self._initial_beta())
        criterion = self._make_criterion(pirls_result)
        grad, hess, score_scale, _ = self._check_convergence(
            criterion,
            params,
            float(score),
            float(score),
            is_pdef=True,
            pirls_result=pirls_result,
        )

        converged = False
        step_failed = False
        n_iter = 0
        consecutive_stuck = 0

        for outer_iter in range(self._max_iter):
            # uconv.ind subsetting: only optimize unconverged dimensions.
            # Both R's fast.REML.fit (line 1802) and newton() (lines
            # 1430-1436) use this. Speeds convergence and prevents
            # poorly-modelled flat directions from corrupting the
            # Newton step in well-conditioned directions.
            uconv = self._compute_uconv_ind(grad, hess, score_scale)
            n_uc = int(np.sum(uconv))

            if n_uc < len(np.asarray(grad)):
                # Subset to unconverged dimensions
                uc_idx = np.where(uconv)[0]
                grad_uc = grad[uc_idx]
                hess_uc = hess[np.ix_(uc_idx, uc_idx)]
                step_uc, is_pdef = _safe_newton_step(
                    grad_uc, hess_uc, self._max_step, self._eig_floor_power
                )
                # Embed back into full space
                step = jnp.zeros_like(grad)
                step = step.at[uc_idx].set(step_uc)
            else:
                step, is_pdef = _safe_newton_step(
                    grad, hess, self._max_step, self._eig_floor_power
                )

            if self._is_gaussian:
                # fast.REML.fit: simpler step acceptance, step.failed
                # breaks immediately (R fast-REML.r lines 1822-1844)
                params_new, pirls_new, score_new, outcome = self._step_halve_gaussian(
                    params,
                    step,
                    float(score),
                    pirls_result.coefficients,
                    score_scale,
                )
            else:
                params_new, pirls_new, score_new, outcome = self._step_halve(
                    params,
                    step,
                    float(score),
                    pirls_result.coefficients,
                    score_scale,
                    grad,
                    hess,
                    outer_iter,
                    bool(is_pdef),
                )

            n_iter = outer_iter + 1

            if outcome is _StepOutcome.FAILED:
                step_failed = True
                break

            if outcome is _StepOutcome.STUCK:
                consecutive_stuck += 1
                if consecutive_stuck >= 3:
                    converged = True
                    break
                continue

            consecutive_stuck = 0
            score_old = score
            params = params_new
            pirls_result = pirls_new
            criterion = self._make_criterion(pirls_result)
            score = score_new

            grad, hess, score_scale, converged = self._check_convergence(
                criterion,
                params,
                float(score),
                float(score_old),
                is_pdef=bool(is_pdef),
                pirls_result=pirls_result,
            )
            if converged:
                break

        return self._build_result(
            params,
            score,
            grad,
            criterion,
            pirls_result,
            converged,
            step_failed,
            n_iter,
        )


# ---------------------------------------------------------------------------
# Public convenience function
# ---------------------------------------------------------------------------


def newton_optimize(
    fd: FittingData,
    method: str = "REML",
    log_lambda_init: jax.Array | None = None,
    max_iter: int = 200,
    tol: float | None = None,
    max_step: float = 5.0,
    lsp_max: float = 40.0,
) -> NewtonResult:
    """Minimize REML/ML criterion over log smoothing parameters.

    Outer Newton loop wrapping PIRLS inner loop. At each iteration:

    1. Compute eigenvalue-safe Newton step from gradient/Hessian
    2. Trial PIRLS fit at proposed log_lambda
    3. Step-halving until criterion decreases
    4. Accept and check convergence

    Follows R's ``newton()`` (gam.fit3.r lines 1290-1719).

    Parameters
    ----------
    fd : FittingData
        Phase 1->2 boundary container with model data and penalties.
    method : str
        ``"REML"`` (default) or ``"ML"``.
    log_lambda_init : jax.Array, shape (m,), optional
        Starting log smoothing parameters. Defaults to ``fd.log_lambda_init``.
    max_iter : int
        Maximum Newton iterations.
    tol : float, optional
        Convergence tolerance. Defaults to ``1e-6`` (R's default).
    max_step : float
        Maximum step component in log-lambda space.
    lsp_max : float or None
        Maximum absolute value for log smoothing parameters. Defaults
        to ``40.0``. ``None`` disables clamping.

    Returns
    -------
    NewtonResult
        Optimization result with converged smoothing parameters,
        final PIRLS fit, and diagnostics.
    """
    if log_lambda_init is not None:
        fd = dataclasses.replace(
            fd,
            log_lambda_init=jnp.asarray(log_lambda_init, dtype=jnp.float64),
        )
    optimizer = NewtonOptimizer(
        fd,
        method=method,
        max_iter=max_iter,
        tol=tol,
        max_step=max_step,
        lsp_max=lsp_max,
    )
    return optimizer.run()
