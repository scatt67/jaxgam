"""Top-level fitting orchestration for jaxgam.

Provides the ``GAM`` class (sklearn-style API) that wires together:
- Phase 1: ``parse_formula()`` → ``ModelSetup.build()``
- Phase 2: ``FittingData.from_setup()`` → ``newton_optimize()`` / ``pirls_loop()``
- Phase 3: ``GAMResults._from_fit()`` materializes the selected result type

``GAM.fit()`` returns a full ``GAMResults`` by default or a lean
``GAMInferenceResult`` when requested.

Design doc reference: docs/refactor_gam_api/design.md §3.3
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

import numpy as np

from jaxgam.families.base import ExponentialFamily
from jaxgam.families.registry import get_family
from jaxgam.fitting.data import FittingData
from jaxgam.fitting.initialization import initialize_beta
from jaxgam.fitting.newton import NewtonResult, newton_optimize
from jaxgam.fitting.pirls import pirls_loop
from jaxgam.fitting.reml import REMLCriterion
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.parser import parse_formula
from jaxgam.results import GAMInferenceResult, GAMResults

if TYPE_CHECKING:
    import jax
    import pandas as pd


# ---------------------------------------------------------------------------
# GAM class
# ---------------------------------------------------------------------------


class GAM:
    """Generalized Additive Model specification.

    This class holds model specification parameters and orchestrates the
    fit pipeline. Calling ``fit()`` returns a frozen full or lean result,
    selected by the keyword-only ``result`` argument.

    Parameters
    ----------
    formula : str
        Model formula in R-style Wilkinson notation, e.g. ``"y ~ s(x)"``.
    family : str or ExponentialFamily
        Distribution family. One of ``'gaussian'``, ``'binomial'``,
        ``'poisson'``, ``'gamma'``, or an ``ExponentialFamily`` instance.
    method : str
        Smoothing parameter estimation method. Only ``'REML'`` is supported
        in v1.0; ``'ML'`` raises ``NotImplementedError`` (see notes below).
    sp : np.ndarray or list, optional
        Fixed smoothing parameters. If provided, skips Newton optimization.
    device : str, optional
        Target device: ``'cpu'``, ``'gpu'``, or ``None`` (auto-detect).
        GPU requires ``jax[cuda12]`` (NVIDIA) or ``jax-metal`` (Apple).
    **kwargs
        Additional arguments. Only the scope-guard keys ``device``,
        ``backend``, ``optimizer``, ``select``, ``gamma``, ``knots`` are
        accepted; any other keyword raises ``TypeError`` rather than being
        silently ignored (so typos like ``tol=`` or ``max_iter=`` are caught).

    Examples
    --------
    >>> model = GAM("y ~ s(x)", family="gaussian")
    >>> results = model.fit(data)
    >>> results.predict(newdata)
    array([...])

    Design doc reference: docs/refactor_gam_api/design.md §3.3
    """

    def __init__(
        self,
        formula: str,
        family: str | ExponentialFamily = "gaussian",
        method: str = "REML",
        sp: np.ndarray | list | None = None,
        **kwargs,
    ) -> None:
        _check_scope_guards(method, kwargs)
        self.formula = formula
        self.family = family
        self.method = method.upper()
        self.sp = sp
        self.device = kwargs.get("device")

    @overload
    def fit(
        self,
        data: pd.DataFrame | dict,
        weights: np.ndarray | None = None,
        offset: np.ndarray | None = None,
        *,
        result: Literal["full"] = "full",
    ) -> GAMResults: ...

    @overload
    def fit(
        self,
        data: pd.DataFrame | dict,
        weights: np.ndarray | None = None,
        offset: np.ndarray | None = None,
        *,
        result: Literal["inference"],
    ) -> GAMInferenceResult: ...

    def fit(
        self,
        data: pd.DataFrame | dict,
        weights: np.ndarray | None = None,
        offset: np.ndarray | None = None,
        *,
        result: Literal["full", "inference"] = "full",
    ) -> GAMResults | GAMInferenceResult:
        """Fit the GAM to data.

        Parameters
        ----------
        data : pandas.DataFrame or dict
            Data frame containing the variables in the formula.
        weights : np.ndarray, optional
            Prior weights, shape ``(n,)``.
        offset : np.ndarray, optional
            Offset vector, shape ``(n,)``.
        result : {"full", "inference"}
            Result materialization mode. ``"full"`` retains training-backed
            diagnostics; ``"inference"`` returns lean prediction state.

        Returns
        -------
        GAMResults or GAMInferenceResult
            Frozen full-diagnostic or lean-inference result.

        Design doc reference: docs/refactor_gam_api/design.md §3.3
        """
        if result not in ("full", "inference"):
            raise ValueError(f"result must be 'full' or 'inference', got {result!r}")

        family_obj = get_family(self.family)

        # Extended families have mutable theta state that is synced by
        # _build_result.put_theta after fitting.  Copy the instance so
        # the registry singleton is never mutated.
        if hasattr(family_obj, "n_theta") and family_obj.n_theta > 0:
            import copy

            family_obj = copy.deepcopy(family_obj)

        # Phase 1: parse + build model setup
        spec = parse_formula(self.formula)
        setup = ModelSetup.build(spec, data, weights, offset)

        # Phase 1→2: transfer to JAX device
        jax_device = _resolve_device(self.device)
        fd = FittingData.from_setup(setup, family_obj, device=jax_device)

        # Phase 2: fit
        if self.sp is not None:
            fit_result = _fit_fixed_sp(fd, self.sp, self.method)
            lambda_strategy = "fixed"
        else:
            fit_result = newton_optimize(fd, self.method)
            lambda_strategy = f"newton_{self.method.lower()}"

        # Phase 2→3: construct the selected result materialization
        return GAMResults._from_fit(
            fit_result=fit_result,
            setup=setup,
            spec=spec,
            data=data,
            family=family_obj,
            fd=fd,
            lambda_strategy=lambda_strategy,
            formula=self.formula,
            method=self.method,
            result_mode=result,
        )


# ---------------------------------------------------------------------------
# Private module-level helpers
# ---------------------------------------------------------------------------


def _resolve_device(device: str | None) -> jax.Device | None:
    """Resolve a device string to a JAX device object."""
    if device is None:
        return None
    import jax

    if device == "cpu":
        return jax.devices("cpu")[0]
    if device == "gpu":
        try:
            gpu_devices = jax.devices("gpu")
        except RuntimeError:
            gpu_devices = []
        if not gpu_devices:
            raise RuntimeError(
                "device='gpu' requested but no GPU backend found. "
                "Install jax[cuda12] (NVIDIA) or jax-metal (Apple Silicon)."
            )
        return gpu_devices[0]
    # _check_scope_guards validates device before this is called, so this
    # line is unreachable.  Raise explicitly for defensive clarity.
    raise ValueError(f"Unrecognized device: {device!r}")


#: Keyword arguments accepted by ``GAM`` beyond the explicit positional params
#: (formula, family, method, sp). Anything else is a typo (e.g. ``tol=``,
#: ``max_iter=``) and must be rejected rather than silently ignored.
_ACCEPTED_KWARGS = frozenset(
    {"device", "backend", "optimizer", "select", "gamma", "knots"}
)


def _check_scope_guards(method: str, kwargs: dict) -> None:
    """Validate v1.0 scope guards."""
    unknown = set(kwargs) - _ACCEPTED_KWARGS
    if unknown:
        raise TypeError(
            f"GAM() got unexpected keyword argument(s) {sorted(unknown)}. "
            f"Accepted arguments: formula, family, method, sp, "
            f"{', '.join(sorted(_ACCEPTED_KWARGS))}."
        )

    method_upper = method.upper()
    if method_upper == "ML":
        raise NotImplementedError(
            "method='ML' is not supported. mgcv's ML criterion uses the "
            "penalty range-space projection of log|X'WX+S| (the C routine "
            "MLpenalty1 in gdi.c), which differs from REML's full-space "
            "determinant. Only method='REML' is available; ML is deferred "
            "until the range-space determinant is implemented. "
            "See docs/design.md Section 4.4."
        )
    if method_upper != "REML":
        raise ValueError(
            f"method must be 'REML', got {method!r}. "
            "ML is not available (see above); GCV/UBRE is planned for v1.1."
        )

    backend = kwargs.get("backend")
    if backend is not None and backend != "jax":
        raise NotImplementedError(
            f"backend={backend!r} is not supported in v1.0. "
            "Only 'jax' backend is available. See docs/design.md Section 10."
        )

    device = kwargs.get("device")
    if device is not None and device not in ("cpu", "gpu"):
        raise ValueError(
            f"device={device!r} is not recognized. Use 'cpu', 'gpu', or None."
        )

    optimizer = kwargs.get("optimizer")
    if optimizer is not None and optimizer != "newton":
        raise NotImplementedError(
            f"optimizer={optimizer!r} is not supported in v1.0. "
            "Only 'newton' optimizer is available."
        )

    if kwargs.get("select", False):
        raise NotImplementedError(
            "select=True (shrinkage smoothing) is planned for v1.1. "
            "See docs/design.md Section 4.6."
        )

    gamma = kwargs.get("gamma", 1.0)
    if gamma != 1.0:
        raise NotImplementedError(
            f"gamma={gamma} is not supported in v1.0. "
            "Only gamma=1.0 (standard REML) is available."
        )

    if kwargs.get("knots") is not None:
        raise NotImplementedError(
            "User-specified knots are planned for v1.1. See docs/design.md Section 5.2."
        )


def _fit_fixed_sp(
    fd: FittingData, sp: np.ndarray | list, method: str = "REML"
) -> NewtonResult:
    """Fit with user-supplied fixed smoothing parameters.

    For standard families this runs a single PIRLS at the given lambda (no
    Newton optimization). For extended families with an estimated dispersion
    parameter (e.g. Negative Binomial theta), the smoothing parameters are
    pinned but theta is still estimated via the outer optimizer — matching
    mgcv, where fixing ``sp`` does not fix theta (``gam.fit4`` keeps family
    parameters in the outer optimization).

    Parameters
    ----------
    fd : FittingData
        Phase 1→2 boundary data.
    sp : array-like
        Smoothing parameters on the original scale, shape ``(n_penalties,)``.
        A scalar is accepted for single-penalty models.
    method : str
        ``"REML"`` — used only when theta is estimated. (ML is not
        supported in v1.0; the public API rejects it before this point.)

    Returns
    -------
    NewtonResult
        For standard families: ``n_iter=0``, ``convergence_info="fixed sp"``.
        For estimated-theta families: the outer-optimizer result with theta
        estimated at the fixed smoothing parameters.
    """
    import jax.numpy as jnp

    sp_arr = np.atleast_1d(np.asarray(sp, dtype=np.float64))
    if sp_arr.shape[0] != fd.n_penalties:
        raise ValueError(
            f"sp has {sp_arr.shape[0]} elements but model has "
            f"{fd.n_penalties} penalty terms."
        )

    # jaxgam treats fixed sp as natural-scale positive values and log-transforms
    # them internally (unlike mgcv's "negative sp means estimate"). Validate
    # eagerly here so sp<=0/non-finite raises a clear error naming the bad value,
    # rather than the cryptic "array must not contain infs or NaNs" that log(-1)
    # / log(0) trigger deep inside PIRLS.
    if np.any(sp_arr <= 0.0) or not np.all(np.isfinite(sp_arr)):
        bad = sp_arr[(sp_arr <= 0.0) | ~np.isfinite(sp_arr)]
        raise ValueError(
            f"sp must contain strictly positive, finite values (smoothing "
            f"parameters are on the natural scale and are log-transformed "
            f"internally); got invalid value(s) {bad.tolist()}."
        )

    log_lambda = jnp.log(jnp.array(sp_arr))

    # Families with an estimated dispersion parameter still co-optimize it with
    # the smoothing parameters held fixed, matching mgcv (gam.outer keeps theta
    # for extended families and appends log.scale as an extra "smoothing
    # parameter" for unknown-scale families, mgcv.r:2025-2039). Route both
    # through the pinned-Newton path so the reported REML score is evaluated at
    # the jointly-optimal (theta, phi) and the real sp stay fixed.
    if fd.family.n_theta > 0 or not fd.family.scale_known:
        return newton_optimize(fd, method, log_lambda_init=log_lambda, pin_lambda=True)

    S_lam = fd.S_lambda(log_lambda)

    # Initialize beta and run PIRLS
    beta_init = initialize_beta(
        np.asarray(fd.X),
        np.asarray(fd.y),
        np.asarray(fd.wt),
        fd.family,
        np.asarray(fd.offset) if fd.offset is not None else None,
    )

    log_theta = None
    if fd.family.n_theta > 0:
        log_theta = jnp.asarray(fd.family.get_theta(transformed=False))

    pirls_result = pirls_loop(
        fd.X,
        fd.y,
        beta_init,
        S_lam,
        fd.family,
        wt=fd.wt,
        offset=fd.offset,
        log_theta=log_theta,
    )

    # Known-scale standard families (Poisson/Binomial): no free dispersion
    # parameter, so the REML score is a single criterion evaluation at the fixed
    # sp (mgcv's no.sps / gam2objective path, mgcv.r:1692-1716). Build the same
    # REMLCriterion the estimated-sp path uses and evaluate it — rather than
    # stubbing score=0.0 — so GAMResults.score is the real REML criterion (R's
    # gcv.ubre). REMLCriterion recomputes EDF/scale internally with the Fisher
    # weighting, identical to the values below.
    criterion = REMLCriterion(fd, pirls_result)
    edf = criterion.edf
    scale = criterion.scale
    score = criterion.score(log_lambda)

    return NewtonResult(
        log_lambda=log_lambda,
        smoothing_params=jnp.exp(log_lambda),
        converged=bool(pirls_result.converged),
        n_iter=0,
        score=score,
        gradient=jnp.zeros_like(log_lambda),
        edf=edf,
        scale=scale,
        pirls_result=pirls_result,
        convergence_info="fixed sp",
    )
