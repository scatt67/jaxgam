"""Top-level fitting orchestration for jaxgam.

Provides the ``GAM`` class (sklearn-style API) that wires together:
- Phase 1: ``parse_formula()`` → ``ModelSetup.build()``
- Phase 2: ``FittingData.from_setup()`` → ``newton_optimize()`` / ``pirls_loop()``
- Phase 3: ``GAMResults._from_fit()``

``GAM.fit()`` returns a ``GAMResults`` frozen dataclass. For backward
compatibility, fitted attributes (``model.coefficients_``) are forwarded
to ``model.results_`` with deprecation warnings.

Design doc reference: docs/refactor_gam_api/design.md §3.3, §3.5
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

import numpy as np

from jaxgam.families.base import ExponentialFamily
from jaxgam.families.registry import get_family
from jaxgam.fitting.data import FittingData
from jaxgam.fitting.initialization import initialize_beta
from jaxgam.fitting.newton import NewtonResult, newton_optimize
from jaxgam.fitting.pirls import pirls_loop
from jaxgam.fitting.reml import estimate_edf, estimate_scale
from jaxgam.formula.design import ModelSetup
from jaxgam.formula.parser import parse_formula
from jaxgam.results import GAMResults

if TYPE_CHECKING:
    import jax
    import matplotlib.figure
    import pandas as pd

    from jaxgam.summary.summary import GAMSummary


# ---------------------------------------------------------------------------
# GAM class
# ---------------------------------------------------------------------------


class GAM:
    """Generalized Additive Model specification.

    This class holds model specification parameters and orchestrates the
    fit pipeline. Calling ``fit()`` returns a ``GAMResults`` frozen
    dataclass containing all fitted state.

    Parameters
    ----------
    formula : str
        Model formula in R-style Wilkinson notation, e.g. ``"y ~ s(x)"``.
    family : str or ExponentialFamily
        Distribution family. One of ``'gaussian'``, ``'binomial'``,
        ``'poisson'``, ``'gamma'``, or an ``ExponentialFamily`` instance.
    method : str
        Smoothing parameter estimation method: ``'REML'`` or ``'ML'``.
    sp : np.ndarray or list, optional
        Fixed smoothing parameters. If provided, skips Newton optimization.
    device : str, optional
        Target device: ``'cpu'``, ``'gpu'``, or ``None`` (auto-detect).
        GPU requires ``jax[cuda12]`` (NVIDIA) or ``jax-metal`` (Apple).
    **kwargs
        Additional arguments. Supported scope guards:
        ``backend``, ``optimizer``, ``select``, ``gamma``, ``knots``.

    Examples
    --------
    >>> model = GAM("y ~ s(x)", family="gaussian")
    >>> results = model.fit(data)
    >>> results.predict(newdata)
    array([...])

    Backward-compatible usage (deprecated):

    >>> model = GAM("y ~ s(x)").fit(data)  # returns GAMResults
    >>> model.predict(newdata)  # works on GAMResults directly

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

    def fit(
        self,
        data: pd.DataFrame | dict,
        weights: np.ndarray | None = None,
        offset: np.ndarray | None = None,
    ) -> GAMResults:
        """Fit the GAM to data.

        Parameters
        ----------
        data : pandas.DataFrame or dict
            Data frame containing the variables in the formula.
        weights : np.ndarray, optional
            Prior weights, shape ``(n,)``.
        offset : np.ndarray, optional
            Offset vector, shape ``(n,)``.

        Returns
        -------
        GAMResults
            Frozen dataclass containing all fitted state. Also stored
            as ``self.results_`` for backward compatibility.

        Design doc reference: docs/refactor_gam_api/design.md §3.3, §9 #1
        """
        family_obj = get_family(self.family)

        # Phase 1: parse + build model setup
        spec = parse_formula(self.formula)
        setup = ModelSetup.build(spec, data, weights, offset)

        # Phase 1→2: transfer to JAX device
        jax_device = _resolve_device(self.device)
        fd = FittingData.from_setup(setup, family_obj, device=jax_device)

        # Phase 2: fit
        if self.sp is not None:
            result = _fit_fixed_sp(fd, self.sp)
            lambda_strategy = "fixed"
        else:
            result = newton_optimize(fd, self.method)
            lambda_strategy = f"newton_{self.method.lower()}"

        # Phase 2→3: construct GAMResults
        results = GAMResults._from_fit(
            result=result,
            setup=setup,
            spec=spec,
            data=data,
            family=family_obj,
            fd=fd,
            lambda_strategy=lambda_strategy,
            formula=self.formula,
            method=self.method,
        )

        self.results_ = results
        return results

    # ------------------------------------------------------------------
    # Backward-compatible forwarding (deprecated)
    # Design doc reference: docs/refactor_gam_api/design.md §3.5
    # ------------------------------------------------------------------

    def __getattr__(self, name: str) -> object:
        """Forward fitted attribute access to results_ with deprecation."""
        if name.endswith("_") and name != "results_":
            stripped = name.rstrip("_")
            if hasattr(self, "results_") and hasattr(
                self.results_, stripped
            ):
                warnings.warn(
                    f"Accessing '{name}' on GAM is deprecated. "
                    f"Use 'results.{stripped}' instead.",
                    DeprecationWarning,
                    stacklevel=2,
                )
                return getattr(self.results_, stripped)
        raise AttributeError(
            f"'{type(self).__name__}' has no attribute '{name}'"
        )

    def predict(
        self,
        newdata: pd.DataFrame | dict | None = None,
        pred_type: str = "response",
        se_fit: bool = False,
        offset: np.ndarray | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Predict from a fitted GAM (deprecated).

        .. deprecated::
            Use ``results.predict()`` instead.
        """
        warnings.warn(
            "Calling predict() on GAM is deprecated. "
            "Use results.predict() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not hasattr(self, "results_"):
            raise RuntimeError(
                "This GAM instance is not fitted yet. "
                "Call fit() first."
            )
        return self.results_.predict(
            newdata=newdata,
            pred_type=pred_type,
            se_fit=se_fit,
            offset=offset,
        )

    def predict_matrix(
        self, newdata: pd.DataFrame | dict
    ) -> np.ndarray:
        """Build prediction matrix (deprecated).

        .. deprecated::
            Use ``results.predict_matrix()`` instead.
        """
        warnings.warn(
            "Calling predict_matrix() on GAM is deprecated. "
            "Use results.predict_matrix() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not hasattr(self, "results_"):
            raise RuntimeError(
                "This GAM instance is not fitted yet. "
                "Call fit() first."
            )
        return self.results_.predict_matrix(newdata)

    def summary(self) -> GAMSummary:
        """Print and return summary (deprecated).

        .. deprecated::
            Use ``results.summary()`` instead.
        """
        warnings.warn(
            "Calling summary() on GAM is deprecated. "
            "Use results.summary() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not hasattr(self, "results_"):
            raise RuntimeError(
                "This GAM instance is not fitted yet. "
                "Call fit() first."
            )
        return self.results_.summary()

    def plot(
        self,
        select: int | list | None = None,
        pages: int = 0,
        rug: bool = True,
        se: bool = True,
        shade: bool = True,
        **kwargs,
    ) -> tuple[matplotlib.figure.Figure, np.ndarray]:
        """Plot smooth components (deprecated).

        .. deprecated::
            Use ``results.plot()`` instead.
        """
        warnings.warn(
            "Calling plot() on GAM is deprecated. "
            "Use results.plot() instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if not hasattr(self, "results_"):
            raise RuntimeError(
                "This GAM instance is not fitted yet. "
                "Call fit() first."
            )
        return self.results_.plot(
            select=select,
            pages=pages,
            rug=rug,
            se=se,
            shade=shade,
            **kwargs,
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


def _check_scope_guards(method: str, kwargs: dict) -> None:
    """Validate v1.0 scope guards."""
    method_upper = method.upper()
    if method_upper not in ("REML", "ML"):
        raise ValueError(
            f"method must be 'REML' or 'ML', got {method!r}. "
            "GCV/UBRE is planned for v1.1."
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
            f"device={device!r} is not recognized. "
            "Use 'cpu', 'gpu', or None."
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
            "Only gamma=1.0 (standard REML/ML) is available."
        )

    if kwargs.get("knots") is not None:
        raise NotImplementedError(
            "User-specified knots are planned for v1.1. "
            "See docs/design.md Section 5.2."
        )


def _fit_fixed_sp(fd: FittingData, sp: np.ndarray | list) -> NewtonResult:
    """Fit with user-supplied fixed smoothing parameters.

    Runs a single PIRLS at the given lambda (no Newton optimization).

    Parameters
    ----------
    fd : FittingData
        Phase 1→2 boundary data.
    sp : array-like
        Smoothing parameters on the original scale, shape ``(n_penalties,)``.

    Returns
    -------
    NewtonResult
        Result with ``n_iter=0``, ``convergence_info="fixed sp"``.
    """
    import jax.numpy as jnp

    sp_arr = np.asarray(sp, dtype=np.float64)
    if sp_arr.shape[0] != fd.n_penalties:
        raise ValueError(
            f"sp has {sp_arr.shape[0]} elements but model has "
            f"{fd.n_penalties} penalty terms."
        )

    log_lambda = jnp.log(jnp.array(sp_arr))
    S_lam = fd.S_lambda(log_lambda)

    # Initialize beta and run PIRLS
    beta_init = initialize_beta(
        np.asarray(fd.X),
        np.asarray(fd.y),
        np.asarray(fd.wt),
        fd.family,
        np.asarray(fd.offset) if fd.offset is not None else None,
    )

    pirls_result = pirls_loop(
        fd.X,
        fd.y,
        beta_init,
        S_lam,
        fd.family,
        wt=fd.wt,
        offset=fd.offset,
    )

    # Compute EDF and scale
    edf = estimate_edf(pirls_result.XtWX, pirls_result.L)
    scale = estimate_scale(
        fd.y,
        pirls_result.mu,
        fd.wt,
        fd.family,
        edf,
    )

    return NewtonResult(
        log_lambda=log_lambda,
        smoothing_params=jnp.exp(log_lambda),
        converged=bool(pirls_result.converged),
        n_iter=0,
        score=jnp.array(0.0),
        gradient=jnp.zeros_like(log_lambda),
        edf=edf,
        scale=scale,
        pirls_result=pirls_result,
        convergence_info="fixed sp",
    )
