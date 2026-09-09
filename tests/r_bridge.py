"""RBridge: interface to R's mgcv for reference comparison.

New reference oracles call R functions in-process through rpy2. The explicit
legacy subprocess mode remains only for pre-stack tests pending migration;
it is never selected automatically.

Usage::

    from tests.r_bridge import RBridge
    import pandas as pd

    bridge = RBridge()
    data = pd.DataFrame({"x": x, "y": y})
    result = bridge.fit_gam("y ~ s(x)", data, family="gaussian")
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import tempfile
from functools import lru_cache
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from jaxgam.formula.terms import SmoothSpec

_REQUIRED_R_VERSION = "4.5.2"
_REQUIRED_MGCV_VERSION = "1.9.3"
_PINNED_MGCV_SOURCE_COMMIT = "fb7e8e718377513e78ba6c6bf7e60757fc6a32a9"

_KERNEL_TO_MGCV_TYPE = {
    "spherical": 1,
    "power_exponential": 2,
    "matern_3_2": 3,
    "matern_5_2": 4,
    "matern_7_2": 5,
}


@lru_cache(maxsize=1)
def _r_packages() -> tuple[Any, Any, Any, Any, Any]:
    """Reuse package wrappers around the single embedded R interpreter."""
    import rpy2.robjects as ro
    from rpy2.robjects.packages import importr

    return ro, importr("mgcv"), importr("base"), importr("stats"), importr("utils")


def gp_config_to_mgcv_m(spec: SmoothSpec, rho: float | None = None) -> list[float]:
    """Translate a GP ``SmoothSpec`` to mgcv's signed ``m`` vector.

    Reads the same ``spec.extra_args`` that ``GaussianProcessSmooth`` consumes.
    R parity tests use this helper so Python-side GP kwargs and R-side ``m=``
    formulas cannot drift.
    """
    kernel = spec.extra_args.get("kernel", "matern_3_2")
    stationary = spec.extra_args.get("stationary", False)
    power = spec.extra_args.get("power", 1.0)
    spec_rho = spec.extra_args.get("rho")

    type_id = _KERNEL_TO_MGCV_TYPE[kernel]
    if stationary:
        type_id = -type_id

    out: list[float] = [float(type_id)]
    resolved_rho = rho if rho is not None else spec_rho
    if resolved_rho is not None:
        out.append(float(resolved_rho))

    if kernel == "power_exponential":
        # Pad rho if absent so the power parameter lands at mgcv's m[3].
        if len(out) == 1:
            out.append(-1.0)
        out.append(float(power))

    return out


class RBridgeError(Exception):
    """Error obtaining a pinned R reference result."""


class RBridge:
    """Interface to R's mgcv for reference comparison.

    Parameters
    ----------
    mode : str
        'auto' and 'rpy2' both require rpy2. 'subprocess' is retained only
        for older tests pending their separate migration.
    """

    _SUBPROCESS_FAMILY_MAP: ClassVar[dict[str, str]] = {
        "gaussian": "gaussian()",
        "binomial": "binomial()",
        "poisson": "poisson()",
        "gamma": "Gamma()",
        "nb": "nb()",
    }

    _ro: Any
    _mgcv: Any
    _base: Any
    _stats: Any

    _utils: Any

    def __init__(self, mode: str = "auto") -> None:
        if mode in {"auto", "rpy2"}:
            self.mode = "rpy2"
            self._setup_rpy2()
        elif mode == "subprocess":
            self.mode = "subprocess"
        else:
            raise ValueError(
                f"Unknown mode: {mode!r}. Use 'auto', 'rpy2', or 'subprocess'."
            )

    def _setup_rpy2(self) -> None:
        """Initialize rpy2 connection and import R packages."""
        self._ro, self._mgcv, self._base, self._stats, self._utils = _r_packages()

    @staticmethod
    def available() -> bool:
        """Check if rpy2 can load R and mgcv in this process."""
        try:
            _r_packages()
            return True
        except (ImportError, ValueError, OSError, RuntimeError):
            return False

    @staticmethod
    def check_versions() -> tuple[bool, str]:
        """Verify R and mgcv match the pinned versions.

        Returns (True, "") if versions match, or (False, reason) if not.
        """
        try:
            _, _, base, _, utils = _r_packages()
            r_ver = str(base.as_character(base.getRversion())[0])
            mgcv_ver = str(base.as_character(utils.packageVersion("mgcv"))[0])
        except Exception as e:
            return False, f"Cannot query R versions: {e}"

        if r_ver != _REQUIRED_R_VERSION:
            return False, f"R {r_ver} != required {_REQUIRED_R_VERSION}"
        if mgcv_ver != _REQUIRED_MGCV_VERSION:
            return False, f"mgcv {mgcv_ver} != required {_REQUIRED_MGCV_VERSION}"
        return True, ""

    # ------------------------------------------------------------------ #
    #  rpy2 helpers                                                       #
    # ------------------------------------------------------------------ #

    def _require_rpy2(self) -> None:
        """Require the pinned in-process reference for migrated oracles."""
        if self.mode != "rpy2":
            raise RBridgeError("This reference oracle requires RBridge(mode='rpy2').")
        ok, reason = self.check_versions()
        if not ok:
            raise RBridgeError(f"Pinned R oracle unavailable: {reason}")

    def _call_internal(self, name: str, *args: Any, **kwargs: Any) -> Any:
        """Call an installed mgcv namespace function without evaluating source."""
        function = self._utils.getFromNamespace(name, "mgcv")
        return function(*args, **kwargs)

    def _to_r_vector(self, values: np.ndarray) -> Any:
        """Transfer a numeric vector directly, preserving double precision."""
        values = np.asarray(values)
        if values.ndim != 1:
            raise ValueError("R vector input must be one-dimensional.")
        if values.dtype.kind == "b":
            return self._ro.BoolVector(values)
        if values.dtype.kind in "iu":
            limit = np.iinfo(np.int32)
            if np.any(values <= limit.min) or np.any(values > limit.max):
                raise ValueError("R integer input exceeds its non-missing range.")
            return self._ro.IntVector(values)
        if values.dtype.kind != "f":
            raise TypeError("R numeric input must contain floating or integer values.")
        return self._ro.FloatVector(np.asarray(values, dtype=np.float64))

    def _to_r_matrix(self, values: np.ndarray) -> Any:
        """Transfer matrix dimensions and column-major numeric values to R."""
        values = np.asarray(values)
        if values.ndim != 2:
            raise ValueError("R matrix input must be two-dimensional.")
        return self._base.matrix(
            self._to_r_vector(values.ravel(order="F")),
            nrow=values.shape[0],
            ncol=values.shape[1],
        )

    def _to_r_dataframe(self, data: pd.DataFrame) -> Any:
        """Convert a pandas DataFrame to an R data.frame via rpy2."""
        from rpy2.robjects import numpy2ri, pandas2ri

        ro = self._ro
        with ro.conversion.localconverter(
            ro.default_converter + pandas2ri.converter + numpy2ri.converter
        ):
            return ro.conversion.py2rpy(data)

    def _fit_r_model(
        self,
        formula: str,
        r_df: Any,
        family: str,
        method: str,
    ) -> Any:
        """Fit a GAM in R and return the R model object."""
        ro = self._ro
        r_family = self._get_r_family_rpy2(family)
        return self._mgcv.gam(
            ro.Formula(formula),
            data=r_df,
            family=r_family,
            method=method,
        )

    def _get_r_family_rpy2(self, family: str) -> Any:
        """Map a Python family string to an R family function call."""
        family_funcs = {
            "gaussian": self._stats.gaussian,
            "binomial": self._stats.binomial,
            "poisson": self._stats.poisson,
            "gamma": self._stats.Gamma,
            # Non-canonical Gamma link (rpy2 path only) for testing the
            # observed-information REML log|H| (Finding 11).
            "gamma_log": lambda: self._stats.Gamma(link="log"),
            "nb": self._mgcv.nb,
            # Non-canonical NB links (rpy2 path only) for testing the signed
            # observed-information REML log|H| (Finding H4).
            "nb_identity": lambda: self._mgcv.nb(link="identity"),
            "nb_sqrt": lambda: self._mgcv.nb(link="sqrt"),
        }
        func = family_funcs.get(family)
        if func is None:
            raise ValueError(
                f"Unknown family: {family!r}. Supported: {list(family_funcs.keys())}"
            )
        return func()

    def _get_subprocess_family(self, family: str) -> str:
        """Map a Python family string to an R family expression for subprocess."""
        r_family = self._SUBPROCESS_FAMILY_MAP.get(family)
        if r_family is None:
            raise ValueError(
                f"Unknown family: {family!r}. "
                f"Supported: {list(self._SUBPROCESS_FAMILY_MAP.keys())}"
            )
        return r_family

    # ------------------------------------------------------------------ #
    #  fit_gam                                                            #
    # ------------------------------------------------------------------ #

    def _get_efs_family_rpy2(self, family: str, theta: float | None) -> Any:
        """Construct the exact pinned EFS family through package functions."""
        nb_links = {"nb": "log", "nb_identity": "identity", "nb_sqrt": "sqrt"}
        if theta is not None:
            if not np.isfinite(theta) or theta <= 0:
                raise ValueError("EFS NB theta must be finite and positive")
            if family not in nb_links:
                raise ValueError(
                    "EFS theta is supported only for family='nb' or its named link keys"
                )
            return self._mgcv.nb(theta=float(theta), link=nb_links[family])
        if family in nb_links:
            return self._mgcv.nb(link=nb_links[family])
        constructors = {
            "gaussian": self._stats.gaussian,
            "binomial": self._stats.binomial,
            "poisson": self._stats.poisson,
            "gamma": self._stats.Gamma,
        }
        links = {
            "identity": "identity",
            "log": "log",
            "logit": "logit",
            "inverse": "inverse",
            "probit": "probit",
            "cloglog": "cloglog",
            "sqrt": "sqrt",
            "inverse_squared": "1/mu^2",
        }
        for name, constructor in constructors.items():
            if family == name:
                return constructor()
            if family.startswith(name + "_"):
                link = family[len(name) + 1 :]
                if link in links:
                    return constructor(link=links[link])
        raise ValueError(f"Unknown EFS family: {family!r}")

    def fit_gam(
        self,
        formula: str,
        data: pd.DataFrame,
        family: str = "gaussian",
        method: str = "REML",
    ) -> dict[str, Any]:
        """Fit a GAM in R and return results as Python objects.

        Parameters
        ----------
        formula : str
            R-style model formula (e.g. "y ~ s(x)").
        data : pd.DataFrame
            Data frame with variables referenced in formula.
        family : str
            Distribution family name.
        method : str
            Smoothing parameter estimation method.

        Returns
        -------
        dict
            Keys: coefficients, fitted_values, smoothing_params, edf,
            deviance, null_deviance, scale, reml_scale, Vp, reml_score.
        """
        if self.mode == "rpy2":
            return self._fit_rpy2(formula, data, family, method)
        return self._fit_subprocess(formula, data, family, method)

    def fix_dependence(
        self,
        X1: np.ndarray,
        X2: np.ndarray,
        tol: float = np.finfo(float).eps ** 0.5,
        rank_def: int = 0,
    ) -> list[int] | None:
        """Call mgcv's unexported ``fixDependence`` reference routine.

        Returned indices are converted from R's one-based convention to
        Python's zero-based convention.
        """
        X1 = np.asarray(X1, dtype=np.float64)
        X2 = np.asarray(X2, dtype=np.float64)
        if X1.ndim != 2 or X2.ndim != 2:
            raise ValueError("X1 and X2 must both be two-dimensional arrays.")
        if X1.shape[0] != X2.shape[0]:
            raise ValueError("X1 and X2 must have the same number of rows.")

        return self._fix_dependence_rpy2(X1, X2, tol, rank_def)

    def _fix_dependence_rpy2(
        self, X1: np.ndarray, X2: np.ndarray, tol: float, rank_def: int
    ) -> list[int] | None:
        """Call ``mgcv:::fixDependence`` through rpy2."""
        self._require_rpy2()
        ind = self._call_internal(
            "fixDependence",
            self._to_r_matrix(X1),
            self._to_r_matrix(X2),
            tol=float(tol),
            **{"rank.def": int(rank_def)},
        )
        if ind is None or ind is self._ro.NULL or len(ind) == 0:
            return None
        return [int(index) - 1 for index in ind]

    # ------------------------------------------------------------------ #
    #  pinned extended Fellner--Schall oracle                            #
    # ------------------------------------------------------------------ #

    def source_qr_update(
        self,
        design: np.ndarray,
        response: np.ndarray,
        split_rows: int,
        penalty_root: np.ndarray,
    ) -> dict[str, np.ndarray | float]:
        """Replay pinned ``qr_update`` batches and a penalty-root append."""
        self._require_rpy2()
        X = np.asarray(design, dtype=np.float64)
        y = np.asarray(response, dtype=np.float64)
        root = np.asarray(penalty_root, dtype=np.float64)
        if (
            X.ndim != 2
            or y.shape != (len(X),)
            or root.ndim != 2
            or root.shape[1] != X.shape[1]
            or not 0 < split_rows < len(X)
        ):
            raise ValueError("QR update oracle dimensions or split are invalid")
        first = self._call_internal(
            "qr_update",
            self._to_r_matrix(X[:split_rows]),
            self._to_r_vector(y[:split_rows]),
        )
        second = self._call_internal(
            "qr_update",
            self._to_r_matrix(X[split_rows:]),
            self._to_r_vector(y[split_rows:]),
            first.rx2("R"),
            first.rx2("f"),
            first.rx2("y.norm2"),
        )
        penalized = self._call_internal(
            "qr_update",
            self._to_r_matrix(root),
            self._to_r_vector(np.zeros(root.shape[0])),
            second.rx2("R"),
            second.rx2("f"),
            second.rx2("y.norm2"),
        )
        factor = penalized.rx2("R")
        crossprod = self._base.crossprod
        normal = crossprod(factor)
        rhs = crossprod(factor, penalized.rx2("f"))
        coefficients = self._base.solve(normal, rhs)
        return {
            "R": np.asarray(second.rx2("R"), dtype=np.float64).copy(),
            "f": np.asarray(second.rx2("f"), dtype=np.float64).copy(),
            "y_norm2": float(second.rx2("y.norm2")[0]),
            "penalized_coefficients": np.asarray(coefficients, dtype=np.float64)
            .ravel()
            .copy(),
        }

    def source_triangular_rank(self, matrix: np.ndarray, tolerance: float) -> int:
        """Call pinned ``Rrank`` on a caller-supplied triangular matrix."""
        self._require_rpy2()
        triangular = np.asarray(matrix, dtype=np.float64)
        if triangular.ndim != 2 or not np.isfinite(tolerance) or tolerance <= 0:
            raise ValueError("Rrank oracle requires a matrix and positive tolerance")
        return int(
            self._call_internal(
                "Rrank", self._to_r_matrix(triangular), tol=float(tolerance)
            )[0]
        )

    def source_weighted_stream_reml_fit(
        self,
        formula: str,
        data: pd.DataFrame,
        family_name: str,
        weights: np.ndarray,
        offset: np.ndarray,
        *,
        newton_tolerance: float,
    ) -> tuple[float, float, np.ndarray]:
        """Fit the installed regular mgcv family with explicit REML controls."""
        from rpy2.robjects import ListVector

        self._require_rpy2()
        constructors = {
            "poisson": self._stats.poisson,
            "binomial": self._stats.binomial,
        }
        if family_name not in constructors:
            raise ValueError("Stream REML oracle supports Poisson or Binomial")
        w = np.asarray(weights, dtype=np.float64)
        off = np.asarray(offset, dtype=np.float64)
        if w.shape != (len(data),) or off.shape != w.shape:
            raise ValueError("Stream REML oracle weights and offset must align")
        control = self._mgcv.gam_control(
            newton=ListVector(
                [("conv.tol", self._to_r_vector(np.array([newton_tolerance])))]
            )
        )
        fit = self._mgcv.gam(
            self._ro.Formula(formula),
            data=self._to_r_dataframe(data),
            weights=self._to_r_vector(w),
            offset=self._to_r_vector(off),
            family=constructors[family_name](),
            method="REML",
            control=control,
        )
        return (
            float(fit.rx2("gcv.ubre")[0]),
            float(fit.rx2("deviance")[0]),
            np.asarray(fit.rx2("fitted.values"), dtype=np.float64).copy(),
        )

    def fit_efs(
        self,
        formula: str,
        data: pd.DataFrame,
        family: str = "gaussian",
        *,
        weights: str | None = None,
        offset: str | None = None,
        controls: dict[str, float] | None = None,
        initial_smoothing: np.ndarray | None = None,
        initial_scale: float | None = None,
        null_coef: bool = False,
        scale: float = -1.0,
        theta: float | None = None,
    ) -> dict[str, Any]:
        """Fit pinned mgcv ``optimizer='efs'`` as an oracle-only bridge call.

        This calls the pinned package in-process with direct rpy2 objects.
        """
        self._require_pinned_efs_versions()
        return self._fit_efs_rpy2(
            formula,
            data,
            family,
            weights,
            offset,
            controls,
            initial_smoothing,
            initial_scale,
            null_coef,
            scale,
            theta,
            skip_offset_null_deviance=False,
        )

    def nb_saturated_likelihood_derivatives(
        self, y: np.ndarray, weight: np.ndarray, theta: float
    ) -> np.ndarray:
        """Return pinned ``nb()$ls`` contractions in log-theta coordinates."""
        self._require_rpy2()
        result = self._mgcv.nb().rx2("ls")(
            self._to_r_vector(y),
            self._to_r_vector(weight),
            self._base.log(self._to_r_vector([theta])),
            1,
        )
        return np.asarray(
            [
                result.rx2("ls")[0],
                result.rx2("lsth1")[0],
                result.rx2("lsth2")[0],
            ],
            dtype=np.float64,
        )

    def regular_first_iteration_source(
        self, family_name: str, link: str, y: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Capture pinned ``gam.fit3`` W/z before its first PLS C call."""
        from rpy2.rinterface_lib.embedded import RRuntimeError

        from tests.r_ast import find_call_paths, instrument_function

        self._require_rpy2()
        constructors = {
            "gaussian": self._stats.gaussian,
            "binomial": self._stats.binomial,
            "poisson": self._stats.poisson,
            "gamma": self._stats.Gamma,
        }
        if family_name not in constructors:
            raise ValueError("unsupported pinned first-iteration family")
        family = self._call_internal(
            "fix.family.var",
            self._call_internal(
                "fix.family.link", constructors[family_name](link=link)
            ),
        )
        source = self._utils.getFromNamespace("gam.fit3", "mgcv")
        calls = find_call_paths(source, ".C", required_symbols=("C_pls_fit1",))
        if len(calls) != 2:
            raise RBridgeError("Pinned gam.fit3 first PLS call anchors changed")
        captured: dict[str, np.ndarray] = {}

        def collect(eta: Any, mu: Any, weight: Any, response: Any) -> None:
            captured.update(
                eta=np.asarray(eta).copy(),
                mu=np.asarray(mu).copy(),
                weight=np.asarray(weight).copy(),
                z=np.asarray(response).copy(),
            )

        fit_function = instrument_function(
            source,
            path=calls[0],
            expected_head=".C",
            capture_symbols=("eta", "mu", "w", "z"),
            callback=collect,
        )
        try:
            fit_function(
                x=self._base.diag(3),
                y=self._to_r_vector(y),
                sp=self._ro.FloatVector([]),
                Eb=self._to_r_matrix(np.zeros((3, 3))),
                UrS=self._ro.baseenv["list"](),
                weights=self._to_r_vector([1.0, 0.8, 1.3]),
                offset=self._to_r_vector([0.1, -0.1, 0.05]),
                U1=self._base.diag(3),
                Mp=0,
                family=family,
                control=self._mgcv.gam_control(maxit=1),
                deriv=0,
                scale=1,
                scoreType="GCV.Cp",
                **{"null.coef": self._to_r_vector([1.0, 1.0, 1.0])},
            )
        except RRuntimeError:
            if not captured:
                raise
        if set(captured) != {"eta", "mu", "weight", "z"}:
            raise RBridgeError("Pinned gam.fit3 did not reach its first PLS call")
        return captured

    def family_constructor_acceptance(
        self, links: tuple[str, ...]
    ) -> dict[tuple[str, str], bool]:
        """Ask pinned R constructors which named family/link pairs they accept."""
        from rpy2.rinterface_lib.embedded import RRuntimeError

        self._require_rpy2()
        constructors = {
            "gaussian": self._stats.gaussian,
            "binomial": self._stats.binomial,
            "poisson": self._stats.poisson,
            "gamma": self._stats.Gamma,
            "nb": self._mgcv.nb,
        }
        accepted: dict[tuple[str, str], bool] = {}
        for name, constructor in constructors.items():
            for link in links:
                try:
                    if name == "nb":
                        constructor(theta=1.2, link=link)
                    else:
                        constructor(link=link)
                except RRuntimeError:
                    accepted[(name, link)] = False
                else:
                    accepted[(name, link)] = True
        return accepted

    def efs_diagnostics(
        self,
        formula: str,
        data: pd.DataFrame,
        family: str = "gaussian",
        *,
        weights: str | None = None,
        offset: str | None = None,
        controls: dict[str, float] | None = None,
        initial_smoothing: np.ndarray | None = None,
        initial_scale: float | None = None,
        null_coef: bool = False,
        scale: float = -1.0,
        theta: float | None = None,
        initial_log_theta: float | None = None,
        initial_beta: np.ndarray | None = None,
        beta_old_init: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Return real per-refit EFS statistics from a private source copy."""
        self._require_pinned_efs_versions()
        return self._efs_diagnostics_rpy2(
            formula,
            data,
            family,
            weights,
            offset,
            controls,
            initial_smoothing,
            initial_scale,
            null_coef,
            scale,
            theta,
            initial_log_theta,
            initial_beta,
            beta_old_init,
        )

    def efs_conditional_theta_reference(
        self,
        y: np.ndarray,
        eta: np.ndarray,
        weights: np.ndarray,
        start: float,
        *,
        mu: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Trace pinned ``estimate.theta`` and its own nested likelihood closure."""
        self._require_rpy2()
        from tests.r_ast import clone_function, find_call_paths, instrument_function

        r_y = self._to_r_vector(np.asarray(y, dtype=np.float64))
        r_weights = self._to_r_vector(np.asarray(weights, dtype=np.float64))
        r_mu = (
            self._base.exp(self._to_r_vector(np.asarray(eta, dtype=np.float64)))
            if mu is None
            else self._to_r_vector(np.asarray(mu, dtype=np.float64))
        )
        r_start = self._to_r_vector(np.asarray([start], dtype=np.float64))
        family = self._mgcv.nb(theta=self._ro.r["-"](self._base.exp(float(start))))
        source = self._utils.getFromNamespace("estimate.theta", "mgcv")
        body = self._ro.r["body"](source)
        nlogl_assignment = body[2]
        if (
            str(nlogl_assignment[0]) != "<-"
            or str(nlogl_assignment[1]) != "nlogl"
            or str(nlogl_assignment[2][0]) != "function"
        ):
            raise RBridgeError("Pinned estimate.theta likelihood anchor changed")
        private = self._ro.r["new.env"](parent=self._ro.r["getNamespace"]("mgcv"))
        nlogl = self._ro.r["eval"](nlogl_assignment[2], envir=private)
        trace: list[np.ndarray] = []

        def record_theta(theta: Any) -> None:
            trace.append(np.array(theta, dtype=np.float64, copy=True))

        accepted_path = (12, 2, 3, 14)
        if accepted_path not in find_call_paths(
            source, "<-", required_symbols=("theta", "step")
        ):
            raise RBridgeError("Pinned estimate.theta accepted-step anchor changed")
        traced = instrument_function(
            clone_function(source, environment=private),
            path=accepted_path,
            expected_head="<-",
            capture_symbols=("theta",),
            callback=record_theta,
            when="after",
        )

        def state(theta: Any) -> np.ndarray:
            value = nlogl(theta, family, r_y, r_mu, scale=1, wt=r_weights, deriv=2)
            return np.asarray(
                [
                    float(np.asarray(value.rx2("nll"))[0]),
                    float(np.asarray(value.rx2("g"))[0]),
                    float(np.asarray(value.rx2("H")).ravel()[0]),
                ],
                dtype=np.float64,
            )

        initial = state(r_start)
        final_theta = traced(r_start, family, r_y, r_mu, scale=1, wt=r_weights)
        final = np.concatenate(
            (np.asarray(final_theta, dtype=np.float64), state(final_theta))
        )
        trace_values = np.asarray(trace, dtype=np.float64).reshape(-1)
        trace_nll = np.asarray(
            [
                float(
                    np.asarray(
                        nlogl(
                            self._to_r_vector(np.asarray([theta])),
                            family,
                            r_y,
                            r_mu,
                            scale=1,
                            wt=r_weights,
                            deriv=0,
                        ).rx2("nll")
                    )[0]
                )
                for theta in trace_values
            ],
            dtype=np.float64,
        )
        return {
            "initial": initial,
            "trace": trace_values,
            "trace_nll": trace_nll,
            "final": final,
        }

    def efs_nb_log_deviance_derivatives(
        self, y: np.ndarray, eta: np.ndarray, weights: np.ndarray, log_theta: float
    ) -> dict[str, np.ndarray | float]:
        """Reduce pinned NB ``dev.resids`` and ``Dd`` in R coordinates."""
        self._require_rpy2()
        r_y = self._to_r_vector(np.asarray(y, dtype=np.float64))
        r_eta = self._to_r_vector(np.asarray(eta, dtype=np.float64))
        r_weights = self._to_r_vector(np.asarray(weights, dtype=np.float64))
        r_theta = self._to_r_vector(np.asarray([log_theta], dtype=np.float64))
        r_mu = self._base.exp(r_eta)
        family = self._mgcv.nb(theta=self._ro.r["-"](self._base.exp(r_theta)))
        derivatives = family.rx2("Dd")(r_y, r_mu, r_theta, wt=r_weights, level=2)
        multiply = self._ro.r["*"]
        add = self._ro.r["+"]
        square = self._ro.r["^"]
        value = self._base.sum(family.rx2("dev.resids")(r_y, r_mu, r_weights, r_theta))
        return {
            "value": float(np.asarray(value)[0]),
            "d_eta": np.asarray(
                multiply(derivatives.rx2("Dmu"), r_mu), dtype=np.float64
            ).copy(),
            "h_eta": np.asarray(
                add(
                    multiply(derivatives.rx2("Dmu2"), square(r_mu, 2)),
                    multiply(derivatives.rx2("Dmu"), r_mu),
                ),
                dtype=np.float64,
            ).copy(),
            "mixed": np.asarray(
                multiply(derivatives.rx2("Dmuth"), r_mu), dtype=np.float64
            ).copy(),
            "d_theta": float(np.asarray(self._base.sum(derivatives.rx2("Dth")))[0]),
            "h_theta": float(np.asarray(self._base.sum(derivatives.rx2("Dth2")))[0]),
        }

    def efs_nb_deviance_is_finite(
        self, y: float, eta: float, theta: float, weight: float = 1.0
    ) -> bool:
        """Check finiteness of the literal pinned NB deviance reduction."""
        self._require_rpy2()
        r_theta = self._to_r_vector(np.asarray([theta], dtype=np.float64))
        family = self._mgcv.nb(theta=self._ro.r["-"](r_theta))
        value = self._base.sum(
            family.rx2("dev.resids")(
                self._to_r_vector(np.asarray([y], dtype=np.float64)),
                self._base.exp(self._to_r_vector(np.asarray([eta], dtype=np.float64))),
                self._to_r_vector(np.asarray([weight], dtype=np.float64)),
                self._base.log(r_theta),
            )
        )
        return bool(self._base.is_finite(value)[0])

    def efs_nb_nonsaturated_inner_reference(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        offset: np.ndarray,
        penalty_diagonal: np.ndarray,
        log_theta: float,
        *,
        mp: int,
    ) -> dict[str, np.ndarray | float]:
        """Run pinned ``gam.fit4`` for one diagonal-penalty NB inner fit."""
        self._require_rpy2()
        from rpy2 import rinterface

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        offset = np.asarray(offset, dtype=np.float64)
        penalty_diagonal = np.asarray(penalty_diagonal, dtype=np.float64)
        n, p = X.shape
        if any(
            value.shape != (n,) for value in (y, weights, offset)
        ) or penalty_diagonal.shape != (p,):
            raise ValueError("Nonsaturated NB oracle arrays have incompatible shapes")
        r_x = self._to_r_matrix(X)
        r_y = self._to_r_vector(y)
        r_weights = self._to_r_vector(weights)
        r_offset = self._to_r_vector(offset)
        r_penalty = self._to_r_vector(penalty_diagonal)
        bracket = self._ro.r["["]
        positive = self._base.which(self._ro.r[">"](r_penalty, 0))
        null = self._base.which(self._ro.r["=="](r_penalty, 0))
        indices = self._base.c(positive, null)
        r_u1 = bracket(self._base.diag(p), rinterface.MissingArg, indices, drop=False)
        compact_root = self._base.diag(
            self._base.sqrt(bracket(r_penalty, positive)), nrow=len(positive)
        )
        r_theta = self._to_r_vector(np.asarray([log_theta], dtype=np.float64))
        r_family = self._mgcv.nb(theta=self._ro.r["-"](self._base.exp(r_theta)))
        for name in ("fix.family.link", "fix.family.var", "fix.family.ls"):
            r_family = self._call_internal(name, r_family)
        fit = self._call_internal(
            "gam.fit4",
            x=r_x,
            y=r_y,
            sp=self._base.c(r_theta, 0.0),
            Eb=self._base.diag(self._base.sqrt(r_penalty)),
            UrS=rinterface.ListSexpVector([compact_root]),
            weights=r_weights,
            offset=r_offset,
            U1=r_u1,
            Mp=mp,
            family=r_family,
            control=self._mgcv.gam_control(epsilon=1e-7, maxit=100),
            deriv=0,
            scoreType="EFS",
            scale=1,
            start=self._base.rep(0.0, p),
            **{"null.coef": self._base.rep(0.0, p)},
        )
        theta_out = r_family.rx2("getTheta")(False)
        r_eta = self._ro.r["+"](
            self._base.drop(self._ro.r["%*%"](r_x, fit.rx2("coefficients"))),
            r_offset,
        )
        r_mu = self._base.exp(r_eta)
        deviance = self._base.sum(
            r_family.rx2("dev.resids")(r_y, r_mu, r_weights, theta_out)
        )
        return {
            "coefficients": np.asarray(
                fit.rx2("coefficients"), dtype=np.float64
            ).copy(),
            "log_theta": float(np.asarray(theta_out)[0]),
            "deviance": float(np.asarray(deviance)[0]),
            "fit_deviance": float(np.asarray(fit.rx2("deviance"))[0]),
        }

    def efs_nb_inner_trace_reference(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        offset: np.ndarray,
        penalty: float,
        log_theta: float,
        *,
        start: np.ndarray,
        null_coef: np.ndarray,
        maxit: int,
    ) -> dict[str, Any]:
        """Trace pinned ``gam.fit4`` through private, validated R AST hooks."""
        self._require_rpy2()
        from rpy2 import rinterface

        from tests.r_ast import clone_function, find_call_paths, instrument_function

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        offset = np.asarray(offset, dtype=np.float64)
        start = np.asarray(start, dtype=np.float64)
        null_coef = np.asarray(null_coef, dtype=np.float64)
        n, p = X.shape
        if any(a.shape != (n,) for a in (y, weights, offset)) or any(
            a.shape != (p,) for a in (start, null_coef)
        ):
            raise ValueError("R EFS inner oracle starts and rows must align with X")
        r_x = self._to_r_matrix(X)
        r_y = self._to_r_vector(y)
        r_weights = self._to_r_vector(weights)
        r_offset = self._to_r_vector(offset)
        r_theta = self._to_r_vector(np.asarray([log_theta], dtype=np.float64))
        family = self._mgcv.nb(theta=self._ro.r["-"](self._base.exp(r_theta)))
        for name in ("fix.family.link", "fix.family.var", "fix.family.ls"):
            family = self._call_internal(name, family)
        trace: dict[str, list[Any]] = {
            name: []
            for name in (
                "pre",
                "post",
                "theta",
                "beta",
                "raw_beta",
                "raw_eta",
                "raw_deviance",
                "nonfinite",
                "domain",
                "divergence",
                "retained",
                "initial_factors",
                "use_wy",
            )
        }

        def copy(value: Any) -> np.ndarray:
            return np.array(value, dtype=np.float64, copy=True).ravel(order="F")

        def scalar(value: Any) -> float:
            return float(np.asarray(value, dtype=np.float64).ravel()[0])

        def record_raw(beta: Any, eta: Any, dev: Any) -> None:
            trace["raw_beta"].append(copy(beta))
            trace["raw_eta"].append(copy(eta))
            trace["raw_deviance"].append(scalar(dev))

        def record_pre(pdev: Any, _theta: Any, beta: Any) -> None:
            trace["pre"].append(scalar(pdev))
            trace["beta"].append(copy(beta)[:p])

        source = self._utils.getFromNamespace("gam.fit4", "mgcv")
        private = self._ro.r["new.env"](parent=self._ro.r["getNamespace"]("mgcv"))
        fit4 = clone_function(source, environment=private)
        anchors = (
            (
                (25,),
                "<-",
                ("coefold", "null.coef"),
                ("start",),
                lambda value: trace["retained"].append(
                    not bool(self._base.is_null(value)[0])
                ),
                "before",
            ),
            (
                (35,),
                "<-",
                ("good", "z", "w"),
                ("w", "wz", "z"),
                lambda w, wz, z: trace["initial_factors"].append(
                    np.column_stack((copy(w), copy(wz), copy(z)))
                ),
                "before",
            ),
            (
                (3, 2, 2),
                "<-",
                ("theta", "sp"),
                ("theta",),
                lambda value: trace["theta"].append(scalar(value)),
                "after",
            ),
            (
                (37, 3, 22, 2, 2, 2),
                "<-",
                ("theta", "estimate.theta"),
                ("theta",),
                lambda value: trace["theta"].append(scalar(value)),
                "after",
            ),
            (
                (37, 3, 16),
                "if",
                ("is.finite", "dev"),
                ("start", "eta", "dev"),
                record_raw,
                "before",
            ),
            (
                (37, 3, 22),
                "if",
                ("scoreType", "n.theta"),
                ("pdev", "theta", "start"),
                record_pre,
                "before",
            ),
            (
                (37, 3, 29),
                "if",
                ("scoreType", "n.theta"),
                ("pdev", "theta", "start"),
                record_pre,
                "before",
            ),
            (
                (37, 3, 29, 2, 2),
                "<-",
                ("old.pdev", "pdev"),
                ("pdev",),
                lambda value: trace["post"].append(scalar(value)),
                "after",
            ),
            (
                (37, 3, 16, 2, 3, 2, 3),
                "<-",
                ("start", "coefold"),
                ("iter",),
                lambda value: trace["nonfinite"].append(int(value[0])),
                "before",
            ),
            (
                (37, 3, 17, 2, 2, 2, 3),
                "<-",
                ("start", "coefold"),
                ("iter",),
                lambda value: trace["domain"].append(int(value[0])),
                "before",
            ),
            (
                (37, 3, 21, 2, 3, 2, 3),
                "<-",
                ("start", "coefold"),
                ("iter",),
                lambda value: trace["divergence"].append(int(value[0])),
                "before",
            ),
            (
                (37, 3, 3, 2, 1),
                "<-",
                ("use.wy",),
                (),
                lambda: trace["use_wy"].append(True),
                "after",
            ),
            (
                (37, 3, 9, 2, 8, 2, 1),
                "<-",
                ("use.wy",),
                (),
                lambda: trace["use_wy"].append(True),
                "after",
            ),
        )
        for path, head, required, captures, callback, when in sorted(
            anchors, key=lambda item: len(item[0]), reverse=True
        ):
            if path not in find_call_paths(source, head, required_symbols=required):
                raise RBridgeError(
                    f"Pinned gam.fit4 inner trace anchor changed: {path!r}"
                )
            fit4 = instrument_function(
                fit4,
                path=path,
                expected_head=head,
                capture_symbols=captures,
                callback=callback,
                when=when,
            )
        fit = fit4(
            x=r_x,
            y=r_y,
            sp=self._base.c(r_theta, self._base.log(float(penalty))),
            Eb=self._base.diag(p),
            UrS=rinterface.ListSexpVector([self._base.diag(p)]),
            weights=r_weights,
            offset=r_offset,
            U1=self._base.diag(p),
            Mp=0,
            family=family,
            control=self._mgcv.gam_control(epsilon=1e-7, maxit=maxit),
            deriv=0,
            scoreType="EFS",
            scale=1,
            start=self._to_r_vector(start),
            **{"null.coef": self._to_r_vector(null_coef)},
        )
        final_theta = self._utils.tail(self._to_r_vector(np.asarray(trace["theta"])), 1)
        mu = fit.rx2("fitted.values")
        deviance = self._base.sum(
            family.rx2("dev.resids")(r_y, mu, r_weights, final_theta)
        )
        derivative = family.rx2("Dd")(r_y, mu, final_theta, wt=r_weights, level=2)
        saturated = family.rx2("ls")(r_y, w=r_weights, theta=final_theta, scale=1)
        subtract, divide = self._ro.r["-"], self._ro.r["/"]
        nll = subtract(divide(deviance, 2), saturated.rx2("ls"))
        gradient = subtract(
            divide(self._base.sum(derivative.rx2("Dth")), 2), saturated.rx2("lsth1")
        )
        threshold = self._ro.r["*"](1e-7, self._ro.r["+"](self._base.abs(nll), 1))
        halvings = {
            "retained": bool(trace["retained"][0]),
            **{
                name: len(trace[name]) for name in ("nonfinite", "domain", "divergence")
            },
            **{
                f"first_{name}": sum(v == 1 for v in trace[name])
                for name in ("nonfinite", "domain", "divergence")
            },
        }
        return {
            "coefficients": copy(fit.rx2("coefficients")),
            "pre": np.asarray(trace["pre"], dtype=np.float64),
            "post": np.asarray(trace["post"], dtype=np.float64),
            "theta": np.asarray(trace["theta"], dtype=np.float64),
            "beta": np.vstack(trace["beta"]),
            "raw_beta": np.vstack(trace["raw_beta"]),
            "raw_eta": np.vstack(trace["raw_eta"]),
            "raw_deviance": np.asarray(trace["raw_deviance"], dtype=np.float64),
            "halvings": halvings,
            "initial_factors": trace["initial_factors"][0],
            "use_wy": np.asarray(trace["use_wy"], dtype=bool),
            "theta_state": {
                "log_theta": scalar(final_theta),
                "nll": scalar(nll),
                "gradient": scalar(gradient),
                "threshold": scalar(threshold),
                "deviance": scalar(deviance),
                "eta_min": scalar(self._base.min(fit.rx2("linear.predictors"))),
                "eta_max": scalar(self._base.max(fit.rx2("linear.predictors"))),
            },
        }

    @staticmethod
    def _require_pinned_efs_versions() -> None:
        ok, reason = RBridge.check_versions()
        if not ok:
            raise RBridgeError(f"Pinned EFS oracle unavailable: {reason}")

    def efs_statistics_algebra(
        self,
        log_smoothing: np.ndarray,
        determinant_roots: list[np.ndarray],
        covariance_roots: list[np.ndarray],
        coefficients: np.ndarray,
        fisher_factor: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Evaluate pinned ``gam.reparam`` and covariance-root contractions.

        This deliberately narrow oracle accepts only already prepared fitting-
        coordinate roots and a lower Fisher Cholesky factor.  It is not an
        arbitrary R evaluation interface: its sole purpose is matched-state
        EFS d/t/q parity.
        """
        self._require_pinned_efs_versions()
        rho = np.asarray(log_smoothing, dtype=np.float64)
        beta = np.asarray(coefficients, dtype=np.float64)
        factor = np.asarray(fisher_factor, dtype=np.float64)
        roots = [np.asarray(root, dtype=np.float64) for root in covariance_roots]
        det_roots = [np.asarray(root, dtype=np.float64) for root in determinant_roots]
        if len(rho) != len(roots) or len(rho) != len(det_roots):
            raise ValueError("EFS roots must have one entry per smoothing parameter")
        if beta.ndim != 1 or factor.shape != (len(beta), len(beta)):
            raise ValueError("coefficients and Fisher factor have incompatible shapes")
        if not all(root.ndim == 2 and root.shape[0] == len(beta) for root in roots):
            raise ValueError("covariance roots must be 2-D with coefficient rows")
        if not all(
            root.ndim == 2 and root.shape[0] == det_roots[0].shape[0]
            for root in det_roots
        ):
            raise ValueError("determinant roots must share their range rows")
        if not all(
            np.all(np.isfinite(value))
            for value in [rho, beta, factor, *roots, *det_roots]
        ):
            raise ValueError("EFS algebra oracle requires finite inputs")
        self._require_rpy2()
        from rpy2 import rinterface

        r_determinant_roots = rinterface.ListSexpVector(
            [self._to_r_matrix(root) for root in det_roots]
        )
        r_covariance_roots = [self._to_r_matrix(root) for root in roots]
        r_beta = self._to_r_vector(beta)
        r_factor = self._to_r_matrix(factor)
        reparam = self._call_internal(
            "gam.reparam", r_determinant_roots, self._to_r_vector(rho), deriv=1
        )
        square = self._ro.r["^"]

        def r_square_sum(value: Any) -> float:
            return float(np.asarray(self._base.sum(square(value, 2)))[0])

        return {
            "d": np.asarray(reparam.rx2("det1"), dtype=np.float64),
            "t": np.asarray(
                [
                    r_square_sum(self._base.forwardsolve(r_factor, root))
                    for root in r_covariance_roots
                ],
                dtype=np.float64,
            ),
            "q": np.asarray(
                [
                    r_square_sum(self._base.crossprod(root, r_beta))
                    for root in r_covariance_roots
                ],
                dtype=np.float64,
            ),
        }

    def efs_scripted_controller_reference(
        self,
        initial_rho: np.ndarray,
        log_ratio: np.ndarray,
        fit_scores: np.ndarray,
        fit_deviances: np.ndarray,
    ) -> dict[str, Any]:
        """Run pinned ``efsudr`` with a private scripted coefficient fitter."""
        self._require_rpy2()
        from rpy2 import rinterface

        from tests.r_ast import clone_function, make_r_callback

        initial_rho = np.asarray(initial_rho, dtype=np.float64)
        log_ratio = np.asarray(log_ratio, dtype=np.float64)
        fit_scores = np.asarray(fit_scores, dtype=np.float64)
        fit_deviances = np.asarray(fit_deviances, dtype=np.float64)
        if initial_rho.ndim != 1 or log_ratio.shape != initial_rho.shape:
            raise ValueError("initial_rho and log_ratio must be equal-length vectors")
        if (
            fit_scores.ndim != 1
            or fit_scores.size == 0
            or fit_deviances.shape != fit_scores.shape
        ):
            raise ValueError(
                "scripted EFS scores and deviances must be nonempty equal-length vectors"
            )
        n_parameters = len(initial_rho)
        private = self._ro.r["new.env"](parent=self._ro.r["getNamespace"]("mgcv"))
        proposals: list[np.ndarray] = []
        r_add = self._ro.r["+"]
        r_ratio = self._to_r_vector(log_ratio)

        def scripted_fit(**arguments: Any) -> Any:
            packed = arguments["sp"]
            proposals.append(np.array(packed, dtype=np.float64, copy=True))
            index = min(len(proposals) - 1, len(fit_scores) - 1)
            return self._ro.ListVector(
                {
                    "coefficients": self._base.rep(1.0, len(packed)),
                    "rV": self._base.matrix(0.0, nrow=len(packed), ncol=len(packed)),
                    "ldetS1": self._base.exp(r_add(packed, r_ratio)),
                    "scale": self._ro.FloatVector([1.0]),
                    "REML": self._ro.FloatVector([fit_scores[index]]),
                    "dev": self._ro.FloatVector([fit_deviances[index]]),
                }
            )

        fit_callback = make_r_callback(scripted_fit)
        private["gam.fit3"] = fit_callback
        efsudr = clone_function(
            self._utils.getFromNamespace("efsudr", "mgcv"), environment=private
        )
        identity = self._base.diag(n_parameters)
        bracket = self._ro.r["["]
        roots = rinterface.ListSexpVector(
            [
                bracket(identity, rinterface.MissingArg, index + 1, drop=False)
                for index in range(n_parameters)
            ]
        )
        try:
            output = efsudr(
                x=self._base.matrix(1.0, nrow=3, ncol=n_parameters),
                y=self._base.rep(1.0, 3),
                lsp=self._to_r_vector(initial_rho),
                Eb=identity,
                UrS=roots,
                weights=self._base.rep(1.0, 3),
                family=self._stats.poisson(),
                U1=identity,
                Mp=0,
                control=self._mgcv.gam_control(),
            )
        finally:
            private["gam.fit3"] = self._ro.NULL
        outer = output.rx2("outer.info")
        return {
            "proposals": np.vstack(proposals),
            "accepted_sp": np.asarray(output.rx2("sp"), dtype=np.float64).copy(),
            "final_score": float(np.asarray(output.rx2("REML"))[0]),
            "score_history": np.asarray(
                outer.rx2("score.hist"), dtype=np.float64
            ).copy(),
            "iter": int(np.asarray(output.rx2("iter"))[0]),
            "convergence": str(outer.rx2("conv")[0]),
        }

    @staticmethod
    def _validate_efs_inputs(
        data: pd.DataFrame,
        weights: str | None,
        offset: str | None,
        controls: dict[str, float] | None,
        initial_smoothing: np.ndarray | None,
    ) -> tuple[dict[str, float], np.ndarray | None]:
        for column_name, label in ((weights, "weights"), (offset, "offset")):
            if column_name is not None and column_name not in data.columns:
                raise ValueError(
                    f"{label} column {column_name!r} is not present in data"
                )
        allowed = {"efs_lspmax", "efs_tol"}
        resolved = {"efs_lspmax": 15.0, "efs_tol": 0.1}
        if controls is not None:
            unknown = set(controls) - allowed
            if unknown:
                raise ValueError(f"Unsupported EFS controls: {sorted(unknown)}")
            resolved.update({key: float(value) for key, value in controls.items()})
        if (
            not all(np.isfinite(value) for value in resolved.values())
            or resolved["efs_tol"] <= 0
        ):
            raise ValueError("EFS controls must be finite and efs_tol must be positive")
        if initial_smoothing is None:
            return resolved, None
        initial = np.asarray(initial_smoothing, dtype=np.float64)
        if (
            initial.ndim != 1
            or not np.all(np.isfinite(initial))
            or np.any(initial <= 0)
        ):
            raise ValueError("initial_smoothing must be a finite positive vector")
        return resolved, initial

    @staticmethod
    def _efs_provenance(data: pd.DataFrame) -> dict[str, str]:
        """Return the source/version/data identity recorded with EFS oracle output."""
        payload = data.to_csv(index=False, float_format="%.17g", lineterminator="\n")
        return {
            "r_version": _REQUIRED_R_VERSION,
            "mgcv_version": _REQUIRED_MGCV_VERSION,
            "source_commit": _PINNED_MGCV_SOURCE_COMMIT,
            "data_hash": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        }

    def _fit_efs_rpy2(
        self,
        formula: str,
        data: pd.DataFrame,
        family: str,
        weights: str | None,
        offset: str | None,
        controls: dict[str, float] | None,
        initial_smoothing: np.ndarray | None,
        initial_scale: float | None,
        null_coef: bool,
        scale: float,
        theta: float | None,
        *,
        skip_offset_null_deviance: bool,
    ) -> dict[str, Any]:
        """Call pinned ``gam`` with EFS controls and optional private AST gate."""
        self._require_rpy2()
        from tests.r_ast import call, clone_function, find_call_paths, replace_call

        resolved, initial = self._validate_efs_inputs(
            data, weights, offset, controls, initial_smoothing
        )
        if initial_scale is not None and (
            not np.isfinite(initial_scale) or initial_scale <= 0
        ):
            raise ValueError("EFS initial_scale must be finite and positive")
        r_data = self._to_r_dataframe(data)
        r_family = self._get_efs_family_rpy2(family, theta)
        r_formula = self._ro.Formula(formula)
        setup_args: dict[str, Any] = {}
        if weights is not None:
            setup_args["weights"] = self._to_r_vector(
                data[weights].to_numpy(dtype=np.float64)
            )
        if offset is not None:
            setup_args["offset"] = self._to_r_vector(
                data[offset].to_numpy(dtype=np.float64)
            )
        r_gam = self._mgcv.gam
        if skip_offset_null_deviance:
            namespace = self._ro.r["getNamespace"]("mgcv")
            private_env = self._ro.r["new.env"](parent=namespace)
            original_estimate = self._utils.getFromNamespace("estimate.gam", "mgcv")
            paths = find_call_paths(
                original_estimate, "if", required_symbols=("null.deviance", "glm")
            )
            if len(paths) != 1:
                raise RBridgeError("Pinned estimate.gam null-deviance anchor changed")
            node = self._ro.r["body"](original_estimate)
            for index in paths[0]:
                node = node[index]
            replacement = call(
                "if",
                call("&&", self._ro.BoolVector([False]), node[1]),
                node[2],
            )
            private_env["estimate.gam"] = replace_call(
                original_estimate,
                path=paths[0],
                expected_head="if",
                replacement=replacement,
            )
            r_gam = clone_function(
                self._utils.getFromNamespace("gam", "mgcv"), environment=private_env
            )
        fit_args: dict[str, Any] = {
            **setup_args,
            "method": "REML",
            "optimizer": "efs",
            "control": self._mgcv.gam_control(
                efs_lspmax=resolved["efs_lspmax"], efs_tol=resolved["efs_tol"]
            ),
            "scale": float(scale),
        }
        if initial is not None and (scale > 0 or initial_scale is not None):
            fit_args["in.out"] = self._ro.ListVector(
                {
                    "sp": self._to_r_vector(initial),
                    "scale": self._ro.FloatVector(
                        [1.0 if initial_scale is None else float(initial_scale)]
                    ),
                }
            )
        if null_coef:
            setup = r_gam(
                r_formula, data=r_data, family=r_family, fit=False, **setup_args
            )
            fit_args["null.coef"] = self._call_internal("get.null.coef", setup).rx2(
                "null.coef"
            )
        from rpy2.rinterface_lib.embedded import RRuntimeError

        try:
            model = r_gam(r_formula, data=r_data, family=r_family, **fit_args)
        except RRuntimeError as exc:
            raise RBridgeError(str(exc)) from exc
        result = self._extract_fit_results_rpy2(model)
        outer = model.rx2("outer.info")
        result.update(
            outer_iterations=int(np.asarray(outer.rx2("iter"))[0]),
            score_history=np.asarray(outer.rx2("score.hist"), dtype=np.float64).ravel(
                order="F"
            ),
            convergence=str(outer.rx2("conv")[0]),
            optimizer="efs",
            controls=resolved,
            provenance=self._efs_provenance(data),
        )
        if skip_offset_null_deviance:
            result["oracle_stage"] = "selected_before_offset_null_deviance"
        return result

    def _efs_diagnostics_rpy2(
        self,
        formula: str,
        data: pd.DataFrame,
        family: str,
        weights: str | None,
        offset: str | None,
        controls: dict[str, float] | None,
        initial_smoothing: np.ndarray | None,
        initial_scale: float | None,
        null_coef: bool,
        scale: float,
        theta: float | None,
        initial_log_theta: float | None,
        initial_beta: np.ndarray | None,
        beta_old_init: np.ndarray | None,
    ) -> dict[str, Any]:
        """Trace pinned EFS through private, object-level mgcv closures."""
        self._require_rpy2()
        from rpy2 import rinterface

        from tests.r_ast import (
            clone_function,
            find_call_paths,
            instrument_function,
            make_r_callback,
        )

        resolved, initial = self._validate_efs_inputs(
            data, weights, offset, controls, initial_smoothing
        )
        if initial_scale is not None and (
            not np.isfinite(initial_scale) or initial_scale <= 0
        ):
            raise ValueError("EFS initial_scale must be finite and positive")
        if initial_log_theta is not None and not np.isfinite(initial_log_theta):
            raise ValueError("EFS initial_log_theta must be finite")

        def optional_vector(value: np.ndarray | None, name: str) -> Any:
            if value is None:
                return self._ro.NULL
            vector = np.asarray(value, dtype=np.float64)
            if vector.ndim != 1 or vector.size == 0 or not np.all(np.isfinite(vector)):
                raise ValueError(f"EFS {name} must be a nonempty finite vector")
            return self._to_r_vector(vector)

        start = optional_vector(initial_beta, "initial_beta")
        old_beta = optional_vector(beta_old_init, "beta_old_init")
        r_data = self._to_r_dataframe(data)
        r_family = self._get_efs_family_rpy2(family, theta)
        setup_args: dict[str, Any] = {}
        if weights is not None:
            setup_args["weights"] = self._to_r_vector(
                data[weights].to_numpy(dtype=np.float64)
            )
        if offset is not None:
            setup_args["offset"] = self._to_r_vector(
                data[offset].to_numpy(dtype=np.float64)
            )
        setup = self._mgcv.gam(
            self._ro.Formula(formula),
            data=r_data,
            family=r_family,
            fit=False,
            **setup_args,
        )
        r_family = setup.rx2("family")
        for function in ("fix.family.link", "fix.family.var", "fix.family.ls"):
            r_family = self._call_internal(function, r_family)
        theta_count = r_family.rx2("n.theta")
        n_theta = (
            0
            if bool(self._base.is_null(theta_count)[0])
            else int(np.asarray(theta_count)[0])
        )
        if initial_log_theta is not None and n_theta > 0:
            r_family.rx2("putTheta")(
                self._to_r_vector(np.asarray([initial_log_theta], dtype=np.float64))
            )

        r_x = setup.rx2("X")
        n_coef = int(self._base.ncol(r_x)[0])
        roots = self._call_internal(
            "mini.roots",
            setup.rx2("S"),
            setup.rx2("off"),
            n_coef,
            setup.rx2("rank"),
        )
        penalty_space = self._call_internal(
            "totalPenaltySpace",
            setup.rx2("S"),
            setup.rx2("H"),
            setup.rx2("off"),
            n_coef,
        )
        y_range = penalty_space.rx2("Y")
        u1 = self._base.cbind(y_range, penalty_space.rx2("Z"))
        mp = int(self._base.ncol(penalty_space.rx2("Z"))[0])
        multiply = self._ro.r["%*%"]
        ur_s = rinterface.ListSexpVector(
            [multiply(self._base.t(y_range), root) for root in roots]
        )
        if initial is not None and (scale > 0 or initial_scale is not None):
            initial_sp = self._to_r_vector(initial)
        else:
            initial_sp = self._call_internal(
                "initial.spg",
                r_x,
                setup.rx2("y"),
                setup.rx2("w"),
                r_family,
                setup.rx2("S"),
                setup.rx2("rank"),
                setup.rx2("off"),
                offset=setup.rx2("offset"),
                E=penalty_space.rx2("E"),
            )
        lsp = self._base.log(initial_sp)
        if n_theta:
            lsp = self._base.c(r_family.rx2("getTheta")(), lsp)
        fit_scale = (
            1.0 if str(r_family.rx2("family")[0]) in {"poisson", "binomial"} else scale
        )
        if fit_scale <= 0:
            if initial_scale is None:
                null_fit = self._call_internal("get.null.coef", setup)
                initial_scale = float(
                    np.asarray(self._ro.r["/"](null_fit.rx2("null.scale"), 10))[0]
                )
            lsp = self._base.c(lsp, self._base.log(float(initial_scale)))
        if old_beta is self._ro.NULL and null_coef:
            old_beta = self._call_internal("get.null.coef", setup).rx2("null.coef")

        trace_log: list[pd.DataFrame] = []
        coefficient_log: list[np.ndarray] = []
        fitted_log: list[np.ndarray] = []
        start_log: list[np.ndarray] = []
        theta_log: list[np.ndarray] = []
        retained_log: list[bool] = []
        initial_state_log: list[pd.DataFrame] = []
        prefix_log: list[dict[str, Any]] = []
        multiplier_log: list[float] = []
        branch_log: list[str] = []
        private = self._ro.r["new.env"](parent=self._ro.r["getNamespace"]("mgcv"))

        def single(value: Any) -> float:
            return float(np.asarray(value, dtype=np.float64).ravel()[0])

        def copied(value: Any) -> np.ndarray:
            return np.array(value, dtype=np.float64, copy=True).ravel(order="F")

        def record_retained(value: Any) -> None:
            retained_log.append(not bool(self._base.is_null(value)[0]))

        def record_initial(
            eta: Any,
            mu: Any,
            mustart: Any,
            null_eta: Any,
            etaold: Any,
            current_offset: Any,
            current_theta: Any,
            response: Any,
        ) -> None:
            rows = len(response)
            initial_state_log.append(
                pd.DataFrame(
                    {
                        "call": np.full(rows, len(trace_log) + 1, dtype=np.int64),
                        "row": np.arange(1, rows + 1),
                        "retained": np.full(rows, retained_log[-1], dtype=bool),
                        "eta": copied(eta),
                        "mu": copied(mu),
                        "mustart": copied(mustart),
                        "null_eta": copied(null_eta),
                        "etaold": copied(etaold),
                        "offset": copied(current_offset),
                        "theta": np.full(rows, single(current_theta)),
                    }
                )
            )

        def record_prefix(iteration: Any, pdev: Any, old_pdev: Any) -> None:
            if int(np.asarray(iteration)[0]) != 1:
                return
            current = single(pdev)
            previous = single(old_pdev)
            r_minus = self._ro.r["-"]
            r_add = self._ro.r["+"]
            r_times = self._ro.r["*"]
            r_threshold = r_times(
                r_times(10, r_add(0.1, self._base.abs(old_pdev))),
                self._base.sqrt(self._ro.r[".Machine"].rx2("double.eps")),
            )
            diverging = self._ro.r[">"](r_minus(pdev, old_pdev), r_threshold)
            prefix_log.append(
                {
                    "call": len(trace_log) + 1,
                    "pdev": current,
                    "old_pdev": previous,
                    "diverging": bool(diverging[0]),
                }
            )

        fit4 = clone_function(
            self._utils.getFromNamespace("gam.fit4", "mgcv"), environment=private
        )
        for path, anchor, symbols, callback, when in (
            ((25,), ("coefold", "null.coef"), ("start",), record_retained, "before"),
            (
                (29,),
                ("mu", "linkinv"),
                (
                    "eta",
                    "mu",
                    "mustart",
                    "null.eta",
                    "etaold",
                    "offset",
                    "theta",
                    "y",
                ),
                record_initial,
                "after",
            ),
            (
                (37, 3, 18),
                ("pdev", "penalty"),
                ("iter", "pdev", "old.pdev"),
                record_prefix,
                "after",
            ),
        ):
            if path not in find_call_paths(fit4, "<-", required_symbols=anchor):
                raise RBridgeError(f"Pinned gam.fit4 trace anchor changed: {path!r}")
            fit4 = instrument_function(
                fit4,
                path=path,
                expected_head="<-",
                capture_symbols=symbols,
                callback=callback,
                when=when,
            )
        private["gam.fit4"] = fit4
        fit3 = clone_function(
            self._utils.getFromNamespace("gam.fit3", "mgcv"), environment=private
        )
        callback_failure: list[Exception] = []

        def record_fit3(**args: Any) -> Any:
            try:
                fit = fit3(**args)
                r_field = self._ro.r["$"]
                nsp = len(args["UrS"])
                fit_family = self._ro.conversion.get_conversion().rpy2py(args["family"])
                nth = (
                    int(np.asarray(fit_family.rx2("n.theta"))[0])
                    if bool(self._base.inherits(args["family"], "extended.family")[0])
                    else 0
                )
                y_space = self._ro.r["["](
                    args["U1"],
                    rinterface.MissingArg,
                    self._base.seq_len(
                        int(self._base.ncol(args["U1"])[0])
                        - int(np.asarray(args["Mp"])[0])
                    ),
                    drop=False,
                )
                y_beta = self._base.drop(
                    multiply(self._base.t(y_space), fit.rx2("coefficients"))
                )
                rv_y = multiply(self._base.t(fit.rx2("rV")), y_space)
                square = self._ro.r["^"]
                bsb = np.asarray(
                    [
                        single(self._base.sum(square(multiply(y_beta, root), 2)))
                        for root in args["UrS"]
                    ]
                )
                trvs = np.asarray(
                    [
                        single(self._base.sum(square(multiply(rv_y, root), 2)))
                        for root in args["UrS"]
                    ]
                )
                packed_sp = copied(args["sp"])
                estimate_scale = len(packed_sp) > nth + nsp
                score_phi = (
                    single(self._base.exp(self._utils.tail(args["sp"], 1)))
                    if estimate_scale
                    else single(args["scale"])
                )
                update_phi = (
                    single(r_field(fit, "scale"))
                    if estimate_scale
                    else single(args["scale"])
                )
                theta_in = packed_sp[0] if nth else np.nan
                theta_out = single(fit_family.rx2("getTheta")()) if nth else np.nan
                number = len(trace_log) + 1
                trace_log.append(
                    pd.DataFrame(
                        {
                            "call": np.full(nsp, number, dtype=np.int64),
                            "parameter": np.arange(1, nsp + 1),
                            "log_smoothing": packed_sp[nth : nth + nsp],
                            "ldetS1": copied(fit.rx2("ldetS1"))[:nsp],
                            "bSb": bsb,
                            "trVS": trvs,
                            "score": np.full(nsp, single(fit.rx2("REML"))),
                            "score_phi": np.full(nsp, score_phi),
                            "update_phi": np.full(nsp, update_phi),
                            "reported_phi": np.full(nsp, single(r_field(fit, "scale"))),
                            "theta_input": np.full(nsp, theta_in),
                            "theta_output": np.full(nsp, theta_out),
                            "deviance": np.full(nsp, single(r_field(fit, "dev"))),
                        }
                    )
                )
                coefficient_log.append(copied(fit.rx2("coefficients")))
                fitted_log.append(copied(fit.rx2("fitted.values")))
                start_log.append(
                    np.full(n_coef, np.nan)
                    if bool(self._base.is_null(args.get("start", self._ro.NULL))[0])
                    else copied(args["start"])
                )
                theta_log.append(np.asarray([theta_in, theta_out]))
                return fit
            except Exception as exc:
                callback_failure.append(exc)
                return self._ro.NULL

        fit3_callback = make_r_callback(record_fit3)
        private["gam.fit3"] = fit3_callback
        efsudr = clone_function(
            self._utils.getFromNamespace("efsudr", "mgcv"), environment=private
        )
        efsudr = self._instrument_efs_branches(
            efsudr, multiplier_log, branch_log, instrument_function
        )
        call_args: dict[str, Any] = {
            "x": r_x,
            "y": setup.rx2("y"),
            "lsp": lsp,
            "Eb": penalty_space.rx2("E"),
            "UrS": ur_s,
            "weights": setup.rx2("w"),
            "family": r_family,
            "offset": setup.rx2("offset"),
            "U1": u1,
            "intercept": setup.rx2("intercept"),
            "scale": fit_scale,
            "Mp": mp,
            "control": self._mgcv.gam_control(
                efs_lspmax=resolved["efs_lspmax"], efs_tol=resolved["efs_tol"]
            ),
            "n.true": setup.rx2("n.true"),
        }
        if start is not self._ro.NULL:
            call_args["start"] = start
        if old_beta is not self._ro.NULL:
            call_args["null.coef"] = old_beta
        try:
            fit = efsudr(**call_args)
        except Exception:
            if callback_failure:
                raise RBridgeError(str(callback_failure[0])) from callback_failure[0]
            raise
        finally:
            # Remove the R -> Python callback -> R private-environment cycle.
            # The copied trace remains independent of these temporary bindings.
            private["gam.fit3"] = self._ro.NULL
            private["gam.fit4"] = self._ro.NULL
        if callback_failure:
            raise RBridgeError(str(callback_failure[0])) from callback_failure[0]
        result = {
            "statistics": pd.concat(trace_log, ignore_index=True),
            "coefficients": np.vstack(coefficient_log),
            "fitted_values": np.vstack(fitted_log),
            "starts": np.vstack(start_log),
            "theta_trace": np.vstack(theta_log),
            "start_retained": np.asarray(retained_log, dtype=bool),
            "initial_states": (
                pd.concat(initial_state_log, ignore_index=True)
                if initial_state_log
                else pd.DataFrame(
                    columns=(
                        "call",
                        "row",
                        "retained",
                        "eta",
                        "mu",
                        "mustart",
                        "null_eta",
                        "etaold",
                        "offset",
                        "theta",
                    )
                )
            ),
            "inner_prefix": pd.DataFrame(
                prefix_log, columns=("call", "pdev", "old_pdev", "diverging")
            ),
            "multipliers": np.asarray(multiplier_log, dtype=np.float64),
            "branches": branch_log,
            "final_score": single(fit.rx2("REML")),
            "selected_packed_sp": copied(fit.rx2("sp")),
            "selected_coefficients": copied(fit.rx2("coefficients")),
            "selected_fitted_values": copied(fit.rx2("fitted.values")),
            "selected_deviance": single(self._ro.r["$"](fit, "dev")),
            "initial_shift": 2.5,
            "source_commit": _PINNED_MGCV_SOURCE_COMMIT,
            "controls": resolved,
            "provenance": self._efs_provenance(data),
        }
        return result

    @staticmethod
    def _instrument_efs_branches(
        efsudr: Any,
        multipliers: list[float],
        branches: list[str],
        instrument_function: Any,
    ) -> Any:
        """Record only actually entered pinned EFS branch and multiplier nodes."""
        from tests.r_ast import find_call_paths

        for path, head, symbols in (
            ((5,), "<-", ("mult",)),
            ((13, 3, 21, 1), "<=", ("old.reml",)),
            ((13, 3, 21, 2, 1, 2, 6, 1), "<", ("fit2", "REML")),
            ((13, 3, 21, 2, 1, 2, 6, 2, 3), "<-", ("mult",)),
            ((13, 3, 21, 3, 1, 2, 1), "<-", ("mult",)),
        ):
            if path not in find_call_paths(efsudr, head, required_symbols=symbols):
                raise RBridgeError(f"Pinned efsudr branch anchor changed: {path!r}")
        anchors = (
            (
                (5,),
                "<-",
                ("mult",),
                lambda value: multipliers.append(float(value[0])),
                "after",
            ),
            (
                (13, 3, 21, 2),
                "{",
                (),
                lambda: branches.append("improvement"),
                "before",
            ),
            (
                (13, 3, 21, 2, 1, 2, 6, 2),
                "{",
                (),
                lambda: branches.append("extension_won"),
                "before",
            ),
            (
                (13, 3, 21, 2, 1, 2, 6, 3),
                "{",
                (),
                lambda: branches.append("extension_lost"),
                "before",
            ),
            (
                (13, 3, 21, 3),
                "{",
                (),
                lambda: branches.append("worsening"),
                "before",
            ),
            (
                (13, 3, 21, 2, 1, 2, 6, 2, 3),
                "<-",
                ("mult",),
                lambda value: multipliers.append(float(value[0])),
                "after",
            ),
            (
                (13, 3, 21, 3, 1, 2, 1),
                "<-",
                ("mult",),
                lambda value: multipliers.append(float(value[0])),
                "after",
            ),
        )
        # Deepest first so each pinned path still refers to the original AST.
        for path, head, captures, callback, when in sorted(
            anchors, key=lambda item: len(item[0]), reverse=True
        ):
            efsudr = instrument_function(
                efsudr,
                path=path,
                expected_head=head,
                capture_symbols=captures,
                callback=callback,
                when=when,
            )
        return efsudr

    def _fit_rpy2(
        self,
        formula: str,
        data: pd.DataFrame,
        family: str,
        method: str,
    ) -> dict[str, Any]:
        """Fit a GAM via rpy2 and extract fit results."""
        r_df = self._to_r_dataframe(data)
        r_model = self._fit_r_model(formula, r_df, family, method)
        return self._extract_fit_results_rpy2(r_model)

    def _extract_fit_results_rpy2(self, r_model: Any) -> dict[str, Any]:
        """Extract fit results from an R model object."""
        coefficients = np.array(r_model.rx2("coefficients"), dtype=np.float64)
        fitted_values = np.array(r_model.rx2("fitted.values"), dtype=np.float64)
        smoothing_params = np.array(r_model.rx2("sp"), dtype=np.float64)
        deviance = float(np.array(r_model.rx2("deviance"))[0])
        scale = float(np.array(r_model.rx2("scale"))[0])
        vp_r = r_model.rx2("Vp")
        n_coef = len(coefficients)
        vp = np.array(vp_r, dtype=np.float64).reshape((n_coef, n_coef))
        reml_score = float(np.array(r_model.rx2("gcv.ubre"))[0])

        # reml.scale is the scale used in the REML criterion (jointly
        # optimized), which differs from model$scale (Fletcher estimate).
        reml_scale_r = r_model.rx2("reml.scale")
        reml_scale = (
            float(np.array(reml_scale_r)[0]) if reml_scale_r is not None else scale
        )

        r_summary = self._base.summary(r_model)
        edf = np.array(r_summary.rx2("edf"), dtype=np.float64)
        edf_total = float(np.sum(np.array(r_model.rx2("edf"), dtype=np.float64)))
        null_deviance = float(np.array(r_model.rx2("null.deviance"))[0])

        # Extract theta for extended families (e.g. NB)
        theta = None
        try:
            family_obj = r_model.rx2("family")
            get_theta = family_obj.rx2("getTheta")
            if get_theta is not None:
                theta = float(np.array(get_theta(True))[0])
        except Exception:
            pass

        return {
            "coefficients": coefficients,
            "fitted_values": fitted_values,
            "smoothing_params": smoothing_params,
            "edf": edf,
            "edf_total": edf_total,
            "deviance": deviance,
            "null_deviance": null_deviance,
            "scale": scale,
            "reml_scale": reml_scale,
            "Vp": vp,
            "reml_score": reml_score,
            "theta": theta,
        }

    def _fit_subprocess(
        self,
        formula: str,
        data: pd.DataFrame,
        family: str,
        method: str,
    ) -> dict[str, Any]:
        """Fit a GAM via Rscript subprocess and parse output files."""
        r_family = self._get_subprocess_family(family)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.csv")
            script_path = os.path.join(tmpdir, "fit.R")

            data.to_csv(data_path, index=False)

            # No jsonlite dependency — serialize via write.csv and writeLines
            script = f"""\
library(mgcv)

data <- read.csv("{data_path}")
model <- gam({formula}, data=data, family={r_family}, method="{method}")
s <- summary(model)

write.csv(data.frame(v=as.numeric(coef(model))), "{tmpdir}/coefficients.csv", row.names=FALSE)
write.csv(data.frame(v=as.numeric(fitted(model))), "{tmpdir}/fitted_values.csv", row.names=FALSE)
write.csv(data.frame(v=as.numeric(model$sp)), "{tmpdir}/smoothing_params.csv", row.names=FALSE)
write.csv(data.frame(v=as.numeric(s$edf)), "{tmpdir}/edf.csv", row.names=FALSE)
writeLines(format(sum(model$edf), digits=15), "{tmpdir}/edf_total.txt")
writeLines(format(deviance(model), digits=15), "{tmpdir}/deviance.txt")
writeLines(format(model$scale, digits=15), "{tmpdir}/scale.txt")
write.csv(as.data.frame(model$Vp), "{tmpdir}/Vp.csv", row.names=FALSE)
writeLines(format(model$gcv.ubre, digits=15), "{tmpdir}/reml_score.txt")
writeLines(format(model$null.deviance, digits=15), "{tmpdir}/null_deviance.txt")
rs <- model$reml.scale
if (!is.null(rs)) {{
    writeLines(format(rs, digits=15), "{tmpdir}/reml_scale.txt")
}}
fam <- model$family
if (!is.null(fam$getTheta)) {{
    writeLines(format(fam$getTheta(TRUE), digits=15), "{tmpdir}/theta.txt")
}}
"""
            with open(script_path, "w") as f:
                f.write(script)

            proc = subprocess.run(
                ["Rscript", script_path],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode != 0:
                raise RBridgeError(
                    f"Rscript failed (exit {proc.returncode}):\n{proc.stderr}"
                )

            def _read_vec(name: str) -> np.ndarray:
                return pd.read_csv(os.path.join(tmpdir, name))["v"].values.astype(
                    np.float64
                )

            def _read_scalar(name: str) -> float:
                with open(os.path.join(tmpdir, name)) as fh:
                    return float(fh.read().strip())

            vp = pd.read_csv(os.path.join(tmpdir, "Vp.csv")).values.astype(np.float64)
            scale = _read_scalar("scale.txt")

            reml_scale_path = os.path.join(tmpdir, "reml_scale.txt")
            reml_scale = (
                _read_scalar("reml_scale.txt")
                if os.path.exists(reml_scale_path)
                else scale
            )

            theta_path = os.path.join(tmpdir, "theta.txt")
            theta = _read_scalar("theta.txt") if os.path.exists(theta_path) else None

            return {
                "coefficients": _read_vec("coefficients.csv"),
                "fitted_values": _read_vec("fitted_values.csv"),
                "smoothing_params": _read_vec("smoothing_params.csv"),
                "edf": _read_vec("edf.csv"),
                "edf_total": _read_scalar("edf_total.txt"),
                "deviance": _read_scalar("deviance.txt"),
                "null_deviance": _read_scalar("null_deviance.txt"),
                "scale": scale,
                "reml_scale": reml_scale,
                "Vp": vp,
                "reml_score": _read_scalar("reml_score.txt"),
                "theta": theta,
            }

    # ------------------------------------------------------------------ #
    #  get_smooth_components                                              #
    # ------------------------------------------------------------------ #

    def get_smooth_components(
        self,
        formula: str,
        data: pd.DataFrame,
        family: str = "gaussian",
        method: str = "REML",
    ) -> dict[str, Any]:
        """Fit a GAM in R and extract per-smooth basis and penalty matrices.

        Returns dict with keys: basis_matrices, penalty_matrices (lists of ndarrays),
        plus all keys from fit_gam().
        """
        if self.mode == "rpy2":
            return self._get_smooth_components_rpy2(formula, data, family, method)
        return self._get_smooth_components_subprocess(formula, data, family, method)

    def _get_smooth_components_rpy2(
        self,
        formula: str,
        data: pd.DataFrame,
        family: str,
        method: str,
    ) -> dict[str, Any]:
        """Fit a GAM via rpy2, extract per-smooth basis/penalty and fit results."""
        ro = self._ro
        r_df = self._to_r_dataframe(data)
        r_model = self._fit_r_model(formula, r_df, family, method)

        # Extract per-smooth basis and penalty from the same model object
        smooth_list = r_model.rx2("smooth")
        n_smooths = len(smooth_list)

        X_full = np.array(ro.r["model.matrix"](r_model), dtype=np.float64)

        basis_matrices = []
        penalty_matrices = []
        for i in range(n_smooths):
            sm = smooth_list[i]
            first_col = int(np.array(sm.rx2("first.para"))[0]) - 1
            last_col = int(np.array(sm.rx2("last.para"))[0])
            basis_matrices.append(X_full[:, first_col:last_col])

            S_list = sm.rx2("S")
            penalties_for_smooth = []
            for j in range(len(S_list)):
                S_flat = np.array(S_list[j], dtype=np.float64)
                # R matrices come as column-major flat arrays via rpy2
                n_cols = last_col - first_col
                if S_flat.ndim == 2:
                    penalties_for_smooth.append(S_flat)
                else:
                    penalties_for_smooth.append(
                        S_flat.reshape((n_cols, n_cols), order="F")
                    )
            penalty_matrices.append(penalties_for_smooth)

        # Extract fit results from the same model (no double-fitting)
        fit_result = self._extract_fit_results_rpy2(r_model)
        fit_result["basis_matrices"] = basis_matrices
        fit_result["penalty_matrices"] = penalty_matrices
        fit_result["model_matrix"] = X_full

        return fit_result

    def _get_smooth_components_subprocess(
        self,
        formula: str,
        data: pd.DataFrame,
        family: str,
        method: str,
    ) -> dict[str, Any]:
        """Fit a GAM via subprocess, extract per-smooth basis/penalty and fit results."""
        r_family = self._get_subprocess_family(family)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.csv")
            script_path = os.path.join(tmpdir, "fit.R")

            data.to_csv(data_path, index=False)

            script = f"""\
library(mgcv)

data <- read.csv("{data_path}")
model <- gam({formula}, data=data, family={r_family}, method="{method}")
s <- summary(model)

# Basic fit results
write.csv(data.frame(v=as.numeric(coef(model))), "{tmpdir}/coefficients.csv", row.names=FALSE)
write.csv(data.frame(v=as.numeric(fitted(model))), "{tmpdir}/fitted_values.csv", row.names=FALSE)
write.csv(data.frame(v=as.numeric(model$sp)), "{tmpdir}/smoothing_params.csv", row.names=FALSE)
write.csv(data.frame(v=as.numeric(s$edf)), "{tmpdir}/edf.csv", row.names=FALSE)
writeLines(format(sum(model$edf), digits=15), "{tmpdir}/edf_total.txt")
writeLines(format(deviance(model), digits=15), "{tmpdir}/deviance.txt")
writeLines(format(model$scale, digits=15), "{tmpdir}/scale.txt")
write.csv(as.data.frame(model$Vp), "{tmpdir}/Vp.csv", row.names=FALSE)
writeLines(format(model$gcv.ubre, digits=15), "{tmpdir}/reml_score.txt")
writeLines(format(model$null.deviance, digits=15), "{tmpdir}/null_deviance.txt")
rs <- model$reml.scale
if (!is.null(rs)) {{
    writeLines(format(rs, digits=15), "{tmpdir}/reml_scale.txt")
}}

# Per-smooth basis and penalty matrices
X <- model.matrix(model)
n_smooths <- length(model$smooth)
writeLines(as.character(n_smooths), "{tmpdir}/n_smooths.txt")

for (i in seq_len(n_smooths)) {{
    sm <- model$smooth[[i]]
    first_col <- sm$first.para
    last_col <- sm$last.para
    Xblock <- X[, first_col:last_col, drop=FALSE]
    write.csv(as.data.frame(Xblock), sprintf("{tmpdir}/basis_%d.csv", i), row.names=FALSE)

    n_penalties <- length(sm$S)
    writeLines(as.character(n_penalties), sprintf("{tmpdir}/n_pen_%d.txt", i))
    for (j in seq_len(n_penalties)) {{
        write.csv(as.data.frame(sm$S[[j]]), sprintf("{tmpdir}/pen_%d_%d.csv", i, j), row.names=FALSE)
    }}
}}
"""
            with open(script_path, "w") as f:
                f.write(script)

            proc = subprocess.run(
                ["Rscript", script_path],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode != 0:
                raise RBridgeError(
                    f"Rscript failed (exit {proc.returncode}):\n{proc.stderr}"
                )

            def _read_vec(name: str) -> np.ndarray:
                return pd.read_csv(os.path.join(tmpdir, name))["v"].values.astype(
                    np.float64
                )

            def _read_scalar(name: str) -> float:
                with open(os.path.join(tmpdir, name)) as fh:
                    return float(fh.read().strip())

            def _read_matrix(name: str) -> np.ndarray:
                return pd.read_csv(os.path.join(tmpdir, name)).values.astype(np.float64)

            n_smooths = int(_read_scalar("n_smooths.txt"))
            basis_matrices = []
            penalty_matrices = []
            for i in range(1, n_smooths + 1):
                basis_matrices.append(_read_matrix(f"basis_{i}.csv"))
                n_pen = int(_read_scalar(f"n_pen_{i}.txt"))
                penalties = [
                    _read_matrix(f"pen_{i}_{j}.csv") for j in range(1, n_pen + 1)
                ]
                penalty_matrices.append(penalties)

            scale = _read_scalar("scale.txt")
            reml_scale_path = os.path.join(tmpdir, "reml_scale.txt")
            reml_scale = (
                _read_scalar("reml_scale.txt")
                if os.path.exists(reml_scale_path)
                else scale
            )

            return {
                "coefficients": _read_vec("coefficients.csv"),
                "fitted_values": _read_vec("fitted_values.csv"),
                "smoothing_params": _read_vec("smoothing_params.csv"),
                "edf": _read_vec("edf.csv"),
                "edf_total": _read_scalar("edf_total.txt"),
                "deviance": _read_scalar("deviance.txt"),
                "null_deviance": _read_scalar("null_deviance.txt"),
                "scale": scale,
                "reml_scale": reml_scale,
                "Vp": _read_matrix("Vp.csv"),
                "reml_score": _read_scalar("reml_score.txt"),
                "basis_matrices": basis_matrices,
                "penalty_matrices": penalty_matrices,
            }

    # ------------------------------------------------------------------ #
    #  smooth_construct                                                   #
    # ------------------------------------------------------------------ #

    def smooth_construct(
        self,
        smooth_expr: str,
        data: pd.DataFrame,
        absorb_cons: bool = False,
        knots: dict[str, np.ndarray] | None = None,
    ) -> dict[str, Any]:
        """Call R's smoothCon() and return smooth construction details.

        Parameters
        ----------
        smooth_expr : str
            Smooth expression, e.g. ``"s(x, bs='tp', k=10)"``.
        data : pd.DataFrame
            Data frame containing the variables.
        absorb_cons : bool
            Whether to absorb identifiability constraints.
        knots : dict[str, np.ndarray] or None
            Optional named knot vectors for the rpy2 path. Used by GP
            smooth-construction parity tests.

        Returns
        -------
        dict
            Keys include X, S (list of penalty matrices), rank,
            null_space_dim, Xu (knots), UZ (mapping matrix), and shift
            (centring values). For GP smooths, knt is the centered knot
            matrix, gp_defn is mgcv's ``c(sign*type, rho, power)`` vector,
            and E is the knot-knot kernel matrix before truncation.
        """
        if self.mode == "rpy2":
            return self._smooth_construct_rpy2(smooth_expr, data, absorb_cons, knots)
        if knots is not None:
            raise NotImplementedError(
                "RBridge.smooth_construct(knots=...) is only supported in rpy2 mode."
            )
        return self._smooth_construct_subprocess(smooth_expr, data, absorb_cons)

    def _smooth_objects_rpy2(
        self,
        smooth_expr: str,
        data: pd.DataFrame,
        absorb_cons: bool,
        knots: dict[str, np.ndarray] | None = None,
    ) -> Any:
        """Pass an interpreted formula term and data objects to smoothCon."""
        self._require_rpy2()
        interpreted = self._call_internal(
            "interpret.gam", self._ro.Formula("~ " + smooth_expr)
        )
        terms = interpreted.rx2("smooth.spec")
        if len(terms) != 1:
            raise ValueError("Smooth construction requires exactly one smooth term.")
        arguments = {"absorb.cons": absorb_cons}
        if knots is not None:
            arguments["knots"] = self._ro.ListVector(
                {name: self._to_r_vector(values) for name, values in knots.items()}
            )
        return self._mgcv.smoothCon(
            terms[0], data=self._to_r_dataframe(data), **arguments
        )

    def _smooth_construct_rpy2(
        self,
        smooth_expr: str,
        data: pd.DataFrame,
        absorb_cons: bool,
        knots: dict[str, np.ndarray] | None,
    ) -> dict[str, Any]:
        """Extract the installed smooth object's fields without generated code."""
        smooth = self._smooth_objects_rpy2(smooth_expr, data, absorb_cons, knots)[0]

        def optional_array(name: str, shape: tuple[int, ...]) -> np.ndarray:
            value = smooth.rx2(name)
            return np.zeros(shape) if value is self._ro.NULL else np.asarray(value)

        ranks = np.asarray(smooth.rx2("rank"), dtype=np.float64).ravel()
        knt = optional_array("knt", (0, 0))
        gp_defn = optional_array("gp.defn", (0,))
        E = (
            np.asarray(
                self._call_internal(
                    "gpE", smooth.rx2("knt"), smooth.rx2("knt"), smooth.rx2("gp.defn")
                ),
                dtype=np.float64,
            )
            if smooth.rx2("knt") is not self._ro.NULL
            else np.zeros((0, 0))
        )
        return {
            "X": np.asarray(smooth.rx2("X"), dtype=np.float64),
            "S": [np.asarray(penalty, dtype=np.float64) for penalty in smooth.rx2("S")],
            "rank": int(ranks[0]),
            "rank_vector": ranks.astype(int),
            "null_space_dim": int(smooth.rx2("null.space.dim")[0]),
            "Xu": optional_array("Xu", (0, 0)),
            "UZ": optional_array("UZ", (0, 0)),
            "shift": optional_array("shift", (0,)),
            "knt": knt,
            "gp_defn": gp_defn,
            "E": E,
        }

    def _smooth_construct_subprocess(
        self,
        smooth_expr: str,
        data: pd.DataFrame,
        absorb_cons: bool,
    ) -> dict[str, Any]:
        """Call smoothCon() via Rscript subprocess and parse output files."""
        absorb_str = "TRUE" if absorb_cons else "FALSE"

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.csv")
            script_path = os.path.join(tmpdir, "smooth.R")

            data.to_csv(data_path, index=False)

            script = f"""\
library(mgcv)

dat <- read.csv("{data_path}")
sm <- smoothCon({smooth_expr}, data=dat, absorb.cons={absorb_str})[[1]]

write.csv(as.data.frame(sm$X), "{tmpdir}/X.csv", row.names=FALSE)
writeLines(as.character(sm$rank), "{tmpdir}/rank.txt")
writeLines(as.character(sm$null.space.dim), "{tmpdir}/null_space_dim.txt")

n_S <- length(sm$S)
writeLines(as.character(n_S), "{tmpdir}/n_S.txt")
for (i in seq_len(n_S)) {{
    write.csv(as.data.frame(sm$S[[i]]), sprintf("{tmpdir}/S_%d.csv", i), row.names=FALSE)
}}

if (!is.null(sm$Xu)) {{
    if (is.matrix(sm$Xu)) {{
        write.csv(as.data.frame(sm$Xu), "{tmpdir}/Xu.csv", row.names=FALSE)
    }} else {{
        write.csv(data.frame(v=as.numeric(sm$Xu)), "{tmpdir}/Xu.csv", row.names=FALSE)
    }}
}} else {{
    writeLines("NULL", "{tmpdir}/Xu.csv")
}}

if (!is.null(sm$UZ)) {{
    write.csv(as.data.frame(sm$UZ), "{tmpdir}/UZ.csv", row.names=FALSE)
}} else {{
    writeLines("NULL", "{tmpdir}/UZ.csv")
}}

if (!is.null(sm$shift)) {{
    write.csv(data.frame(v=as.numeric(sm$shift)), "{tmpdir}/shift.csv", row.names=FALSE)
}} else {{
    writeLines("NULL", "{tmpdir}/shift.csv")
}}
"""
            with open(script_path, "w") as f:
                f.write(script)

            proc = subprocess.run(
                ["Rscript", script_path],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode != 0:
                raise RBridgeError(
                    f"Rscript failed (exit {proc.returncode}):\n{proc.stderr}"
                )

            def _read_matrix(name: str) -> np.ndarray:
                path = os.path.join(tmpdir, name)
                with open(path) as fh:
                    first_line = fh.readline().strip()
                if first_line == "NULL":
                    return np.array([])
                return pd.read_csv(path).values.astype(np.float64)

            def _read_scalar(name: str) -> float:
                with open(os.path.join(tmpdir, name)) as fh:
                    return float(fh.read().strip())

            def _read_vector(name: str) -> np.ndarray:
                with open(os.path.join(tmpdir, name)) as fh:
                    values = [float(line.strip()) for line in fh if line.strip()]
                return np.array(values, dtype=np.float64)

            X = _read_matrix("X.csv")
            rank_vector = _read_vector("rank.txt").astype(int)
            rank = int(rank_vector[0])
            null_space_dim = int(_read_scalar("null_space_dim.txt"))

            n_S = int(_read_scalar("n_S.txt"))
            S_matrices = [_read_matrix(f"S_{i}.csv") for i in range(1, n_S + 1)]

            Xu_raw = _read_matrix("Xu.csv")
            shift_raw = _read_matrix("shift.csv")

            # Handle 1D knots stored as single column
            Xu = Xu_raw.ravel() if Xu_raw.ndim == 2 and Xu_raw.shape[1] == 1 else Xu_raw

            if shift_raw.ndim == 2 and shift_raw.shape[1] == 1:
                shift = shift_raw.ravel()
            else:
                shift = shift_raw

            UZ = _read_matrix("UZ.csv")

            return {
                "X": X,
                "S": S_matrices,
                "rank": rank,
                "rank_vector": rank_vector,
                "null_space_dim": null_space_dim,
                "Xu": Xu,
                "UZ": UZ,
                "shift": shift,
            }

    # ------------------------------------------------------------------ #
    #  smooth_construct_list                                              #
    # ------------------------------------------------------------------ #

    def smooth_construct_list(
        self,
        smooth_expr: str,
        data: pd.DataFrame,
        absorb_cons: bool = False,
    ) -> list[dict[str, Any]]:
        """Call R's smoothCon() and return ALL smooth objects.

        Unlike ``smooth_construct`` which returns only ``[[1]]``, this
        returns every element. Essential for factor-by smooths where
        ``smoothCon()`` returns one smooth per factor level.

        Parameters
        ----------
        smooth_expr : str
            Smooth expression, e.g. ``"s(x, by=fac, bs='tp', k=10)"``.
        data : pd.DataFrame
            Data frame containing the variables.
        absorb_cons : bool
            Whether to absorb identifiability constraints.

        Returns
        -------
        list[dict]
            One dict per smooth returned by smoothCon(). Each dict has keys:
            X, S, rank, null_space_dim, by_level (str or None), label.
        """
        if self.mode == "rpy2":
            return self._smooth_construct_list_rpy2(smooth_expr, data, absorb_cons)
        return self._smooth_construct_list_subprocess(smooth_expr, data, absorb_cons)

    def _smooth_construct_list_rpy2(
        self,
        smooth_expr: str,
        data: pd.DataFrame,
        absorb_cons: bool,
    ) -> list[dict[str, Any]]:
        """Extract all factor-by smooths from their installed R objects."""
        smooths = self._smooth_objects_rpy2(smooth_expr, data, absorb_cons)
        results = []
        for smooth in smooths:
            by_level = smooth.rx2("by.level")
            label = smooth.rx2("label")
            results.append(
                {
                    "X": np.asarray(smooth.rx2("X"), dtype=np.float64),
                    "S": [
                        np.asarray(penalty, dtype=np.float64)
                        for penalty in smooth.rx2("S")
                    ],
                    "rank": int(smooth.rx2("rank")[0]),
                    "null_space_dim": int(smooth.rx2("null.space.dim")[0]),
                    "by_level": None if by_level is self._ro.NULL else str(by_level[0]),
                    "label": "" if label is self._ro.NULL else str(label[0]),
                }
            )
        return results

    def _smooth_construct_list_subprocess(
        self,
        smooth_expr: str,
        data: pd.DataFrame,
        absorb_cons: bool,
    ) -> list[dict[str, Any]]:
        """Call smoothCon() via Rscript subprocess and return all smooth objects."""
        absorb_str = "TRUE" if absorb_cons else "FALSE"

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.csv")
            script_path = os.path.join(tmpdir, "smooth.R")

            data.to_csv(data_path, index=False)

            script = f"""\
library(mgcv)

dat <- read.csv("{data_path}")
## Ensure factor columns are treated as factors in R
for (cn in names(dat)) {{
    if (is.character(dat[[cn]])) dat[[cn]] <- factor(dat[[cn]])
}}
sml <- smoothCon({smooth_expr}, data=dat, absorb.cons={absorb_str})
n_sm <- length(sml)
writeLines(as.character(n_sm), "{tmpdir}/n_sm.txt")

for (i in seq_len(n_sm)) {{
    sm <- sml[[i]]
    write.csv(as.data.frame(sm$X), sprintf("{tmpdir}/X_%d.csv", i), row.names=FALSE)
    writeLines(as.character(sm$rank), sprintf("{tmpdir}/rank_%d.txt", i))
    writeLines(as.character(sm$null.space.dim), sprintf("{tmpdir}/nsd_%d.txt", i))

    n_S <- length(sm$S)
    writeLines(as.character(n_S), sprintf("{tmpdir}/nS_%d.txt", i))
    for (j in seq_len(n_S)) {{
        write.csv(as.data.frame(sm$S[[j]]), sprintf("{tmpdir}/S_%d_%d.csv", i, j), row.names=FALSE)
    }}

    by_lev <- if (!is.null(sm$by.level)) sm$by.level else "NONE"
    writeLines(by_lev, sprintf("{tmpdir}/bylevel_%d.txt", i))

    lab <- if (!is.null(sm$label)) sm$label else ""
    writeLines(lab, sprintf("{tmpdir}/label_%d.txt", i))
}}
"""
            with open(script_path, "w") as f:
                f.write(script)

            proc = subprocess.run(
                ["Rscript", script_path],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode != 0:
                raise RBridgeError(
                    f"Rscript failed (exit {proc.returncode}):\n{proc.stderr}"
                )

            def _read_matrix(name: str) -> np.ndarray:
                return pd.read_csv(os.path.join(tmpdir, name)).values.astype(np.float64)

            def _read_scalar(name: str) -> float:
                with open(os.path.join(tmpdir, name)) as fh:
                    return float(fh.read().strip())

            def _read_text(name: str) -> str:
                with open(os.path.join(tmpdir, name)) as fh:
                    return fh.read().strip()

            n_sm = int(_read_scalar("n_sm.txt"))
            results = []
            for i in range(1, n_sm + 1):
                X = _read_matrix(f"X_{i}.csv")
                rank = int(_read_scalar(f"rank_{i}.txt"))
                null_space_dim = int(_read_scalar(f"nsd_{i}.txt"))

                n_S = int(_read_scalar(f"nS_{i}.txt"))
                S_matrices = [_read_matrix(f"S_{i}_{j}.csv") for j in range(1, n_S + 1)]

                by_level_str = _read_text(f"bylevel_{i}.txt")
                if by_level_str == "NONE":
                    by_level_str = None

                label = _read_text(f"label_{i}.txt")

                results.append(
                    {
                        "X": X,
                        "S": S_matrices,
                        "rank": rank,
                        "null_space_dim": null_space_dim,
                        "by_level": by_level_str,
                        "label": label,
                    }
                )

            return results

    # ------------------------------------------------------------------ #
    #  predict_gam                                                        #
    # ------------------------------------------------------------------ #

    def predict_gam(
        self,
        formula: str,
        train_data: pd.DataFrame,
        newdata: pd.DataFrame,
        family: str = "gaussian",
        method: str = "REML",
        pred_type: str = "response",
        se_fit: bool = False,
    ) -> dict[str, Any]:
        """Fit a GAM in R and predict on new data.

        Parameters
        ----------
        formula : str
            R-style model formula.
        train_data : pd.DataFrame
            Training data.
        newdata : pd.DataFrame
            New data for prediction.
        family : str
            Distribution family name.
        method : str
            Smoothing parameter estimation method.
        pred_type : str
            Prediction type: ``'response'`` or ``'link'``.
        se_fit : bool
            Whether to return standard errors.

        Returns
        -------
        dict
            Keys: ``'predictions'``, optionally ``'se'``.
        """
        if self.mode == "rpy2":
            return self._predict_gam_rpy2(
                formula, train_data, newdata, family, method, pred_type, se_fit
            )
        return self._predict_gam_subprocess(
            formula, train_data, newdata, family, method, pred_type, se_fit
        )

    def _predict_gam_rpy2(
        self,
        formula: str,
        train_data: pd.DataFrame,
        newdata: pd.DataFrame,
        family: str,
        method: str,
        pred_type: str,
        se_fit: bool,
    ) -> dict[str, Any]:
        """Predict via rpy2."""
        r_df = self._to_r_dataframe(train_data)
        r_new = self._to_r_dataframe(newdata)
        r_model = self._fit_r_model(formula, r_df, family, method)

        pred = self._stats.predict(
            r_model,
            newdata=r_new,
            type=pred_type,
            **{"se.fit": se_fit},
        )

        result: dict[str, Any] = {}
        if se_fit:
            result["predictions"] = np.array(pred.rx2("fit"), dtype=np.float64)
            result["se"] = np.array(pred.rx2("se.fit"), dtype=np.float64)
        else:
            result["predictions"] = np.array(pred, dtype=np.float64)

        return result

    def _predict_gam_subprocess(
        self,
        formula: str,
        train_data: pd.DataFrame,
        newdata: pd.DataFrame,
        family: str,
        method: str,
        pred_type: str,
        se_fit: bool,
    ) -> dict[str, Any]:
        """Predict via Rscript subprocess."""
        r_family = self._get_subprocess_family(family)
        se_str = "TRUE" if se_fit else "FALSE"

        with tempfile.TemporaryDirectory() as tmpdir:
            train_path = os.path.join(tmpdir, "train.csv")
            new_path = os.path.join(tmpdir, "newdata.csv")
            script_path = os.path.join(tmpdir, "predict.R")

            train_data.to_csv(train_path, index=False)
            newdata.to_csv(new_path, index=False)

            script = f"""\
library(mgcv)

train <- read.csv("{train_path}")
newdata <- read.csv("{new_path}")
model <- gam({formula}, data=train, family={r_family}, method="{method}")
pred <- predict(model, newdata=newdata, type="{pred_type}", se.fit={se_str})

if ({se_str}) {{
    write.csv(data.frame(v=as.numeric(pred$fit)), "{tmpdir}/predictions.csv", row.names=FALSE)
    write.csv(data.frame(v=as.numeric(pred$se.fit)), "{tmpdir}/se.csv", row.names=FALSE)
}} else {{
    write.csv(data.frame(v=as.numeric(pred)), "{tmpdir}/predictions.csv", row.names=FALSE)
}}
"""
            with open(script_path, "w") as f:
                f.write(script)

            proc = subprocess.run(
                ["Rscript", script_path],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode != 0:
                raise RBridgeError(
                    f"Rscript failed (exit {proc.returncode}):\n{proc.stderr}"
                )

            def _read_vec(name: str) -> np.ndarray:
                return pd.read_csv(os.path.join(tmpdir, name))["v"].values.astype(
                    np.float64
                )

            result: dict[str, Any] = {}
            result["predictions"] = _read_vec("predictions.csv")
            if se_fit:
                result["se"] = _read_vec("se.csv")

            return result

    # ------------------------------------------------------------------ #
    #  summary_gam                                                        #
    # ------------------------------------------------------------------ #

    def summary_gam(
        self,
        formula: str,
        data: pd.DataFrame,
        family: str = "gaussian",
        method: str = "REML",
    ) -> dict[str, Any]:
        """Fit a GAM in R and return summary statistics.

        Parameters
        ----------
        formula : str
            R-style model formula.
        data : pd.DataFrame
            Data frame with variables referenced in formula.
        family : str
            Distribution family name.
        method : str
            Smoothing parameter estimation method.

        Returns
        -------
        dict
            Keys: p_table, s_table, r_sq, dev_explained, scale,
            residual_df, n, edf (per smooth), sp_criterion.
        """
        if self.mode == "rpy2":
            return self._summary_gam_rpy2(formula, data, family, method)
        return self._summary_gam_subprocess(formula, data, family, method)

    def _summary_gam_rpy2(
        self,
        formula: str,
        data: pd.DataFrame,
        family: str,
        method: str,
    ) -> dict[str, Any]:
        """Extract GAM summary statistics via rpy2."""
        r_df = self._to_r_dataframe(data)
        r_model = self._fit_r_model(formula, r_df, family, method)

        r_summary = self._base.summary(r_model)

        result: dict[str, Any] = {}

        # Parametric coefficients table
        p_table_r = r_summary.rx2("p.table")
        if p_table_r is not None and len(p_table_r) > 0:
            p_arr = np.array(p_table_r, dtype=np.float64)
            n_rows = len(r_summary.rx2("p.coeff"))
            if p_arr.ndim == 1:
                result["p_table"] = p_arr.reshape(n_rows, -1)
            else:
                result["p_table"] = p_arr
        else:
            result["p_table"] = None

        # Smooth terms table
        s_table_r = r_summary.rx2("s.table")
        if s_table_r is not None and len(s_table_r) > 0:
            s_arr = np.array(s_table_r, dtype=np.float64)
            n_smooths = int(np.array(r_summary.rx2("m"))[0])
            if n_smooths > 0:
                result["s_table"] = s_arr.reshape(n_smooths, -1)
            else:
                result["s_table"] = None
        else:
            result["s_table"] = None

        # R-squared
        r_sq = r_summary.rx2("r.sq")
        result["r_sq"] = float(np.array(r_sq)[0]) if r_sq is not None else None

        result["dev_explained"] = float(np.array(r_summary.rx2("dev.expl"))[0])
        result["scale"] = float(np.array(r_summary.rx2("scale"))[0])
        result["residual_df"] = float(np.array(r_summary.rx2("residual.df"))[0])
        result["n"] = int(np.array(r_summary.rx2("n"))[0])

        edf_r = r_summary.rx2("edf")
        if edf_r is not None:
            result["edf"] = np.array(edf_r, dtype=np.float64)
        else:
            result["edf"] = np.array([])

        sp_crit = r_summary.rx2("sp.criterion")
        if sp_crit is not None:
            result["sp_criterion"] = float(np.array(sp_crit)[0])
        else:
            result["sp_criterion"] = None

        return result

    def _summary_gam_subprocess(
        self,
        formula: str,
        data: pd.DataFrame,
        family: str,
        method: str,
    ) -> dict[str, Any]:
        """Extract GAM summary statistics via Rscript subprocess."""
        r_family = self._get_subprocess_family(family)

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.csv")
            script_path = os.path.join(tmpdir, "summary.R")

            data.to_csv(data_path, index=False)

            script = f"""\
library(mgcv)

data <- read.csv("{data_path}")
model <- gam({formula}, data=data, family={r_family}, method="{method}")
s <- summary(model)

# Parametric table
if (!is.null(s$p.table)) {{
    write.csv(as.data.frame(s$p.table), "{tmpdir}/p_table.csv", row.names=TRUE)
}} else {{
    writeLines("NULL", "{tmpdir}/p_table.csv")
}}

# Smooth table
if (!is.null(s$s.table) && s$m > 0) {{
    write.csv(as.data.frame(s$s.table), "{tmpdir}/s_table.csv", row.names=TRUE)
}} else {{
    writeLines("NULL", "{tmpdir}/s_table.csv")
}}

# Scalars
writeLines(format(s$r.sq, digits=15), "{tmpdir}/r_sq.txt")
writeLines(format(s$dev.expl, digits=15), "{tmpdir}/dev_expl.txt")
writeLines(format(s$scale, digits=15), "{tmpdir}/scale.txt")
writeLines(format(s$residual.df, digits=15), "{tmpdir}/residual_df.txt")
writeLines(as.character(s$n), "{tmpdir}/n.txt")
write.csv(data.frame(v=as.numeric(s$edf)), "{tmpdir}/edf.csv", row.names=FALSE)
if (!is.null(s$sp.criterion)) {{
    writeLines(format(s$sp.criterion, digits=15), "{tmpdir}/sp_criterion.txt")
}}
"""
            with open(script_path, "w") as f:
                f.write(script)

            proc = subprocess.run(
                ["Rscript", script_path],
                capture_output=True,
                text=True,
                timeout=120,
            )
            if proc.returncode != 0:
                raise RBridgeError(
                    f"Rscript failed (exit {proc.returncode}):\n{proc.stderr}"
                )

            def _read_scalar(name: str) -> float:
                with open(os.path.join(tmpdir, name)) as fh:
                    return float(fh.read().strip())

            def _read_matrix(name: str) -> np.ndarray | None:
                path = os.path.join(tmpdir, name)
                with open(path) as fh:
                    first = fh.readline().strip()
                if first == "NULL":
                    return None
                return pd.read_csv(path, index_col=0).values.astype(np.float64)

            result: dict[str, Any] = {}
            result["p_table"] = _read_matrix("p_table.csv")
            result["s_table"] = _read_matrix("s_table.csv")
            result["r_sq"] = _read_scalar("r_sq.txt")
            result["dev_explained"] = _read_scalar("dev_expl.txt")
            result["scale"] = _read_scalar("scale.txt")
            result["residual_df"] = _read_scalar("residual_df.txt")
            result["n"] = int(_read_scalar("n.txt"))
            result["edf"] = pd.read_csv(os.path.join(tmpdir, "edf.csv"))[
                "v"
            ].values.astype(np.float64)

            sp_path = os.path.join(tmpdir, "sp_criterion.txt")
            if os.path.exists(sp_path):
                result["sp_criterion"] = _read_scalar("sp_criterion.txt")
            else:
                result["sp_criterion"] = None

            return result
