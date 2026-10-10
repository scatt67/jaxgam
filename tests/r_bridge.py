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
