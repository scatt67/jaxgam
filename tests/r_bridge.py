"""RBridge: interface to R's mgcv for reference comparison.

Two modes:
1. rpy2 (preferred): Direct R execution in-process
2. subprocess: Run Rscript and parse output (fallback)

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
from pathlib import Path
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
    """Error communicating with R via subprocess mode."""


class RBridge:
    """Interface to R's mgcv for reference comparison.

    Parameters
    ----------
    mode : str
        One of 'auto', 'rpy2', 'subprocess'. 'auto' tries rpy2 first,
        falls back to subprocess.
    """

    _SUBPROCESS_FAMILY_MAP: ClassVar[dict[str, str]] = {
        "gaussian": "gaussian()",
        "binomial": "binomial()",
        "poisson": "poisson()",
        "gamma": "Gamma()",
        "gamma_log": "Gamma(link='log')",
        "nb": "nb()",
        "nb_identity": "nb(link='identity')",
        "nb_sqrt": "nb(link='sqrt')",
    }

    _ro: Any
    _mgcv: Any
    _base: Any
    _stats: Any

    def __init__(self, mode: str = "auto") -> None:
        if mode == "auto":
            try:
                import rpy2.robjects  # noqa: F401

                self.mode = "rpy2"
                self._setup_rpy2()
            except (ImportError, ValueError, OSError):
                self.mode = "subprocess"
        elif mode == "rpy2":
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
        import rpy2.robjects as ro
        from rpy2.robjects.packages import importr

        self._ro = ro
        self._mgcv = importr("mgcv")
        self._base = importr("base")
        self._stats = importr("stats")

    @staticmethod
    def available() -> bool:
        """Check if R and mgcv are available via either mode."""
        try:
            import rpy2.robjects  # noqa: F401

            return True
        except (ImportError, ValueError, OSError):
            pass
        try:
            result = subprocess.run(
                ["Rscript", "-e", "library(mgcv); cat('ok')"],
                capture_output=True,
                text=True,
                timeout=30,
            )
            return result.returncode == 0 and "ok" in result.stdout
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False

    @staticmethod
    def check_versions() -> tuple[bool, str]:
        """Verify R and mgcv match the pinned versions.

        Returns (True, "") if versions match, or (False, reason) if not.
        """
        try:
            r_ver = subprocess.check_output(
                ["Rscript", "-e", "cat(R.version$major, R.version$minor, sep='.')"],
                text=True,
                timeout=10,
            ).strip()
            mgcv_ver = subprocess.check_output(
                ["Rscript", "-e", "cat(as.character(packageVersion('mgcv')))"],
                text=True,
                timeout=10,
            ).strip()
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

    def _get_efs_subprocess_family(self, family: str, theta: float | None) -> str:
        """Resolve an EFS oracle family, including a fixed NB size.

        ``mgcv::nb(theta=...)`` is the fixed-theta EFS route.  Keep the
        historical ``nb()`` mapping when callers do not request a size.
        """
        if theta is not None:
            if family != "nb":
                raise ValueError("EFS theta is supported only for family='nb'")
            if not np.isfinite(theta) or theta <= 0:
                raise ValueError("EFS NB theta must be finite and positive")
            return f"nb(theta={float(theta)!r})"
        return self._get_subprocess_family(family)

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

        if self.mode == "rpy2":
            return self._fix_dependence_rpy2(X1, X2, tol, rank_def)
        return self._fix_dependence_subprocess(X1, X2, tol, rank_def)

    def _fix_dependence_rpy2(
        self, X1: np.ndarray, X2: np.ndarray, tol: float, rank_def: int
    ) -> list[int] | None:
        """Call ``mgcv:::fixDependence`` through rpy2."""
        from rpy2.robjects import FloatVector

        ro = self._ro
        matrix = ro.r["matrix"]
        X1_r = matrix(
            FloatVector(X1.ravel(order="F")), nrow=X1.shape[0], ncol=X1.shape[1]
        )
        X2_r = matrix(
            FloatVector(X2.ravel(order="F")), nrow=X2.shape[0], ncol=X2.shape[1]
        )
        fix_dependence = ro.r("mgcv:::fixDependence")
        ind = fix_dependence(X1_r, X2_r, tol=float(tol), **{"rank.def": int(rank_def)})
        if ind is None or ind is ro.NULL or len(ind) == 0:
            return None
        return [int(index) - 1 for index in ind]

    def _fix_dependence_subprocess(
        self, X1: np.ndarray, X2: np.ndarray, tol: float, rank_def: int
    ) -> list[int] | None:
        """Call ``mgcv:::fixDependence`` through Rscript."""
        with tempfile.TemporaryDirectory() as tmpdir:
            x1_path = os.path.join(tmpdir, "X1.csv")
            x2_path = os.path.join(tmpdir, "X2.csv")
            script_path = os.path.join(tmpdir, "fix_dependence.R")
            result_path = os.path.join(tmpdir, "indices.csv")
            np.savetxt(x1_path, X1, delimiter=",")
            np.savetxt(x2_path, X2, delimiter=",")

            script = "\n".join(
                [
                    "library(mgcv)",
                    f'X1 <- as.matrix(read.csv("{x1_path}", header=FALSE))',
                    f'X2 <- as.matrix(read.csv("{x2_path}", header=FALSE))',
                    (
                        "ind <- mgcv:::fixDependence(X1, X2, "
                        f"tol={tol:.17g}, rank.def={rank_def})"
                    ),
                    "if (is.null(ind)) {",
                    f'    writeLines("", "{result_path}")',
                    "} else {",
                    (
                        "    write.csv(data.frame(index=as.integer(ind)), "
                        f'"{result_path}", row.names=FALSE)'
                    ),
                    "}",
                ]
            )
            with open(script_path, "w") as f:
                f.write(script)

            proc = subprocess.run(
                ["Rscript", script_path],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode != 0:
                raise RBridgeError(
                    f"Rscript failed (exit {proc.returncode}):\n{proc.stderr}"
                )
            with open(result_path) as f:
                if not f.read().strip():
                    return None
            return (pd.read_csv(result_path)["index"].astype(int) - 1).tolist()

    # ------------------------------------------------------------------ #
    #  pinned extended Fellner--Schall oracle                            #
    # ------------------------------------------------------------------ #

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

        This uses ``Rscript`` in either bridge mode so the separate diagnostic
        can use identical pinned local-source instrumentation. Existing
        :meth:`fit_gam` mode selection and defaults are unchanged.
        """
        self._require_pinned_efs_versions()
        return self._fit_efs_subprocess(
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
        )

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
    ) -> dict[str, Any]:
        """Return real per-refit EFS statistics from a private source copy."""
        self._require_pinned_efs_versions()
        return self._efs_diagnostics_subprocess(
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
        )

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
        with tempfile.TemporaryDirectory() as tmpdir:
            base = Path(tmpdir)
            np.savetxt(base / "rho.csv", rho, delimiter=",")
            np.savetxt(base / "beta.csv", beta, delimiter=",")
            np.savetxt(base / "factor.csv", factor, delimiter=",")
            for index, root in enumerate(roots):
                np.savetxt(base / f"root_{index}.csv", root, delimiter=",")
            for index, root in enumerate(det_roots):
                np.savetxt(base / f"det_{index}.csv", root, delimiter=",")
            shapes = ";".join(f"{root.shape[0]},{root.shape[1]}" for root in roots)
            det_shapes = ";".join(
                f"{root.shape[0]},{root.shape[1]}" for root in det_roots
            )
            output = base / "statistics.csv"
            script = "\n".join(
                [
                    "library(mgcv)",
                    f"base <- {str(base)!r}",
                    f"nsp <- {len(rho)}L",
                    f"shapes <- strsplit({shapes!r}, ';', fixed=TRUE)[[1]]",
                    f"det.shapes <- strsplit({det_shapes!r}, ';', fixed=TRUE)[[1]]",
                    "read.root <- function(prefix, i, shape) { d <- as.integer(strsplit(shape, ',', fixed=TRUE)[[1]]); matrix(scan(file.path(base, sprintf(paste0(prefix, '_%d.csv'), i-1L)), sep=',', quiet=TRUE), d[1], d[2], byrow=TRUE) }",
                    "rho <- scan(file.path(base, 'rho.csv'), sep=',', quiet=TRUE); beta <- scan(file.path(base, 'beta.csv'), sep=',', quiet=TRUE)",
                    "L <- as.matrix(read.csv(file.path(base, 'factor.csv'), header=FALSE))",
                    "rS <- lapply(seq_len(nsp), function(i) read.root('det', i, det.shapes[i]))",
                    "roots <- lapply(seq_len(nsp), function(i) read.root('root', i, shapes[i]))",
                    "d <- mgcv:::gam.reparam(rS, rho, deriv=1)$det1",
                    "q <- vapply(roots, function(B) sum((crossprod(B, beta))^2), 0.0)",
                    "t <- vapply(roots, function(B) sum(forwardsolve(L, B)^2), 0.0)",
                    f"write.csv(data.frame(d=d,t=t,q=q), {str(output)!r}, row.names=FALSE)",
                ]
            )
            script_path = base / "efs_statistics.R"
            script_path.write_text(script, encoding="utf-8")
            proc = subprocess.run(
                ["Rscript", str(script_path)],
                capture_output=True,
                text=True,
                timeout=30,
            )
            if proc.returncode != 0:
                raise RBridgeError(f"R EFS algebra oracle failed: {proc.stderr}")
            result = pd.read_csv(output)
            return {
                key: result[key].to_numpy(dtype=np.float64) for key in ("d", "t", "q")
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
    def _pinned_efsudr_source() -> str:
        """Deparse the installed, version-gated function into private source."""
        try:
            source = subprocess.check_output(
                [
                    "Rscript",
                    "-e",
                    "library(mgcv); cat('efsudr <- ', deparse(mgcv:::efsudr), sep='\\n')",
                ],
                text=True,
                timeout=20,
            )
        except (
            OSError,
            subprocess.CalledProcessError,
            subprocess.TimeoutExpired,
        ) as exc:
            raise RBridgeError("Cannot deparse installed pinned mgcv efsudr") from exc
        if "efsudr <-" not in source:
            raise RBridgeError("Installed pinned mgcv does not expose efsudr")
        return source

    @staticmethod
    def _run_efs_rscript(script_path: str) -> None:
        proc = subprocess.run(
            ["Rscript", script_path], capture_output=True, text=True, timeout=120
        )
        if proc.returncode != 0:
            raise RBridgeError(
                f"Pinned EFS Rscript failed (exit {proc.returncode}):\\n{proc.stderr}"
            )

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

    def _efs_r_arguments(
        self,
        weights: str | None,
        offset: str | None,
        controls: dict[str, float],
        initial_smoothing: np.ndarray | None,
        initial_scale: float | None,
        scale: float,
    ) -> str:
        arguments = [
            'method="REML"',
            'optimizer="efs"',
            f"control=gam.control(efs.lspmax={controls['efs_lspmax']!r}, efs.tol={controls['efs_tol']!r})",
            f"scale={float(scale)!r}",
        ]
        if weights is not None:
            arguments.append(f"weights=data[[{weights!r}]]")
        if offset is not None:
            arguments.append(f"offset=data[[{offset!r}]]")
        # Unknown-scale ``in.out`` must carry both starting values.  Supplying
        # only sp made mgcv silently use an unrelated phi=1 and changed the
        # EFS branch trace.  Retain the historical phi=1 fallback for the
        # known-scale callers that supplied only smoothing values.
        if initial_smoothing is not None and (scale > 0 or initial_scale is not None):
            smoothing = ", ".join(repr(float(value)) for value in initial_smoothing)
            initial_scale_r = 1.0 if initial_scale is None else float(initial_scale)
            arguments.append(
                f"in.out=list(sp=c({smoothing}), scale={initial_scale_r!r})"
            )
        return ", ".join(arguments)

    def _fit_efs_subprocess(
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
    ) -> dict[str, Any]:
        resolved, initial = self._validate_efs_inputs(
            data, weights, offset, controls, initial_smoothing
        )
        r_family = self._get_efs_subprocess_family(family, theta)
        if initial_scale is not None and (
            not np.isfinite(initial_scale) or initial_scale <= 0
        ):
            raise ValueError("EFS initial_scale must be finite and positive")
        r_arguments = self._efs_r_arguments(
            weights, offset, resolved, initial, initial_scale, scale
        )
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.csv")
            script_path = os.path.join(tmpdir, "fit_efs.R")
            data.to_csv(data_path, index=False)
            setup_arguments = []
            if weights is not None:
                setup_arguments.append(f"weights=data[[{weights!r}]]")
            if offset is not None:
                setup_arguments.append(f"offset=data[[{offset!r}]]")
            setup_suffix = ", " + ", ".join(setup_arguments) if setup_arguments else ""
            null_coef_prefix = (
                f"G <- gam({formula}, data=data, family={r_family}, fit=FALSE{setup_suffix})"
                if null_coef
                else ""
            )
            null_coef_argument = (
                ", null.coef=mgcv:::get.null.coef(G)$null.coef" if null_coef else ""
            )
            outputs = {
                "coefficients": os.path.join(tmpdir, "coefficients.csv"),
                "fitted": os.path.join(tmpdir, "fitted_values.csv"),
                "sp": os.path.join(tmpdir, "smoothing_params.csv"),
                "edf": os.path.join(tmpdir, "edf.csv"),
                "history": os.path.join(tmpdir, "score_history.csv"),
                "iterations": os.path.join(tmpdir, "outer_iterations.txt"),
                "convergence": os.path.join(tmpdir, "convergence.txt"),
                "edf_total": os.path.join(tmpdir, "edf_total.txt"),
                "deviance": os.path.join(tmpdir, "deviance.txt"),
                "scale": os.path.join(tmpdir, "scale.txt"),
                "vp": os.path.join(tmpdir, "Vp.csv"),
                "score": os.path.join(tmpdir, "reml_score.txt"),
                "null_deviance": os.path.join(tmpdir, "null_deviance.txt"),
                "reml_scale": os.path.join(tmpdir, "reml_scale.txt"),
                "theta": os.path.join(tmpdir, "theta.txt"),
            }
            script = "\n".join(
                [
                    "library(mgcv)",
                    f"data <- read.csv({data_path!r})",
                    null_coef_prefix,
                    f"model <- gam({formula}, data=data, family={r_family}, {r_arguments}{null_coef_argument})",
                    "s <- summary(model)",
                    f"write.csv(data.frame(v=as.numeric(coef(model))), {outputs['coefficients']!r}, row.names=FALSE)",
                    f"write.csv(data.frame(v=as.numeric(fitted(model))), {outputs['fitted']!r}, row.names=FALSE)",
                    f"write.csv(data.frame(v=as.numeric(model$sp)), {outputs['sp']!r}, row.names=FALSE)",
                    f"write.csv(data.frame(v=as.numeric(s$edf)), {outputs['edf']!r}, row.names=FALSE)",
                    f"write.csv(data.frame(v=as.numeric(model$outer.info$score.hist)), {outputs['history']!r}, row.names=FALSE)",
                    f"writeLines(as.character(model$outer.info$iter), {outputs['iterations']!r})",
                    f"writeLines(as.character(model$outer.info$conv), {outputs['convergence']!r})",
                    f"writeLines(format(sum(model$edf), digits=17), {outputs['edf_total']!r})",
                    f"writeLines(format(deviance(model), digits=17), {outputs['deviance']!r})",
                    f"writeLines(format(model$scale, digits=17), {outputs['scale']!r})",
                    f"write.csv(as.data.frame(model$Vp), {outputs['vp']!r}, row.names=FALSE)",
                    f"writeLines(format(model$gcv.ubre, digits=17), {outputs['score']!r})",
                    f"writeLines(format(model$null.deviance, digits=17), {outputs['null_deviance']!r})",
                    f"if (!is.null(model$reml.scale)) writeLines(format(model$reml.scale, digits=17), {outputs['reml_scale']!r})",
                    f"if (!is.null(model$family$getTheta)) writeLines(format(model$family$getTheta(TRUE), digits=17), {outputs['theta']!r})",
                ]
            )
            Path(script_path).write_text(script, encoding="utf-8")
            self._run_efs_rscript(script_path)

            def vector(key: str) -> np.ndarray:
                return pd.read_csv(outputs[key])["v"].to_numpy(dtype=np.float64)

            def scalar(key: str) -> float:
                return float(Path(outputs[key]).read_text(encoding="utf-8").strip())

            return {
                "coefficients": vector("coefficients"),
                "fitted_values": vector("fitted"),
                "smoothing_params": vector("sp"),
                "edf": vector("edf"),
                "edf_total": scalar("edf_total"),
                "deviance": scalar("deviance"),
                "null_deviance": scalar("null_deviance"),
                "scale": scalar("scale"),
                "reml_scale": scalar("reml_scale")
                if Path(outputs["reml_scale"]).exists()
                else scalar("scale"),
                "Vp": pd.read_csv(outputs["vp"]).to_numpy(dtype=np.float64),
                "reml_score": scalar("score"),
                "theta": scalar("theta") if Path(outputs["theta"]).exists() else None,
                "outer_iterations": int(scalar("iterations")),
                "score_history": vector("history"),
                "convergence": Path(outputs["convergence"])
                .read_text(encoding="utf-8")
                .strip(),
                "optimizer": "efs",
                "controls": resolved,
                "provenance": self._efs_provenance(data),
            }

    def _efs_diagnostics_subprocess(
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
    ) -> dict[str, Any]:
        """Execute a private instrumented pinned ``efsudr`` source function."""
        resolved, initial = self._validate_efs_inputs(
            data, weights, offset, controls, initial_smoothing
        )
        if initial_scale is not None and (
            not np.isfinite(initial_scale) or initial_scale <= 0
        ):
            raise ValueError("EFS initial_scale must be finite and positive")
        r_family = self._get_efs_subprocess_family(family, theta)
        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.csv")
            source_path = os.path.join(tmpdir, "efsudr_pinned.R")
            script_path = os.path.join(tmpdir, "diagnose_efs.R")
            trace_path = os.path.join(tmpdir, "trace.csv")
            coefficient_path = os.path.join(tmpdir, "coefficient_trace.csv")
            fitted_path = os.path.join(tmpdir, "fitted_trace.csv")
            multiplier_path = os.path.join(tmpdir, "multipliers.csv")
            branch_path = os.path.join(tmpdir, "branches.txt")
            score_path = os.path.join(tmpdir, "score.txt")
            data.to_csv(data_path, index=False)
            source = self._pinned_efsudr_source()

            def instrument(old: str, new: str) -> None:
                nonlocal source
                if source.count(old) != 1:
                    raise RBridgeError(
                        f"Pinned efsudr instrumentation anchor changed: {old!r}"
                    )
                source = source.replace(old, new)

            instrument(
                "    mult <- 1\n    fit <- gam.fit3",
                "    mult <- 1; efs_record_multiplier(mult)\n    fit <- gam.fit3",
            )
            instrument(
                "if (fit$REML <= old.reml) {",
                "if (fit$REML <= old.reml) { efs_record_branch('improvement')",
            )
            instrument(
                "if (fit2$REML < fit$REML) {",
                "if (fit2$REML < fit$REML) { efs_record_branch('extension_won')",
            )
            instrument(
                "else {\n                  lsp <- lsp1",
                "else {\n                  efs_record_branch('extension_lost'); lsp <- lsp1",
            )
            instrument(
                "else {\n            while",
                "else {\n            efs_record_branch('worsening'); while",
            )
            instrument(
                "mult <- mult * 2", "mult <- mult * 2; efs_record_multiplier(mult)"
            )
            instrument("mult <- mult/2", "mult <- mult/2; efs_record_multiplier(mult)")
            Path(source_path).write_text(source, encoding="utf-8")
            setup_arguments = []
            if weights is not None:
                setup_arguments.append(f"weights=data[[{weights!r}]]")
            if offset is not None:
                setup_arguments.append(f"offset=data[[{offset!r}]]")
            setup_suffix = ", " + ", ".join(setup_arguments) if setup_arguments else ""
            matched_initial = initial is not None and (
                scale > 0 or initial_scale is not None
            )
            initial_text = (
                "NULL"
                if not matched_initial
                else "c(" + ", ".join(repr(float(value)) for value in initial) + ")"
            )
            initial_scale_text = (
                "NULL" if initial_scale is None else repr(float(initial_scale))
            )
            null_coef_argument = (
                ", null.coef=mgcv:::get.null.coef(G)$null.coef" if null_coef else ""
            )
            script = "\n".join(
                [
                    "library(mgcv)",
                    f"data <- read.csv({data_path!r})",
                    f"G <- gam({formula}, data=data, family={r_family}, fit=FALSE{setup_suffix})",
                    "family <- mgcv:::fix.family.ls(mgcv:::fix.family.var(mgcv:::fix.family.link(G$family)))",
                    "G$rS <- mgcv:::mini.roots(G$S, G$off, ncol(G$X), G$rank)",
                    "Ssp <- mgcv:::totalPenaltySpace(G$S, G$H, G$off, ncol(G$X))",
                    "G$Eb <- Ssp$E; G$U1 <- cbind(Ssp$Y, Ssp$Z); G$Mp <- ncol(Ssp$Z)",
                    "G$UrS <- lapply(G$rS, function(root) t(Ssp$Y) %*% root)",
                    f"initial_sp <- {initial_text}",
                    "if (is.null(initial_sp)) initial_sp <- mgcv:::initial.spg(G$X, G$y, G$w, family, G$S, G$rank, G$off, offset=G$offset, E=G$Eb)",
                    "lsp <- log(initial_sp)",
                    f"fit_scale <- {float(scale)!r}",
                    "if (family$family[1] %in% c('poisson', 'binomial')) fit_scale <- 1",
                    f"initial_phi <- {initial_scale_text}",
                    "if (fit_scale <= 0) { if (is.null(initial_phi)) { null_fit <- mgcv:::get.null.coef(G); initial_phi <- null_fit$null.scale / 10 }; lsp <- c(lsp, log(initial_phi)) }",
                    "trace_env <- new.env(parent=asNamespace('mgcv'))",
                    "trace_env$trace_log <- list(); trace_env$coefficient_log <- list(); trace_env$fitted_log <- list(); trace_env$multiplier_history <- numeric(); trace_env$branch_history <- character()",
                    "trace_env$efs_record_multiplier <- function(value) trace_env$multiplier_history <- c(trace_env$multiplier_history, value)",
                    "trace_env$efs_record_branch <- function(value) trace_env$branch_history <- c(trace_env$branch_history, value)",
                    "trace_env$gam.fit3 <- function(...) {",
                    "  args <- list(...); fit <- do.call(get('gam.fit3', envir=asNamespace('mgcv')), args)",
                    "  nsp <- length(args$UrS); Y <- args$U1[, seq_len(ncol(args$U1) - args$Mp), drop=FALSE]",
                    "  Yb <- drop(t(Y) %*% fit$coefficients); rVY <- t(fit$rV) %*% Y",
                    "  bSb <- trVS <- rep(NA_real_, nsp)",
                    "  if (nsp > 0) for (i in seq_len(nsp)) { bSb[i] <- sum((Yb %*% args$UrS[[i]])^2); trVS[i] <- sum((rVY %*% args$UrS[[i]])^2) }",
                    "  score_phi <- if (length(args$sp) > nsp) exp(tail(args$sp, 1)) else args$scale",
                    "  update_phi <- if (length(args$sp) > nsp) fit$scale else args$scale",
                    "  trace_env$trace_log[[length(trace_env$trace_log) + 1]] <- data.frame(call=rep(length(trace_env$trace_log) + 1L, nsp), parameter=seq_len(nsp), log_smoothing=as.numeric(args$sp[seq_len(nsp)]), ldetS1=as.numeric(fit$ldetS1[seq_len(nsp)]), bSb=as.numeric(bSb), trVS=as.numeric(trVS), score=rep(as.numeric(fit$REML)[1], nsp), score_phi=rep(as.numeric(score_phi)[1], nsp), update_phi=rep(as.numeric(update_phi)[1], nsp), reported_phi=rep(as.numeric(fit$scale)[1], nsp), deviance=rep(as.numeric(fit$dev)[1], nsp))",
                    "  trace_env$coefficient_log[[length(trace_env$coefficient_log) + 1]] <- as.numeric(fit$coefficients)",
                    "  trace_env$fitted_log[[length(trace_env$fitted_log) + 1]] <- as.numeric(fit$fitted.values)",
                    "  fit",
                    "}",
                    f"source({source_path!r}, local=trace_env)",
                    f"fit <- trace_env$efsudr(x=G$X, y=G$y, lsp=lsp, Eb=G$Eb, UrS=G$UrS, weights=G$w, family=family, offset=G$offset, U1=G$U1, intercept=G$intercept, scale=fit_scale, Mp=G$Mp, control=gam.control(efs.lspmax={resolved['efs_lspmax']!r}, efs.tol={resolved['efs_tol']!r}), n.true=G$n.true{null_coef_argument})",
                    "trace <- do.call(rbind, trace_env$trace_log)",
                    f"write.csv(trace, {trace_path!r}, row.names=FALSE)",
                    f"write.csv(do.call(rbind, trace_env$coefficient_log), {coefficient_path!r}, row.names=FALSE)",
                    f"write.csv(do.call(rbind, trace_env$fitted_log), {fitted_path!r}, row.names=FALSE)",
                    f"write.csv(data.frame(multiplier=trace_env$multiplier_history), {multiplier_path!r}, row.names=FALSE)",
                    f"writeLines(trace_env$branch_history, {branch_path!r})",
                    f"writeLines(format(fit$REML, digits=17), {score_path!r})",
                ]
            )
            Path(script_path).write_text(script, encoding="utf-8")
            self._run_efs_rscript(script_path)
            trace = pd.read_csv(trace_path)
            coefficients = pd.read_csv(coefficient_path).to_numpy(dtype=np.float64)
            fitted = pd.read_csv(fitted_path).to_numpy(dtype=np.float64)
            multipliers = pd.read_csv(multiplier_path)["multiplier"].to_numpy(
                dtype=np.float64
            )
            branches = Path(branch_path).read_text(encoding="utf-8").splitlines()
            return {
                "statistics": trace,
                "coefficients": coefficients,
                "fitted_values": fitted,
                "multipliers": multipliers,
                "branches": branches,
                "final_score": float(
                    Path(score_path).read_text(encoding="utf-8").strip()
                ),
                "initial_shift": 2.5,
                "source_commit": _PINNED_MGCV_SOURCE_COMMIT,
                "controls": resolved,
                "provenance": self._efs_provenance(data),
            }

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

    def _smooth_construct_rpy2(
        self,
        smooth_expr: str,
        data: pd.DataFrame,
        absorb_cons: bool,
        knots: dict[str, np.ndarray] | None,
    ) -> dict[str, Any]:
        """Call smoothCon() via rpy2 and extract smooth construction details."""
        from rpy2.robjects import FloatVector, ListVector

        ro = self._ro
        r_df = self._to_r_dataframe(data)

        # Python booleans → R boolean strings for embedded R code
        absorb_str = "TRUE" if absorb_cons else "FALSE"
        knots_arg = ", knots=knots_input" if knots is not None else ""
        r_code = f"""
        library(mgcv)
        dat <- as.data.frame(dat_input)
        sm <- smoothCon({smooth_expr}, data=dat{knots_arg}, absorb.cons={absorb_str})[[1]]
        list(
            X = sm$X,
            S = sm$S,
            rank = sm$rank,
            null_space_dim = sm$null.space.dim,
            Xu = if (!is.null(sm$Xu)) sm$Xu else matrix(0, 0, 0),
            UZ = if (!is.null(sm$UZ)) sm$UZ else matrix(0, 0, 0),
            shift = if (!is.null(sm$shift)) sm$shift else numeric(0),
            knt = if (!is.null(sm$knt)) sm$knt else matrix(0, 0, 0),
            gp_defn = if (!is.null(sm$gp.defn)) sm$gp.defn else numeric(0),
            # mgcv:::gpE is version-pinned by RBridge.check_versions().
            E = if (!is.null(sm$knt))
                    mgcv:::gpE(sm$knt, sm$knt, sm$gp.defn)
                else matrix(0, 0, 0)
        )
        """
        ro.globalenv["dat_input"] = r_df
        if knots is not None:
            ro.globalenv["knots_input"] = ListVector(
                {
                    name: FloatVector(np.asarray(values, dtype=np.float64).ravel())
                    for name, values in knots.items()
                }
            )
        try:
            result = ro.r(r_code)
        finally:
            del ro.globalenv["dat_input"]
            if knots is not None:
                del ro.globalenv["knots_input"]

        X = np.array(result.rx2("X"), dtype=np.float64)
        rank_arr = np.array(result.rx2("rank"), dtype=np.float64).ravel()
        rank = int(rank_arr[0])
        rank_vector = rank_arr.astype(int)
        nsd_arr = np.array(result.rx2("null_space_dim"), dtype=np.float64).ravel()
        null_space_dim = int(nsd_arr[0])

        S_list = result.rx2("S")
        S_matrices = [np.array(S_list[i], dtype=np.float64) for i in range(len(S_list))]

        Xu = np.array(result.rx2("Xu"), dtype=np.float64)
        UZ = np.array(result.rx2("UZ"), dtype=np.float64)
        shift = np.array(result.rx2("shift"), dtype=np.float64)
        knt = np.array(result.rx2("knt"), dtype=np.float64)
        gp_defn = np.array(result.rx2("gp_defn"), dtype=np.float64)
        E = np.array(result.rx2("E"), dtype=np.float64)

        return {
            "X": X,
            "S": S_matrices,
            "rank": rank,
            "rank_vector": rank_vector,
            "null_space_dim": null_space_dim,
            "Xu": Xu,
            "UZ": UZ,
            "shift": shift,
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
        """Call smoothCon() via rpy2 and return all smooth objects."""
        ro = self._ro
        r_df = self._to_r_dataframe(data)

        absorb_str = "TRUE" if absorb_cons else "FALSE"
        r_code = f"""
        library(mgcv)
        dat <- as.data.frame(dat_input)
        sml <- smoothCon({smooth_expr}, data=dat, absorb.cons={absorb_str})
        n_sm <- length(sml)
        result <- list(n_sm=n_sm, smooths=list())
        for (i in seq_len(n_sm)) {{
            sm <- sml[[i]]
            by_lev <- if (!is.null(sm$by.level)) sm$by.level else "NONE"
            lab <- if (!is.null(sm$label)) sm$label else ""
            result$smooths[[i]] <- list(
                X = sm$X,
                S = sm$S,
                rank = sm$rank,
                null_space_dim = sm$null.space.dim,
                by_level = by_lev,
                label = lab
            )
        }}
        result
        """
        ro.globalenv["dat_input"] = r_df
        try:
            result = ro.r(r_code)
        finally:
            del ro.globalenv["dat_input"]

        n_sm = int(np.array(result.rx2("n_sm"))[0])
        smooths_r = result.rx2("smooths")

        results = []
        for i in range(n_sm):
            sm = smooths_r[i]
            X = np.array(sm.rx2("X"), dtype=np.float64)
            rank_arr = np.array(sm.rx2("rank"), dtype=np.float64).ravel()
            rank = int(rank_arr[0])
            nsd_arr = np.array(sm.rx2("null_space_dim"), dtype=np.float64).ravel()
            null_space_dim = int(nsd_arr[0])

            S_list_r = sm.rx2("S")
            S_matrices = [
                np.array(S_list_r[j], dtype=np.float64) for j in range(len(S_list_r))
            ]

            by_level_arr = np.array(sm.rx2("by_level"))
            by_level_str = str(by_level_arr[0]) if by_level_arr.size > 0 else None
            if by_level_str == "NONE":
                by_level_str = None

            label_arr = np.array(sm.rx2("label"))
            label = str(label_arr[0]) if label_arr.size > 0 else ""

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
