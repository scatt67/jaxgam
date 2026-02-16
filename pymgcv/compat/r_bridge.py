"""RBridge: interface to R's mgcv for reference comparison.

Two modes:
1. rpy2 (preferred): Direct R execution in-process
2. subprocess: Run Rscript and parse output (fallback)

Usage::

    from pymgcv.compat.r_bridge import RBridge
    import pandas as pd

    bridge = RBridge()
    data = pd.DataFrame({"x": x, "y": y})
    result = bridge.fit_gam("y ~ s(x)", data, family="gaussian")
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Any

import numpy as np
import pandas as pd


class RBridgeError(Exception):
    """Error communicating with R."""


class RBridge:
    """Interface to R's mgcv for reference comparison.

    Parameters
    ----------
    mode : str
        One of 'auto', 'rpy2', 'subprocess'. 'auto' tries rpy2 first,
        falls back to subprocess.
    """

    def __init__(self, mode: str = "auto") -> None:
        if mode == "auto":
            try:
                import rpy2.robjects  # noqa: F401

                self.mode = "rpy2"
                self._setup_rpy2()
            except Exception:
                # rpy2 may raise ImportError, ValueError, or OSError
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
        except Exception:
            # rpy2 may raise ImportError, ValueError (R_HOME not set),
            # or OSError (shared library not found)
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
            deviance, scale, Vp, reml_score.
        """
        if self.mode == "rpy2":
            return self._fit_rpy2(formula, data, family, method)
        return self._fit_subprocess(formula, data, family, method)

    def _fit_rpy2(
        self,
        formula: str,
        data: pd.DataFrame,
        family: str,
        method: str,
    ) -> dict[str, Any]:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri, pandas2ri

        r_family = self._get_r_family_rpy2(family)

        with ro.conversion.localconverter(
            ro.default_converter + pandas2ri.converter + numpy2ri.converter
        ):
            r_df = ro.conversion.py2rpy(data)

        r_model = self._mgcv.gam(
            ro.Formula(formula),
            data=r_df,
            family=r_family,
            method=method,
        )

        coefficients = np.array(r_model.rx2("coefficients"), dtype=np.float64)
        fitted_values = np.array(r_model.rx2("fitted.values"), dtype=np.float64)
        smoothing_params = np.array(r_model.rx2("sp"), dtype=np.float64)
        deviance = float(np.array(r_model.rx2("deviance"))[0])
        scale = float(np.array(r_model.rx2("scale"))[0])
        vp_r = r_model.rx2("Vp")
        p = len(coefficients)
        vp = np.array(vp_r, dtype=np.float64).reshape((p, p))
        reml_score = float(np.array(r_model.rx2("gcv.ubre"))[0])

        # EDF from summary
        r_summary = self._base.summary(r_model)
        edf = np.array(r_summary.rx2("edf"), dtype=np.float64)

        return {
            "coefficients": coefficients,
            "fitted_values": fitted_values,
            "smoothing_params": smoothing_params,
            "edf": edf,
            "deviance": deviance,
            "scale": scale,
            "Vp": vp,
            "reml_score": reml_score,
        }

    def _get_r_family_rpy2(self, family: str) -> Any:
        family_funcs = {
            "gaussian": self._stats.gaussian,
            "binomial": self._stats.binomial,
            "poisson": self._stats.poisson,
            "gamma": self._stats.Gamma,
        }
        func = family_funcs.get(family)
        if func is None:
            raise ValueError(
                f"Unknown family: {family!r}. Supported: {list(family_funcs.keys())}"
            )
        return func()

    def _fit_subprocess(
        self,
        formula: str,
        data: pd.DataFrame,
        family: str,
        method: str,
    ) -> dict[str, Any]:
        family_map = {
            "gaussian": "gaussian()",
            "binomial": "binomial()",
            "poisson": "poisson()",
            "gamma": "Gamma()",
        }
        r_family = family_map.get(family)
        if r_family is None:
            raise ValueError(
                f"Unknown family: {family!r}. Supported: {list(family_map.keys())}"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.csv")
            script_path = os.path.join(tmpdir, "fit.R")
            out = tmpdir  # output directory for CSVs

            data.to_csv(data_path, index=False)

            # No jsonlite dependency — serialize via write.csv and writeLines
            script = f"""\
library(mgcv)

data <- read.csv("{data_path}")
model <- gam({formula}, data=data, family={r_family}, method="{method}")
s <- summary(model)

write.csv(data.frame(v=as.numeric(coef(model))), "{out}/coefficients.csv", row.names=FALSE)
write.csv(data.frame(v=as.numeric(fitted(model))), "{out}/fitted_values.csv", row.names=FALSE)
write.csv(data.frame(v=as.numeric(model$sp)), "{out}/smoothing_params.csv", row.names=FALSE)
write.csv(data.frame(v=as.numeric(s$edf)), "{out}/edf.csv", row.names=FALSE)
writeLines(format(deviance(model), digits=15), "{out}/deviance.txt")
writeLines(format(model$scale, digits=15), "{out}/scale.txt")
write.csv(as.data.frame(model$Vp), "{out}/Vp.csv", row.names=FALSE)
writeLines(format(model$gcv.ubre, digits=15), "{out}/reml_score.txt")
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
                return pd.read_csv(os.path.join(out, name))["v"].values.astype(
                    np.float64
                )

            def _read_scalar(name: str) -> float:
                with open(os.path.join(out, name)) as f:
                    return float(f.read().strip())

            vp = pd.read_csv(os.path.join(out, "Vp.csv")).values.astype(np.float64)

            return {
                "coefficients": _read_vec("coefficients.csv"),
                "fitted_values": _read_vec("fitted_values.csv"),
                "smoothing_params": _read_vec("smoothing_params.csv"),
                "edf": _read_vec("edf.csv"),
                "deviance": _read_scalar("deviance.txt"),
                "scale": _read_scalar("scale.txt"),
                "Vp": vp,
                "reml_score": _read_scalar("reml_score.txt"),
            }

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
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri, pandas2ri

        r_family = self._get_r_family_rpy2(family)

        with ro.conversion.localconverter(
            ro.default_converter + pandas2ri.converter + numpy2ri.converter
        ):
            r_df = ro.conversion.py2rpy(data)

        r_model = self._mgcv.gam(
            ro.Formula(formula),
            data=r_df,
            family=r_family,
            method=method,
        )

        # Extract per-smooth basis and penalty
        smooth_list = r_model.rx2("smooth")
        n_smooths = len(smooth_list)

        # Get full model matrix once
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

        fit_result = self._fit_rpy2(formula, data, family, method)
        fit_result["basis_matrices"] = basis_matrices
        fit_result["penalty_matrices"] = penalty_matrices

        return fit_result

    def _get_smooth_components_subprocess(
        self,
        formula: str,
        data: pd.DataFrame,
        family: str,
        method: str,
    ) -> dict[str, Any]:
        family_map = {
            "gaussian": "gaussian()",
            "binomial": "binomial()",
            "poisson": "poisson()",
            "gamma": "Gamma()",
        }
        r_family = family_map.get(family)
        if r_family is None:
            raise ValueError(
                f"Unknown family: {family!r}. Supported: {list(family_map.keys())}"
            )

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.csv")
            script_path = os.path.join(tmpdir, "fit.R")
            out = tmpdir

            data.to_csv(data_path, index=False)

            script = f"""\
library(mgcv)

data <- read.csv("{data_path}")
model <- gam({formula}, data=data, family={r_family}, method="{method}")
s <- summary(model)

# Basic fit results
write.csv(data.frame(v=as.numeric(coef(model))), "{out}/coefficients.csv", row.names=FALSE)
write.csv(data.frame(v=as.numeric(fitted(model))), "{out}/fitted_values.csv", row.names=FALSE)
write.csv(data.frame(v=as.numeric(model$sp)), "{out}/smoothing_params.csv", row.names=FALSE)
write.csv(data.frame(v=as.numeric(s$edf)), "{out}/edf.csv", row.names=FALSE)
writeLines(format(deviance(model), digits=15), "{out}/deviance.txt")
writeLines(format(model$scale, digits=15), "{out}/scale.txt")
write.csv(as.data.frame(model$Vp), "{out}/Vp.csv", row.names=FALSE)
writeLines(format(model$gcv.ubre, digits=15), "{out}/reml_score.txt")

# Per-smooth basis and penalty matrices
X <- model.matrix(model)
n_smooths <- length(model$smooth)
writeLines(as.character(n_smooths), "{out}/n_smooths.txt")

for (i in seq_len(n_smooths)) {{
    sm <- model$smooth[[i]]
    first_col <- sm$first.para
    last_col <- sm$last.para
    Xblock <- X[, first_col:last_col, drop=FALSE]
    write.csv(as.data.frame(Xblock), sprintf("{out}/basis_%d.csv", i), row.names=FALSE)

    n_penalties <- length(sm$S)
    writeLines(as.character(n_penalties), sprintf("{out}/n_pen_%d.txt", i))
    for (j in seq_len(n_penalties)) {{
        write.csv(as.data.frame(sm$S[[j]]), sprintf("{out}/pen_%d_%d.csv", i, j), row.names=FALSE)
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
                return pd.read_csv(os.path.join(out, name))["v"].values.astype(
                    np.float64
                )

            def _read_scalar(name: str) -> float:
                with open(os.path.join(out, name)) as f:
                    return float(f.read().strip())

            def _read_matrix(name: str) -> np.ndarray:
                return pd.read_csv(os.path.join(out, name)).values.astype(np.float64)

            n_smooths = int(_read_scalar("n_smooths.txt"))
            basis_matrices = []
            penalty_matrices = []
            for i in range(1, n_smooths + 1):
                basis_matrices.append(_read_matrix(f"basis_{i}.csv"))
                n_pen = int(_read_scalar(f"n_pen_{i}.txt"))
                penalties = []
                for j in range(1, n_pen + 1):
                    penalties.append(_read_matrix(f"pen_{i}_{j}.csv"))
                penalty_matrices.append(penalties)

            return {
                "coefficients": _read_vec("coefficients.csv"),
                "fitted_values": _read_vec("fitted_values.csv"),
                "smoothing_params": _read_vec("smoothing_params.csv"),
                "edf": _read_vec("edf.csv"),
                "deviance": _read_scalar("deviance.txt"),
                "scale": _read_scalar("scale.txt"),
                "Vp": _read_matrix("Vp.csv"),
                "reml_score": _read_scalar("reml_score.txt"),
                "basis_matrices": basis_matrices,
                "penalty_matrices": penalty_matrices,
            }
