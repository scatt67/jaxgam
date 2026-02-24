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

        # reml.scale is the scale used in the REML criterion (jointly
        # optimized), which differs from model$scale (Fletcher estimate).
        reml_scale_r = r_model.rx2("reml.scale")
        reml_scale = (
            float(np.array(reml_scale_r)[0]) if reml_scale_r is not None else scale
        )

        # EDF from summary
        r_summary = self._base.summary(r_model)
        edf = np.array(r_summary.rx2("edf"), dtype=np.float64)

        null_deviance = float(np.array(r_model.rx2("null.deviance"))[0])

        return {
            "coefficients": coefficients,
            "fitted_values": fitted_values,
            "smoothing_params": smoothing_params,
            "edf": edf,
            "deviance": deviance,
            "null_deviance": null_deviance,
            "scale": scale,
            "reml_scale": reml_scale,
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
writeLines(format(model$null.deviance, digits=15), "{out}/null_deviance.txt")
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
                "null_deviance": _read_scalar("null_deviance.txt"),
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
        fit_result["model_matrix"] = X_full

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
                penalties = [
                    _read_matrix(f"pen_{i}_{j}.csv") for j in range(1, n_pen + 1)
                ]
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

    def smooth_construct(
        self,
        smooth_expr: str,
        data: pd.DataFrame,
        absorb_cons: bool = False,
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

        Returns
        -------
        dict
            Keys: X, S (list of penalty matrices), rank, null_space_dim,
            Xu (knots), UZ (mapping matrix), shift (centring values).
        """
        if self.mode == "rpy2":
            return self._smooth_construct_rpy2(smooth_expr, data, absorb_cons)
        return self._smooth_construct_subprocess(smooth_expr, data, absorb_cons)

    def _smooth_construct_rpy2(
        self,
        smooth_expr: str,
        data: pd.DataFrame,
        absorb_cons: bool,
    ) -> dict[str, Any]:
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri, pandas2ri

        with ro.conversion.localconverter(
            ro.default_converter + pandas2ri.converter + numpy2ri.converter
        ):
            r_df = ro.conversion.py2rpy(data)

        absorb_str = "TRUE" if absorb_cons else "FALSE"
        r_code = f"""
        library(mgcv)
        dat <- as.data.frame(dat_input)
        sm <- smoothCon({smooth_expr}, data=dat, absorb.cons={absorb_str})[[1]]
        list(
            X = sm$X,
            S = sm$S,
            rank = sm$rank,
            null_space_dim = sm$null.space.dim,
            Xu = if (!is.null(sm$Xu)) sm$Xu else matrix(0, 0, 0),
            UZ = if (!is.null(sm$UZ)) sm$UZ else matrix(0, 0, 0),
            shift = if (!is.null(sm$shift)) sm$shift else numeric(0)
        )
        """
        ro.globalenv["dat_input"] = r_df
        result = ro.r(r_code)

        X = np.array(result.rx2("X"), dtype=np.float64)
        rank_arr = np.array(result.rx2("rank"), dtype=np.float64).ravel()
        rank = int(rank_arr[0])
        nsd_arr = np.array(result.rx2("null_space_dim"), dtype=np.float64).ravel()
        null_space_dim = int(nsd_arr[0])

        S_list = result.rx2("S")
        S_matrices = [np.array(S_list[i], dtype=np.float64) for i in range(len(S_list))]

        Xu = np.array(result.rx2("Xu"), dtype=np.float64)
        UZ = np.array(result.rx2("UZ"), dtype=np.float64)
        shift = np.array(result.rx2("shift"), dtype=np.float64)

        return {
            "X": X,
            "S": S_matrices,
            "rank": rank,
            "null_space_dim": null_space_dim,
            "Xu": Xu,
            "UZ": UZ,
            "shift": shift,
        }

    def _smooth_construct_subprocess(
        self,
        smooth_expr: str,
        data: pd.DataFrame,
        absorb_cons: bool,
    ) -> dict[str, Any]:
        absorb_str = "TRUE" if absorb_cons else "FALSE"

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.csv")
            script_path = os.path.join(tmpdir, "smooth.R")
            out = tmpdir

            data.to_csv(data_path, index=False)

            script = f"""\
library(mgcv)

dat <- read.csv("{data_path}")
sm <- smoothCon({smooth_expr}, data=dat, absorb.cons={absorb_str})[[1]]

write.csv(as.data.frame(sm$X), "{out}/X.csv", row.names=FALSE)
writeLines(as.character(sm$rank), "{out}/rank.txt")
writeLines(as.character(sm$null.space.dim), "{out}/null_space_dim.txt")

n_S <- length(sm$S)
writeLines(as.character(n_S), "{out}/n_S.txt")
for (i in seq_len(n_S)) {{
    write.csv(as.data.frame(sm$S[[i]]), sprintf("{out}/S_%d.csv", i), row.names=FALSE)
}}

if (!is.null(sm$Xu)) {{
    if (is.matrix(sm$Xu)) {{
        write.csv(as.data.frame(sm$Xu), "{out}/Xu.csv", row.names=FALSE)
    }} else {{
        write.csv(data.frame(v=as.numeric(sm$Xu)), "{out}/Xu.csv", row.names=FALSE)
    }}
}} else {{
    writeLines("NULL", "{out}/Xu.csv")
}}

if (!is.null(sm$UZ)) {{
    write.csv(as.data.frame(sm$UZ), "{out}/UZ.csv", row.names=FALSE)
}} else {{
    writeLines("NULL", "{out}/UZ.csv")
}}

if (!is.null(sm$shift)) {{
    write.csv(data.frame(v=as.numeric(sm$shift)), "{out}/shift.csv", row.names=FALSE)
}} else {{
    writeLines("NULL", "{out}/shift.csv")
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
                path = os.path.join(out, name)
                with open(path) as f:
                    first_line = f.readline().strip()
                if first_line == "NULL":
                    return np.array([])
                return pd.read_csv(path).values.astype(np.float64)

            def _read_scalar(name: str) -> float:
                with open(os.path.join(out, name)) as f:
                    return float(f.read().strip())

            X = _read_matrix("X.csv")
            rank = int(_read_scalar("rank.txt"))
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
                "null_space_dim": null_space_dim,
                "Xu": Xu,
                "UZ": UZ,
                "shift": shift,
            }

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
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri, pandas2ri

        with ro.conversion.localconverter(
            ro.default_converter + pandas2ri.converter + numpy2ri.converter
        ):
            r_df = ro.conversion.py2rpy(data)

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
        result = ro.r(r_code)

        n_sm = int(np.array(result.rx2("n_sm"))[0])
        smooths_r = result.rx2("smooths")

        out = []
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

            out.append(
                {
                    "X": X,
                    "S": S_matrices,
                    "rank": rank,
                    "null_space_dim": null_space_dim,
                    "by_level": by_level_str,
                    "label": label,
                }
            )

        return out

    def _smooth_construct_list_subprocess(
        self,
        smooth_expr: str,
        data: pd.DataFrame,
        absorb_cons: bool,
    ) -> list[dict[str, Any]]:
        absorb_str = "TRUE" if absorb_cons else "FALSE"

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.csv")
            script_path = os.path.join(tmpdir, "smooth.R")
            out = tmpdir

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
writeLines(as.character(n_sm), "{out}/n_sm.txt")

for (i in seq_len(n_sm)) {{
    sm <- sml[[i]]
    write.csv(as.data.frame(sm$X), sprintf("{out}/X_%d.csv", i), row.names=FALSE)
    writeLines(as.character(sm$rank), sprintf("{out}/rank_%d.txt", i))
    writeLines(as.character(sm$null.space.dim), sprintf("{out}/nsd_%d.txt", i))

    n_S <- length(sm$S)
    writeLines(as.character(n_S), sprintf("{out}/nS_%d.txt", i))
    for (j in seq_len(n_S)) {{
        write.csv(as.data.frame(sm$S[[j]]), sprintf("{out}/S_%d_%d.csv", i, j), row.names=FALSE)
    }}

    by_lev <- if (!is.null(sm$by.level)) sm$by.level else "NONE"
    writeLines(by_lev, sprintf("{out}/bylevel_%d.txt", i))

    lab <- if (!is.null(sm$label)) sm$label else ""
    writeLines(lab, sprintf("{out}/label_%d.txt", i))
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
                return pd.read_csv(os.path.join(out, name)).values.astype(np.float64)

            def _read_scalar(name: str) -> float:
                with open(os.path.join(out, name)) as f:
                    return float(f.read().strip())

            def _read_text(name: str) -> str:
                with open(os.path.join(out, name)) as f:
                    return f.read().strip()

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

    def predict_gam(
        self,
        formula: str,
        train_data: pd.DataFrame,
        newdata: pd.DataFrame,
        family: str = "gaussian",
        method: str = "REML",
        type: str = "response",
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
        type : str
            Prediction type: ``'response'`` or ``'link'``.
        se_fit : bool
            Whether to return standard errors.

        Returns
        -------
        dict
            Keys: ``'predictions'``, optionally ``'se'``.
        """
        import rpy2.robjects as ro
        from rpy2.robjects import numpy2ri, pandas2ri

        r_family = self._get_r_family_rpy2(family)

        with ro.conversion.localconverter(
            ro.default_converter + pandas2ri.converter + numpy2ri.converter
        ):
            r_train = ro.conversion.py2rpy(train_data)
            r_new = ro.conversion.py2rpy(newdata)

        r_model = self._mgcv.gam(
            ro.Formula(formula),
            data=r_train,
            family=r_family,
            method=method,
        )

        pred = self._stats.predict(
            r_model,
            newdata=r_new,
            type=type,
            **{"se.fit": se_fit},
        )

        result: dict[str, Any] = {}
        if se_fit:
            result["predictions"] = np.array(pred.rx2("fit"), dtype=np.float64)
            result["se"] = np.array(pred.rx2("se.fit"), dtype=np.float64)
        else:
            result["predictions"] = np.array(pred, dtype=np.float64)

        return result

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

        # Deviance explained
        result["dev_explained"] = float(np.array(r_summary.rx2("dev.expl"))[0])

        # Scale
        result["scale"] = float(np.array(r_summary.rx2("scale"))[0])

        # Residual df
        result["residual_df"] = float(np.array(r_summary.rx2("residual.df"))[0])

        # N
        result["n"] = int(np.array(r_summary.rx2("n"))[0])

        # Per-smooth EDF
        edf_r = r_summary.rx2("edf")
        if edf_r is not None:
            result["edf"] = np.array(edf_r, dtype=np.float64)
        else:
            result["edf"] = np.array([])

        # SP criterion (REML/ML score)
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

        import os
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            data_path = os.path.join(tmpdir, "data.csv")
            script_path = os.path.join(tmpdir, "summary.R")
            out = tmpdir

            data.to_csv(data_path, index=False)

            script = f"""\
library(mgcv)

data <- read.csv("{data_path}")
model <- gam({formula}, data=data, family={r_family}, method="{method}")
s <- summary(model)

# Parametric table
if (!is.null(s$p.table)) {{
    write.csv(as.data.frame(s$p.table), "{out}/p_table.csv", row.names=TRUE)
}} else {{
    writeLines("NULL", "{out}/p_table.csv")
}}

# Smooth table
if (!is.null(s$s.table) && s$m > 0) {{
    write.csv(as.data.frame(s$s.table), "{out}/s_table.csv", row.names=TRUE)
}} else {{
    writeLines("NULL", "{out}/s_table.csv")
}}

# Scalars
writeLines(format(s$r.sq, digits=15), "{out}/r_sq.txt")
writeLines(format(s$dev.expl, digits=15), "{out}/dev_expl.txt")
writeLines(format(s$scale, digits=15), "{out}/scale.txt")
writeLines(format(s$residual.df, digits=15), "{out}/residual_df.txt")
writeLines(as.character(s$n), "{out}/n.txt")
write.csv(data.frame(v=as.numeric(s$edf)), "{out}/edf.csv", row.names=FALSE)
if (!is.null(s$sp.criterion)) {{
    writeLines(format(s$sp.criterion, digits=15), "{out}/sp_criterion.txt")
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
                with open(os.path.join(out, name)) as f:
                    return float(f.read().strip())

            def _read_matrix(name: str) -> np.ndarray | None:
                path = os.path.join(out, name)
                with open(path) as f:
                    first = f.readline().strip()
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
            result["edf"] = pd.read_csv(os.path.join(out, "edf.csv"))[
                "v"
            ].values.astype(np.float64)

            sp_path = os.path.join(out, "sp_criterion.txt")
            if os.path.exists(sp_path):
                result["sp_criterion"] = _read_scalar("sp_criterion.txt")
            else:
                result["sp_criterion"] = None

            return result
