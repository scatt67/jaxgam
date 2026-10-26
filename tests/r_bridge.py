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

import ctypes
import hashlib
import os
import subprocess
import tempfile
import time
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
import pandas as pd

from jaxgam.formula.terms import SmoothSpec
from tests.tolerances import STRICT

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


def _r_array(value: Any, dtype: Any = None) -> np.ndarray:
    """Copy an R vector, matrix or array into NumPy-owned memory.

    rpy2 exposes R double, integer and logical vectors through buffers that
    ``np.asarray`` wraps without copying, so such a view reads R-managed memory
    that R's garbage collector may reuse after any later R allocation once the
    R object is unreferenced. Every R-to-NumPy conversion in this bridge goes
    through this copy. Values, shape and dtype (unless ``dtype`` is given) are
    those ``np.asarray`` would produce; ``order="K"`` keeps R's column-major
    layout for matrices and arrays.
    """
    return np.array(value, dtype=dtype, copy=True, order="K")


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


@dataclass(frozen=True)
class DiscreteOperatorLayout:
    """Zero-based translation of mgcv's compact discrete operator metadata.

    ``marginal_tables``/``row_indices`` correspond to mgcv ``Xd``/``kd``;
    ``index_spans`` is its half-open Python form of ``ks``; ``term_starts``
    and ``term_dimensions`` translate ``ts``/``dt``.  Constraints, dropped
    columns and an R-to-public coefficient permutation are explicit.  The
    operator gate exercises a constrained tensor with a dropped coordinate
    and a different compact/public term order.
    """

    marginal_tables: tuple[np.ndarray, ...]
    row_indices: np.ndarray
    index_spans: np.ndarray
    term_starts: tuple[int, ...]
    term_dimensions: tuple[int, ...]
    constraint_vectors: tuple[np.ndarray, ...]
    constraint_codes: np.ndarray
    drop: np.ndarray | None = None
    r_to_public: np.ndarray | None = None

    def __post_init__(self) -> None:
        tables = tuple(np.asarray(table, dtype=float) for table in self.marginal_tables)
        kd = np.asarray(self.row_indices)
        ks = np.asarray(self.index_spans)
        qc = np.asarray(self.constraint_codes)
        if not tables or any(table.ndim != 2 for table in tables):
            raise ValueError(
                "discrete layout needs non-empty two-dimensional Xd tables"
            )
        if kd.ndim != 2 or kd.dtype.kind not in "iu" or kd.shape[1] == 0:
            raise ValueError("discrete layout kd must be a non-empty integer matrix")
        if ks.shape != (len(tables), 2) or ks.dtype.kind not in "iu":
            raise ValueError("discrete layout ks must be an integer (n_Xd, 2) matrix")
        if len(self.term_starts) != len(self.term_dimensions) or len(qc) != len(
            self.term_starts
        ):
            raise ValueError("discrete layout ts/dt/qc lengths are incompatible")
        if self.constraint_vectors and len(self.constraint_vectors) != len(
            self.term_starts
        ):
            raise ValueError(
                "discrete layout v entries must match term count when supplied"
            )
        if any(
            start < 0 or width <= 0 or start + width > len(tables)
            for start, width in zip(self.term_starts, self.term_dimensions, strict=True)
        ):
            raise ValueError("discrete layout ts/dt reference invalid Xd tables")
        if (
            np.any(ks[:, 0] < 0)
            or np.any(ks[:, 1] <= ks[:, 0])
            or np.any(ks[:, 1] > kd.shape[1])
        ):
            raise ValueError("discrete layout ks has invalid half-open selector spans")
        for table, span in zip(tables, ks, strict=True):
            selectors = kd[:, span[0] : span[1]]
            if np.any(selectors < 0) or np.any(selectors >= table.shape[0]):
                raise ValueError("discrete layout kd selector is out of bounds")
        object.__setattr__(self, "marginal_tables", tables)
        object.__setattr__(self, "row_indices", kd.astype(np.int32, copy=True))
        object.__setattr__(self, "index_spans", ks.astype(np.int32, copy=True))
        object.__setattr__(self, "constraint_codes", qc.astype(np.int32, copy=True))
        if self.drop is not None:
            object.__setattr__(self, "drop", np.asarray(self.drop, dtype=np.int32))
        if self.r_to_public is not None:
            permutation = np.asarray(self.r_to_public, dtype=np.int32)
            if not np.array_equal(np.sort(permutation), np.arange(len(permutation))):
                raise ValueError("r_to_public must be a coefficient permutation")
            object.__setattr__(self, "r_to_public", permutation)


@dataclass(frozen=True)
class CompressedDataFrame:
    """Pinned ``mgcv:::compress.df`` representatives and zero-based selector."""

    values: pd.DataFrame
    selector: np.ndarray

    def reconstruct(self) -> pd.DataFrame:
        """Expand representative rows into the original scan order."""
        return self.values.iloc[self.selector].reset_index(drop=True)


@dataclass(frozen=True)
class DiscreteModelFrame:
    """Pinned ``discrete.mf`` padded frame and translated index metadata."""

    frame: pd.DataFrame
    row_indices: np.ndarray
    n_unique: dict[str, int]
    index_spans: dict[str, tuple[int, int]]


@dataclass(frozen=True)
class BamDiscretePrefit:
    """Pinned ``bam(fit=FALSE)`` made with a fit's ``weights=``/``offset=``.

    ``frame`` is ``discrete.mf``'s padded compact frame (it includes the
    ``(weights)``/``(offset)`` columns when supplied), ``row_indices`` its
    zero-based index matrix ``k``, ``n_unique`` its ``nr`` and
    ``parametric_names`` bam's ``pmf.names``.
    """

    expanded_design: np.ndarray
    frame: pd.DataFrame
    row_indices: np.ndarray
    n_unique: dict[str, int]
    parametric_names: tuple[str, ...]


@dataclass(frozen=True)
class BamDiscreteSetup:
    """Pinned ``bam(fit=FALSE, discrete=...)`` metadata and expanded design."""

    layout: DiscreteOperatorLayout
    expanded_design: np.ndarray
    penalties: tuple[np.ndarray, ...]
    term_names: tuple[str, ...]


@dataclass(frozen=True)
class BamDiscretizedVariable:
    """One variable of pinned ``bam``'s own ``discrete.mf`` call.

    ``span`` is the zero-based half-open ``ks`` index-column span (jointly
    discretized variables share it), ``representatives`` the first ``nr``
    rows of the padded compact frame, and ``data_range`` R's ``range`` of the
    model-frame column (``None`` for factors).
    """

    name: str
    n_unique: int
    span: tuple[int, int]
    representatives: np.ndarray
    data_range: tuple[float, float] | None


@dataclass(frozen=True)
class BamDiscretization:
    """Pinned ``bam(discrete=)`` per-variable grids and term-to-margin map.

    ``term_margins`` maps every ``ts`` label to the variables of its ``Xd``
    margins in order (``(Intercept)`` is its own margin), and ``table_rows``
    holds ``nrow(Xd[[j]])``.
    """

    variables: dict[str, BamDiscretizedVariable]
    term_margins: dict[str, tuple[tuple[str, ...], ...]]
    table_rows: tuple[int, ...]


@dataclass(frozen=True)
class BamDiscreteFit:
    """Pinned fixed-sp ``bam(discrete=, method='fREML')`` result."""

    coefficients: np.ndarray
    fitted_values: np.ndarray
    deviance: float
    scale: float
    edf_total: float
    score: float
    converged: bool
    mgcv_converged: bool
    smoothing_params: np.ndarray
    n_iter: int
    trial_records: np.ndarray | None = None
    covariance: np.ndarray | None = None
    prediction: np.ndarray | None = None
    prediction_se: np.ndarray | None = None
    log_theta: float | None = None
    coefficient_edf: np.ndarray | None = None


@dataclass(frozen=True)
class SourceLinearAlgebra:
    """Pinned R's own LAPACK/BLAS libraries, from typed R queries.

    ``openblas_config`` is OpenBLAS's ``openblas_get_config()`` string read
    from the library R reports in ``La_library()``, or ``None`` when that
    library is not OpenBLAS (for example R's own libRlapack).
    """

    lapack_library: str
    lapack_version: str
    blas_library: str
    openblas_config: str | None

    def lapack_is_openblas(self, prefix: str) -> bool:
        """Whether R's LAPACK is the OpenBLAS build whose config has ``prefix``."""
        return self.openblas_config is not None and self.openblas_config.startswith(
            prefix
        )


@dataclass(frozen=True)
class SourceWorkingPenaltyBlock:
    """One pinned ``Sl.setup`` block with zero-based half-open coefficients.

    ``penalties`` are ``Sl[[b]]$S`` exactly as ``Sl.addS`` adds them at the
    ``penalized`` coordinates; ``roots`` and ``root_products`` are ``rS`` and
    ``pcrossprod(rS, trans=TRUE)`` as ``ldetSblock`` uses them.
    """

    start: int
    stop: int
    transform: np.ndarray
    penalties: tuple[np.ndarray, ...]
    roots: tuple[np.ndarray, ...]
    root_products: tuple[np.ndarray, ...]
    rank: int
    penalized: np.ndarray
    reparameterized: bool


@dataclass(frozen=True)
class SourceWorkingModelState:
    """Pinned ``Sl.fitChol`` inputs after ``Sl.initial.repara``."""

    blocks: tuple[SourceWorkingPenaltyBlock, ...]
    gram: np.ndarray
    rhs: np.ndarray
    response_square_sum: float
    n_obs: int
    null_dimension: int


@dataclass(frozen=True)
class SourceWorkingProposal:
    """One pinned ``Sl.fitChol`` evaluation and the exact state it used.

    ``smoothing`` is R's own ``exp(log_smoothing)``; ``coefficients`` are in
    the initial-reparameterization coordinates of ``SourceWorkingModelState``.
    """

    log_smoothing: np.ndarray
    smoothing: np.ndarray
    log_scale: float
    tolerance: float
    coefficients: np.ndarray
    gradient: np.ndarray
    hessian: np.ndarray
    step: np.ndarray
    logdet_penalty: float
    logdet_hessian: float


@dataclass(frozen=True)
class SourceSlFitCholEvaluation:
    """Matched pinned state, proposal and ``ldetS`` derivatives at one rho."""

    state: SourceWorkingModelState
    proposal: SourceWorkingProposal
    penalty_first: np.ndarray
    penalty_second: np.ndarray


@dataclass(frozen=True)
class SourceWorkingTrial:
    """``bgam.fitd`` locals immediately after one ``Sl.fitChol`` trial.

    ``accepted``/``step``/``deviance``/``previous_deviance``/``criterion``
    are ``lsp0``, ``Nstep``, ``dev``, ``devold`` and ``crit`` *before* this
    trial's acceptance decision, so each trial is a replayable source state.
    ``penalty_first``/``penalty_second`` are the trial's ``ldetS`` derivatives
    from the same frozen ``Sl``, as ``Sl.fitChol`` evaluates them.
    """

    iteration: int
    accepted: np.ndarray
    step: np.ndarray
    deviance: float
    previous_deviance: float
    criterion: float
    proposal: SourceWorkingProposal
    penalty_first: np.ndarray
    penalty_second: np.ndarray


@dataclass(frozen=True)
class SourceWorkingModelTrace:
    """Pinned additive-Gaussian ``bgam.fitd`` trial sequence and final fit."""

    state: SourceWorkingModelState
    trials: tuple[SourceWorkingTrial, ...]
    fit: BamDiscreteFit


@dataclass(frozen=True)
class SourceRegularWorkingTrial:
    """Non-additive ``bgam.fitd`` locals after one ``Sl.fitChol`` trial.

    ``coefficients`` is ``coef`` after this iteration's step control (``None``
    at iteration one, which starts from ``linkfun(mustart)``) and
    ``coefficient_trials`` its ``kk``. ``gram``/``rhs``/``response_square_sum``
    are ``qrx$R``, ``qrx$f`` and ``qrx$y.norm2`` in model coordinates, and
    ``state`` holds the same moments after ``Sl.initial.repara`` with the
    fit's ``Sl``. ``accepted``/``step``/``deviance``/``previous_deviance``/
    ``criterion`` are ``lsp0``, ``Nstep``, ``dev``, ``devold`` and ``crit``
    before this trial's acceptance decision.
    """

    iteration: int
    coefficient_trials: int
    coefficients: np.ndarray | None
    gram: np.ndarray
    rhs: np.ndarray
    state: SourceWorkingModelState
    accepted: np.ndarray
    step: np.ndarray
    deviance: float
    previous_deviance: float
    criterion: float
    proposal: SourceWorkingProposal
    log_theta: float | None = None


@dataclass(frozen=True)
class SourceRegularWorkingModelTrace:
    """Pinned non-additive ``bgam.fitd`` trial sequence and final fit.

    ``final_iteration``/``final_coefficient_trials``/``converged`` are
    ``iter``, ``kk`` and ``conv`` when the fitting loop ends: at convergence
    ``kk`` belongs to the converging iteration, which makes no proposal.
    When pinned R stops with an error instead (and the caller allowed it),
    ``failure`` holds R's message, ``trials`` the trials made before it, and
    ``fit`` and the loop-exit fields are ``None``. For ``nb`` every trial's
    ``log_theta`` is ``theta`` after that iteration's ``estimate.theta``, and
    ``final_log_theta`` is ``theta`` at the loop exit.
    """

    trials: tuple[SourceRegularWorkingTrial, ...]
    fit: BamDiscreteFit | None
    final_iteration: int | None
    final_coefficient_trials: int | None
    converged: bool | None
    failure: str | None = None
    final_log_theta: float | None = None


def _same_source_working_state(
    first: SourceWorkingModelState, second: SourceWorkingModelState
) -> bool:
    """Bitwise equality of two captured frozen ``bgam.fitd`` states."""
    if (
        first.response_square_sum != second.response_square_sum
        or first.n_obs != second.n_obs
        or first.null_dimension != second.null_dimension
        or len(first.blocks) != len(second.blocks)
        or not np.array_equal(first.gram, second.gram)
        or not np.array_equal(first.rhs, second.rhs)
    ):
        return False
    for left, right in zip(first.blocks, second.blocks, strict=True):
        if (
            (left.start, left.stop, left.rank, left.reparameterized)
            != (right.start, right.stop, right.rank, right.reparameterized)
            or not np.array_equal(left.transform, right.transform)
            or not np.array_equal(left.penalized, right.penalized)
        ):
            return False
        for name in ("penalties", "roots", "root_products"):
            values, others = getattr(left, name), getattr(right, name)
            if len(values) != len(others) or not all(
                np.array_equal(a, b) for a, b in zip(values, others, strict=True)
            ):
                return False
    return True


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

    @contextmanager
    def _preserved_r_seed(self, seed: int):
        """Keep a pinned random fixture private to the embedded R session."""
        base = self._ro.baseenv
        environment = self._ro.globalenv
        had_seed = bool(
            base["exists"](".Random.seed", envir=environment, inherits=False)[0]
        )
        old_seed = base["get"](".Random.seed", envir=environment) if had_seed else None
        try:
            base["set.seed"](seed)
            yield
        finally:
            if had_seed:
                base["assign"](".Random.seed", old_seed, envir=environment)
            else:
                base["rm"](".Random.seed", envir=environment)

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

    def source_numeric_tokens(self, values: np.ndarray) -> np.ndarray:
        """Render numeric values with pinned R's ``paste0`` coercion."""
        self._require_rpy2()
        numbers = np.asarray(values, dtype=np.float64)
        if numbers.ndim != 1:
            raise ValueError("Numeric-token oracle requires a vector")
        return _r_array(self._base.paste0(self._to_r_vector(numbers)), dtype=str)

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

    def benchmark_efs(
        self,
        formula: str,
        data_path: Path,
        initial_sp: np.ndarray,
        *,
        repeats: int,
        threads: int,
        pirls_tolerance: float,
        pirls_max_iter: int,
        log_lambda_max: float,
        score_tolerance: float,
    ) -> dict[str, Any]:
        """Measure repeated Poisson EFS fits of the saved benchmark input.

        R still reads the identical saved CSV. Formula, data, family and
        controls are prepared once; timings include the synchronous rpy2 call
        boundary and the complete R fit, with result extraction afterward.
        """
        from rpy2.rinterface_lib.embedded import RRuntimeError

        self._require_rpy2()
        if repeats < 1 or threads < 1:
            raise ValueError("Benchmark repeats and threads must be positive.")
        r_data = self._utils.read_csv(str(data_path))
        r_formula = self._ro.Formula(formula)
        r_family = self._stats.poisson()
        r_control = self._mgcv.gam_control(
            epsilon=pirls_tolerance,
            maxit=pirls_max_iter,
            nthreads=threads,
            **{"efs.lspmax": log_lambda_max, "efs.tol": score_tolerance},
        )
        initial = self._ro.ListVector(
            {"sp": self._to_r_vector(initial_sp), "scale": self._ro.FloatVector([1.0])}
        )
        durations: list[float] = []
        validity: list[bool] = []
        for _ in range(repeats + 1):
            start = time.perf_counter()
            try:
                model = self._mgcv.gam(
                    r_formula,
                    data=r_data,
                    family=r_family,
                    method="REML",
                    optimizer="efs",
                    scale=1.0,
                    control=r_control,
                    **{"in.out": initial},
                )
            except RRuntimeError as error:
                raise RBridgeError(f"Pinned R benchmark failed: {error}") from error
            durations.append(time.perf_counter() - start)
            outer = model.rx2("outer.info")
            converged = (
                bool(model.rx2("converged")[0])
                and str(outer.rx2("conv")[0]) == "full convergence"
            )
            finite = all(
                np.all(np.isfinite(_r_array(model.rx2(name), dtype=np.float64)))
                for name in (
                    "coefficients",
                    "fitted.values",
                    "sp",
                    "edf",
                    "Vp",
                    "linear.predictors",
                    "deviance",
                    "gcv.ubre",
                    "scale",
                )
            )
            validity.append(bool(converged and finite))
        summary = self._call_internal("summary.gam", model)
        return {
            "cold_seconds": durations[0],
            "warm_seconds": durations[1:],
            "validity": validity,
            "result": {
                "coefficients": _r_array(model.rx2("coefficients")).tolist(),
                "fitted_values": _r_array(model.rx2("fitted.values")).tolist(),
                "smoothing_params": _r_array(model.rx2("sp")).tolist(),
                "edf": _r_array(summary.rx2("edf")).tolist(),
                "deviance": float(model.rx2("deviance")[0]),
                "score": float(model.rx2("gcv.ubre")[0]),
                "scale": float(model.rx2("scale")[0]),
                "outer_iterations": int(outer.rx2("iter")[0]),
                "converged": converged,
                "convergence_info": str(outer.rx2("conv")[0]),
            },
            "session_info": list(self._utils.capture_output(self._utils.sessionInfo())),
        }

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

    def efs_nb_null_deviance(
        self, y: np.ndarray, mu: float, weights: np.ndarray, theta: float
    ) -> float:
        """Evaluate pinned NB ``dev.resids`` with the source's R reduction."""
        self._require_rpy2()
        y = np.asarray(y, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if y.ndim != 1 or weights.shape != y.shape:
            raise ValueError("NB null-deviance inputs need matching vectors")
        if not np.isfinite(mu) or not np.isfinite(theta) or theta <= 0:
            raise ValueError("NB null-deviance needs finite mu and positive theta")
        family = self._mgcv.nb(theta=float(theta))
        deviance = family.rx2("dev.resids")(
            self._to_r_vector(y),
            self._base.rep(float(mu), len(y)),
            self._to_r_vector(weights),
            self._base.log(float(theta)),
        )
        return float(self._base.sum(deviance)[0])

    def efs_startup_reference(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        penalties: list[np.ndarray],
        ranks: list[int],
        offsets: list[int],
        *,
        family: str,
        link: str,
        theta: float,
    ) -> dict[str, np.ndarray | float]:
        """Run pinned ``get.null.coef`` and ``initial.spg`` on supplied arrays."""
        self._require_rpy2()
        from rpy2 import rinterface

        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        if X.ndim != 2 or y.shape != (X.shape[0],) or weights.shape != y.shape:
            raise ValueError("EFS startup arrays have incompatible dimensions")
        if len(penalties) != len(ranks) or len(ranks) != len(offsets):
            raise ValueError("EFS startup penalty metadata lengths differ")
        if family == "nb":
            r_family = self._mgcv.nb(theta=float(theta), link=link)
        else:
            constructors = {
                "Gamma": self._stats.Gamma,
                "gaussian": self._stats.gaussian,
                "poisson": self._stats.poisson,
                "binomial": self._stats.binomial,
            }
            try:
                r_family = constructors[family](link=link)
            except KeyError:
                raise ValueError(
                    f"Unsupported EFS startup family: {family!r}"
                ) from None
        r_family = self._call_internal("fix.family", r_family)
        r_x = self._to_r_matrix(X)
        r_y = self._to_r_vector(y)
        r_weight = self._to_r_vector(weights)
        r_penalties = rinterface.ListSexpVector(
            [self._to_r_matrix(value) for value in penalties]
        )
        r_ranks = self._to_r_vector(np.asarray(ranks, dtype=np.int32))
        r_offsets = self._to_r_vector(np.asarray(offsets, dtype=np.int32))
        setup = self._ro.ListVector(
            {
                "X": r_x,
                "y": r_y,
                "w": r_weight,
                "n": self._ro.IntVector([len(y)]),
                "family": r_family,
            }
        )
        null = self._call_internal("get.null.coef", setup)
        sp = self._call_internal(
            "initial.spg",
            r_x,
            r_y,
            r_weight,
            r_family,
            r_penalties,
            r_ranks,
            r_offsets,
        )
        r_phi = (
            self._ro.FloatVector([1.0])
            if family in {"poisson", "binomial", "nb"}
            else self._ro.r["/"](null.rx2("null.scale"), 10)
        )
        return {
            "log_smoothing": _r_array(self._base.log(sp), dtype=np.float64),
            "null_coefficients": _r_array(null.rx2("null.coef"), dtype=np.float64),
            "scale": float(_r_array(r_phi, dtype=np.float64)[0]),
        }

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
            The typed rpy2 route also returns inner_converged,
            outer_converged, and outer_status from the fitted R object.
        """
        if self.mode == "rpy2":
            return self._fit_rpy2(formula, data, family, method)
        return self._fit_subprocess(formula, data, family, method)

    def _source_cubic_spec(self, *, basis: str, k: int) -> Any:
        """Build an unevaluated cubic specification with ``x`` as an R symbol."""
        self._require_rpy2()
        from rpy2 import rinterface

        from tests.r_ast import symbol

        arguments = rinterface.ListSexpVector(
            [
                self._mgcv.s,
                symbol("x"),
                self._ro.StrVector([basis]),
                self._ro.IntVector([k]),
            ]
        )
        arguments.names = self._ro.StrVector(["", "", "bs", "k"])
        return self._ro.r["evalq"](self._ro.r["as.call"](arguments))

    def source_cubic_smooth(self, x: np.ndarray, *, basis: str, k: int) -> Any:
        """Construct one installed mgcv cubic smooth as an R object."""
        spec = self._source_cubic_spec(basis=basis, k=k)
        frame = self._to_r_dataframe(pd.DataFrame({"x": x}))
        return self._call_internal("smooth.construct", spec, frame, knots=self._ro.NULL)

    def source_cubic_prediction(
        self, x: np.ndarray, new_x: np.ndarray, *, k: int
    ) -> np.ndarray:
        """Evaluate pinned cubic extrapolation through ``PredictMat``."""
        spec = self._source_cubic_spec(basis="cr", k=k)
        source = self._to_r_dataframe(pd.DataFrame({"x": x}))
        smooth = self._mgcv.smoothCon(spec, source, absorb_cons=False)[0]
        frame = self._to_r_dataframe(pd.DataFrame({"x": new_x}))
        return _r_array(self._mgcv.PredictMat(smooth, frame))

    def source_cubic_knots(self, x: np.ndarray, *, basis: str, k: int) -> np.ndarray:
        """Return pinned cubic knot placement from ``smooth.construct``."""
        return _r_array(self.source_cubic_smooth(x, basis=basis, k=k).rx2("xp"))

    def source_cubic_shrinkage_penalty(self, penalty: np.ndarray) -> np.ndarray:
        """Apply pinned R eigenvalue shrinkage to an identical raw penalty."""
        self._require_rpy2()
        base = self._ro.baseenv
        eigensystem = self._base.eigen(self._to_r_matrix(penalty), symmetric=True)
        values = eigensystem.rx2("values")
        count = len(values)
        values = base["[<-"](
            values,
            count - 1,
            value=base["*"](base["["](values, count - 2), 0.1),
        )
        values = base["[<-"](
            values,
            count,
            value=base["*"](base["["](values, count - 1), 0.1),
        )
        vectors = eigensystem.rx2("vectors")
        return _r_array(
            base["%*%"](
                vectors,
                base["*"](self._base.as_numeric(values), self._base.t(vectors)),
            )
        )

    def source_penalty_frobenius_norms(
        self, penalties: tuple[np.ndarray, ...]
    ) -> np.ndarray:
        """Reduce source penalty squares in R's column-major sum order."""
        self._require_rpy2()
        multiply = self._ro.baseenv["*"]
        values = []
        for penalty in penalties:
            matrix = np.asarray(penalty, dtype=np.float64)
            if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
                raise ValueError("Source penalty norm requires square matrices")
            source_matrix = self._to_r_matrix(matrix)
            values.append(
                float(
                    self._base.sqrt(
                        self._base.sum(multiply(source_matrix, source_matrix))
                    )[0]
                )
            )
        return np.asarray(values, dtype=np.float64)

    def source_gaussian_rank_fit(
        self,
        formula: str,
        data: pd.DataFrame,
        smoothing_parameters: np.ndarray,
        weights: np.ndarray,
        offset: np.ndarray,
    ) -> dict[str, Any]:
        """Call pinned ``gam.setup``/``gam.fit3`` in its public rank gauge."""
        from rpy2.robjects import ListVector

        self._require_rpy2()
        smoothing = np.atleast_1d(np.asarray(smoothing_parameters, dtype=np.float64))
        weight_array = np.asarray(weights, dtype=np.float64)
        offset_array = np.asarray(offset, dtype=np.float64)
        if (
            smoothing.ndim != 1
            or not len(smoothing)
            or np.any(smoothing <= 0)
            or weight_array.shape != (len(data),)
            or offset_array.shape != (len(data),)
        ):
            raise ValueError("Gaussian rank oracle inputs are invalid")
        r_formula = self._ro.Formula(formula)
        frame = self._to_r_dataframe(data)
        interpreted = self._call_internal("interpret.gam", r_formula)
        setup = self._call_internal(
            "gam.setup", interpreted, self._stats.terms(interpreted.rx2("pf")), frame
        )
        design = setup.rx2("X")
        width = int(self._base.ncol(design)[0])
        roots = self._call_internal(
            "mini.roots",
            setup.rx2("S"),
            setup.rx2("off"),
            width,
            setup.rx2("rank"),
        )
        penalty_space = self._call_internal(
            "totalPenaltySpace",
            setup.rx2("S"),
            setup.rx2("H"),
            setup.rx2("off"),
            width,
        )
        range_basis = penalty_space.rx2("Y")
        null_basis = penalty_space.rx2("Z")
        transform = self._base.cbind(range_basis, null_basis)
        null_width = int(self._base.ncol(null_basis)[0])
        matrix_product = self._ro.baseenv["%*%"]
        projected_roots = ListVector(
            [
                (str(index), matrix_product(self._base.t(range_basis), root))
                for index, root in enumerate(roots)
            ]
        )
        family = self._stats.gaussian()
        for patch in ("fix.family.link", "fix.family.var", "fix.family.ls"):
            family = self._call_internal(patch, family)
        response_name = formula.split("~", 1)[0].strip()
        fit = self._call_internal(
            "gam.fit3",
            x=design,
            y=self._to_r_vector(data[response_name].to_numpy(dtype=np.float64)),
            sp=self._base.log(self._to_r_vector(smoothing)),
            Eb=penalty_space.rx2("E"),
            UrS=projected_roots,
            weights=self._to_r_vector(weight_array),
            offset=self._to_r_vector(offset_array),
            U1=transform,
            Mp=null_width,
            family=family,
            control=self._mgcv.gam_control(maxit=1),
            deriv=0,
            scale=1,
        )
        public_factor = fit.rx2("rV")
        covariance = matrix_product(public_factor, self._base.t(public_factor))
        # gam.fit3's fitting coordinates: rp <- gam.reparam(UrS, sp),
        # T <- U1 %*% blockdiag(Qs, I), rS[[i]] <- rbind(rp$rS[[i]], 0) and
        # Eb <- Eb %*% T, each formed with R's own operations.
        reparam = self._call_internal(
            "gam.reparam",
            projected_roots,
            self._base.log(self._to_r_vector(smoothing)),
            0,
        )
        qs = _r_array(reparam.rx2("Qs"), dtype=np.float64)
        block = np.eye(width)
        block[: qs.shape[0], : qs.shape[1]] = qs
        fitting_transform = matrix_product(transform, self._to_r_matrix(block))
        roots_in_fit = [
            np.vstack(
                (
                    _r_array(root, dtype=np.float64),
                    np.zeros((null_width, int(self._base.ncol(root)[0]))),
                )
            )
            for root in reparam.rx2("rS")
        ]
        balanced = matrix_product(penalty_space.rx2("E"), fitting_transform)
        # gdi1 zeroes the rows of rV for the coordinates it drops; gam.fit3
        # returns T %*% rV, so t(T) %*% rV recovers them to rounding.
        fitting_factor = _r_array(
            matrix_product(self._base.t(fitting_transform), public_factor),
            dtype=np.float64,
        )
        # Dropped rows are rounding-level against the largest row; STRICT's
        # relative precision separates them from every retained row.
        row_norms = np.sqrt(np.sum(fitting_factor**2, axis=1))
        drop = np.flatnonzero(row_norms <= STRICT.rtol * float(np.max(row_norms)))
        return {
            "X": _r_array(design, dtype=np.float64),
            "S": [_r_array(penalty, dtype=np.float64) for penalty in setup.rx2("S")],
            "coefficients": _r_array(fit.rx2("coefficients"), dtype=np.float64),
            "covariance": _r_array(covariance, dtype=np.float64),
            "score": float(fit.rx2("REML")[0]),
            "fitted_values": _r_array(fit.rx2("fitted.values"), dtype=np.float64),
            "deviance": float(_r_array(fit.rx2("deviance"), dtype=np.float64)[0]),
            # trA is gdi1's influence trace (the EDF); with the default
            # Fletcher estimator and Gaussian dvar = 0, scale.est is
            # Pearson / (n - trA).
            "edf": float(_r_array(fit.rx2("trA"), dtype=np.float64)[0]),
            "scale": float(_r_array(fit.rx2("scale.est"), dtype=np.float64)[0]),
            "transform": _r_array(fitting_transform, dtype=np.float64),
            "penalty_space": _r_array(transform, dtype=np.float64),
            "balanced_root": _r_array(balanced, dtype=np.float64),
            "fitting_roots": roots_in_fit,
            "drop": drop.astype(np.int64),
        }

    def source_gaussian_fixed_sp_fit(
        self,
        formula: str,
        data: pd.DataFrame,
        smoothing: list[float] | np.ndarray,
        weights: np.ndarray,
        offset: np.ndarray,
    ) -> dict[str, Any]:
        """Fit pinned weighted/offset Gaussian ``gam`` at fixed smoothing."""
        self._require_rpy2()
        model = self._mgcv.gam(
            self._ro.Formula(formula),
            data=self._to_r_dataframe(data),
            family=self._stats.gaussian(),
            method="REML",
            sp=self._ro.FloatVector(np.asarray(smoothing, dtype=np.float64)),
            weights=self._ro.FloatVector(np.asarray(weights, dtype=np.float64)),
            offset=self._ro.FloatVector(np.asarray(offset, dtype=np.float64)),
        )
        return self._extract_fit_results_rpy2(model)

    def source_gaussian_sl_fit(
        self,
        data: pd.DataFrame,
        weights: np.ndarray,
        offset: np.ndarray,
        rho: np.ndarray,
        log_phi: float,
    ) -> dict[str, Any]:
        """Evaluate pinned ``qr_update``/``Sl.fit`` on a weighted tensor fit."""
        self._require_rpy2()
        weight_array = np.asarray(weights, dtype=np.float64)
        offset_array = np.asarray(offset, dtype=np.float64)
        parameters = np.asarray(rho, dtype=np.float64)
        if (
            weight_array.shape != (len(data),)
            or offset_array.shape != (len(data),)
            or parameters.shape != (2,)
            or np.any(weight_array <= 0)
        ):
            raise ValueError("Gaussian Sl.fit oracle inputs are invalid")
        formula = self._ro.Formula("y ~ te(x,z,k=4)")
        frame = self._to_r_dataframe(data)
        interpreted = self._call_internal("interpret.gam", formula)
        setup = self._call_internal(
            "gam.setup", interpreted, self._stats.terms(interpreted.rx2("pf")), frame
        )
        design = _r_array(setup.rx2("X"), dtype=np.float64)
        response = _r_array(setup.rx2("y"), dtype=np.float64)
        square_root_weight = np.sqrt(weight_array)
        qr = self._call_internal(
            "qr_update",
            self._to_r_matrix(square_root_weight[:, None] * design),
            self._to_r_vector(square_root_weight * (response - offset_array)),
        )
        fitted = _r_array(qr.rx2("f"), dtype=np.float64)
        tail = float(qr.rx2("y.norm2")[0]) - float(np.sum(fitted * fitted))
        prepared = self._call_internal(
            "Sl.Xprep", self._call_internal("Sl.setup", setup), qr.rx2("R")
        )
        result = self._call_internal(
            "Sl.fit",
            prepared.rx2("Sl"),
            prepared.rx2("X"),
            qr.rx2("f"),
            self._to_r_vector(parameters),
            self._to_r_vector(np.zeros(2, dtype=bool)),
            float(log_phi),
            False,
            tail,
            len(data),
            prepared.rx2("Mp"),
        )
        return {
            "X": design.copy(),
            "S": [_r_array(penalty, dtype=np.float64) for penalty in setup.rx2("S")],
            "score": float(result.rx2("reml")[0]),
            "gradient": _r_array(result.rx2("reml1"), dtype=np.float64),
            "hessian": _r_array(result.rx2("reml2"), dtype=np.float64),
        }

    def source_gaussian_fixed_rho_qr_refit(
        self,
        factor: np.ndarray,
        transformed_response: np.ndarray,
        residual_tail: float,
        log_smoothing: np.ndarray,
        scale: float,
        n_positive_weights: int,
        sum_log_positive_weights: float,
    ) -> float:
        """Refit a compact Gaussian QR system with pinned base-R arithmetic."""
        self._require_rpy2()
        R = np.asarray(factor, dtype=np.float64)
        qtz = np.asarray(transformed_response, dtype=np.float64)
        rho = np.asarray(log_smoothing, dtype=np.float64)
        if R.shape != (3, 3) or qtz.shape != (3,) or rho.shape != (2,):
            raise ValueError("Fixed-rho QR oracle has incompatible dimensions")
        if scale <= 0 or n_positive_weights <= 0:
            raise ValueError(
                "Fixed-rho QR oracle requires positive scale and weight count"
            )
        penalty_root = np.array(
            ((0.0, np.exp(rho[0] / 2), 0.0), (0.0, 0.0, np.exp(rho[1] / 2)))
        )
        source_factor = self._to_r_matrix(R)
        source_root = self._to_r_matrix(penalty_root)
        response = self._to_r_vector(qtz)
        base = self._ro.baseenv
        hessian = base["+"](
            self._base.crossprod(source_factor), self._base.crossprod(source_root)
        )
        coefficients = self._base.solve(
            hessian, self._base.crossprod(source_factor, response)
        )
        residual = base["-"](response, base["%*%"](source_factor, coefficients))
        penalty = base["%*%"](source_root, coefficients)
        rss = residual_tail + float(self._base.sum(base["*"](residual, residual))[0])
        quadratic = float(self._base.sum(base["*"](penalty, penalty))[0])
        logdet_h = float(
            self._base.determinant(hessian, logarithm=True).rx2("modulus")[0]
        )
        penalty_qr = self._base.qr(self._to_r_matrix(penalty_root[:, 1:]))
        penalty_factor = base["qr.R"](penalty_qr)
        logdet_s = 2 * float(
            self._base.sum(
                self._base.log(self._base.abs(self._base.diag(penalty_factor)))
            )[0]
        )
        saturated = (
            -n_positive_weights * np.log(2 * np.pi * scale) / 2
            + sum_log_positive_weights / 2
        )
        return (
            (rss + quadratic) / (2 * scale)
            - saturated
            + (logdet_h - logdet_s) / 2
            - np.log(2 * np.pi * scale) / 2
        )

    def source_gaussian_tensor_sl_fit(
        self, log_smoothing: np.ndarray, log_phi: float
    ) -> dict[str, Any]:
        """Return pinned tensor ``Sl.fit`` and its frozen source coordinates."""
        self._require_rpy2()
        rho = np.asarray(log_smoothing, dtype=np.float64)
        if rho.shape != (2,) or not np.all(np.isfinite(rho)):
            raise ValueError("Tensor Sl.fit requires two finite log penalties")
        n = 53
        with self._preserved_r_seed(742):
            x = _r_array(self._stats.runif(n), dtype=np.float64)
            z = _r_array(self._stats.runif(n), dtype=np.float64)
            noise = _r_array(self._stats.rnorm(n, sd=0.17), dtype=np.float64)
        offset = np.linspace(-0.15, 0.25, n)
        weights = 0.4 + (np.arange(1, n + 1) % 9) / 7
        response = 0.3 + np.sin(3 * x) - 0.5 * np.cos(4 * z) + offset + noise
        frame = self._to_r_dataframe(pd.DataFrame({"x": x, "z": z, "y": response}))
        formula = self._ro.Formula("y ~ te(x,z,bs=c('cr','cr'),k=c(4,4))")
        interpreted = self._call_internal("interpret.gam", formula)
        setup = self._call_internal(
            "gam.setup", interpreted, self._stats.terms(interpreted.rx2("pf")), frame
        )
        base = self._ro.baseenv
        setup = base["[[<-"](setup, "w", value=self._to_r_vector(weights))
        setup = base["[[<-"](setup, "offset", value=self._to_r_vector(offset))
        design = _r_array(setup.rx2("X"), dtype=np.float64)
        target = _r_array(setup.rx2("y"), dtype=np.float64) - offset
        square_root_weight = np.sqrt(weights)
        qr = self._call_internal(
            "qr_update",
            self._to_r_matrix(square_root_weight[:, None] * design),
            self._to_r_vector(square_root_weight * target),
        )
        transformed_response = _r_array(qr.rx2("f"), dtype=np.float64)
        residual_tail = float(qr.rx2("y.norm2")[0]) - float(
            np.sum(transformed_response * transformed_response)
        )
        prepared = self._call_internal(
            "Sl.Xprep", self._call_internal("Sl.setup", setup), qr.rx2("R")
        )
        fit = self._call_internal(
            "Sl.fit",
            prepared.rx2("Sl"),
            prepared.rx2("X"),
            qr.rx2("f"),
            self._to_r_vector(rho),
            self._to_r_vector(np.zeros(2, dtype=bool)),
            float(log_phi),
            False,
            residual_tail,
            n,
            prepared.rx2("Mp"),
        )
        anchor = self._to_r_vector(np.array((-2.0, 1.0)))
        width = int(self._base.ncol(prepared.rx2("X"))[0])
        determinant = self._call_internal(
            "ldetS",
            prepared.rx2("Sl"),
            anchor,
            self._to_r_vector(np.ones(2, dtype=bool)),
            width,
            root=True,
        )
        block = prepared.rx2("Sl")[0]
        grouped = self._call_internal(
            "gam.reparam", block.rx2("rS"), lsp=anchor, deriv=2
        )
        block_columns = np.arange(
            int(block.rx2("start")[0]), int(block.rx2("stop")[0]) + 1
        )
        selected = _r_array(block.rx2("ind"))
        active = (
            block_columns[
                selected.astype(bool)
                if selected.size == block_columns.size
                and np.all(np.isin(selected, (0, 1)))
                else selected.astype(int) - 1
            ]
            - 1
        )
        roots = []
        for value in grouped.rx2("rS"):
            matrix = _r_array(value, dtype=np.float64)
            root = np.zeros((matrix.shape[0], width))
            root[:, active] = matrix.T
            roots.append(root)
        refit_coefficients = self._call_internal(
            "Sl.repara", fit.rx2("rp"), fit.rx2("beta"), inverse=True
        )
        prediction = base["%*%"](prepared.rx2("X"), refit_coefficients)
        return {
            "R": _r_array(
                self._call_internal(
                    "Sl.repara", determinant.rx2("rp"), prepared.rx2("X")
                ),
                dtype=np.float64,
            ),
            "qtz": transformed_response.copy(),
            "residual_tail": residual_tail,
            "root": _r_array(determinant.rx2("E"), dtype=np.float64),
            "score": float(fit.rx2("reml")[0]),
            "roots": roots,
            "root_ranks": _r_array(setup.rx2("smooth")[0].rx2("rank"), dtype=int),
            "gradient": _r_array(fit.rx2("reml1"), dtype=np.float64),
            "hessian": _r_array(fit.rx2("reml2"), dtype=np.float64),
            "penalty_nullity": int(prepared.rx2("Mp")[0]),
            "n_obs": n,
            "n_positive_weights": int(np.count_nonzero(weights > 0)),
            "sum_log_positive_weights": float(np.log(weights).sum()),
            "scale": float(np.exp(log_phi)),
            "prediction": _r_array(prediction, dtype=np.float64).ravel().copy(),
        }

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

    def qr_constant_coefficients(
        self, design: np.ndarray, constant: float
    ) -> np.ndarray:
        """Pinned R ``qr.coef(qr(X), rep(constant,n))`` with NA pivot zeros.

        Use the source QR ordering for rank-deficient no-intercept anchors.
        """
        self._require_rpy2()
        design = np.asarray(design, dtype=np.float64)
        if design.ndim != 2 or not np.all(np.isfinite(design)):
            raise ValueError("QR design must be a finite matrix.")
        if not np.isfinite(constant):
            raise ValueError("QR constant must be finite.")
        qr = self._base.qr(self._to_r_matrix(design))
        response = self._base.rep(float(constant), design.shape[0])
        coefficients = self._ro.r["qr.coef"](qr, response)
        coefficients = self._base.replace(
            coefficients, self._base.is_na(coefficients), 0.0
        )
        result = _r_array(coefficients, dtype=np.float64)
        if result.shape != (design.shape[1],):
            raise RBridgeError("Pinned QR oracle returned the wrong coefficient size")
        return result

    def sample_int_permutation(self, size: int, seed: int) -> np.ndarray:
        """Pinned R RNG reference for ``sample.int(size,size)`` (one-based)."""
        self._require_rpy2()
        if size < 1 or seed < 0:
            raise ValueError("R sample size must be positive and seed nonnegative")
        with self._preserved_r_seed(seed):
            return _r_array(self._base.sample_int(size, size), dtype=np.int64)

    def fit_fixed_sp(
        self,
        formula: str,
        data: pd.DataFrame,
        family: str,
        sp: np.ndarray,
        *,
        weights: str | None = None,
        offset: str | None = None,
        theta: float | None = None,
        with_uncertainty: bool = False,
    ) -> dict[str, Any]:
        """Pinned fixed-sp fit, optionally including covariance and link SE.

        The same declared categorical levels and source weights/offsets are
        retained. This layer owns coefficient fitting, not smoothing selection
        or final result summaries.
        """
        self._require_rpy2()
        sp = np.asarray(sp, dtype=np.float64)
        if sp.ndim != 1 or np.any(~np.isfinite(sp) | (sp <= 0)):
            raise ValueError("Fixed-sp oracle requires a positive finite vector.")
        for name in (weights, offset):
            if name is not None and name not in data:
                raise ValueError(f"Oracle column {name!r} is missing.")
        r_family = self._get_efs_family_rpy2(family, theta)
        r_data = self._to_r_dataframe(data)
        arguments: dict[str, Any] = {
            "data": r_data,
            "family": r_family,
            "sp": self._to_r_vector(sp),
            "method": "REML",
            "drop.unused.levels": False,
            "control": self._mgcv.gam_control(epsilon=1e-10, maxit=200),
        }
        if weights is not None:
            arguments["weights"] = r_data.rx2(weights)
        if offset is not None:
            arguments["offset"] = r_data.rx2(offset)
        fit = self._mgcv.gam(self._ro.Formula(formula), **arguments)
        result = {
            "coefficients": _r_array(fit.rx2("coefficients"), dtype=np.float64),
            "fitted_values": _r_array(fit.rx2("fitted.values"), dtype=np.float64),
            "deviance": float(fit.rx2("deviance")[0]),
            "converged": bool(fit.rx2("converged")[0]),
        }
        if with_uncertainty:
            prediction = self._stats.predict(fit, type="link", **{"se.fit": True})
            result.update(
                scale=float(fit.rx2("scale")[0]),
                Vp=_r_array(fit.rx2("Vp"), dtype=np.float64),
                link_se=_r_array(prediction.rx2("se.fit"), dtype=np.float64),
            )
        return result

    def discrete_operators(
        self,
        layout: DiscreteOperatorLayout,
        weights: np.ndarray,
        response: np.ndarray,
        beta: np.ndarray,
        covariance: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Run pinned mgcv compact ``XW*`` actions on explicit metadata.

        This deliberately calls the unexported operation wrappers directly;
        it is not a ``bam(discrete=TRUE)`` comparison and therefore does not
        ask R to construct a different unique-row basis or constraint setup.
        """
        ok, reason = self.check_versions()
        if not ok:
            raise RBridgeError(f"Pinned discrete oracle unavailable: {reason}")
        weights = np.asarray(weights, dtype=float)
        response = np.asarray(response, dtype=float)
        beta = np.asarray(beta, dtype=float)
        covariance = np.asarray(covariance, dtype=float)
        if weights.ndim != 1 or response.shape != weights.shape:
            raise ValueError("discrete weights and response must be matching vectors")
        p = covariance.shape[0]
        if covariance.shape != (p, p) or beta.shape != (p,):
            raise ValueError("discrete beta/covariance shapes are incompatible")
        if layout.r_to_public is not None:
            if len(layout.r_to_public) != p:
                raise ValueError(
                    "discrete permutation does not match coefficient width"
                )
            beta_r = beta[layout.r_to_public]
            covariance_r = covariance[np.ix_(layout.r_to_public, layout.r_to_public)]
        else:
            beta_r = beta
            covariance_r = covariance
        result = self._discrete_operators_rpy2(
            layout, weights, response, beta_r, covariance_r
        )
        if layout.r_to_public is not None:
            permutation = layout.r_to_public
            public_xwyd = np.empty_like(result["xwyd"])
            public_xwyd[permutation] = result["xwyd"]
            public_xwxd = np.empty_like(result["xwxd"])
            public_xwxd[np.ix_(permutation, permutation)] = result["xwxd"]
            result["xwyd"] = public_xwyd
            result["xwxd"] = public_xwxd
        return result

    def _discrete_operators_rpy2(
        self,
        layout: DiscreteOperatorLayout,
        weights: np.ndarray,
        response: np.ndarray,
        beta: np.ndarray,
        covariance: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Call the pinned compact operators with typed R matrices and vectors."""
        from rpy2.robjects import ListVector

        self._require_rpy2()
        tables = ListVector(
            [
                (str(index), self._to_r_matrix(table))
                for index, table in enumerate(layout.marginal_tables)
            ]
        )
        kd = self._to_r_matrix(layout.row_indices + 1)
        ks = self._to_r_matrix(layout.index_spans + 1)
        ts = self._to_r_vector(np.asarray(layout.term_starts, dtype=np.int32) + 1)
        dt = self._to_r_vector(np.asarray(layout.term_dimensions, dtype=np.int32))
        qc = self._to_r_vector(np.asarray(layout.constraint_codes, dtype=np.int32))
        flattened = (
            np.concatenate(layout.constraint_vectors)
            if layout.constraint_vectors
            else np.empty(0, dtype=np.float64)
        )
        vectors = self._to_r_vector(np.asarray(flattened, dtype=np.float64))
        drop = (
            self._ro.NULL
            if layout.drop is None
            else self._to_r_vector(np.asarray(layout.drop, dtype=np.int32) + 1)
        )
        arguments = (tables, kd, ks, ts, dt, vectors, qc)
        xbd = self._call_internal(
            "Xbd",
            tables,
            self._to_r_vector(beta),
            kd,
            ks,
            ts,
            dt,
            vectors,
            qc,
            drop=drop,
        )
        xwyd = self._call_internal(
            "XWyd",
            tables,
            self._to_r_vector(weights),
            self._to_r_vector(response),
            *arguments[1:],
            drop=drop,
        )
        xwxd = self._call_internal(
            "XWXd", tables, self._to_r_vector(weights), *arguments[1:], drop=drop
        )
        diagonal = self._call_internal(
            "diagXVXd",
            tables,
            self._to_r_matrix(covariance),
            *arguments[1:],
            drop=drop,
        )
        return {
            "xbd": _r_array(xbd, dtype=np.float64),
            "xwyd": _r_array(xwyd, dtype=np.float64),
            "xwxd": _r_array(xwxd, dtype=np.float64),
            "diag_xvxd": _r_array(diagonal, dtype=np.float64),
        }

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
            "R": _r_array(second.rx2("R"), dtype=np.float64),
            "f": _r_array(second.rx2("f"), dtype=np.float64),
            "y_norm2": float(second.rx2("y.norm2")[0]),
            "penalized_coefficients": _r_array(coefficients, dtype=np.float64)
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

    def source_signed_pls_fit1(
        self,
        design: np.ndarray,
        weights: np.ndarray,
        pseudodata: np.ndarray,
        weighted_response: np.ndarray,
        penalty_root: np.ndarray,
        *,
        use_weighted_response: bool,
    ) -> tuple[int, bool, np.ndarray]:
        """Call pinned ``C_pls_fit1`` with its original typed ``.C`` arguments."""
        self._require_rpy2()
        X = np.asarray(design, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        z = np.asarray(pseudodata, dtype=np.float64)
        wz = np.asarray(weighted_response, dtype=np.float64)
        E = np.asarray(penalty_root, dtype=np.float64)
        if (
            X.ndim != 2
            or w.shape != (len(X),)
            or z.shape != w.shape
            or wz.shape != w.shape
            or E.ndim != 2
            or E.shape[1] != X.shape[1]
        ):
            raise ValueError("Signed fitter oracle dimensions are incompatible")
        symbol = self._utils.getFromNamespace("C_pls_fit1", "mgcv")
        call = self._ro.baseenv[".C"]
        result = call(
            symbol,
            y=self._to_r_vector(z),
            X=self._to_r_vector(X.ravel(order="F")),
            w=self._to_r_vector(w),
            wy=self._to_r_vector(wz),
            E=self._to_r_vector(E.ravel(order="F")),
            Es=self._to_r_vector(E.ravel(order="F")),
            n=self._to_r_vector(np.array([len(X)], dtype=np.int32)),
            q=self._to_r_vector(np.array([X.shape[1]], dtype=np.int32)),
            rE=self._to_r_vector(np.array([E.shape[0]], dtype=np.int32)),
            eta=self._to_r_vector(z),
            penalty=self._to_r_vector(np.array([1.0])),
            **{"rank.tol": self._to_r_vector(np.array([100 * np.finfo(float).eps]))},
            nt=self._to_r_vector(np.array([1], dtype=np.int32)),
            **{
                "use.wy": self._to_r_vector(
                    np.array([int(use_weighted_response)], dtype=np.int32)
                )
            },
        )
        return (
            int(result.rx2("n")[0]),
            bool(result.rx2("use.wy")[0]),
            _r_array(result.rx2("y"), dtype=np.float64)[: X.shape[1]].copy(),
        )

    def source_gaussian_aic(
        self, response: np.ndarray, mean: np.ndarray, weights: np.ndarray
    ) -> float:
        """Evaluate stats Gaussian raw AIC, including zero-weight boundaries."""
        self._require_rpy2()
        y = np.asarray(response, dtype=np.float64)
        mu = np.asarray(mean, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        if y.ndim != 1 or mu.shape != y.shape or w.shape != y.shape:
            raise ValueError("Gaussian AIC oracle vectors must align")
        family = self._stats.gaussian()
        y_r, mu_r, w_r = map(self._to_r_vector, (y, mu, w))
        deviance = self._base.sum(family.rx2("dev.resids")(y_r, mu_r, w_r))
        result = family.rx2("aic")(
            y_r, self._to_r_vector(np.ones(len(y))), mu_r, w_r, deviance
        )
        return float(result[0])

    def source_gaussian_saturated_likelihood(
        self, response: np.ndarray, weights: np.ndarray, scale: float
    ) -> np.ndarray:
        """Evaluate mgcv's patched Gaussian saturated likelihood derivatives."""
        self._require_rpy2()
        y = np.asarray(response, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        if y.ndim != 1 or w.shape != y.shape or not np.isfinite(scale) or scale <= 0:
            raise ValueError("Gaussian likelihood oracle inputs are invalid")
        family = self._call_internal("fix.family.ls", self._stats.gaussian())
        return _r_array(
            family.rx2("ls")(
                self._to_r_vector(y), self._to_r_vector(w), len(y), float(scale)
            ),
            dtype=np.float64,
        )

    def source_gamma_saturated_likelihood(
        self, response: np.ndarray, weights: np.ndarray, scale: float
    ) -> np.ndarray:
        """Return mgcv's Gamma ls value and first two scale derivatives."""
        self._require_rpy2()
        y = np.asarray(response, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        if (
            y.ndim != 1
            or w.shape != y.shape
            or not np.all(np.isfinite(y))
            or not np.all(np.isfinite(w))
            or np.any(y <= 0)
            or np.any(w < 0)
            or not np.isfinite(scale)
            or scale <= 0
        ):
            raise ValueError("Gamma likelihood oracle inputs are invalid")
        family = self._call_internal("fix.family.ls", self._stats.Gamma())
        return _r_array(
            family.rx2("ls")(
                self._to_r_vector(y), self._to_r_vector(w), len(y), float(scale)
            ),
            dtype=np.float64,
        )

    def source_gaussian_initial_values(
        self, response: np.ndarray, link: str
    ) -> tuple[float, np.ndarray]:
        """Evaluate the installed family's ``initialize`` language object."""
        self._require_rpy2()
        y = np.asarray(response, dtype=np.float64)
        if y.ndim != 1 or link not in {"log", "inverse"}:
            raise ValueError("Gaussian initializer requires a response and known link")
        family = self._call_internal("fix.family", self._stats.gaussian(link=link))
        environment = self._base.new_env(parent=self._ro.globalenv)
        environment["y"] = self._to_r_vector(y)
        environment["nobs"] = self._to_r_vector(np.array([len(y)], dtype=np.int32))
        environment["family"] = family
        self._base.eval(family.rx2("initialize"), envir=environment)
        return (
            float(self._stats.sd(self._to_r_vector(y))[0]),
            _r_array(environment["mustart"], dtype=np.float64),
        )

    def source_gaussian_public_fit(
        self,
        formula: str,
        data: pd.DataFrame,
        link: str,
        weights: np.ndarray,
        offset: np.ndarray,
        *,
        epsilon: float,
    ) -> dict[str, Any]:
        """Fit pinned Gaussian GAM with an explicit link and convergence control."""
        self._require_rpy2()
        weight_array = np.asarray(weights, dtype=np.float64)
        offset_array = np.asarray(offset, dtype=np.float64)
        if (
            weight_array.shape != (len(data),)
            or offset_array.shape != (len(data),)
            or link not in {"log", "inverse"}
            or not np.isfinite(epsilon)
            or epsilon <= 0
        ):
            raise ValueError("Gaussian public fit oracle inputs are invalid")
        fit = self._mgcv.gam(
            self._ro.Formula(formula),
            data=self._to_r_dataframe(data),
            weights=self._to_r_vector(weight_array),
            offset=self._to_r_vector(offset_array),
            family=self._stats.gaussian(link=link),
            method="REML",
            control=self._mgcv.gam_control(epsilon=float(epsilon)),
        )
        if not bool(fit.rx2("converged")[0]):
            raise RBridgeError("Pinned Gaussian GAM did not converge")
        return {
            "coefficients": _r_array(fit.rx2("coefficients"), dtype=np.float64),
            "deviance": float(fit.rx2("deviance")[0]),
            "scale": float(fit.rx2("sig2")[0]),
            "edf": float(self._base.sum(fit.rx2("edf"))[0]),
            "score": float(fit.rx2("gcv.ubre")[0]),
            "fitted": _r_array(fit.rx2("fitted.values"), dtype=np.float64),
            "covariance": _r_array(fit.rx2("Vp"), dtype=np.float64),
            "reml_scale": float(fit.rx2("reml.scale")[0]),
        }

    def source_gaussian_efs_initial(
        self,
        formula: str,
        data: pd.DataFrame,
        link: str,
        weights: np.ndarray,
    ) -> np.ndarray:
        """Return pinned ``initial.spg`` and null scale from a setup object."""
        self._require_rpy2()
        weight_array = np.asarray(weights, dtype=np.float64)
        if weight_array.shape != (len(data),) or link not in {"log", "inverse"}:
            raise ValueError("Gaussian EFS initial oracle inputs are invalid")
        setup = self._mgcv.gam(
            self._ro.Formula(formula),
            data=self._to_r_dataframe(data),
            weights=self._to_r_vector(weight_array),
            family=self._stats.gaussian(link=link),
            fit=False,
        )
        patched = self._call_internal("fix.family", setup.rx2("family"))
        setup = self._ro.baseenv["[[<-"](setup, "family", patched)
        initial = self._call_internal(
            "initial.spg",
            setup.rx2("X"),
            setup.rx2("y"),
            setup.rx2("w"),
            patched,
            setup.rx2("S"),
            setup.rx2("rank"),
            setup.rx2("off"),
        )
        null = self._call_internal("get.null.coef", setup)
        divide = self._ro.baseenv["/"]
        return _r_array(
            self._base.c(
                self._base.log(initial),
                self._base.log(divide(null.rx2("null.scale"), 10.0)),
            ),
            dtype=np.float64,
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
            _r_array(fit.rx2("fitted.values"), dtype=np.float64),
        )

    def source_regular_gdi_ratio(
        self,
        design: np.ndarray,
        response: np.ndarray,
        weights: np.ndarray,
        offset: np.ndarray,
        coefficients: np.ndarray,
        covariance_direction: np.ndarray,
        *,
        family_name: str,
        link: str,
        fisher_equals_observed: bool,
    ) -> np.ndarray:
        """Form gdi1's first cotangent from installed family derivatives."""
        self._require_rpy2()
        constructors = {
            "gaussian": self._stats.gaussian,
            "Gamma": self._stats.Gamma,
            "poisson": self._stats.poisson,
            "binomial": self._stats.binomial,
        }
        if family_name not in constructors:
            raise ValueError("Unknown regular family for source gdi ratio")
        X = np.asarray(design, dtype=np.float64)
        y = np.asarray(response, dtype=np.float64)
        w = np.asarray(weights, dtype=np.float64)
        off = np.asarray(offset, dtype=np.float64)
        beta = np.asarray(coefficients, dtype=np.float64)
        direction = np.asarray(covariance_direction, dtype=np.float64)
        if (
            X.ndim != 2
            or y.shape != (len(X),)
            or w.shape != y.shape
            or off.shape != y.shape
            or beta.shape != (X.shape[1],)
            or direction.shape != (X.shape[1], X.shape[1])
        ):
            raise ValueError("Regular gdi ratio oracle inputs are misaligned")
        r_link = "1/mu^2" if link == "inverse_squared" else link
        family = constructors[family_name](link=r_link)
        for patch in ("fix.family.link", "fix.family.var", "fix.family.ls"):
            family = self._call_internal(patch, family)
        add = self._ro.baseenv["+"]
        subtract = self._ro.baseenv["-"]
        multiply = self._ro.baseenv["*"]
        divide = self._ro.baseenv["/"]
        power = self._ro.baseenv["^"]
        matrix_product = self._ro.baseenv["%*%"]
        X_r = self._to_r_matrix(X)
        y_r = self._to_r_vector(y)
        weight_r = self._to_r_vector(w)
        eta = self._base.drop(
            add(matrix_product(X_r, self._to_r_vector(beta)), self._to_r_vector(off))
        )
        mu_r = family.rx2("linkinv")(eta)
        me_r = family.rx2("mu.eta")(eta)
        good = (w > 0) & (_r_array(me_r, dtype=np.float64) != 0)
        X_r = self._to_r_matrix(X[good])
        y_r = self._to_r_vector(y[good])
        weight_r = self._to_r_vector(w[good])
        mu_r = self._to_r_vector(_r_array(mu_r, dtype=np.float64)[good])
        me_r = self._to_r_vector(_r_array(me_r, dtype=np.float64)[good])

        def derivative(name: str):
            return family.rx2(name)(mu_r)

        variance = derivative("variance")
        g1 = divide(1.0, me_r)
        g2 = divide(derivative("d2link"), g1)
        g3 = divide(derivative("d3link"), g1)
        variance_first = divide(derivative("dvar"), variance)
        variance_second = divide(derivative("d2var"), variance)
        fisher_weight = divide(multiply(weight_r, power(me_r, 2.0)), variance)
        if fisher_equals_observed:
            weight = fisher_weight
            derivative_weight = divide(
                multiply(subtract(0.0, weight), add(variance_first, multiply(2.0, g2))),
                g1,
            )
        else:
            alpha = add(1.0, multiply(subtract(y_r, mu_r), add(variance_first, g2)))
            alpha_values = _r_array(alpha, dtype=np.float64)
            alpha_values[alpha_values == 0] = np.finfo(float).eps
            alpha = self._to_r_vector(alpha_values)
            weight = multiply(fisher_weight, alpha)
            second = subtract(
                add(subtract(variance_second, power(variance_first, 2.0)), g3),
                power(g2, 2.0),
            )
            alpha_first = divide(
                add(
                    subtract(0.0, add(variance_first, g2)),
                    multiply(subtract(y_r, mu_r), second),
                ),
                alpha,
            )
            derivative_weight = divide(
                multiply(
                    weight,
                    subtract(subtract(alpha_first, variance_first), multiply(2.0, g2)),
                ),
                g1,
            )
        row_direction = self._base.rowSums(
            multiply(matrix_product(X_r, self._to_r_matrix(direction)), X_r)
        )
        deviance_derivative = divide(
            multiply(multiply(-2.0, weight_r), subtract(y_r, mu_r)),
            multiply(variance, g1),
        )
        beta_cotangent = self._base.drop(
            self._base.crossprod(
                X_r,
                add(
                    multiply(row_direction, derivative_weight),
                    multiply(0.7, deviance_derivative),
                ),
            )
        )
        saturated = family.rx2("ls")(y_r, weight_r, len(y_r), 0.8)
        log_scale_cotangent = multiply(multiply(-0.6, saturated[1]), 0.8)
        return _r_array(
            self._base.c(beta_cotangent, log_scale_cotangent), dtype=np.float64
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

    def fit_efs_selected_before_offset_null_deviance(
        self,
        formula: str,
        data: pd.DataFrame,
        family: str,
        *,
        weights: str | None = None,
        offset: str | None = None,
        controls: dict[str, float] | None = None,
        initial_smoothing: np.ndarray | None = None,
        initial_scale: float | None = None,
        scale: float = -1.0,
        theta: float | None = None,
    ) -> dict[str, Any]:
        """Return a selected EFS fit before offset null-deviance postprocessing.

        Pinned ``estimate.gam`` refits an offset-only GLM after the complete
        EFS fit. Some bounded noncanonical links reject that auxiliary GLM
        even though ``efsudr`` and ``gam.fit3.post.proc`` returned a valid
        selected model. This oracle clones the pinned closure and disables
        only that post-fit assignment. Callers must separately preserve the
        public ``gam`` rejection as boundary evidence.
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
            False,
            scale,
            theta,
            skip_offset_null_deviance=True,
        )

    def _public_fixed_sp_gam_arguments(
        self,
        data: pd.DataFrame,
        weight: np.ndarray,
        offset: np.ndarray,
        family_name: str,
        link: str,
        *,
        theta: float | None,
        sp: float | None,
        r_epsilon: float,
        r_newton_tolerance: float,
    ) -> tuple[Any, dict[str, Any]]:
        """Build the pinned public fixed-sp ``gam`` formula and arguments."""
        ro = self._ro
        source_link = "1/mu^2" if link == "inverse_squared" else link
        constructors = {
            "gamma": self._stats.Gamma,
            "gaussian": self._stats.gaussian,
            "poisson": self._stats.poisson,
            "binomial": self._stats.binomial,
        }
        if family_name == "nb":
            if theta is None:
                raise ValueError("fixed-sp NB source oracle needs theta")
            family = self._mgcv.nb(theta=theta, link=source_link)
        else:
            try:
                constructor = constructors[family_name]
            except KeyError:
                raise ValueError("unsupported fixed-sp source family") from None
            family = constructor(link=source_link)
        smooth = sp is not None
        control = self._mgcv.gam_control(
            epsilon=r_epsilon,
            maxit=200,
            newton=ro.ListVector(
                {
                    "conv.tol": r_newton_tolerance,
                    "maxNstep": 5.0,
                    "maxSstep": 2.0,
                    "maxHalf": 30,
                }
            ),
        )
        formula = ro.Formula('y ~ s(x, bs="cr", k=8)' if smooth else "y ~ x")
        return formula, {
            "data": self._to_r_dataframe(data),
            "weights": self._to_r_vector(weight),
            "offset": self._to_r_vector(offset),
            "family": family,
            "sp": self._to_r_vector([sp]) if smooth else ro.NULL,
            "method": "REML",
            "control": control,
        }

    def public_fixed_sp_scale_derivatives(
        self,
        data: pd.DataFrame,
        weight: np.ndarray,
        offset: np.ndarray,
        family_name: str,
        link: str,
        *,
        sp: float | None,
        scale: float | None = None,
        r_epsilon: float = 1e-11,
        r_newton_tolerance: float = 1e-10,
    ) -> dict[str, Any]:
        """Evaluate pinned gam.fit3's log-scale REML derivatives at one scale.

        The public fit of :meth:`public_fixed_sp_fit` runs through private
        clones of ``gam`` -> ``estimate.gam`` -> ``gam.outer`` -> ``newton``
        whose ``gam.fit3`` records newton's first call: the prepared model
        matrix, penalty roots, null coefficients, fixed family, weights,
        offset and newton's tightened control. One fresh ``gam.fit3``
        (``deriv=2``) at ``sp = c(log(sp), log(scale))`` then returns the
        objective newton differentiates, independent of newton's path or of
        the trial object it may return after a step failure. ``sp=None``
        selects the parametric ``y ~ x`` model, whose only newton parameter is
        ``log(scale)``. ``scale=None`` evaluates at the fit's own returned
        ``reml.scale``.
        """
        self._require_rpy2()
        from tests.r_ast import clone_function, make_r_callback

        if family_name not in {"gaussian", "gamma"}:
            raise ValueError("scale derivatives need an unknown-scale regular family")
        formula, arguments = self._public_fixed_sp_gam_arguments(
            data,
            weight,
            offset,
            family_name,
            link,
            theta=None,
            sp=sp,
            r_epsilon=r_epsilon,
            r_newton_tolerance=r_newton_tolerance,
        )
        ro = self._ro
        fit3 = self._utils.getFromNamespace("gam.fit3", "mgcv")
        recorded: list[dict[str, Any]] = []

        def record_fit3(**fit_arguments: Any) -> Any:
            if not recorded:
                recorded.append(dict(fit_arguments))
            return fit3(**fit_arguments)

        namespace = ro.r["getNamespace"]("mgcv")
        private = ro.r["new.env"](parent=namespace)
        callback = make_r_callback(record_fit3)
        names = ("newton", "gam.outer", "estimate.gam")
        try:
            private["gam.fit3"] = callback
            for name in names:
                private[name] = clone_function(
                    self._utils.getFromNamespace(name, "mgcv"), environment=private
                )
            r_gam = clone_function(
                self._utils.getFromNamespace("gam", "mgcv"), environment=private
            )
            fit = r_gam(formula, **arguments)
        finally:
            for name in ("gam.fit3", *names):
                private[name] = ro.NULL
        outer = fit.rx2("outer.info")
        if not recorded or outer is ro.NULL:
            raise RBridgeError("Pinned public fit did not call newton's gam.fit3")
        first = recorded[0]
        recorded_sp = _r_array(first["sp"], dtype=np.float64).ravel()
        if sp is None:
            if recorded_sp.shape != (1,):
                raise RBridgeError("newton's parametric gam.fit3 sp is not log(scale)")
        elif recorded_sp.shape != (2,) or abs(recorded_sp[0] - np.log(sp)) > 4 * (
            np.spacing(abs(np.log(sp)))
        ):
            raise RBridgeError("newton's gam.fit3 sp is not c(log(sp), log(scale))")
        returned_scale = float(fit.rx2("reml.scale")[0])
        evaluation_scale = returned_scale if scale is None else float(scale)
        if not np.isfinite(evaluation_scale) or evaluation_scale <= 0.0:
            raise ValueError("scale must be finite and positive")
        evaluated = fit3(
            **{
                **first,
                "sp": self._to_r_vector(
                    np.r_[recorded_sp[:-1], np.log(evaluation_scale)]
                ),
                "deriv": 2,
            }
        )
        gradient = _r_array(evaluated.rx2("REML1"), dtype=np.float64)
        hessian = _r_array(evaluated.rx2("REML2"), dtype=np.float64)
        return {
            "scale": evaluation_scale,
            "reml_scale": float(evaluated.rx2("reml.scale")[0]),
            "returned_scale": returned_scale,
            "outer_status": str(outer.rx2("conv")[0]),
            "outer_gradient": float(_r_array(outer.rx2("grad")).ravel()[-1]),
            "outer_hessian": float(_r_array(outer.rx2("hess")).ravel()[-1]),
            "score": float(evaluated.rx2("REML")[0]),
            "gradient": float(gradient[-1]),
            "hessian": float(hessian[-1, -1]),
            "inner_converged": bool(evaluated.rx2("converged")[0]),
        }

    def public_fixed_sp_fit(
        self,
        data: pd.DataFrame,
        weight: np.ndarray,
        offset: np.ndarray,
        family_name: str,
        link: str,
        *,
        theta: float | None = None,
        sp: float | None = None,
        r_epsilon: float = 1e-11,
        r_newton_tolerance: float = 1e-10,
    ) -> dict[str, Any]:
        """Run a tight pinned mgcv public fit for one fixed-sp release cell."""
        self._require_rpy2()
        ro = self._ro
        formula, arguments = self._public_fixed_sp_gam_arguments(
            data,
            weight,
            offset,
            family_name,
            link,
            theta=theta,
            sp=sp,
            r_epsilon=r_epsilon,
            r_newton_tolerance=r_newton_tolerance,
        )
        r_data = arguments["data"]
        fit = self._mgcv.gam(formula, **arguments)
        score_scale = fit.rx2("reml.scale")
        if score_scale is ro.NULL:
            score_scale = fit.rx2("sig2")
        link_prediction = self._stats.predict(
            fit, newdata=r_data, type="link", **{"se.fit": True}
        )
        outer = fit.rx2("outer.info")
        outer_status = (
            "none"
            if outer is ro.NULL or outer.rx2("conv") is ro.NULL
            else str(outer.rx2("conv")[0])
        )
        outer_gradient = (
            None
            if outer is ro.NULL or outer.rx2("grad") is ro.NULL
            else _r_array(outer.rx2("grad"), dtype=np.float64)
        )
        outer_hessian = (
            None
            if outer is ro.NULL or outer.rx2("hess") is ro.NULL
            else _r_array(outer.rx2("hess"), dtype=np.float64)
        )
        # gam() reports every smoothing parameter multiplying a penalty as
        # full.sp = exp(L %*% log(free sp) + lsp0); a supplied sp enters only
        # lsp0 = log(sp) (gam.setup). NULL for the unpenalized "y ~ x" model.
        full_sp = fit.rx2("full.sp")
        return {
            "full_sp": (
                None if full_sp is ro.NULL else _r_array(full_sp, dtype=np.float64)
            ),
            "theta": (
                float(fit.rx2("family").rx2("getTheta")(True)[0])
                if family_name == "nb"
                else np.nan
            ),
            "score": float(fit.rx2("gcv.ubre")[0]),
            "scale": float(fit.rx2("sig2")[0]),
            "score_scale": float(score_scale[0]),
            "deviance": float(fit.rx2("deviance")[0]),
            "edf": float(np.sum(_r_array(fit.rx2("edf"), dtype=np.float64))),
            "coefficients": _r_array(fit.rx2("coefficients")),
            "fitted_values": _r_array(fit.rx2("fitted.values")),
            "link_se": _r_array(link_prediction.rx2("se.fit")),
            "inner_converged": bool(fit.rx2("converged")[0]),
            "inner_iterations": int(fit.rx2("iter")[0]),
            "outer_status": outer_status,
            "outer_gradient": outer_gradient,
            "outer_hessian": outer_hessian,
            "outer_iterations": (
                None
                if outer is ro.NULL or outer.rx2("iter") is ro.NULL
                else int(outer.rx2("iter")[0])
            ),
            "r_epsilon": r_epsilon,
            "r_newton_tolerance": r_newton_tolerance,
        }

    def nb_theta_diagnostics(
        self,
        start: np.ndarray,
        y: np.ndarray,
        mu: np.ndarray,
        weight: np.ndarray,
        link: str = "log",
    ) -> dict[str, np.ndarray]:
        """Read source contractions and the traced theta path through pinned R."""
        from rpy2 import rinterface

        from tests.r_ast import find_call_paths, instrument_function, symbol

        self._require_rpy2()
        start_r = self._to_r_vector(np.asarray(start, dtype=np.float64))
        y_r = self._to_r_vector(np.asarray(y, dtype=np.float64))
        mu_r = self._to_r_vector(np.asarray(mu, dtype=np.float64))
        weight_r = self._to_r_vector(np.asarray(weight, dtype=np.float64))
        base = self._ro.baseenv
        family = self._mgcv.nb(theta=base["-"](self._base.exp(start_r)), link=link)

        def fields(theta_r: Any) -> np.ndarray:
            likelihood = family.rx2("ls")(y_r, w=weight_r, theta=theta_r, scale=1)
            derivatives = family.rx2("Dd")(y_r, mu_r, theta_r, wt=weight_r, level=2)
            deviance = self._base.sum(
                family.rx2("dev.resids")(y_r, mu_r, weight_r, theta_r)
            )
            half = self._to_r_vector([2.0])
            return np.asarray(
                [
                    base["-"](base["/"](deviance, half), likelihood.rx2("ls"))[0],
                    base["-"](
                        base["/"](self._base.sum(derivatives.rx2("Dth")), half),
                        likelihood.rx2("lsth1"),
                    )[0],
                    base["-"](
                        base["/"](self._base.sum(derivatives.rx2("Dth2")), half),
                        likelihood.rx2("lsth2"),
                    )[0],
                ],
                dtype=np.float64,
            )

        source = self._utils.getFromNamespace("estimate.theta", "mgcv")

        def assignment_path(target: str, increment: str) -> tuple[int, ...]:
            paths = find_call_paths(source, "<-", required_symbols=(target, increment))
            matched = []
            for path in paths:
                node = self._ro.r["body"](source)
                for index in path:
                    node = node[index]
                rhs = node[2]
                if (
                    node[1].rsame(symbol(target))
                    and rhs.typeof == rinterface.RTYPES.LANGSXP
                    and rhs[0].rsame(symbol("+"))
                    and rhs[1].rsame(symbol(target))
                    and (
                        rhs[2].rsame(symbol(increment))
                        if increment == "step"
                        else float(rhs[2][0]) == 1.0
                    )
                ):
                    matched.append(path)
            if len(matched) != 1:
                raise RBridgeError(f"Pinned estimate.theta {target} assignment changed")
            return matched[0]

        theta_path = [float(start_r[0])]
        halvings = [0]
        theta_assignment = assignment_path("theta", "step")
        iter_assignment = assignment_path("iter", "iter")
        traced = instrument_function(
            source,
            path=theta_assignment,
            expected_head="<-",
            capture_symbols=("theta",),
            callback=lambda theta: theta_path.append(float(theta[0])),
            when="after",
        )
        traced = instrument_function(
            traced,
            path=iter_assignment,
            expected_head="<-",
            capture_symbols=("iter",),
            callback=lambda _iter: halvings.__setitem__(0, halvings[0] + 1),
            when="after",
        )
        end = traced(start_r, family, y_r, mu_r, scale=1, wt=weight_r)
        return {
            "initial": fields(start_r),
            "final": np.r_[_r_array(end, dtype=np.float64), fields(end)],
            "path": np.asarray(theta_path, dtype=np.float64),
            "halvings": np.asarray(halvings, dtype=np.float64),
        }

    def nb_pirls_controller_reference(
        self,
        X: np.ndarray,
        X_public: np.ndarray,
        transform: np.ndarray,
        E: np.ndarray,
        y: np.ndarray,
        weight: np.ndarray,
        offset: np.ndarray,
        link: str,
        start: np.ndarray | None,
        rank: int,
        tolerance: float,
    ) -> dict[str, np.ndarray]:
        """Trace a private pinned ``gam.fit4``/``estimate.theta`` controller."""
        from rpy2 import rinterface

        from tests.r_ast import (
            clone_function,
            find_call_paths,
            instrument_function,
            symbol,
        )

        self._require_rpy2()
        ro, base = self._ro, self._ro.baseenv
        r_X, r_X_public, r_transform, r_E = (
            self._to_r_matrix(values) for values in (X, X_public, transform, E)
        )
        r_y, r_weight, r_offset = (
            self._to_r_vector(values) for values in (y, weight, offset)
        )
        p = X.shape[1]
        theta_initial = self._base.log(self._to_r_vector([0.7]))
        family = self._call_internal(
            "fix.family.link",
            self._mgcv.nb(theta=base["-"](self._to_r_vector([0.7])), link=link),
        )
        U1 = (
            self._base.eigen(self._base.crossprod(r_E), symmetric=True).rx2("vectors")
            if rank
            else self._base.diag(p)
        )
        if rank:
            root = base["%*%"](
                self._base.t(self._to_r_matrix(_r_array(U1)[:, :rank])),
                self._base.t(r_E),
            )
            roots = base["list"](root)
        else:
            roots = base["list"]()
        target = self._base.rep(
            family.rx2("linkfun")(self._base.mean(r_y)), times=len(y)
        )
        null = _r_array(
            self._base.qr_coef(self._base.qr(r_X_public), target), dtype=np.float64
        )
        null[np.isnan(null)] = 0.0
        null_r = self._base.solve(r_transform, self._to_r_vector(null))

        theta_updates = [0]
        theta_history = [float(theta_initial[0])]
        pre_theta = [float(theta_initial[0])]
        recoveries = [0]
        final_locals: dict[str, float] = {}

        estimate = self._utils.getFromNamespace("estimate.theta", "mgcv")
        theta_paths = find_call_paths(
            estimate, "<-", required_symbols=("theta", "step")
        )
        accepted_paths = []
        for path in theta_paths:
            node = ro.r["body"](estimate)
            for index in path:
                node = node[index]
            rhs = node[2]
            if (
                node[1].rsame(symbol("theta"))
                and rhs.typeof == rinterface.RTYPES.LANGSXP
                and rhs[0].rsame(symbol("+"))
                and rhs[1].rsame(symbol("theta"))
                and rhs[2].rsame(symbol("step"))
            ):
                accepted_paths.append(path)
        if len(accepted_paths) != 1:
            raise RBridgeError("Pinned estimate.theta accepted-step anchor changed")
        traced_estimate = instrument_function(
            estimate,
            path=accepted_paths[0],
            expected_head="<-",
            capture_symbols=("theta",),
            callback=lambda _theta: theta_updates.__setitem__(0, theta_updates[0] + 1),
            when="after",
        )

        fit_source = self._utils.getFromNamespace("gam.fit4", "mgcv")
        theta_calls = find_call_paths(
            fit_source, "<-", required_symbols=("theta", "estimate.theta")
        )
        if len(theta_calls) != 1:
            raise RBridgeError("Pinned gam.fit4 theta-call anchor changed")
        fit_source = instrument_function(
            fit_source,
            path=theta_calls[0],
            expected_head="<-",
            capture_symbols=("theta",),
            callback=lambda theta: pre_theta.__setitem__(0, float(theta[0])),
            when="before",
        )
        theta_calls = find_call_paths(
            fit_source, "<-", required_symbols=("theta", "estimate.theta")
        )
        if len(theta_calls) != 1:
            raise RBridgeError("Pinned gam.fit4 theta return anchor changed")
        fit_source = instrument_function(
            fit_source,
            path=theta_calls[0],
            expected_head="<-",
            capture_symbols=("theta",),
            callback=lambda theta: theta_history.append(float(theta[0])),
            when="after",
        )
        recovery_paths = []
        for path in find_call_paths(fit_source, "cat"):
            node = ro.r["body"](fit_source)
            for index in path:
                node = node[index]
            if (
                len(node) > 1
                and node[1].typeof == rinterface.RTYPES.STRSXP
                and str(node[1][0]) == "**using positive weights\n"
            ):
                recovery_paths.append(path)
        if len(recovery_paths) != 1:
            raise RBridgeError("Pinned gam.fit4 positive-weight anchor changed")
        fit_source = instrument_function(
            fit_source,
            path=recovery_paths[0],
            expected_head="cat",
            capture_symbols=(),
            callback=lambda: recoveries.__setitem__(0, recoveries[0] + 1),
        )
        body = ro.r["body"](fit_source)
        fit_source = instrument_function(
            fit_source,
            path=(len(body) - 1,),
            expected_head="list",
            capture_symbols=("oo", "pdev"),
            callback=lambda oo, pdev: final_locals.update(
                penalty=float(ro.ListVector(oo).rx2("P")[0]),
                pdev=float(pdev[0]),
            ),
        )
        private_environment = ro.r["new.env"](parent=ro.r["environment"](fit_source))
        private_environment["estimate.theta"] = traced_estimate
        fit_function = clone_function(fit_source, environment=private_environment)
        smoothing = self._base.c(
            theta_initial,
            self._base.log(self._to_r_vector([0.35])) if rank else ro.FloatVector([]),
        )
        fit = fit_function(
            x=r_X,
            y=r_y,
            sp=smoothing,
            Eb=r_E,
            UrS=roots,
            weights=r_weight,
            offset=r_offset,
            start=ro.NULL if start is None else self._to_r_vector(start),
            U1=U1,
            Mp=p - rank,
            family=family,
            control=self._mgcv.gam_control(epsilon=tolerance, maxit=100, trace=True),
            deriv=0,
            scale=1,
            scoreType="EFS",
            **{"null.coef": null_r},
        )
        theta = family.rx2("getTheta")()
        mean = fit.rx2("fitted.values")
        derivatives = self._call_internal(
            "dDeta", r_y, mean, r_weight, theta, family, 0
        )
        half = self._to_r_vector([0.5])
        observed = self._base.crossprod(
            r_X,
            base["*"](base["*"](half, derivatives.rx2("Deta2")), r_X),
        )
        fisher = self._base.crossprod(
            r_X,
            base["*"](base["*"](half, derivatives.rx2("EDeta2")), r_X),
        )
        final_deviance = self._base.sum(
            family.rx2("dev.resids")(r_y, mean, r_weight, theta)
        )
        return {
            "beta": _r_array(fit.rx2("coefficients")),
            "theta": _r_array(theta),
            "source_deviance": _r_array(fit.rx2("deviance")),
            "final_deviance": _r_array(final_deviance),
            "source_theta": np.asarray(pre_theta, dtype=np.float64),
            "gdi_penalty": np.asarray([final_locals["penalty"]]),
            "score": _r_array(fit.rx2("REML")),
            "stopping_pdev": np.asarray([final_locals["pdev"]]),
            "G": _r_array(observed),
            "F": _r_array(fisher),
            "V": _r_array(self._base.tcrossprod(fit.rx2("rV"))),
            "theta_history": np.asarray(theta_history, dtype=np.float64),
            "iter": _r_array(fit.rx2("iter")),
            "theta_iter": np.asarray([theta_updates[0]]),
            "null_coefficients": _r_array(null_r),
            "recoveries": np.asarray([recoveries[0]]),
        }

    def nb_fixed_pirls_reference(
        self,
        X: np.ndarray,
        E: np.ndarray,
        y: np.ndarray,
        weight: np.ndarray,
        offset: np.ndarray,
        link: str,
        theta: float,
    ) -> dict[str, np.ndarray]:
        """Evaluate pinned fixed-theta ``gam.fit4`` with private source hooks."""
        from rpy2 import rinterface

        from tests.r_ast import find_call_paths, instrument_function

        self._require_rpy2()
        ro, base = self._ro, self._ro.baseenv
        r_X, r_E = self._to_r_matrix(X), self._to_r_matrix(E)
        r_y, r_weight, r_offset = (
            self._to_r_vector(values) for values in (y, weight, offset)
        )
        family = self._call_internal(
            "fix.family.link", self._mgcv.nb(theta=theta, link=link)
        )
        p, rank = X.shape[1], E.shape[0]
        U1 = (
            self._base.eigen(self._base.crossprod(r_E), symmetric=True).rx2("vectors")
            if rank
            else self._base.diag(p)
        )
        roots = (
            base["list"](
                base["%*%"](
                    self._base.t(self._to_r_matrix(_r_array(U1)[:, :rank])),
                    self._base.t(r_E),
                )
            )
            if rank
            else base["list"]()
        )
        target = self._base.rep(
            family.rx2("linkfun")(self._base.mean(r_y)), times=len(y)
        )
        null = _r_array(self._base.qr_coef(self._base.qr(r_X), target))
        null[np.isnan(null)] = 0.0
        recovery_count = [0]
        penalty = [np.nan]
        fit_source = self._utils.getFromNamespace("gam.fit4", "mgcv")
        recovery_paths = []
        for path in find_call_paths(fit_source, "cat"):
            node = ro.r["body"](fit_source)
            for index in path:
                node = node[index]
            if (
                len(node) > 1
                and node[1].typeof == rinterface.RTYPES.STRSXP
                and str(node[1][0]) == "**using positive weights\n"
            ):
                recovery_paths.append(path)
        if len(recovery_paths) != 1:
            raise RBridgeError("Pinned gam.fit4 positive-weight anchor changed")
        fit_source = instrument_function(
            fit_source,
            path=recovery_paths[0],
            expected_head="cat",
            capture_symbols=(),
            callback=lambda: recovery_count.__setitem__(0, recovery_count[0] + 1),
        )
        body = ro.r["body"](fit_source)
        fit_function = instrument_function(
            fit_source,
            path=(len(body) - 1,),
            expected_head="list",
            capture_symbols=("oo",),
            callback=lambda oo: penalty.__setitem__(
                0, float(ro.ListVector(oo).rx2("P")[0])
            ),
        )
        fit = fit_function(
            x=r_X,
            y=r_y,
            sp=self._base.log(self._to_r_vector([0.35]))
            if rank
            else ro.FloatVector([]),
            Eb=r_E,
            UrS=roots,
            weights=r_weight,
            offset=r_offset,
            U1=U1,
            Mp=p - rank,
            family=family,
            control=self._mgcv.gam_control(epsilon=1e-11, maxit=100, trace=True),
            deriv=0,
            scale=1,
            scoreType="REML",
            **{"null.coef": self._to_r_vector(null)},
        )
        derivatives = self._call_internal(
            "dDeta",
            r_y,
            fit.rx2("fitted.values"),
            r_weight,
            self._base.log(self._to_r_vector([theta])),
            family,
            0,
        )
        half = self._to_r_vector([0.5])
        observed = self._base.crossprod(
            r_X, base["*"](base["*"](half, derivatives.rx2("Deta2")), r_X)
        )
        fisher = self._base.crossprod(
            r_X, base["*"](base["*"](half, derivatives.rx2("EDeta2")), r_X)
        )
        covariance = self._base.tcrossprod(fit.rx2("rV"))
        edf = self._base.sum(base["diag"](base["%*%"](covariance, fisher)))
        return {
            "beta": _r_array(fit.rx2("coefficients")),
            "dev": _r_array(fit.rx2("deviance")),
            "G": _r_array(observed),
            "F": _r_array(fisher),
            "V": _r_array(covariance),
            "edf": _r_array(edf),
            "score": _r_array(fit.rx2("REML")),
            "gdi": np.asarray(penalty, dtype=np.float64),
            "iter": _r_array(fit.rx2("iter")),
            "converged": _r_array(fit.rx2("converged")),
            "recovery": np.asarray(recovery_count),
        }

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

    def nb_working_source_factors(
        self,
        link: str,
        theta: float,
        mu: np.ndarray,
        eta: np.ndarray,
        y: np.ndarray,
        weight: np.ndarray,
        offset: np.ndarray,
        *,
        direct_design: np.ndarray | None = None,
        direct_penalty: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Evaluate pinned ``dDeta`` and source positive-observed retry in R."""
        self._require_rpy2()
        base = self._ro.baseenv
        mu_r, eta_r, y_r, weight_r, offset_r = (
            self._to_r_vector(values) for values in (mu, eta, y, weight, offset)
        )
        family = self._call_internal(
            "fix.family.link", self._mgcv.nb(theta=theta, link=link)
        )
        derivatives = self._call_internal(
            "dDeta",
            y_r,
            mu_r,
            weight_r,
            self._base.log(self._to_r_vector([theta])),
            family,
            0,
        )
        half = self._to_r_vector([0.5])
        weight_observed = base["*"](half, derivatives.rx2("Deta2"))
        fisher = base["*"](half, derivatives.rx2("EDeta2"))
        centered_eta = base["-"](eta_r, offset_r)
        weighted_response = base["-"](
            base["*"](weight_observed, centered_eta),
            base["*"](half, derivatives.rx2("Deta")),
        )
        response = base["-"](centered_eta, derivatives.rx2("Deta.Deta2"))
        deviance = self._base.sum(family.rx2("dev.resids")(y_r, mu_r, weight_r))
        invalid = base["|"](
            base["!"](base["is.finite"](weight_observed)),
            base["!"](base["is.finite"](response)),
        )
        retry_invalid = base["|"](
            base["!"](base["is.finite"](weight_observed)),
            base["<="](weight_observed, 0),
        )
        retried_weight = base["ifelse"](
            retry_invalid, self._to_r_vector([0.0]), weight_observed
        )
        retried_response = base["-"](
            base["*"](retried_weight, centered_eta),
            base["*"](half, derivatives.rx2("Deta")),
        )
        good = base["&"](
            base["is.finite"](retried_weight),
            base["is.finite"](retried_response),
        )
        output: dict[str, Any] = {
            "w": _r_array(weight_observed),
            "fisher": _r_array(fisher),
            "wz": _r_array(weighted_response),
            "z": _r_array(response),
            "dev": float(deviance[0]),
            "use.wy": bool(self._base.any(invalid)[0]),
            "retry_w": _r_array(retried_weight),
            "retry_wz": _r_array(retried_response),
            "good.count": int(self._base.sum(good)[0]),
        }
        if direct_design is not None:
            if direct_penalty is None:
                raise ValueError("direct_penalty is required with direct_design")
            design_r = self._to_r_matrix(direct_design)
            penalty_r = self._to_r_matrix(direct_penalty)
            gram = self._base.crossprod(design_r, base["*"](retried_weight, design_r))
            rhs = self._base.crossprod(design_r, retried_response)
            beta = self._base.solve(base["+"](gram, penalty_r), rhs)
            output["beta"] = _r_array(beta).ravel().copy()
        return output

    def nb_batch_derivative_contractions(
        self,
        link: str,
        theta: float,
        beta: np.ndarray,
        X: np.ndarray,
        y: np.ndarray,
        weight: np.ndarray,
        offset: np.ndarray,
        observed_bar: np.ndarray,
        deviance_bar: float,
        likelihood_bar: float,
    ) -> dict[str, Any]:
        """Contract pinned ``dDeta`` and ``nb()$ls`` at one streamed state."""
        self._require_rpy2()
        base = self._ro.baseenv
        X_r, C_r = self._to_r_matrix(X), self._to_r_matrix(observed_bar)
        beta_r, y_r, weight_r, offset_r = (
            self._to_r_vector(values) for values in (beta, y, weight, offset)
        )
        half = self._to_r_vector([0.5])
        theta_r = self._to_r_vector([theta])
        dbar_r = self._to_r_vector([deviance_bar])
        lsbar_r = self._to_r_vector([likelihood_bar])
        family = self._call_internal(
            "fix.family.link",
            self._mgcv.nb(theta=base["-"](theta_r), link=link),
        )
        eta = base["+"](base["drop"](base["%*%"](X_r, beta_r)), offset_r)
        mu = family.rx2("linkinv")(eta)
        log_theta = self._base.log(theta_r)
        derivatives = self._call_internal(
            "dDeta", y_r, mu, weight_r, log_theta, family, 1
        )
        cc = base["rowSums"](base["*"](base["%*%"](X_r, base["t"](C_r)), X_r))
        likelihood = family.rx2("ls")(y_r, weight_r, log_theta, 1)
        beta_term = base["+"](
            base["*"](base["*"](half, cc), derivatives.rx2("Deta3")),
            base["*"](dbar_r, derivatives.rx2("Deta")),
        )
        theta_term = base["+"](
            base["+"](
                self._base.sum(
                    base["*"](base["*"](half, cc), derivatives.rx2("Deta2th"))
                ),
                base["*"](dbar_r, self._base.sum(derivatives.rx2("Dth"))),
            ),
            base["*"](lsbar_r, likelihood.rx2("lsth1")),
        )
        half_curvature = base["*"](half, derivatives.rx2("Deta2"))
        weighted_response = base["-"](
            base["*"](half_curvature, base["-"](eta, offset_r)),
            base["*"](half, derivatives.rx2("Deta")),
        )
        good = base["&"](
            base["is.finite"](half_curvature),
            base["is.finite"](weighted_response),
        )
        return {
            "beta": _r_array(base["drop"](self._base.crossprod(X_r, beta_term))),
            "log.theta": _r_array(theta_term),
            "stationarity": _r_array(
                base["drop"](
                    self._base.crossprod(
                        X_r, base["*"](half, derivatives.rx2("Detath"))
                    )
                )
            ),
            "good": int(self._base.sum(good)[0]),
        }

    def binomial_likelihood_aic(
        self, y: np.ndarray, weight: np.ndarray, mu: np.ndarray
    ) -> np.ndarray:
        """Return pinned saturated likelihood and binomial AIC for one response."""
        self._require_rpy2()
        y_r, weight_r, mu_r = (self._to_r_vector(values) for values in (y, weight, mu))
        trials = self._base.rep(1.0, times=len(y))
        family = self._stats.binomial()
        saturated = self._call_internal("fix.family.ls", family).rx2("ls")(
            y_r, weight_r, trials, 1
        )
        aic = family.rx2("aic")(y_r, trials, mu_r, weight_r, 0)
        return np.asarray([saturated[0], aic[0]], dtype=np.float64)

    def binomial_deviance_curvature(
        self, y: np.ndarray, weight: np.ndarray, mu: np.ndarray
    ) -> np.ndarray:
        """Evaluate stats binomial deviance and direct mean-scale curvature in R."""
        self._require_rpy2()
        y_r, weight_r, mu_r = (self._to_r_vector(values) for values in (y, weight, mu))
        base = self._ro.baseenv
        one = self._to_r_vector([1.0])
        squared_mu = base["^"](mu_r, 2)
        squared_complement = base["^"](base["-"](one, mu_r), 2)
        curvature = base["*"](
            weight_r,
            base["+"](
                base["/"](y_r, squared_mu),
                base["/"](base["-"](one, y_r), squared_complement),
            ),
        )
        deviance = self._stats.binomial().rx2("dev.resids")(y_r, mu_r, weight_r)
        return np.column_stack((_r_array(deviance), _r_array(curvature)))

    def binomial_aic_rows(
        self, y: np.ndarray, weight: np.ndarray, mu: np.ndarray
    ) -> np.ndarray:
        """Call stats binomial AIC independently on each source row."""
        self._require_rpy2()
        aic = self._stats.binomial().rx2("aic")
        return np.asarray(
            [
                aic(
                    self._to_r_vector(y[index : index + 1]),
                    self._to_r_vector([1.0]),
                    self._to_r_vector(mu[index : index + 1]),
                    self._to_r_vector(weight[index : index + 1]),
                    0,
                )[0]
                for index in range(len(y))
            ],
            dtype=np.float64,
        )

    def binomial_log_initial_terms(
        self,
        y: np.ndarray,
        weight: np.ndarray,
        offset: np.ndarray,
        *,
        eta: np.ndarray | None = None,
    ) -> dict[str, np.ndarray]:
        """Evaluate pinned ``gam.fit3`` Binomial/log initial raw W/z algebra."""
        from tests.r_ast import call, symbol

        self._require_rpy2()
        ro, base = self._ro, self._ro.baseenv
        family = self._call_internal(
            "fix.family.link",
            self._call_internal("fix.family.var", self._stats.binomial(link="log")),
        )
        r_y, r_weight, r_offset = (
            self._to_r_vector(values) for values in (y, weight, offset)
        )
        if eta is None:
            environment = ro.r["new.env"](parent=ro.r["globalenv"]())
            environment["family"] = family
            environment["y"] = r_y
            environment["weights"] = r_weight
            environment["nobs"] = ro.IntVector([len(y)])
            environment["mustart"] = ro.NULL
            environment[".jaxgam_initialize"] = family.rx2("initialize")
            # Evaluate only the installed family expression object in its data
            # environment; the call itself is constructed as an R AST.
            ro.r["eval"](
                call("quote", call("eval", symbol(".jaxgam_initialize"))),
                envir=environment,
            )
            r_eta = family.rx2("linkfun")(environment["mustart"])
        else:
            r_eta = self._to_r_vector(eta)
        mu = family.rx2("linkinv")(r_eta)
        variance = family.rx2("variance")(mu)
        mu_eta = family.rx2("mu.eta")(r_eta)
        dvar = family.rx2("dvar")(mu)
        d2link = family.rx2("d2link")(mu)
        residual = base["-"](r_y, mu)
        one = self._to_r_vector([1.0])
        alpha_raw = base["+"](
            one,
            base["*"](
                residual,
                base["+"](base["/"](dvar, variance), base["*"](d2link, mu_eta)),
            ),
        )
        machine_epsilon = ro.r[".Machine"].rx2("double.eps")
        alpha = base["ifelse"](base["=="](alpha_raw, 0), machine_epsilon, alpha_raw)
        working_weight = base["/"](
            base["*"](base["*"](r_weight, alpha), base["^"](mu_eta, 2)),
            variance,
        )
        response = base["+"](
            base["-"](r_eta, r_offset),
            base["/"](residual, base["*"](mu_eta, alpha)),
        )
        return {
            name: _r_array(value)
            for name, value in (
                ("eta", r_eta),
                ("mu", mu),
                ("variance", variance),
                ("mu_eta", mu_eta),
                ("dvar", dvar),
                ("d2link", d2link),
                ("alpha_raw", alpha_raw),
                ("alpha", alpha),
                ("weight", working_weight),
                ("response", response),
            )
        }

    def binomial_log_cancellation_terms(self) -> dict[str, np.ndarray]:
        """Build the pinned five-row cancellation case with R arithmetic."""
        self._require_rpy2()
        base = self._ro.baseenv
        one = self._to_r_vector([1.0])
        epsilon = self._ro.r[".Machine"].rx2("double.eps")
        neighbor = base["-"](one, base["/"](epsilon, 2))
        near_one = base["-"](one, self._to_r_vector([1e-12]))
        y = base["c"](one, neighbor, near_one, one, neighbor)
        thirteen_eighteenths = base["/"](
            self._to_r_vector([13.0]), self._to_r_vector([18.0])
        )
        eta = self._base.log(
            base["c"](
                thirteen_eighteenths,
                thirteen_eighteenths,
                thirteen_eighteenths,
                self._to_r_vector([0.3, 0.9]),
            )
        )
        return self.binomial_log_initial_terms(
            _r_array(y),
            np.asarray([0.8, 1.7, 0.25, 2.3, 0.4]),
            np.asarray([-0.1, 0.2, 0.0, -0.4, 0.7]),
            eta=_r_array(eta),
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
                eta=_r_array(eta),
                mu=_r_array(mu),
                weight=_r_array(weight),
                z=_r_array(response),
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

    def binomial_log_admission_reference(
        self, y: np.ndarray, weight: np.ndarray, tolerance: float
    ) -> dict[str, Any]:
        """Run the pinned intercept-only ``gam.fit3`` boundary model."""
        from rpy2.rinterface_lib.embedded import RRuntimeError

        self._require_rpy2()
        family = self._call_internal(
            "fix.family.ls",
            self._call_internal(
                "fix.family.var",
                self._call_internal(
                    "fix.family.link", self._stats.binomial(link="log")
                ),
            ),
        )
        r_y = self._to_r_vector(y)
        r_weight = self._to_r_vector(weight)
        try:
            fit = self._call_internal(
                "gam.fit3",
                x=self._to_r_matrix(np.ones((len(y), 1))),
                y=r_y,
                sp=self._ro.FloatVector([]),
                Eb=self._to_r_vector([0.0]),
                UrS=self._ro.baseenv["list"](),
                weights=r_weight,
                offset=self._to_r_vector(np.zeros(len(y))),
                Mp=1,
                family=family,
                control=self._mgcv.gam_control(epsilon=tolerance, maxit=100),
                deriv=0,
                scale=1,
                scoreType="REML",
                **{
                    "null.coef": self._base.log(self._base.mean(r_y)),
                },
            )
        except RRuntimeError as error:
            return {"error": str(error)}
        if not bool(fit.rx2("converged")[0]):
            raise RBridgeError("Pinned Binomial/log admission fit did not converge")
        return {
            "reference": np.r_[
                _r_array(fit.rx2("coefficients")),
                fit.rx2("deviance")[0],
                1.0,
                fit.rx2("trA")[0],
                fit.rx2("REML")[0],
            ],
            "covariance": _r_array(self._base.tcrossprod(fit.rx2("rV"))),
            "fitted": _r_array(fit.rx2("fitted.values")),
        }

    def gamma_fisher_recovery_reference(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weight: np.ndarray,
        offset: np.ndarray,
    ) -> dict[str, Any]:
        """Trace the pinned Gamma/identity switch from observed to Fisher W."""
        from tests.r_ast import find_call_paths, instrument_function, symbol

        self._require_rpy2()
        ro = self._ro
        source = self._utils.getFromNamespace("gam.fit3", "mgcv")
        paths = find_call_paths(source, "<", required_symbols=("oo", "n"))
        if len(paths) != 1:
            raise RBridgeError("Pinned gam.fit3 indefinite-PLS anchor changed")
        node = ro.r["body"](source)
        for index in paths[0]:
            node = node[index]
        if not (
            node[1][0].rsame(symbol("$"))
            and node[1][1].rsame(symbol("oo"))
            and node[1][2].rsame(symbol("n"))
        ):
            raise RBridgeError("Pinned gam.fit3 indefinite-PLS condition changed")
        recoveries = [0]

        def record(oo: Any) -> None:
            if int(ro.ListVector(oo).rx2("n")[0]) < 0:
                recoveries[0] += 1

        fit_function = instrument_function(
            source,
            path=paths[0],
            expected_head="<",
            capture_symbols=("oo",),
            callback=record,
        )
        family = self._call_internal(
            "fix.family.ls",
            self._call_internal(
                "fix.family.var",
                self._call_internal(
                    "fix.family.link", self._stats.Gamma(link="identity")
                ),
            ),
        )
        r_X = self._to_r_matrix(X)
        r_y, r_weight, r_offset = (
            self._to_r_vector(values) for values in (y, weight, offset)
        )
        null = self._base.c(
            family.rx2("linkfun")(self._base.mean(r_y)),
            self._to_r_vector([0.0]),
        )

        def run(max_iter: int) -> Any:
            return fit_function(
                x=r_X,
                y=r_y,
                sp=self._base.log(self._to_r_vector([0.7])),
                Eb=self._to_r_vector([0.0]),
                UrS=ro.baseenv["list"](),
                weights=r_weight,
                offset=r_offset,
                Mp=2,
                family=family,
                control=self._mgcv.gam_control(epsilon=1e-10, maxit=max_iter),
                deriv=0,
                scale=0,
                scoreType="REML",
                **{"null.coef": null},
            )

        fit = run(100)
        successful_recoveries = recoveries[0]
        if not bool(fit.rx2("converged")[0]) or successful_recoveries <= 0:
            raise RBridgeError("Pinned Gamma source recovery did not converge")
        limited = run(2)
        if bool(limited.rx2("converged")[0]):
            raise RBridgeError("Pinned Gamma limited control unexpectedly converged")
        return {
            "reference": np.r_[
                _r_array(fit.rx2("coefficients")),
                fit.rx2("deviance")[0],
                fit.rx2("scale.est")[0],
                fit.rx2("trA")[0],
                fit.rx2("REML")[0],
            ],
            "covariance": _r_array(self._base.tcrossprod(fit.rx2("rV"))),
            "fitted": _r_array(fit.rx2("fitted.values")),
            "recoveries": successful_recoveries,
        }

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

    def qr_null_projection(
        self, X: np.ndarray, target: float
    ) -> tuple[int, np.ndarray, np.ndarray]:
        """Return pinned R QR rank, natural pivots and null coefficients."""
        self._require_rpy2()
        r_X = self._to_r_matrix(X)
        factor = self._base.qr(r_X)
        coefficients = _r_array(
            self._base.qr_coef(
                factor, self._base.rep(target, times=np.asarray(X).shape[0])
            ),
            dtype=np.float64,
        )
        coefficients[np.isnan(coefficients)] = 0.0
        return (
            int(factor.rx2("rank")[0]),
            _r_array(factor.rx2("pivot"), dtype=np.int32) - 1,
            coefficients,
        )

    def _regular_source_family(
        self, family_name: str, link: str, *, fix_base: bool
    ) -> Any:
        """Construct the installed family with a caller-owned source modifier set."""
        self._require_rpy2()
        constructors = {
            "gaussian": self._stats.gaussian,
            "gamma": self._stats.Gamma,
            "poisson": self._stats.poisson,
            "binomial": self._stats.binomial,
        }
        family_key = family_name.lower()
        if family_key not in constructors:
            raise ValueError("unsupported regular source family")
        source_link = "1/mu^2" if link == "inverse_squared" else link
        family = constructors[family_key](link=source_link)
        modifiers = ["fix.family.link", "fix.family.var", "fix.family.ls"]
        if fix_base:
            modifiers.insert(0, "fix.family")
        for modifier in modifiers:
            family = self._call_internal(modifier, family)
        return family

    def _regular_gam_fit3_inputs(
        self,
        family_name: str,
        link: str,
        X: np.ndarray,
        y: np.ndarray,
        weight: np.ndarray,
        offset: np.ndarray,
    ) -> tuple[Any, Any, Any, Any, Any, np.ndarray, bool]:
        """Initialize a pinned family and project its natural null start."""
        family = self._regular_source_family(family_name, link, fix_base=True)
        ro = self._ro
        r_X = self._to_r_matrix(X)
        r_y, r_weight, r_offset = (
            self._to_r_vector(values) for values in (y, weight, offset)
        )
        # Pinned get.null.coef evaluates the installed initialization expression
        # in a local data environment before projecting its normalized response.
        initialization = ro.r["new.env"](parent=ro.r["globalenv"]())
        initialization["family"] = family
        initialization["y"] = r_y
        initialization["weights"] = r_weight
        initialization["nobs"] = ro.IntVector([len(y)])
        ro.r["evalq"](family.rx2("initialize"), envir=initialization)
        r_y = initialization["y"]
        normalized_target = self._base.rep(
            family.rx2("linkfun")(self._base.mean(r_y)), times=len(y)
        )
        null = _r_array(
            self._base.qr_coef(self._base.qr(r_X), normalized_target), dtype=np.float64
        )
        null[np.isnan(null)] = 0.0
        known_scale = family_name.lower() in {"binomial", "poisson"}
        return family, r_X, r_y, r_weight, r_offset, null, known_scale

    def regular_gam_fit3_start(
        self,
        family_name: str,
        link: str,
        X: np.ndarray,
        y: np.ndarray,
        weight: np.ndarray,
        offset: np.ndarray,
        start: np.ndarray | None = None,
        *,
        tolerance: float = 1e-7,
        max_iter: int = 200,
        require_convergence: bool = True,
        score_scale: float = 0.7,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Run pinned unpenalized ``gam.fit3`` with the natural QR null start."""
        if not np.isfinite(score_scale) or score_scale <= 0:
            raise ValueError("score_scale must be positive and finite")
        family, r_X, r_y, r_weight, r_offset, null, known_scale = (
            self._regular_gam_fit3_inputs(family_name, link, X, y, weight, offset)
        )
        ro = self._ro
        fit = self._call_internal(
            "gam.fit3",
            x=r_X,
            y=r_y,
            sp=ro.FloatVector([])
            if known_scale
            else self._base.log(ro.FloatVector([score_scale])),
            Eb=ro.FloatVector([0.0]),
            UrS=ro.baseenv["list"](),
            weights=r_weight,
            offset=r_offset,
            U1=self._base.diag(X.shape[1]),
            Mp=X.shape[1],
            family=family,
            control=self._mgcv.gam_control(epsilon=tolerance, maxit=max_iter),
            deriv=0,
            scale=1 if known_scale else 0,
            scoreType="REML",
            **{"null.coef": self._to_r_vector(null)},
            start=ro.NULL if start is None else self._to_r_vector(start),
        )
        status = np.asarray(
            [bool(fit.rx2("converged")[0]), int(fit.rx2("iter")[0])],
            dtype=np.float64,
        )
        if require_convergence and not bool(status[0]):
            raise RBridgeError("Pinned gam.fit3 start oracle did not converge")
        reference = np.r_[  # same source field order as the old binary oracle
            _r_array(fit.rx2("coefficients")),
            float(fit.rx2("deviance")[0]),
            1.0 if known_scale else float(fit.rx2("scale.est")[0]),
            float(fit.rx2("trA")[0]),
            float(fit.rx2("REML")[0]),
        ]
        covariance = _r_array(self._base.tcrossprod(fit.rx2("rV")))
        return reference, null, covariance, status

    def regular_coefficient_fit(
        self,
        family_name: str,
        link: str,
        X: np.ndarray,
        y: np.ndarray,
        weight: np.ndarray,
        offset: np.ndarray,
        start: np.ndarray | None = None,
        *,
        tolerance: float = 1e-7,
        max_iter: int = 200,
        require_convergence: bool = True,
        score_scale: float = 0.7,
    ) -> dict[str, np.ndarray]:
        """Expose the shared pinned regular start reference by field name."""
        reference, null, covariance, status = self.regular_gam_fit3_start(
            family_name,
            link,
            X,
            y,
            weight,
            offset,
            start,
            tolerance=tolerance,
            max_iter=max_iter,
            require_convergence=require_convergence,
            score_scale=score_scale,
        )
        return {
            "reference": reference,
            "null": null,
            "covariance": covariance,
            "status": status,
        }

    def regular_gam_fit3_penalized(
        self,
        family_name: str,
        link: str,
        X: np.ndarray,
        y: np.ndarray,
        weight: np.ndarray,
        offset: np.ndarray,
        start: np.ndarray,
        rho: np.ndarray,
        *,
        score_phi: float = 0.7,
        derivatives: bool = False,
    ) -> dict[str, np.ndarray]:
        """Run the pinned rank-one regular system and inspect its gdi candidate."""
        from tests.r_ast import find_call_paths, instrument_function

        family, r_X, r_y, r_weight, r_offset, null, known_scale = (
            self._regular_gam_fit3_inputs(family_name, link, X, y, weight, offset)
        )
        ro = self._ro
        source = self._utils.getFromNamespace("gam.fit3", "mgcv")
        paths = find_call_paths(source, "<-", required_symbols=("oo", "C_gdi1"))
        if len(paths) != 1:
            raise RBridgeError("Pinned gam.fit3 gdi candidate assignment changed")
        candidates: list[bool] = []

        def capture(
            output: Any,
            matrix: Any,
            model_offset: Any,
            valid_eta: Any,
            valid_mu: Any,
            inverse_link: Any,
        ) -> None:
            beta = ro.ListVector(output).rx2("beta")
            base = ro.baseenv
            candidate_eta = base["drop"](
                base["+"](base["%*%"](matrix, beta), model_offset)
            )
            candidates.append(
                bool(valid_eta(candidate_eta)[0])
                and bool(valid_mu(inverse_link(candidate_eta))[0])
            )

        fit_source = instrument_function(
            source,
            path=paths[0],
            expected_head="<-",
            capture_symbols=("oo", "x", "offset", "valideta", "validmu", "linkinv"),
            callback=capture,
            when="after",
        )
        weighted_rho = self._to_r_vector(np.atleast_1d(rho))
        if not known_scale:
            weighted_rho = self._base.c(
                weighted_rho, self._base.log(self._to_r_vector([score_phi]))
            )
        penalty_root = self._base.matrix(self._to_r_vector([0.0, 1.0]), 1, 2)
        coordinates = self._base.matrix(self._to_r_vector([0.0, 1.0, 1.0, 0.0]), 2, 2)
        fit = fit_source(
            x=r_X,
            y=r_y,
            sp=weighted_rho,
            Eb=penalty_root,
            UrS=ro.baseenv["list"](self._base.matrix(self._to_r_vector([1.0]), 1, 1)),
            weights=r_weight,
            offset=r_offset,
            U1=coordinates,
            Mp=1,
            family=family,
            control=self._mgcv.gam_control(epsilon=1e-7, maxit=200),
            deriv=int(derivatives),
            scale=1 if known_scale else 0,
            scoreType="REML",
            **{"null.coef": self._to_r_vector(null)},
            start=self._to_r_vector(start),
        )
        if not bool(fit.rx2("converged")[0]):
            raise RBridgeError("Pinned penalized gam.fit3 oracle did not converge")
        reference = np.r_[
            _r_array(fit.rx2("coefficients")),
            float(fit.rx2("deviance")[0]),
            1.0 if known_scale else float(fit.rx2("scale.est")[0]),
            float(fit.rx2("trA")[0]),
            float(fit.rx2("REML")[0]),
        ]
        return {
            "reference": reference,
            "null": null,
            "covariance": _r_array(self._base.tcrossprod(fit.rx2("rV"))),
            "status": np.asarray(
                [bool(fit.rx2("converged")[0]), int(fit.rx2("iter")[0])],
                dtype=np.float64,
            ),
            "candidate_valid": np.asarray(candidates[-1:], dtype=np.float64),
            "gradient": (
                _r_array(fit.rx2("REML1"), dtype=np.float64)
                if derivatives
                else np.empty(0)
            ),
        }

    def regular_source_zero_null(self, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
        """Compare a raw zero null anchor in pinned Poisson and Gamma fits."""
        from rpy2.rinterface_lib.embedded import RRuntimeError

        self._require_rpy2()
        ro = self._ro
        base = ro.baseenv
        r_y = self._to_r_vector(np.asarray(y, dtype=np.float64))
        r_X = base["cbind"](self._base.rep(1.0, times=len(y)), self._to_r_vector(x))
        weight = self._base.rep(1.0, times=len(y))
        offset = self._base.rep(0.0, times=len(y))
        results = {}
        for name, scale in (("poisson", 1), ("Gamma", 0)):
            family = self._regular_source_family(name, "identity", fix_base=False)
            environment = self._base.new_env(parent=ro.globalenv)
            environment["family"] = family
            environment["y"] = r_y
            environment["weights"] = weight
            environment["nobs"] = ro.IntVector([len(y)])
            ro.r["evalq"](family.rx2("initialize"), envir=environment)
            try:
                fit = self._call_internal(
                    "gam.fit3",
                    x=r_X,
                    y=environment["y"],
                    sp=ro.FloatVector([]),
                    Eb=0,
                    UrS=base["list"](),
                    weights=weight,
                    offset=offset,
                    U1=self._base.diag(self._base.ncol(r_X)),
                    Mp=self._base.ncol(r_X),
                    family=family,
                    control=self._mgcv.gam_control(epsilon=1e-7, maxit=200),
                    deriv=0,
                    scale=scale,
                    scoreType="REML",
                    **{"null.coef": self._base.numeric(self._base.ncol(r_X))},
                )
            except RRuntimeError:
                if name != "Gamma":
                    raise
                results["gamma_failed"] = True
            else:
                if name == "Gamma":
                    results["gamma_failed"] = False
                else:
                    results["beta"] = _r_array(fit.rx2("coefficients"))
                    results["deviance"] = _r_array(fit.rx2("deviance"))
        return results

    def regular_source_gamma_fixed_trial(
        self,
        link: str,
        X: np.ndarray,
        y: np.ndarray,
        weight: np.ndarray,
        offset: np.ndarray,
    ) -> np.ndarray:
        """Evaluate pinned Gamma ``gam.fit3`` with the declared scale trial."""
        self._require_rpy2()
        ro = self._ro
        family = self._regular_source_family("Gamma", link, fix_base=False)
        r_X = self._to_r_matrix(X)
        r_y = self._to_r_vector(np.asarray(y, dtype=np.float64))
        null = self._base.c(
            family.rx2("linkfun")(self._base.mean(r_y)), self._to_r_vector([0.0])
        )
        fit = self._call_internal(
            "gam.fit3",
            x=r_X,
            y=r_y,
            sp=self._base.log(self._to_r_vector([0.7])),
            Eb=0,
            UrS=ro.baseenv["list"](),
            weights=self._to_r_vector(np.asarray(weight, dtype=np.float64)),
            offset=self._to_r_vector(np.asarray(offset, dtype=np.float64)),
            U1=self._base.diag(2),
            Mp=2,
            family=family,
            control=self._mgcv.gam_control(epsilon=1e-12),
            deriv=0,
            scale=0,
            scoreType="REML",
            **{"null.coef": null},
        )
        return np.r_[
            _r_array(fit.rx2("coefficients")),
            _r_array(fit.rx2("deviance")),
            _r_array(fit.rx2("scale.est")),
            _r_array(fit.rx2("trA")),
            _r_array(fit.rx2("REML")),
        ]

    def regular_source_penalized_gamma_score(
        self,
        X: np.ndarray,
        E: np.ndarray,
        y: np.ndarray,
        weight: np.ndarray,
        offset: np.ndarray,
        *,
        controlled_invalid_refit: bool,
    ) -> dict[str, np.ndarray]:
        """Trace one private pinned gdi return, with an optional invalid solve."""
        from rpy2 import rinterface

        from tests.r_ast import (
            call,
            find_call_paths,
            make_r_callback,
            replace_call,
            symbol,
        )

        self._require_rpy2()
        ro = self._ro
        base = ro.baseenv
        r_X, r_E = self._to_r_matrix(X), self._to_r_matrix(E)
        r_y, r_weight, r_offset = (
            self._to_r_vector(np.asarray(values, dtype=np.float64))
            for values in (y, weight, offset)
        )
        width, rank = X.shape[1], E.shape[0]
        coordinates = self._base.eigen(self._base.crossprod(r_E), symmetric=True).rx2(
            "vectors"
        )
        range_coordinates = base["["](
            coordinates,
            rinterface.MissingArg,
            self._base.seq_len(rank),
            drop=False,
        )
        root = base["%*%"](self._base.t(range_coordinates), self._base.t(r_E))
        family = self._regular_source_family("Gamma", "identity", fix_base=False)
        null = self._base.c(
            family.rx2("linkfun")(self._base.mean(r_y)),
            self._base.rep(0.0, times=width - 1),
        )
        source = self._utils.getFromNamespace("gam.fit3", "mgcv")
        paths = find_call_paths(source, "<-", required_symbols=("oo", "C_gdi1"))
        if len(paths) != 1:
            raise RBridgeError("Pinned gam.fit3 gdi return assignment changed")
        captured: list[np.ndarray] = []

        def inspect_return(
            output: Any,
            raw_deviance: Any,
            stopping_pdev: Any,
            score_phi: Any,
            transform: Any,
            penalty: Any,
            matrix: Any,
            model_offset: Any,
            valid_eta: Any,
            valid_mu: Any,
            inverse_link: Any,
        ) -> Any:
            result = ro.ListVector(output)
            if controlled_invalid_refit:
                bad = self._to_r_vector(np.r_[-10.0, 3.0, np.zeros(width - 2)])
                beta = self._base.c(self._base.crossprod(transform, bad))
                result = base["$<-"](result, "beta", value=beta)
                penalized = base["drop"](
                    base["%*%"](base["%*%"](self._base.t(beta), penalty), beta)
                )
                result = base["$<-"](result, "conv.tol", value=penalized)
            candidate_eta = base["drop"](
                base["+"](base["%*%"](matrix, result.rx2("beta")), model_offset)
            )
            valid = bool(valid_eta(candidate_eta)[0]) and bool(
                valid_mu(inverse_link(candidate_eta))[0]
            )
            captured.append(
                np.asarray(
                    [
                        raw_deviance[0],
                        stopping_pdev[0],
                        result.rx2("conv.tol")[0],
                        valid,
                        score_phi[0],
                    ],
                    dtype=np.float64,
                )
            )
            return result

        callback = make_r_callback(inspect_return)
        original = ro.r["body"](source)
        assignment = original
        for index in paths[0]:
            assignment = assignment[index]
        replacement = call(
            "{",
            assignment,
            call(
                "<-",
                symbol("oo"),
                call(
                    callback,
                    *(
                        symbol(name)
                        for name in (
                            "oo",
                            "dev",
                            "pdev",
                            "scale",
                            "T",
                            "St",
                            "x",
                            "offset",
                            "valideta",
                            "validmu",
                            "linkinv",
                        )
                    ),
                ),
            ),
            symbol("oo"),
        )
        private = replace_call(
            source,
            path=paths[0],
            expected_head="<-",
            replacement=replacement,
        )
        private._jaxgam_callback_handles = (inspect_return, callback)
        fit = private(
            x=r_X,
            y=r_y,
            sp=self._base.c(
                self._base.log(self._to_r_vector([0.35])),
                self._base.log(self._to_r_vector([0.7])),
            ),
            Eb=r_E,
            UrS=base["list"](root),
            weights=r_weight,
            offset=r_offset,
            U1=coordinates,
            Mp=width - rank,
            family=family,
            control=self._mgcv.gam_control(epsilon=1e-12),
            deriv=0,
            scale=0,
            scoreType="REML",
            **{"null.coef": null},
        )
        if not bool(fit.rx2("converged")[0]):
            raise RBridgeError("Pinned signed-score gam.fit3 oracle did not converge")
        if not captured:
            raise RBridgeError("Pinned signed-score gdi return was not reached")
        return {
            "reference": np.r_[
                _r_array(fit.rx2("coefficients")),
                _r_array(fit.rx2("deviance")),
                _r_array(fit.rx2("scale.est")),
                _r_array(fit.rx2("trA")),
                _r_array(fit.rx2("REML")),
            ],
            "fisher_inverse": _r_array(
                self._base.tcrossprod(fit.rx2("rV")), dtype=np.float64
            ),
            "source_payload": np.r_[captured[-1], _r_array(fit.rx2("scale.est"))],
        }

    def nb_selected_fit(
        self,
        X: np.ndarray,
        E: np.ndarray,
        y: np.ndarray,
        weight: np.ndarray,
        offset: np.ndarray,
        params: np.ndarray,
        link: str,
        *,
        estimated: bool,
        epsilon: float = 1e-11,
        max_iter: int = 200,
        deriv: int = 1,
    ) -> dict[str, Any]:
        """Evaluate pinned gam.fit4 in supplied NB fitting coordinates (theta 2.7).

        ``deriv=2`` also returns gam.fit4's exact REML Hessian ``REML2`` in the
        same (log sp, log theta) order as the gradient.
        """
        self._require_rpy2()
        if deriv not in (1, 2):
            raise ValueError("deriv must be 1 or 2")
        ro = self._ro
        X, E, y, weight, offset, params = (
            np.asarray(value, dtype=np.float64)
            for value in (X, E, y, weight, offset, params)
        )
        r_X, r_E = self._to_r_matrix(X), self._to_r_matrix(E)
        r_y = self._to_r_vector(y)
        family = self._call_internal(
            "fix.family.link",
            self._mgcv.nb(
                theta=-float(self._base.exp(self._to_r_vector(params[-1:]))[0])
                if estimated
                else 2.7,
                link=link,
            ),
        )
        rank = E.shape[0]
        U1 = self._base.eigen(self._base.crossprod(r_E), symmetric=True).rx2("vectors")
        U1_penalized = self._to_r_matrix(_r_array(U1)[:, :rank])
        root = ro.baseenv["%*%"](self._base.t(U1_penalized), self._base.t(r_E))
        target = self._base.rep(
            family.rx2("linkfun")(self._base.mean(r_y)), times=len(y)
        )
        null = _r_array(
            self._base.qr_coef(self._base.qr(r_X), target), dtype=np.float64
        )
        null[np.isnan(null)] = 0.0
        smoothing = params[[1, 0]] if estimated else params[:1]
        fit = self._call_internal(
            "gam.fit4",
            x=r_X,
            y=r_y,
            sp=self._to_r_vector(smoothing),
            Eb=r_E,
            UrS=ro.baseenv["list"](root),
            weights=self._to_r_vector(weight),
            offset=self._to_r_vector(offset),
            U1=U1,
            Mp=X.shape[1] - rank,
            family=family,
            control=self._mgcv.gam_control(epsilon=epsilon, maxit=max_iter),
            deriv=deriv,
            scale=1,
            scoreType="REML",
            **{"null.coef": self._to_r_vector(null)},
        )
        if not bool(fit.rx2("converged")[0]):
            raise RBridgeError("Pinned gam.fit4 coefficient oracle did not converge.")
        raw_gradient = _r_array(fit.rx2("REML1"), dtype=np.float64)
        # gam.fit4 orders (log theta, log sp); JaxGAM orders (log sp, log theta).
        order = [1, 0] if estimated else list(range(raw_gradient.size))
        gradient = raw_gradient[order].copy()
        result = {
            "score": float(fit.rx2("REML")[0]),
            "iter": int(fit.rx2("iter")[0]),
            "deviance": float(fit.rx2("deviance")[0]),
            "beta": _r_array(fit.rx2("coefficients")),
            "gradient": gradient,
        }
        if deriv == 2:
            hessian = _r_array(fit.rx2("REML2"), dtype=np.float64).reshape(
                len(order), len(order)
            )
            result["hessian"] = hessian[np.ix_(order, order)].copy()
        return result

    def nb_cubic_outer_fit(
        self,
        data: pd.DataFrame,
        weight: np.ndarray,
        offset: np.ndarray,
        link: str,
        *,
        estimated: bool,
        pinned_rho: float | None = None,
    ) -> dict[str, Any]:
        """Fit the NB cr(k=8) outer-optimizer model with pinned review controls."""
        self._require_rpy2()
        ro = self._ro
        r_data = self._to_r_dataframe(data[["x", "y"]])
        family = self._mgcv.nb(theta=-2.7 if estimated else 2.7, link=link)
        control = self._mgcv.gam_control(
            epsilon=1e-11,
            maxit=200,
            newton=ro.ListVector(
                {"conv.tol": 1e-6, "maxNstep": 5.0, "maxSstep": 2.0, "maxHalf": 30}
            ),
        )
        options: dict[str, Any] = {}
        if pinned_rho is not None:
            options["sp"] = self._base.exp(self._to_r_vector([pinned_rho]))
        fit = self._mgcv.gam(
            ro.Formula('y ~ s(x, bs="cr", k=8)'),
            data=r_data,
            weights=self._to_r_vector(weight),
            offset=self._to_r_vector(offset),
            family=family,
            method="REML",
            optimizer=ro.StrVector(["outer", "newton"]),
            control=control,
            **options,
        )
        outer = fit.rx2("outer.info")
        if outer is ro.NULL:
            # With both theta and smoothing fixed there is no outer step.
            outer_converged = not estimated and pinned_rho is not None
            outer_iterations = 0
        else:
            outer_converged = str(outer.rx2("conv")[0]) == "full convergence"
            outer_iterations = int(outer.rx2("iter")[0])
        return {
            "rho": (
                pinned_rho
                if pinned_rho is not None
                else float(self._base.log(fit.rx2("sp"))[0])
            ),
            "log_theta": float(fit.rx2("family").rx2("getTheta")()[0]),
            "score": float(fit.rx2("gcv.ubre")[0]),
            "fitted_values": _r_array(fit.rx2("fitted.values")),
            "deviance": float(fit.rx2("deviance")[0]),
            "edf": float(self._base.sum(fit.rx2("edf"))[0]),
            "scale": float(fit.rx2("sig2")[0]),
            "inner_converged": bool(fit.rx2("converged")[0]),
            "outer_converged": outer_converged,
            "outer_iterations": outer_iterations,
        }

    def nb_cubic_setup(
        self,
        data: pd.DataFrame,
        weight: np.ndarray,
        offset: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return gam.setup X, S, offsets and ranks for the NB cr(k=8) model."""
        self._require_rpy2()
        ro = self._ro
        setup = self._mgcv.gam(
            ro.Formula('y ~ s(x, bs="cr", k=8)'),
            data=self._to_r_dataframe(data[["x", "y"]]),
            weights=self._to_r_vector(weight),
            offset=self._to_r_vector(offset),
            family=self._mgcv.nb(theta=2.7, link="log"),
            method="REML",
            fit=False,
        )
        metadata = np.r_[_r_array(setup.rx2("off")), _r_array(setup.rx2("rank"))]
        return (
            _r_array(setup.rx2("X")),
            _r_array(setup.rx2("S")[0]),
            metadata,
        )

    def efs_nb_working_factors(
        self,
        link: str,
        y: np.ndarray,
        mu: np.ndarray,
        weights: np.ndarray,
        theta: float,
        offset: np.ndarray,
    ) -> dict[str, np.ndarray]:
        """Evaluate pinned ``dDeta`` observed W and Wz for NB links."""
        self._require_pinned_efs_versions()
        if link not in {"identity", "sqrt", "log"}:
            raise ValueError(
                "NB working-factor oracle link must be identity, sqrt or log"
            )
        arrays = [
            np.asarray(value, dtype=np.float64) for value in (y, mu, weights, offset)
        ]
        if (
            any(value.ndim != 1 for value in arrays)
            or len({len(value) for value in arrays}) != 1
        ):
            raise ValueError("NB working-factor oracle inputs must be aligned vectors")
        if not np.isfinite(theta) or theta <= 0.0:
            raise ValueError(
                "NB working-factor oracle theta must be finite and positive"
            )
        self._require_rpy2()
        r_y, r_mu, r_weights, r_offset = map(self._to_r_vector, arrays)
        r_family = self._call_internal("fix.family.link", self._mgcv.nb(link=link))
        derivatives = self._call_internal(
            "dDeta", r_y, r_mu, r_weights, self._base.log(theta), r_family, deriv=0
        )
        multiply = self._ro.r["*"]
        subtract = self._ro.r["-"]
        half = self._ro.FloatVector([0.5])
        r_weight = multiply(half, derivatives.rx2("Deta2"))
        r_response = subtract(
            multiply(r_weight, subtract(r_family.rx2("linkfun")(r_mu), r_offset)),
            multiply(half, derivatives.rx2("Deta")),
        )
        return {
            "weight": _r_array(r_weight, dtype=np.float64),
            "weighted_response": _r_array(r_response, dtype=np.float64),
        }

    def efs_nb_deviance_derivatives(
        self,
        link: str,
        y: np.ndarray,
        mu: np.ndarray,
        weights: np.ndarray,
        theta: float,
    ) -> dict[str, np.ndarray]:
        """Evaluate pinned NB deviance and its first two eta derivatives."""
        self._require_pinned_efs_versions()
        if link not in {"identity", "sqrt"}:
            raise ValueError("NB deviance oracle link must be identity or sqrt")
        arrays = [np.asarray(value, dtype=np.float64) for value in (y, mu, weights)]
        if (
            any(value.ndim != 1 for value in arrays)
            or len({len(value) for value in arrays}) != 1
        ):
            raise ValueError("NB deviance oracle inputs must be aligned vectors")
        if not np.isfinite(theta) or theta <= 0.0:
            raise ValueError("NB deviance oracle theta must be finite and positive")
        self._require_rpy2()
        r_y, r_mu, r_weights = map(self._to_r_vector, arrays)
        r_theta = self._base.log(theta)
        r_family = self._call_internal("fix.family.link", self._mgcv.nb(link=link))
        derivatives = self._call_internal(
            "dDeta", r_y, r_mu, r_weights, r_theta, r_family, deriv=0
        )
        return {
            "deviance": _r_array(
                r_family.rx2("dev.resids")(r_y, r_mu, r_weights, r_theta),
                dtype=np.float64,
            ),
            "deta": _r_array(derivatives.rx2("Deta"), dtype=np.float64),
            "deta2": _r_array(derivatives.rx2("Deta2"), dtype=np.float64),
        }

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
            trace.append(_r_array(theta, dtype=np.float64))

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
                    float(_r_array(value.rx2("nll"))[0]),
                    float(_r_array(value.rx2("g"))[0]),
                    float(_r_array(value.rx2("H")).ravel()[0]),
                ],
                dtype=np.float64,
            )

        initial = state(r_start)
        final_theta = traced(r_start, family, r_y, r_mu, scale=1, wt=r_weights)
        final = np.concatenate(
            (_r_array(final_theta, dtype=np.float64), state(final_theta))
        )
        trace_values = np.asarray(trace, dtype=np.float64).reshape(-1)
        trace_nll = np.asarray(
            [
                float(
                    _r_array(
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
            "value": float(_r_array(value)[0]),
            "d_eta": _r_array(multiply(derivatives.rx2("Dmu"), r_mu), dtype=np.float64),
            "h_eta": _r_array(
                add(
                    multiply(derivatives.rx2("Dmu2"), square(r_mu, 2)),
                    multiply(derivatives.rx2("Dmu"), r_mu),
                ),
                dtype=np.float64,
            ),
            "mixed": _r_array(
                multiply(derivatives.rx2("Dmuth"), r_mu), dtype=np.float64
            ),
            "d_theta": float(_r_array(self._base.sum(derivatives.rx2("Dth")))[0]),
            "h_theta": float(_r_array(self._base.sum(derivatives.rx2("Dth2")))[0]),
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
            "coefficients": _r_array(fit.rx2("coefficients"), dtype=np.float64),
            "log_theta": float(_r_array(theta_out)[0]),
            "deviance": float(_r_array(deviance)[0]),
            "fit_deviance": float(_r_array(fit.rx2("deviance"))[0]),
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
            return _r_array(value, dtype=np.float64).ravel(order="F")

        def scalar(value: Any) -> float:
            return float(_r_array(value, dtype=np.float64).ravel()[0])

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

    def efs_streamed_default_start_reference(
        self,
        x: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray,
        offset: np.ndarray,
        *,
        family: str,
        link: str,
        theta: float,
        smoothing: np.ndarray,
        scale: float,
        trace_null_coef: bool = False,
    ) -> dict[str, Any]:
        """Fit the pinned cubic EFS model with explicit source startup state."""
        self._require_rpy2()
        from rpy2 import rinterface

        from tests.r_ast import clone_function, make_r_callback

        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        weights = np.asarray(weights, dtype=np.float64)
        offset = np.asarray(offset, dtype=np.float64)
        smoothing = np.asarray(smoothing, dtype=np.float64)
        if any(
            value.ndim != 1 or value.shape != y.shape for value in (x, weights, offset)
        ):
            raise ValueError("EFS streamed reference columns must align")
        if smoothing.ndim != 1 or smoothing.size == 0:
            raise ValueError("EFS streamed reference needs smoothing parameters")
        if family == "nb":
            r_family = self._mgcv.nb(theta=float(theta), link=link)
        else:
            constructors = {
                "gaussian": self._stats.gaussian,
                "gamma": self._stats.Gamma,
                "poisson": self._stats.poisson,
                "binomial": self._stats.binomial,
            }
            try:
                r_family = constructors[family](link=link)
            except KeyError:
                raise ValueError(
                    f"Unsupported EFS streamed family: {family!r}"
                ) from None
        data = self._ro.DataFrame(
            {
                "x": self._to_r_vector(x),
                "y": self._to_r_vector(y),
                "w": self._to_r_vector(weights),
                "off": self._to_r_vector(offset),
            }
        )
        fit_args: dict[str, Any] = {
            "data": data,
            "weights": self._to_r_vector(weights),
            "offset": self._to_r_vector(offset),
            "family": r_family,
            "method": "REML",
            "optimizer": "efs",
            "control": self._mgcv.gam_control(
                epsilon=1e-7, maxit=200, efs_tol=0.1, efs_lspmax=15
            ),
            "in.out": self._ro.ListVector(
                {
                    "sp": self._to_r_vector(smoothing),
                    "scale": self._ro.FloatVector([scale]),
                }
            ),
        }
        r_gam = self._mgcv.gam
        null_trace: list[np.ndarray] = []
        null_omitted: list[bool] = []
        if trace_null_coef:
            namespace = self._ro.r["getNamespace"]("mgcv")
            private = self._ro.r["new.env"](parent=namespace)
            fit3 = self._utils.getFromNamespace("gam.fit3", "mgcv")
            formals = self._ro.r["formals"](fit3)
            default_null = formals[list(formals.names).index("null.coef")][0]
            r_name = self._ro.r["as.name"]
            if (
                not default_null[0].rsame(r_name("rep"))
                or float(_r_array(default_null[1])[0]) != 0.0
                or not default_null[2][0].rsame(r_name("ncol"))
                or not default_null[2][1].rsame(r_name("x"))
            ):
                raise RBridgeError("Pinned gam.fit3 null-coef default changed")

            def record_fit3(**arguments: Any) -> Any:
                null_omitted.append("null.coef" not in arguments)
                null_trace.append(
                    _r_array(
                        arguments.get(
                            "null.coef",
                            self._base.rep(
                                0.0, int(self._base.ncol(arguments["x"])[0])
                            ),
                        ),
                        dtype=np.float64,
                    )
                )
                return fit3(**arguments)

            fit_callback = make_r_callback(record_fit3)
            private["gam.fit3"] = fit_callback
            private["efsudr"] = clone_function(
                self._utils.getFromNamespace("efsudr", "mgcv"), environment=private
            )
            private["gam.outer"] = clone_function(
                self._utils.getFromNamespace("gam.outer", "mgcv"), environment=private
            )
            private["estimate.gam"] = clone_function(
                self._utils.getFromNamespace("estimate.gam", "mgcv"),
                environment=private,
            )
            r_gam = clone_function(
                self._utils.getFromNamespace("gam", "mgcv"), environment=private
            )
        try:
            model = r_gam(self._ro.Formula('y ~ s(x, bs="cr", k=6)'), **fit_args)
        finally:
            if trace_null_coef:
                private["gam.fit3"] = self._ro.NULL
                private["efsudr"] = self._ro.NULL
                private["gam.outer"] = self._ro.NULL
                private["estimate.gam"] = self._ro.NULL
        reference = self._extract_fit_results_rpy2(model)
        outer = model.rx2("outer.info")
        reference.update(
            outer_iterations=int(_r_array(outer.rx2("iter"))[0]),
            convergence=str(outer.rx2("conv")[0]),
            null_coef_trace=tuple(null_trace),
            null_coef_omitted=tuple(null_omitted),
        )
        if trace_null_coef:
            if not null_trace:
                raise RBridgeError("Pinned EFS produced no gam.fit3 null-coef trace")
            r_trace = rinterface.ListSexpVector(
                [self._to_r_vector(value) for value in null_trace]
            )
            reference["null_coef_trace_summary"] = np.asarray(
                [
                    int(self._base.length(r_trace)[0]),
                    int(self._base.length(r_trace[0])[0]),
                    float(
                        self._base.max(self._base.abs(self._base.unlist(r_trace)))[0]
                    ),
                ],
                dtype=np.float64,
            )
        return reference

    def efs_regular_gdi1_diagnostics(
        self,
        X: np.ndarray,
        y: np.ndarray,
        penalty: np.ndarray,
        start: np.ndarray,
        null_coef: np.ndarray,
        *,
        family: str = "gaussian",
        link: str = "log",
        weights: np.ndarray | None = None,
        offset: np.ndarray | None = None,
        scale: float = 1.0,
        tolerance: float = 1e-7,
    ) -> dict[str, Any]:
        """Trace one pinned regular-family ``gam.fit3`` final ``gdi1`` solve.

        This narrow layer oracle uses a private pinned R closure. It
        accepts one already materialized positive-semidefinite penalty so tests
        can distinguish the PIRLS stopping state, C_gdi1 candidate state, and
        returned feasible state without constructing a formula or outer loop.
        """
        self._require_pinned_efs_versions()
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        penalty = np.asarray(penalty, dtype=np.float64)
        start = np.asarray(start, dtype=np.float64)
        null_coef = np.asarray(null_coef, dtype=np.float64)
        if weights is None:
            weights = np.ones_like(y)
        if offset is None:
            offset = np.zeros_like(y)
        weights = np.asarray(weights, dtype=np.float64)
        offset = np.asarray(offset, dtype=np.float64)
        if (
            X.ndim != 2
            or y.shape != (X.shape[0],)
            or penalty.shape != (X.shape[1], X.shape[1])
            or start.shape != (X.shape[1],)
            or null_coef.shape != (X.shape[1],)
            or weights.shape != y.shape
            or offset.shape != y.shape
        ):
            raise ValueError("Regular gdi1 oracle inputs have incompatible shapes")
        if not all(
            np.all(np.isfinite(value))
            for value in (X, y, penalty, start, null_coef, weights, offset)
        ):
            raise ValueError("Regular gdi1 oracle requires finite array inputs")
        if np.any(weights <= 0.0) or not np.isfinite(scale) or scale <= 0.0:
            raise ValueError("Regular gdi1 oracle requires positive weights and scale")
        if not np.isfinite(tolerance) or tolerance <= 0.0:
            raise ValueError("Regular gdi1 oracle requires a positive tolerance")
        if (family, link) not in {("gaussian", "log"), ("poisson", "identity")}:
            raise ValueError(
                "Regular gdi1 oracle supports Gaussian/log or Poisson/identity"
            )
        np.testing.assert_allclose(penalty, penalty.T, rtol=0.0, atol=1e-14)
        if np.min(np.linalg.eigvalsh(penalty)) < -1e-12:
            raise ValueError(
                "Regular gdi1 oracle penalty must be positive semidefinite"
            )

        self._require_rpy2()
        from rpy2 import rinterface

        from tests.r_ast import find_call_paths, instrument_function

        r_x = self._to_r_matrix(X)
        r_y = self._to_r_vector(y)
        r_penalty = self._to_r_matrix(penalty)
        r_start = self._to_r_vector(start)
        r_null = self._to_r_vector(null_coef)
        r_weight = self._to_r_vector(weights)
        r_offset = self._to_r_vector(offset)
        r_add = self._ro.r["+"]
        r_divide = self._ro.r["/"]
        r_multiply = self._ro.r["%*%"]
        symmetric = r_divide(r_add(r_penalty, self._base.t(r_penalty)), 2)
        eigen = self._base.eigen(symmetric, symmetric=True)
        values = eigen.rx2("values")
        machine_epsilon = self._ro.r[".Machine"].rx2("double.eps")
        cutoff = self._ro.r["*"](
            self._base.max(values), self._ro.r["^"](machine_epsilon, 0.75)
        )
        keep = _r_array(self._ro.r[">"](values, cutoff), dtype=bool)
        if not np.any(keep):
            raise ValueError("regular gdi1 oracle requires a nonzero penalty")
        r_keep = self._ro.BoolVector(keep)
        bracket = self._ro.r["["]
        vectors = eigen.rx2("vectors")
        y_space = bracket(vectors, rinterface.MissingArg, r_keep, drop=False)
        z_space = bracket(
            vectors, rinterface.MissingArg, self._ro.BoolVector(~keep), drop=False
        )
        selected_values = bracket(eigen.rx2("values"), r_keep)
        root = self._base.sweep(y_space, 2, self._base.sqrt(selected_values), "*")
        eb = self._base.t(root)
        u1 = self._base.cbind(y_space, z_space)
        mp = int(self._base.ncol(z_space)[0])
        ur_s = rinterface.ListSexpVector([r_multiply(self._base.t(y_space), root)])
        r_family = (
            self._stats.gaussian(link="log")
            if (family, link) == ("gaussian", "log")
            else self._stats.poisson(link="identity")
        )
        for function in ("fix.family.link", "fix.family.var", "fix.family.ls"):
            r_family = self._call_internal(function, r_family)

        captured: dict[str, np.ndarray | float] = {}

        def copy(value: Any) -> np.ndarray:
            return _r_array(value, dtype=np.float64).ravel(order="F")

        def pre_gdi(
            transform: Any,
            current_start: Any,
            eta: Any,
            mu: Any,
            dev: Any,
            pdev: Any,
        ) -> None:
            captured["pre_beta"] = copy(
                self._base.drop(r_multiply(transform, current_start))
            )
            captured["pre_eta"] = copy(eta)
            captured["pre_mu"] = copy(mu)
            captured["pre_deviance"] = float(_r_array(dev).ravel()[0])
            captured["pre_pdev"] = float(_r_array(pdev).ravel()[0])

        def gdi_candidate(transform: Any, current: Any) -> None:
            current = self._ro.conversion.get_conversion().rpy2py(current)
            captured["gdi_beta"] = copy(
                self._base.drop(r_multiply(transform, current.rx2("beta")))
            )
            captured["gdi_penalty"] = float(_r_array(current.rx2("conv.tol"))[0])

        fit3 = self._utils.getFromNamespace("gam.fit3", "mgcv")
        for head, symbols, capture, callback, when in (
            (
                "<-",
                ("wdr", "dev.resids", "y", "mu", "weights"),
                ("T", "start", "eta", "mu", "dev", "pdev"),
                pre_gdi,
                "before",
            ),
            (
                "<-",
                ("coef", "oo", "beta"),
                ("T", "oo"),
                gdi_candidate,
                "before",
            ),
        ):
            paths = find_call_paths(fit3, head, required_symbols=symbols)
            if len(paths) != 1:
                raise RBridgeError("Pinned gam.fit3 regular gdi1 anchor changed")
            fit3 = instrument_function(
                fit3,
                path=paths[0],
                expected_head=head,
                capture_symbols=capture,
                callback=callback,
                when=when,
            )
        fit = fit3(
            r_x,
            r_y,
            sp=0,
            Eb=eb,
            UrS=ur_s,
            weights=r_weight,
            start=r_start,
            offset=r_offset,
            U1=u1,
            Mp=mp,
            family=r_family,
            control=self._mgcv.gam_control(maxit=100, epsilon=float(tolerance)),
            intercept=True,
            deriv=0,
            gamma=1,
            scale=float(scale),
            scoreType="EFS",
            **{"null.coef": r_null, "n.true": len(y)},
        )
        observed = fit.rx2("working.weights")
        fisher = fit.rx2("weights")
        weighted = self._ro.r["*"]
        captured.update(
            reported_scale=float(_r_array(fit.rx2("scale.est"))[0]),
            score=float(_r_array(fit.rx2("REML"))[0]),
            selected_beta=copy(fit.rx2("coefficients")),
            selected_eta=copy(fit.rx2("linear.predictors")),
            selected_mu=copy(fit.rx2("fitted.values")),
            observed_weight=copy(observed),
            fisher_weight=copy(fisher),
            XtWX=_r_array(
                self._base.crossprod(r_x, weighted(observed, r_x)), dtype=np.float64
            ),
            XtWX_fisher=_r_array(
                self._base.crossprod(r_x, weighted(fisher, r_x)), dtype=np.float64
            ),
            source_commit=_PINNED_MGCV_SOURCE_COMMIT,
        )
        return captured

    @staticmethod
    def _require_pinned_efs_versions() -> None:
        ok, reason = RBridge.check_versions()
        if not ok:
            raise RBridgeError(f"Pinned EFS oracle unavailable: {reason}")

    def _public_exact_reml_gam_arguments(
        self,
        data: pd.DataFrame,
        weight: np.ndarray,
        offset: np.ndarray,
        family_name: str,
        link: str,
        *,
        smooth: bool,
        estimated_theta: bool,
        initial_smoothing: float | None,
    ) -> tuple[Any, dict[str, Any]]:
        """Build the pinned public exact-REML ``gam`` formula and arguments."""
        if family_name not in {"gaussian", "gamma", "poisson", "binomial", "nb"}:
            raise ValueError(f"Unsupported public exact R family: {family_name}")
        if len(data) != len(weight) or len(data) != len(offset):
            raise ValueError("Public exact R rows, weights and offsets differ.")
        if initial_smoothing is not None and (
            not smooth or not np.isfinite(initial_smoothing) or initial_smoothing <= 0.0
        ):
            raise ValueError("Initial R smoothing must be positive for a smooth fit.")
        ro = self._ro
        source_link = "1/mu^2" if link == "inverse_squared" else link
        if family_name == "nb":
            family = self._mgcv.nb(
                theta=-2.7 if estimated_theta else 2.7, link=source_link
            )
        else:
            constructors = {
                "gaussian": self._stats.gaussian,
                "gamma": self._stats.Gamma,
                "poisson": self._stats.poisson,
                "binomial": self._stats.binomial,
            }
            family = constructors[family_name](link=source_link)
        formula = (
            "y ~ x"
            if not smooth
            else 'y ~ s(x, bs="cr", k=8)'
            if family_name == "nb"
            else 'y ~ s(x, bs="cr", k=6)'
        )
        control = self._mgcv.gam_control(
            epsilon=1e-11,
            maxit=200,
            newton=ro.ListVector(
                {"conv.tol": 1e-10, "maxNstep": 5.0, "maxSstep": 2.0, "maxHalf": 30}
            ),
        )
        startup = (
            {}
            if initial_smoothing is None
            else {
                "in.out": ro.ListVector(
                    {
                        "sp": ro.FloatVector([initial_smoothing]),
                        "scale": ro.FloatVector([1.0]),
                    }
                )
            }
        )
        return ro.Formula(formula), {
            "data": self._to_r_dataframe(data),
            "weights": self._to_r_vector(weight),
            "offset": self._to_r_vector(offset),
            "family": family,
            "method": "REML",
            "optimizer": ro.StrVector(["outer", "newton"]),
            "control": control,
            **startup,
        }

    def public_exact_reml_fit(
        self,
        data: pd.DataFrame,
        weight: np.ndarray,
        offset: np.ndarray,
        family_name: str,
        link: str,
        *,
        smooth: bool = True,
        estimated_theta: bool = False,
        initial_smoothing: float | None = None,
    ) -> dict[str, Any]:
        """Fit the public cubic or linear REML model in live pinned mgcv.

        Both regular and NB release cells use the same source-matched Newton
        controls and uncertainty. ``initial_smoothing`` maps to the documented
        ``gam(in.out=list(sp=..., scale=1))`` initial state in ``estimate.gam``;
        it leaves the rows, model, criterion and convergence controls intact.
        """
        self._require_rpy2()
        formula, arguments = self._public_exact_reml_gam_arguments(
            data,
            weight,
            offset,
            family_name,
            link,
            smooth=smooth,
            estimated_theta=estimated_theta,
            initial_smoothing=initial_smoothing,
        )
        ro = self._ro
        r_data = arguments["data"]
        fit = self._mgcv.gam(formula, **arguments)
        score_scale = fit.rx2("reml.scale")
        if score_scale is ro.NULL:
            score_scale = fit.rx2("sig2")
        link_prediction = self._stats.predict(
            fit, newdata=r_data, type="link", **{"se.fit": True}
        )
        outer = fit.rx2("outer.info")
        outer_converged = (
            outer is not ro.NULL
            and outer.rx2("conv") is not ro.NULL
            and str(outer.rx2("conv")[0]) == "full convergence"
        )
        outer_status = (
            str(outer.rx2("conv")[0])
            if outer is not ro.NULL and outer.rx2("conv") is not ro.NULL
            else None
        )
        return {
            "theta": (
                float(fit.rx2("family").rx2("getTheta")(True)[0])
                if family_name == "nb"
                else np.nan
            ),
            "score": float(fit.rx2("gcv.ubre")[0]),
            "scale": float(fit.rx2("sig2")[0]),
            "score_scale": float(score_scale[0]),
            "deviance": float(fit.rx2("deviance")[0]),
            "smoothing": float(fit.rx2("sp")[0]) if smooth else np.nan,
            "edf": float(self._base.sum(fit.rx2("edf"))[0]),
            "inner_converged": bool(fit.rx2("converged")[0]),
            "outer_converged": outer_converged,
            "outer_status": outer_status,
            "coefficients": _r_array(fit.rx2("coefficients")),
            "fitted_values": _r_array(fit.rx2("fitted.values")),
            "link_se": _r_array(link_prediction.rx2("se.fit")),
        }

    def public_exact_reml_derivatives(
        self,
        data: pd.DataFrame,
        weight: np.ndarray,
        offset: np.ndarray,
        family_name: str,
        link: str,
    ) -> dict[str, Any]:
        """Evaluate pinned REML derivatives at R's returned estimated endpoint.

        The public smooth fit of :meth:`public_exact_reml_fit` runs through
        private clones of ``gam`` -> ``estimate.gam`` -> ``gam.outer`` ->
        ``newton`` whose ``gam.fit3`` records newton's first call: the prepared
        model matrix, penalty roots, null coefficients, family, weights, offset
        and newton's tightened control. For one smoothing parameter and an
        unknown scale ``estimate.gam`` appends ``log(scale)`` to newton's
        ``lsp`` and leaves ``L`` NULL, so ``gam.fit3`` receives
        ``sp = c(log(sp), log(scale))``. One fresh ``gam.fit3`` (``deriv=2``)
        at R's returned ``c(log(sp), log(reml.scale))`` then returns the outer
        gradient ``REML1`` and Hessian ``REML2`` over both free log parameters,
        independent of newton's path or of the trial object it may describe
        after a step failure.
        """
        self._require_rpy2()
        from tests.r_ast import clone_function, make_r_callback

        if family_name not in {"gaussian", "gamma"}:
            raise ValueError(
                "endpoint derivatives need an unknown-scale regular family"
            )
        formula, arguments = self._public_exact_reml_gam_arguments(
            data,
            weight,
            offset,
            family_name,
            link,
            smooth=True,
            estimated_theta=False,
            initial_smoothing=None,
        )
        ro = self._ro
        fit3 = self._utils.getFromNamespace("gam.fit3", "mgcv")
        recorded: list[dict[str, Any]] = []

        def record_fit3(**fit_arguments: Any) -> Any:
            if not recorded:
                recorded.append(dict(fit_arguments))
            return fit3(**fit_arguments)

        namespace = ro.r["getNamespace"]("mgcv")
        private = ro.r["new.env"](parent=namespace)
        callback = make_r_callback(record_fit3)
        names = ("newton", "gam.outer", "estimate.gam")
        try:
            private["gam.fit3"] = callback
            for name in names:
                private[name] = clone_function(
                    self._utils.getFromNamespace(name, "mgcv"), environment=private
                )
            r_gam = clone_function(
                self._utils.getFromNamespace("gam", "mgcv"), environment=private
            )
            fit = r_gam(formula, **arguments)
        finally:
            for name in ("gam.fit3", *names):
                private[name] = ro.NULL
        outer = fit.rx2("outer.info")
        if not recorded or outer is ro.NULL:
            raise RBridgeError("Pinned public fit did not call newton's gam.fit3")
        first = recorded[0]
        recorded_sp = _r_array(first["sp"], dtype=np.float64).ravel()
        if recorded_sp.shape != (2,):
            raise RBridgeError("newton's gam.fit3 sp is not c(log(sp), log(scale))")
        smoothing = _r_array(fit.rx2("sp"), dtype=np.float64).ravel()
        if smoothing.shape != (1,):
            raise RBridgeError(
                "public exact fit did not return one smoothing parameter"
            )
        returned_scale = float(fit.rx2("reml.scale")[0])
        log_parameters = np.array([np.log(smoothing[0]), np.log(returned_scale)])
        if not np.all(np.isfinite(log_parameters)):
            raise RBridgeError("public exact fit returned a non-finite endpoint")
        evaluated = fit3(
            **{**first, "sp": self._to_r_vector(log_parameters), "deriv": 2}
        )
        gradient = _r_array(evaluated.rx2("REML1"), dtype=np.float64).ravel()
        hessian = _r_array(evaluated.rx2("REML2"), dtype=np.float64)
        if gradient.shape != (2,) or hessian.shape != (2, 2):
            raise RBridgeError("gam.fit3 did not return two-parameter REML derivatives")
        return {
            "log_parameters": log_parameters,
            "smoothing": float(smoothing[0]),
            "returned_scale": returned_scale,
            "reml_scale": float(evaluated.rx2("reml.scale")[0]),
            "returned_score": float(fit.rx2("gcv.ubre")[0]),
            "score": float(evaluated.rx2("REML")[0]),
            "gradient": gradient.copy(),
            "hessian": hessian.copy(),
            "outer_status": str(outer.rx2("conv")[0]),
            "outer_gradient": _r_array(outer.rx2("grad"), dtype=np.float64),
            "outer_hessian": _r_array(outer.rx2("hess"), dtype=np.float64),
            "inner_converged": bool(evaluated.rx2("converged")[0]),
        }

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
            return float(_r_array(self._base.sum(square(value, 2)))[0])

        return {
            "d": _r_array(reparam.rx2("det1"), dtype=np.float64),
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
            proposals.append(_r_array(packed, dtype=np.float64))
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
            "accepted_sp": _r_array(output.rx2("sp"), dtype=np.float64),
            "final_score": float(_r_array(output.rx2("REML"))[0]),
            "score_history": _r_array(outer.rx2("score.hist"), dtype=np.float64),
            "iter": int(_r_array(output.rx2("iter"))[0]),
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
            outer_iterations=int(_r_array(outer.rx2("iter"))[0]),
            score_history=_r_array(outer.rx2("score.hist"), dtype=np.float64).ravel(
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
            else int(_r_array(theta_count)[0])
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
                    _r_array(self._ro.r["/"](null_fit.rx2("null.scale"), 10))[0]
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
            return float(_r_array(value, dtype=np.float64).ravel()[0])

        def copied(value: Any) -> np.ndarray:
            return _r_array(value, dtype=np.float64).ravel(order="F")

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
            if int(_r_array(iteration)[0]) != 1:
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
                    int(_r_array(fit_family.rx2("n.theta"))[0])
                    if bool(self._base.inherits(args["family"], "extended.family")[0])
                    else 0
                )
                y_space = self._ro.r["["](
                    args["U1"],
                    rinterface.MissingArg,
                    self._base.seq_len(
                        int(self._base.ncol(args["U1"])[0])
                        - int(_r_array(args["Mp"])[0])
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
        coefficients = _r_array(r_model.rx2("coefficients"), dtype=np.float64)
        fitted_values = _r_array(r_model.rx2("fitted.values"), dtype=np.float64)
        smoothing_params = _r_array(r_model.rx2("sp"), dtype=np.float64)
        deviance = float(_r_array(r_model.rx2("deviance"))[0])
        scale = float(_r_array(r_model.rx2("scale"))[0])
        vp_r = r_model.rx2("Vp")
        n_coef = len(coefficients)
        vp = _r_array(vp_r, dtype=np.float64).reshape((n_coef, n_coef))
        reml_score = float(_r_array(r_model.rx2("gcv.ubre"))[0])

        # reml.scale is the scale used in the REML criterion (jointly
        # optimized), which differs from model$scale (Fletcher estimate).
        reml_scale_r = r_model.rx2("reml.scale")
        reml_scale = (
            float(_r_array(reml_scale_r)[0]) if reml_scale_r is not None else scale
        )

        r_summary = self._base.summary(r_model)
        edf = _r_array(r_summary.rx2("edf"), dtype=np.float64)
        edf_total = float(np.sum(_r_array(r_model.rx2("edf"), dtype=np.float64)))
        null_deviance = float(_r_array(r_model.rx2("null.deviance"))[0])
        outer = r_model.rx2("outer.info")
        outer_status = (
            str(outer.rx2("conv")[0])
            if outer is not self._ro.NULL and outer.rx2("conv") is not self._ro.NULL
            else None
        )

        # Extract theta for extended families (e.g. NB)
        theta = None
        try:
            family_obj = r_model.rx2("family")
            get_theta = family_obj.rx2("getTheta")
            if get_theta is not None:
                theta = float(_r_array(get_theta(True))[0])
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
            "inner_converged": bool(r_model.rx2("converged")[0]),
            "outer_converged": outer_status == "full convergence",
            "outer_status": outer_status,
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

        X_full = _r_array(ro.r["model.matrix"](r_model), dtype=np.float64)

        basis_matrices = []
        penalty_matrices = []
        for i in range(n_smooths):
            sm = smooth_list[i]
            first_col = int(_r_array(sm.rx2("first.para"))[0]) - 1
            last_col = int(_r_array(sm.rx2("last.para"))[0])
            basis_matrices.append(X_full[:, first_col:last_col])

            S_list = sm.rx2("S")
            penalties_for_smooth = []
            for j in range(len(S_list)):
                S_flat = _r_array(S_list[j], dtype=np.float64)
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

    def source_slanczos(
        self, matrix: np.ndarray, k: int, tol: float
    ) -> dict[str, np.ndarray | int]:
        """Pinned ``slanczos(A, k, kl=-1, tol, nt=1)``: tprs_setup's Rlanczos call."""
        self._require_rpy2()
        A = np.asarray(matrix, dtype=np.float64)
        if A.ndim != 2 or A.shape[0] != A.shape[1] or not 1 <= k <= A.shape[0]:
            raise ValueError("slanczos requires a square matrix and 1 <= k <= n.")
        result = self._mgcv.slanczos(
            self._to_r_matrix(A),
            k=self._ro.IntVector([k]),
            kl=self._ro.IntVector([-1]),
            tol=self._ro.FloatVector([tol]),
            nt=self._ro.IntVector([1]),
        )
        return {
            "values": _r_array(result.rx2("values"), dtype=np.float64),
            "vectors": _r_array(result.rx2("vectors"), dtype=np.float64),
            "iterations": int(result.rx2("iter")[0]),
        }

    def source_smooth_basis(
        self,
        smooth_expr: str,
        data: pd.DataFrame,
        new_data: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Pinned unconstrained ``smoothCon`` basis and ``PredictMat`` rows.

        Returns ``smoothCon(..., absorb.cons=FALSE)``'s scaled ``X`` and
        ``S`` and, when ``new_data`` is given, ``PredictMat`` on it, each
        copied into NumPy-owned memory before any further R allocation.
        """
        smooth = self._smooth_objects_rpy2(smooth_expr, data, False)[0]
        result: dict[str, Any] = {
            "X": _r_array(smooth.rx2("X"), dtype=np.float64),
            "S": [_r_array(penalty, dtype=np.float64) for penalty in smooth.rx2("S")],
        }
        if new_data is not None:
            result["prediction"] = _r_array(
                self._mgcv.PredictMat(smooth, self._to_r_dataframe(new_data)),
                dtype=np.float64,
            )
        return result

    def linked_lapack(self) -> str:
        """Resolved path of the LAPACK library R links (``La_library()``)."""
        self._require_rpy2()
        return os.path.realpath(str(self._base.La_library()[0]))

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

    def smooth_predict_matrix(
        self, smooth_expr: str, data: pd.DataFrame, new_data: pd.DataFrame
    ) -> np.ndarray:
        """Evaluate installed ``PredictMat`` for one unabsorbed smooth."""
        smooth = self._smooth_objects_rpy2(smooth_expr, data, False)[0]
        prediction = self._call_internal(
            "PredictMat", smooth, self._to_r_dataframe(new_data)
        )
        return _r_array(prediction, dtype=np.float64)

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
            return np.zeros(shape) if value is self._ro.NULL else _r_array(value)

        ranks = _r_array(smooth.rx2("rank"), dtype=np.float64).ravel()
        knt = optional_array("knt", (0, 0))
        gp_defn = optional_array("gp.defn", (0,))
        E = (
            _r_array(
                self._call_internal(
                    "gpE", smooth.rx2("knt"), smooth.rx2("knt"), smooth.rx2("gp.defn")
                ),
                dtype=np.float64,
            )
            if smooth.rx2("knt") is not self._ro.NULL
            else np.zeros((0, 0))
        )
        return {
            "X": _r_array(smooth.rx2("X"), dtype=np.float64),
            "S": [_r_array(penalty, dtype=np.float64) for penalty in smooth.rx2("S")],
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
                    "X": _r_array(smooth.rx2("X"), dtype=np.float64),
                    "S": [
                        _r_array(penalty, dtype=np.float64)
                        for penalty in smooth.rx2("S")
                    ],
                    "rank": int(smooth.rx2("rank")[0]),
                    "rank_vector": _r_array(smooth.rx2("rank"), dtype=int).ravel(),
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
            result["predictions"] = _r_array(pred.rx2("fit"), dtype=np.float64)
            result["se"] = _r_array(pred.rx2("se.fit"), dtype=np.float64)
        else:
            result["predictions"] = _r_array(pred, dtype=np.float64)

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
            p_arr = _r_array(p_table_r, dtype=np.float64)
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
            s_arr = _r_array(s_table_r, dtype=np.float64)
            n_smooths = int(_r_array(r_summary.rx2("m"))[0])
            if n_smooths > 0:
                result["s_table"] = s_arr.reshape(n_smooths, -1)
            else:
                result["s_table"] = None
        else:
            result["s_table"] = None

        # R-squared
        r_sq = r_summary.rx2("r.sq")
        result["r_sq"] = float(_r_array(r_sq)[0]) if r_sq is not None else None

        result["dev_explained"] = float(_r_array(r_summary.rx2("dev.expl"))[0])
        result["scale"] = float(_r_array(r_summary.rx2("scale"))[0])
        result["residual_df"] = float(_r_array(r_summary.rx2("residual.df"))[0])
        result["n"] = int(_r_array(r_summary.rx2("n"))[0])

        edf_r = r_summary.rx2("edf")
        if edf_r is not None:
            result["edf"] = _r_array(edf_r, dtype=np.float64)
        else:
            result["edf"] = np.array([])

        sp_crit = r_summary.rx2("sp.criterion")
        if sp_crit is not None:
            result["sp_criterion"] = float(_r_array(sp_crit)[0])
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

    def _to_r_discrete_dataframe(
        self,
        data: pd.DataFrame,
        factor_names: tuple[str, ...],
        *,
        preserve_order: bool,
    ) -> Any:
        """Transfer CSV-equivalent columns and declare source factor levels."""
        raw = data.copy()
        for name in raw.columns:
            if isinstance(raw[name].dtype, pd.CategoricalDtype):
                raw[name] = raw[name].astype(object)
        frame = self._to_r_dataframe(raw)
        for name in factor_names:
            series = data[name]
            levels = (
                list(series.dtype.categories)
                if isinstance(series.dtype, pd.CategoricalDtype)
                else list(pd.unique(series))
            )
            constructor = (
                self._base.ordered
                if preserve_order
                and isinstance(series.dtype, pd.CategoricalDtype)
                and series.dtype.ordered
                else self._base.factor
            )
            frame = self._ro.baseenv["[[<-"](
                frame,
                name,
                value=constructor(
                    frame.rx2(name),
                    levels=self._ro.StrVector([str(level) for level in levels]),
                ),
            )
        return frame

    def _r_discrete_grid(self, m: int | tuple[int, ...] | None, *, default: Any) -> Any:
        """Validate and transfer mgcv's scalar or per-term grid control."""
        if isinstance(m, tuple):
            if not m or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 2
                for value in m
            ):
                raise ValueError("m tuple entries must be integers of at least 2")
            return self._to_r_vector(np.asarray(m, dtype=np.int32))
        if m is None:
            return default
        if isinstance(m, bool) or not isinstance(m, int) or m < 2:
            raise ValueError("m must be an integer of at least 2")
        return int(m)

    def compress_dataframe(
        self,
        data: pd.DataFrame,
        *,
        factor_names: tuple[str, ...] = (),
        m: int | None = None,
        seed: int = 8547,
    ) -> CompressedDataFrame:
        """Call pinned ``compress.df`` with an isolated, restored R RNG state."""
        self._require_rpy2()
        if data.empty or data.shape[1] == 0:
            raise ValueError("compress_dataframe requires a non-empty data frame")
        if len(factor_names) != len(set(factor_names)) or not set(factor_names) <= set(
            data.columns
        ):
            raise ValueError("factor_names must be distinct data-frame columns")
        if m is not None and (isinstance(m, bool) or not isinstance(m, int) or m < 2):
            raise ValueError("m must be an integer of at least 2")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("seed must be an integer")
        frame = self._to_r_discrete_dataframe(data, factor_names, preserve_order=False)
        with self._preserved_r_seed(1907):
            expected_after = self._stats.runif(8)
            self._ro.baseenv["set.seed"](1907)
            old_seed = self._call_internal("temp.seed", seed)
            try:
                compressed = self._call_internal(
                    "compress.df", frame, m=self._ro.NULL if m is None else m
                )
            finally:
                self._call_internal("temp.seed", old_seed)
            observed_after = self._stats.runif(8)
            if not bool(
                self._ro.baseenv["identical"](expected_after, observed_after)[0]
            ):
                raise RBridgeError("compress RNG state not restored")
        selector = self._base.attr(compressed, "index")
        if bool(self._ro.baseenv["is.matrix"](selector)[0]):
            raise RBridgeError("matrix-valued terms are not supported by this oracle")
        columns = {}
        for name in compressed.names:
            values = compressed.rx2(name)
            if bool(self._ro.baseenv["is.factor"](values)[0]):
                values = self._base.as_character(values)
            columns[str(name)] = _r_array(values)
        return CompressedDataFrame(
            pd.DataFrame(columns), _r_array(selector, dtype=np.int32) - 1
        )

    def discrete_model_frame(
        self,
        formula: str,
        data: pd.DataFrame,
        *,
        factor_names: tuple[str, ...] = (),
        m: int | tuple[int, ...] | None = None,
    ) -> DiscreteModelFrame:
        """Call pinned ``discrete.mf`` and translate its one-based metadata."""
        self._require_rpy2()
        if len(factor_names) != len(set(factor_names)) or not set(factor_names) <= set(
            data.columns
        ):
            raise ValueError("factor_names must be distinct data-frame columns")
        grid = self._r_discrete_grid(m, default=self._ro.NULL)
        frame = self._to_r_discrete_dataframe(data, factor_names, preserve_order=True)
        interpreted = self._call_internal("interpret.gam", self._ro.Formula(formula))
        model_frame = self._stats.model_frame(
            interpreted.rx2("fake.formula"),
            frame,
            **{"drop.unused.levels": False},
        )
        parametric_frame = self._stats.model_frame(
            interpreted.rx2("pf"), frame, **{"drop.unused.levels": False}
        )
        with self._preserved_r_seed(1907):
            expected_after = self._stats.runif(8)
            self._ro.baseenv["set.seed"](1907)
            result = self._call_internal(
                "discrete.mf",
                interpreted,
                model_frame,
                self._base.names(parametric_frame),
                m=grid,
                full=True,
            )
            observed_after = self._stats.runif(8)
            if not bool(
                self._ro.baseenv["identical"](expected_after, observed_after)[0]
            ):
                raise RBridgeError("discrete.mf RNG not restored")
        returned_frame = result.rx2("mf")
        columns = {}
        for name in returned_frame.names:
            values = returned_frame.rx2(name)
            if bool(self._ro.baseenv["is.factor"](values)[0]):
                values = self._base.as_character(values)
            columns[str(name)] = _r_array(values)
        unique = result.rx2("nr")
        spans = result.rx2("ks")
        span_names = tuple(self._base.rownames(spans))
        span_values = _r_array(spans, dtype=np.int64)
        return DiscreteModelFrame(
            pd.DataFrame(columns),
            _r_array(result.rx2("k"), dtype=np.int32) - 1,
            dict(zip(map(str, unique.names), map(int, unique), strict=True)),
            {
                str(name): (int(bounds[0]) - 1, int(bounds[1]) - 1)
                for name, bounds in zip(span_names, span_values, strict=True)
            },
        )

    def bam_discrete_setup(
        self,
        formula: str,
        data: pd.DataFrame,
        *,
        factor_names: tuple[str, ...] = (),
        m: int | tuple[int, ...] | None = None,
    ) -> BamDiscreteSetup:
        """Return pinned ``bam(fit=FALSE)`` compact metadata and design."""
        self._require_rpy2()
        from rpy2 import rinterface

        if len(factor_names) != len(set(factor_names)) or not set(factor_names) <= set(
            data.columns
        ):
            raise ValueError("factor_names must be distinct data-frame columns")
        grid = self._r_discrete_grid(m, default=True)
        frame = self._to_r_discrete_dataframe(data, factor_names, preserve_order=True)
        with self._preserved_r_seed(1907):
            expected_after = self._stats.runif(8)
            self._ro.baseenv["set.seed"](1907)
            setup = self._mgcv.bam(
                self._ro.Formula(formula), data=frame, discrete=grid, fit=False
            )
            observed_after = self._stats.runif(8)
            if not bool(
                self._ro.baseenv["identical"](expected_after, observed_after)[0]
            ):
                raise RBridgeError("bam setup RNG state not restored")
        tables_r = setup.rx2("Xd")
        kd_r, ks_r = setup.rx2("kd"), setup.rx2("ks")
        ts_r, dt_r, qc_r = (setup.rx2(name) for name in ("ts", "dt", "qc"))
        if any(len(values) == 0 for values in (tables_r, kd_r, ks_r, ts_r, dt_r, qc_r)):
            raise RBridgeError(
                "bam did not return the required compact discrete operator metadata"
            )
        tables = tuple(_r_array(table, dtype=np.float64) for table in tables_r)
        vectors = tuple(
            _r_array(vector, dtype=np.float64).ravel(order="F")
            if vector.typeof != rinterface.RTYPES.NILSXP and len(vector)
            else np.empty(0, dtype=np.float64)
            for vector in setup.rx2("v")
        )
        ts = _r_array(ts_r, dtype=np.int64)
        dt = _r_array(dt_r, dtype=np.int64)
        qc = _r_array(qc_r, dtype=np.int32)
        for term in np.flatnonzero(qc > 0):
            margins = slice(int(ts[term]) - 1, int(ts[term]) + int(dt[term]) - 1)
            raw_width = int(np.prod([table.shape[1] for table in tables[margins]]))
            if len(vectors[term]) != raw_width:
                raise RBridgeError(
                    f"rank-one constraint {term + 1} has {len(vectors[term])} "
                    f"Householder entries for raw width {raw_width}"
                )
        expanded = self._call_internal(
            "Xbd",
            tables_r,
            self._base.diag(self._base.ncol(setup.rx2("X"))),
            kd_r,
            ks_r,
            ts_r,
            dt_r,
            setup.rx2("v"),
            qc_r,
            drop=setup.rx2("drop"),
        )
        drop_r = setup.rx2("drop")
        drop = (
            None
            if drop_r.typeof == rinterface.RTYPES.NILSXP
            else _r_array(drop_r, dtype=np.int32) - 1
        )
        layout = DiscreteOperatorLayout(
            tables,
            _r_array(kd_r, dtype=np.int32) - 1,
            _r_array(ks_r, dtype=np.int32) - 1,
            tuple(ts - 1),
            tuple(dt),
            vectors,
            qc,
            drop,
        )
        return BamDiscreteSetup(
            layout,
            _r_array(expanded, dtype=np.float64),
            tuple(_r_array(penalty, dtype=np.float64) for penalty in setup.rx2("S")),
            tuple(map(str, self._base.names(ts_r))),
        )

    def bam_discretization(
        self,
        formula: str,
        data: pd.DataFrame,
        *,
        factor_names: tuple[str, ...] = (),
        m: int | tuple[int, ...] | None = None,
    ) -> BamDiscretization:
        """Return pinned ``bam(discrete=)``'s grids and term-to-margin map.

        ``bam(fit=FALSE)`` retains the (``tero``-ordered) ``gp``, the model
        frame and ``dinfo$pmf.names`` it passed to ``discrete.mf``.  That
        call seeds its own private stream (``temp.seed(8547)``), so replaying
        it on the retained inputs reproduces bam's discretization exactly;
        the replay is checked against bam's own ``kd`` index before use.
        """
        self._require_rpy2()
        from rpy2 import rinterface

        if len(factor_names) != len(set(factor_names)) or not set(factor_names) <= set(
            data.columns
        ):
            raise ValueError("factor_names must be distinct data-frame columns")
        grid = self._r_discrete_grid(m, default=True)
        frame = self._to_r_discrete_dataframe(data, factor_names, preserve_order=True)
        with self._preserved_r_seed(1907):
            expected_after = self._stats.runif(8)
            self._ro.baseenv["set.seed"](1907)
            setup = self._mgcv.bam(
                self._ro.Formula(formula), data=frame, discrete=grid, fit=False
            )
            # bam passes NULL for discrete=TRUE and the numeric grid otherwise.
            discrete = self._call_internal(
                "discrete.mf",
                setup.rx2("gp"),
                setup.rx2("mf"),
                setup.rx2("dinfo").rx2("pmf.names"),
                m=self._ro.NULL if m is None else grid,
                full=True,
            )
            observed_after = self._stats.runif(8)
            if not bool(
                self._ro.baseenv["identical"](expected_after, observed_after)[0]
            ):
                raise RBridgeError("bam discretization RNG state not restored")
        bam_index = _r_array(setup.rx2("kd"), dtype=np.int64)
        replay_index = _r_array(discrete.rx2("k"), dtype=np.int64)
        if bam_index.ndim == 1:
            bam_index = bam_index[:, None]
        if not np.array_equal(replay_index[:, : bam_index.shape[1]], bam_index):
            raise RBridgeError("replayed discrete.mf differs from bam's own index")
        unique = discrete.rx2("nr")
        counts = dict(zip(map(str, unique.names), map(int, unique), strict=True))
        span_names = tuple(map(str, self._base.rownames(discrete.rx2("ks"))))
        span_values = _r_array(discrete.rx2("ks"), dtype=np.int64)
        spans = {
            name: (int(bounds[0]) - 1, int(bounds[1]) - 1)
            for name, bounds in zip(span_names, span_values, strict=True)
        }
        padded = discrete.rx2("mf")
        model_frame = setup.rx2("mf")
        variables = {}
        for name in map(str, padded.names):
            values = padded.rx2(name)
            factor = bool(self._ro.baseenv["is.factor"](values)[0])
            representatives = _r_array(
                self._base.as_character(values) if factor else values
            )[: counts[name]]
            data_range = None
            if not factor:
                bounds = _r_array(
                    self._base.range(model_frame.rx2(name)), dtype=np.float64
                )
                data_range = (float(bounds[0]), float(bounds[1]))
            variables[name] = BamDiscretizedVariable(
                name,
                counts[name],
                spans[name],
                np.array(representatives, copy=True),
                data_range,
            )
        span_variables: dict[int, tuple[str, ...]] = {}
        for name in span_names:
            start = spans[name][0]
            span_variables[start] = (*span_variables.get(start, ()), name)
        margin_spans = _r_array(setup.rx2("ks"), dtype=np.int64)
        ts_r = setup.rx2("ts")
        dt = _r_array(setup.rx2("dt"), dtype=np.int64)
        term_margins = {}
        for label, first, count in zip(
            map(str, self._base.names(ts_r)),
            _r_array(ts_r, dtype=np.int64) - 1,
            dt,
            strict=True,
        ):
            term_margins[label] = tuple(
                span_variables[int(margin_spans[row, 0]) - 1]
                for row in range(int(first), int(first + count))
            )
        tables = setup.rx2("Xd")
        if tables.typeof == rinterface.RTYPES.NILSXP:
            raise RBridgeError("bam did not return discrete tables")
        return BamDiscretization(
            variables,
            term_margins,
            tuple(int(self._base.nrow(table)[0]) for table in tables),
        )

    def fit_bam_discrete_fixed_sp(
        self,
        formula: str,
        data: pd.DataFrame,
        sp: np.ndarray,
        *,
        factor_names: tuple[str, ...] = (),
        m: int | tuple[int, ...] | None = None,
        weights: np.ndarray | None = None,
        offset: np.ndarray | None = None,
        newdata: pd.DataFrame | None = None,
        discrete_prediction: bool = False,
        family: str = "gaussian",
        link: str | None = None,
    ) -> BamDiscreteFit:
        """Fit pinned discrete ``bam`` at caller-owned smoothing parameters."""
        if sp is None:
            raise ValueError("sp must contain positive finite values")
        return self.fit_bam_discrete(
            formula,
            data,
            family=family,
            link=link,
            sp=sp,
            factor_names=factor_names,
            m=m,
            weights=weights,
            offset=offset,
            newdata=newdata,
            discrete_prediction=discrete_prediction,
        )

    def bam_discrete_prefit(
        self,
        formula: str,
        data: pd.DataFrame,
        *,
        family: str = "gaussian",
        link: str | None = None,
        factor_names: tuple[str, ...] = (),
        m: int | tuple[int, ...] | None = 30,
        weights: np.ndarray | None = None,
        offset: np.ndarray | None = None,
    ) -> BamDiscretePrefit:
        """Pinned ``bam(fit=FALSE)`` with a fit's own ``weights=``/``offset=``.

        Unlike :meth:`bam_discrete_setup`, which passes neither, the setup is
        made with the arguments :meth:`fit_bam_discrete` passes (``None``
        omits an argument; an array, all ones included, supplies it), so
        ``pmf`` carries ``(weights)``/``(offset)`` exactly as in the fit.
        ``discrete.mf`` is then replayed on the setup's retained ``gp``,
        ``mf`` and ``dinfo$pmf.names`` (it seeds its own private stream) and
        checked against bam's own ``kd``, to expose the padded frame.
        """
        _, arguments = self._bam_discrete_arguments(
            data,
            family=family,
            link=link,
            sp=None,
            initial_sp=None,
            initial_scale=None,
            scale=None,
            factor_names=factor_names,
            m=m,
            weights=weights,
            offset=offset,
            max_iterations=200,
            epsilon=1e-7,
        )
        with self._preserved_r_seed(1907):
            expected_after = self._stats.runif(8)
            self._ro.baseenv["set.seed"](1907)
            setup = self._mgcv.bam(self._ro.Formula(formula), fit=False, **arguments)
            replay = self._call_internal(
                "discrete.mf",
                setup.rx2("gp"),
                setup.rx2("mf"),
                setup.rx2("dinfo").rx2("pmf.names"),
                m=self._ro.NULL
                if m is None
                else self._r_discrete_grid(m, default=True),
                full=True,
            )
            observed_after = self._stats.runif(8)
            if not bool(
                self._ro.baseenv["identical"](expected_after, observed_after)[0]
            ):
                raise RBridgeError("bam prefit RNG state not restored")
        bam_index = _r_array(setup.rx2("kd"), dtype=np.int64)
        replay_index = _r_array(replay.rx2("k"), dtype=np.int64)
        if bam_index.ndim == 1:
            bam_index = bam_index[:, None]
        if not np.array_equal(replay_index[:, : bam_index.shape[1]], bam_index):
            raise RBridgeError("replayed discrete.mf differs from bam's own index")
        expanded = self._call_internal(
            "Xbd",
            setup.rx2("Xd"),
            self._base.diag(self._base.ncol(setup.rx2("X"))),
            setup.rx2("kd"),
            setup.rx2("ks"),
            setup.rx2("ts"),
            setup.rx2("dt"),
            setup.rx2("v"),
            setup.rx2("qc"),
            drop=setup.rx2("drop"),
        )
        padded = replay.rx2("mf")
        columns = {}
        for name in padded.names:
            values = padded.rx2(name)
            if bool(self._ro.baseenv["is.factor"](values)[0]):
                values = self._base.as_character(values)
            columns[str(name)] = _r_array(values)
        unique = replay.rx2("nr")
        return BamDiscretePrefit(
            _r_array(expanded, dtype=np.float64),
            pd.DataFrame(columns),
            replay_index.astype(np.int32) - 1,
            dict(zip(map(str, unique.names), map(int, unique), strict=True)),
            tuple(map(str, setup.rx2("dinfo").rx2("pmf.names"))),
        )

    def fit_bam_discrete_gaussian(
        self, formula: str, data: pd.DataFrame, **kwargs: Any
    ) -> BamDiscreteFit:
        """Pinned Gaussian identity :meth:`fit_bam_discrete`."""
        return self.fit_bam_discrete(formula, data, family="gaussian", **kwargs)

    def fit_bam_discrete(
        self,
        formula: str,
        data: pd.DataFrame,
        *,
        family: str = "gaussian",
        link: str | None = None,
        sp: np.ndarray | None = None,
        initial_sp: np.ndarray | None = None,
        initial_scale: float | None = None,
        scale: float | None = None,
        factor_names: tuple[str, ...] = (),
        m: int | tuple[int, ...] | None = None,
        weights: np.ndarray | None = None,
        offset: np.ndarray | None = None,
        max_iterations: int = 200,
        epsilon: float = 1e-7,
        capture_trials: bool = False,
        newdata: pd.DataFrame | None = None,
        discrete_prediction: bool = False,
        theta: float | None = None,
        theta_fixed: bool = False,
    ) -> BamDiscreteFit:
        """Fit pinned discrete bam with fixed or performance-iteration smoothing.

        ``family``/``link`` name a pinned ``stats`` family constructor
        (``gaussian``, ``poisson``, ``binomial`` or ``gamma``) or mgcv's
        ``nb``, and its link. For ``nb``, ``theta`` is the fixed theta when
        ``theta_fixed`` and otherwise the start of its estimation (default 1).
        ``sp`` fixes smoothing. ``initial_sp`` and ``initial_scale`` instead
        supply mgcv's ``in.out`` start for a selected working-model replay.
        ``scale`` is R's scale control: negative estimates scale, while a
        positive value fixes it. ``newdata`` is predicted on the link
        scale with standard errors by ``predict.bam``: exactly
        (``discrete=FALSE``, ``predict.gam``) by default, or through
        ``predict.bamd``'s request compression when ``discrete_prediction``.
        """
        if not isinstance(discrete_prediction, bool):
            raise ValueError("discrete_prediction must be a boolean")
        smoothing, arguments = self._bam_discrete_arguments(
            data,
            family=family,
            link=link,
            sp=sp,
            initial_sp=initial_sp,
            initial_scale=initial_scale,
            scale=scale,
            factor_names=factor_names,
            m=m,
            weights=weights,
            offset=offset,
            max_iterations=max_iterations,
            epsilon=epsilon,
            theta=theta,
            theta_fixed=theta_fixed,
        )
        trials: list[np.ndarray] = []
        fitter = self._traced_bam_fit_chol(trials) if capture_trials else self._mgcv.bam
        fit = fitter(self._ro.Formula(formula), **arguments)
        return self._bam_discrete_result(
            fit,
            smoothing,
            newdata=newdata,
            trial_records=np.vstack(trials) if capture_trials else None,
            discrete_prediction=discrete_prediction,
        )

    _BAM_FAMILY_LINKS: ClassVar[dict[str, tuple[str, ...]]] = {
        "gaussian": ("identity", "log", "inverse"),
        "poisson": ("log", "identity", "sqrt"),
        "binomial": ("logit", "probit", "cloglog", "log"),
        "gamma": ("inverse", "identity", "log"),
        "nb": ("log", "identity", "sqrt"),
    }

    def _bam_family(
        self,
        family: str,
        link: str | None,
        *,
        theta: float | None = None,
        theta_fixed: bool = False,
    ) -> Any:
        """Construct one pinned family object by typed name.

        ``nb`` is ``mgcv::nb``: ``theta`` fixes theta when ``theta_fixed``,
        otherwise it starts the estimation (``nb(theta=-theta)``; default 1).
        """
        links = self._BAM_FAMILY_LINKS.get(family)
        if links is None:
            raise ValueError(
                f"Unknown bam family {family!r}; expected one of "
                f"{sorted(self._BAM_FAMILY_LINKS)}"
            )
        if link is not None and link not in links:
            raise ValueError(f"link {link!r} is not a pinned {family} link")
        if not isinstance(theta_fixed, bool):
            raise ValueError("theta_fixed must be a boolean")
        if family != "nb":
            if theta is not None or theta_fixed:
                raise ValueError("theta applies to the nb family only")
            constructor = {
                "gaussian": self._stats.gaussian,
                "poisson": self._stats.poisson,
                "binomial": self._stats.binomial,
                "gamma": self._stats.Gamma,
            }[family]
            return constructor() if link is None else constructor(link=link)
        if theta is not None and (not np.isfinite(theta) or theta <= 0.0):
            raise ValueError("nb theta must be positive and finite")
        if theta_fixed and theta is None:
            raise ValueError("a fixed nb theta needs its value")
        arguments: dict[str, Any] = {} if link is None else {"link": link}
        if theta is not None:
            arguments["theta"] = float(theta) if theta_fixed else -float(theta)
        return self._mgcv.nb(**arguments)

    def family_dev_resids(
        self,
        family: str,
        y: np.ndarray,
        mu: np.ndarray,
        weight: np.ndarray,
        *,
        link: str | None = None,
    ) -> np.ndarray:
        """Evaluate one pinned ``stats`` family's ``dev.resids`` per row.

        No domain check is applied: means outside the family domain return
        whatever R's arithmetic returns (``NaN`` included), which is what
        ``bgam.fitd``'s step control sums.
        """
        self._require_rpy2()
        arrays = [np.asarray(values, dtype=np.float64) for values in (y, mu, weight)]
        if any(
            values.ndim != 1 or values.shape != arrays[0].shape for values in arrays
        ):
            raise ValueError("y, mu and weight must be matching 1-D arrays")
        constructor = self._bam_family(family, link)
        y_r, mu_r, weight_r = (self._to_r_vector(values) for values in arrays)
        return _r_array(
            constructor.rx2("dev.resids")(y_r, mu_r, weight_r), dtype=np.float64
        )

    def nb_working_terms(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        eta: np.ndarray,
        offset: np.ndarray,
        weight: np.ndarray,
        log_theta: float,
        *,
        link: str = "log",
    ) -> dict[str, np.ndarray]:
        """Pinned ``nb()`` row terms at explicit theta, as ``bgam.fitd`` uses them.

        ``fix.family.link(nb(theta, link))``'s ``dev.resids``, ``Dd`` (level
        2: ``Dth``, ``Dth2``), the ``ls`` saturated likelihood totals and the
        extended-family working model ``w = dDeta$Deta2*.5``,
        ``z = (eta-offset) - dDeta$Deta.Deta2`` with non-finite rows zeroed.
        """
        self._require_rpy2()
        arrays = [
            np.asarray(values, dtype=np.float64)
            for values in (y, mu, eta, offset, weight)
        ]
        if any(
            values.ndim != 1 or values.shape != arrays[0].shape for values in arrays
        ):
            raise ValueError(
                "y, mu, eta, offset and weight must be matching 1-D arrays"
            )
        if not np.isfinite(log_theta):
            raise ValueError("log_theta must be finite")
        family = self._call_internal(
            "fix.family.link", self._bam_family("nb", link, theta=1.0, theta_fixed=True)
        )
        y_r, mu_r, eta_r, offset_r, weight_r = (self._to_r_vector(v) for v in arrays)
        theta_r = self._to_r_vector(np.array([log_theta]))
        base = self._ro.baseenv
        deviance = family.rx2("dev.resids")(y_r, mu_r, weight_r, theta_r)
        derivatives = self._ro.ListVector(
            family.rx2("Dd")(y_r, mu_r, theta_r, weight_r, level=2)
        )
        saturated = self._ro.ListVector(family.rx2("ls")(y_r, weight_r, theta_r, 1.0))
        working = self._ro.ListVector(
            self._call_internal("dDeta", y_r, mu_r, weight_r, theta_r, family, 0)
        )
        w = base["*"](working.rx2("Deta2"), self._to_r_vector(np.array([0.5])))
        z = base["-"](base["-"](eta_r, offset_r), working.rx2("Deta.Deta2"))
        good = base["&"](base["is.finite"](z), base["is.finite"](w))
        zero = self._to_r_vector(np.array([0.0]))
        return {
            "deviance": _r_array(deviance, dtype=np.float64),
            "dth": _r_array(derivatives.rx2("Dth"), dtype=np.float64).ravel(),
            "dth2": _r_array(derivatives.rx2("Dth2"), dtype=np.float64).ravel(),
            "ls": np.array([saturated.rx2("ls")[0]], dtype=np.float64),
            "lsth1": np.array([saturated.rx2("lsth1")[0]], dtype=np.float64),
            "lsth2": np.array([saturated.rx2("lsth2")[0]], dtype=np.float64),
            "w": _r_array(base["ifelse"](good, w, zero), dtype=np.float64),
            "z": _r_array(base["ifelse"](good, z, zero), dtype=np.float64),
        }

    def nb_estimate_theta(
        self,
        y: np.ndarray,
        mu: np.ndarray,
        weight: np.ndarray,
        log_theta: float,
        *,
        link: str = "log",
        tolerance: float = 1e-7,
    ) -> float:
        """Pinned ``estimate.theta(theta, nb, y, mu, scale=1, wt, tol)``: log theta."""
        self._require_rpy2()
        arrays = [np.asarray(values, dtype=np.float64) for values in (y, mu, weight)]
        if any(
            values.ndim != 1 or values.shape != arrays[0].shape for values in arrays
        ):
            raise ValueError("y, mu and weight must be matching 1-D arrays")
        family = self._call_internal(
            "fix.family.link", self._bam_family("nb", link, theta=1.5)
        )
        y_r, mu_r, weight_r = (self._to_r_vector(v) for v in arrays)
        result = self._call_internal(
            "estimate.theta",
            self._to_r_vector(np.array([float(log_theta)])),
            family,
            y_r,
            mu_r,
            scale=1.0,
            wt=weight_r,
            tol=float(tolerance),
        )
        return float(_r_array(result, dtype=np.float64)[0])

    def _bam_discrete_arguments(
        self,
        data: pd.DataFrame,
        *,
        family: str = "gaussian",
        link: str | None = None,
        sp: np.ndarray | None,
        initial_sp: np.ndarray | None,
        initial_scale: float | None,
        scale: float | None,
        factor_names: tuple[str, ...],
        m: int | tuple[int, ...] | None,
        weights: np.ndarray | None,
        offset: np.ndarray | None,
        max_iterations: int,
        epsilon: float,
        theta: float | None = None,
        theta_fixed: bool = False,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Validate pinned discrete ``bam`` controls and build its arguments."""
        if family not in self._BAM_FAMILY_LINKS or (
            link is not None and link not in self._BAM_FAMILY_LINKS[family]
        ):
            raise ValueError(f"unsupported bam family/link {family!r}/{link!r}")
        self._require_rpy2()
        ok, reason = self.check_versions()
        if not ok:
            raise RBridgeError(f"Pinned bam discrete fit unavailable: {reason}")
        if len(factor_names) != len(set(factor_names)) or not set(factor_names) <= set(
            data.columns
        ):
            raise ValueError("factor_names must be distinct data-frame columns")
        smoothing = (
            np.empty(0)
            if sp is None
            else np.atleast_1d(np.asarray(sp, dtype=np.float64))
        )
        if sp is not None:
            if smoothing.ndim != 1 or not len(smoothing):
                raise ValueError("sp must contain at least one smoothing parameter")
            if np.any(smoothing <= 0) or not np.all(np.isfinite(smoothing)):
                raise ValueError("sp must contain positive finite values")
        if sp is not None and initial_sp is not None:
            raise ValueError("fixed sp and initial_sp are mutually exclusive")
        if initial_sp is None:
            if initial_scale is not None:
                raise ValueError("initial_scale requires initial_sp")
            start = None
        else:
            start = np.atleast_1d(np.asarray(initial_sp, dtype=np.float64))
            if (
                start.ndim != 1
                or not len(start)
                or np.any(start <= 0)
                or not np.all(np.isfinite(start))
                or initial_scale is None
                or not np.isfinite(initial_scale)
                or initial_scale <= 0
            ):
                raise ValueError("initial_sp and initial_scale must be positive finite")
            initial_scale = float(initial_scale)
        if scale is not None and (not np.isfinite(scale) or scale == 0):
            raise ValueError("scale must be finite and nonzero")
        if scale is not None:
            scale = float(scale)
        if (
            isinstance(max_iterations, bool)
            or not isinstance(max_iterations, int)
            or max_iterations < 1
        ):
            raise ValueError("max_iterations must be a positive integer")
        if (
            isinstance(epsilon, bool)
            or not isinstance(epsilon, (int, float))
            or not np.isfinite(epsilon)
            or epsilon <= 0.0
        ):
            raise ValueError("epsilon must be positive and finite")
        if isinstance(m, tuple):
            if not m or any(
                isinstance(value, bool) or not isinstance(value, int) or value < 2
                for value in m
            ):
                raise ValueError("m tuple entries must be integers of at least 2")
            m_value = self._to_r_vector(np.asarray(m, dtype=np.int32))
        elif m is None:
            m_value = True
        elif isinstance(m, bool) or not isinstance(m, int) or m < 2:
            raise ValueError("m must be an integer of at least 2")
        else:
            m_value = int(m)
        prior_weights = (
            np.ones(len(data), dtype=np.float64)
            if weights is None
            else np.asarray(weights, dtype=np.float64)
        )
        model_offset = (
            np.zeros(len(data), dtype=np.float64)
            if offset is None
            else np.asarray(offset, dtype=np.float64)
        )
        if prior_weights.shape != (len(data),) or model_offset.shape != (len(data),):
            raise ValueError("weights and offset must have one value per data row")
        if (
            np.any(prior_weights < 0.0)
            or not np.all(np.isfinite(prior_weights))
            or not np.all(np.isfinite(model_offset))
        ):
            raise ValueError("weights must be finite and nonnegative; offset finite")
        frame = self._to_r_dataframe(data)
        for name in factor_names:
            series = data[name]
            levels = (
                list(series.dtype.categories)
                if isinstance(series.dtype, pd.CategoricalDtype)
                else list(pd.unique(series))
            )
            constructor = (
                self._base.ordered
                if isinstance(series.dtype, pd.CategoricalDtype)
                and series.dtype.ordered
                else self._base.factor
            )
            frame = self._ro.baseenv["[[<-"](
                frame,
                name,
                value=constructor(
                    frame.rx2(name), levels=self._ro.StrVector([str(x) for x in levels])
                ),
            )
        arguments: dict[str, Any] = {
            "data": frame,
            "family": self._bam_family(
                family, link, theta=theta, theta_fixed=theta_fixed
            ),
            "method": "fREML",
            "discrete": m_value,
            "control": self._call_internal(
                "gam.control", maxit=max_iterations, epsilon=epsilon
            ),
        }
        if weights is not None:
            arguments["weights"] = self._to_r_vector(prior_weights)
        if offset is not None:
            arguments["offset"] = self._to_r_vector(model_offset)
        if sp is not None:
            arguments["sp"] = self._to_r_vector(smoothing)
        if start is not None:
            arguments["in.out"] = self._ro.ListVector(
                {"sp": self._to_r_vector(start), "scale": initial_scale}
            )
        if scale is not None:
            arguments["scale"] = scale
        return smoothing, arguments

    def _bam_discrete_result(
        self,
        fit: Any,
        smoothing: np.ndarray,
        *,
        newdata: pd.DataFrame | None,
        trial_records: np.ndarray | None,
        discrete_prediction: bool = False,
    ) -> BamDiscreteFit:
        """Translate one pinned discrete ``bam`` fit into typed fields."""
        coefficients = _r_array(self._stats.coef(fit), dtype=np.float64)
        covariance = _r_array(fit.rx2("Vp"), dtype=np.float64)
        prediction = prediction_se = None
        if newdata is not None:
            predicted = self._stats.predict(
                fit,
                newdata=self._to_r_dataframe(newdata),
                type="link",
                discrete=discrete_prediction,
                **{"se.fit": True},
            )
            prediction = _r_array(predicted.rx2("fit"), dtype=np.float64)
            prediction_se = _r_array(predicted.rx2("se.fit"), dtype=np.float64)
        fitted_sp = _r_array(fit.rx2("sp"), dtype=np.float64)
        if not len(fitted_sp):
            fitted_sp = smoothing
        fitted_family = fit.rx2("family")
        log_theta = (
            float(fitted_family.rx2("getTheta")()[0])
            if bool(self._base.inherits(fitted_family, "extended.family")[0])
            else None
        )
        return BamDiscreteFit(
            coefficients=coefficients,
            fitted_values=_r_array(self._stats.fitted(fit), dtype=np.float64),
            deviance=float(self._stats.deviance(fit)[0]),
            scale=float(fit.rx2("sig2")[0]),
            edf_total=float(self._base.sum(fit.rx2("edf"))[0]),
            score=float(fit.rx2("gcv.ubre")[0]),
            converged=bool(self._ro.baseenv["isTRUE"](fit.rx2("converged"))[0]),
            mgcv_converged=bool(self._ro.baseenv["isTRUE"](fit.rx2("mgcv.conv"))[0]),
            smoothing_params=fitted_sp,
            n_iter=int(fit.rx2("iter")[0]),
            trial_records=trial_records,
            covariance=covariance,
            prediction=prediction,
            prediction_se=prediction_se,
            log_theta=log_theta,
            coefficient_edf=_r_array(fit.rx2("edf"), dtype=np.float64),
        )

    def _traced_bam_fit_chol(self, trials: list[np.ndarray]) -> Any:
        """Clone bam's discrete caller chain and trace source proposals privately."""

        def capture(
            lsp: Any, n_sp: Any, log_phi: Any, crit: Any, proposal: Any
        ) -> None:
            proposal = self._ro.ListVector(proposal)
            rho = _r_array(lsp, dtype=np.float64)[: int(n_sp[0])]
            trials.append(
                np.concatenate(
                    (
                        rho,
                        _r_array(log_phi, dtype=np.float64),
                        np.asarray(self._source_fitchol_tolerance(crit)),
                        _r_array(proposal.rx2("step"), dtype=np.float64),
                        _r_array(proposal.rx2("grad"), dtype=np.float64),
                        _r_array(proposal.rx2("hess"), dtype=np.float64).ravel(
                            order="F"
                        ),
                    )
                )
            )

        return self._instrumented_bam_fit_chol(
            ("lsp", "n.sp", "log.phi", "crit", "prop"), capture
        )

    def _source_fitchol_tolerance(self, crit: Any) -> np.ndarray:
        """bgam.fitd's ``abs(crit)*.Machine$double.eps^.5`` evaluated in R."""
        base = self._ro.baseenv
        epsilon = base[".Machine"].rx2("double.eps")
        return _r_array(
            base["*"](
                self._base.abs(crit), base["^"](epsilon, self._to_r_vector([0.5]))
            ),
            dtype=np.float64,
        )

    def _instrumented_bam_fit_chol(
        self,
        capture_symbols: tuple[str, ...],
        callback: Any,
        *,
        final_symbols: tuple[str, ...] = (),
        final_callback: Any = None,
    ) -> Any:
        """Private bam whose bgam.fitd calls back after each Sl.fitChol trial.

        ``final_callback`` optionally receives ``final_symbols`` once, when
        the fitting loop has ended (before ``if (!conv) warning(...)``).
        """
        from tests.r_ast import clone_function, find_call_paths, instrument_function

        namespace = self._base.asNamespace("mgcv")
        private = self._base.new_env(parent=namespace)
        source = self._utils.getFromNamespace("bgam.fitd", "mgcv")
        paths = find_call_paths(source, "<-", required_symbols=("prop", "Sl.fitChol"))
        if len(paths) != 1:
            raise RBridgeError("Pinned bgam.fitd Sl.fitChol proposal changed")
        traced = instrument_function(
            source,
            path=paths[0],
            expected_head="<-",
            capture_symbols=capture_symbols,
            callback=callback,
            when="after",
        )
        if final_callback is not None:
            ends = tuple(
                path
                for path in find_call_paths(
                    traced, "if", required_symbols=("conv", "warning")
                )
                if len(path) == 1
            )
            if len(ends) != 1:
                raise RBridgeError("Pinned bgam.fitd loop exit changed")
            traced = instrument_function(
                traced,
                path=ends[0],
                expected_head="if",
                capture_symbols=final_symbols,
                callback=final_callback,
                when="before",
            )
        private_fit = clone_function(traced, environment=private)
        private["bgam.fitd"] = private_fit
        bam = self._utils.getFromNamespace("bam", "mgcv")
        fitter = clone_function(bam, environment=private)
        fitter._jaxgam_callback_handles = (
            traced,
            private_fit,
            *traced._jaxgam_callback_handles,
        )
        return fitter

    def trace_bam_working_model(
        self,
        formula: str,
        data: pd.DataFrame,
        *,
        initial_sp: np.ndarray,
        initial_scale: float,
        scale: float,
        m: int = 30,
        weights: np.ndarray | None = None,
        offset: np.ndarray | None = None,
        max_iterations: int = 200,
        epsilon: float = 1e-7,
    ) -> SourceWorkingModelTrace:
        """Replayable pinned ``bgam.fitd`` state at every ``Sl.fitChol`` trial.

        Additive Gaussian ``bgam.fitd`` freezes ``qrx`` and ``Sl`` after its
        first iteration; every trial must report the identical frozen state.
        """
        smoothing, arguments = self._bam_discrete_arguments(
            data,
            sp=None,
            initial_sp=initial_sp,
            initial_scale=initial_scale,
            scale=scale,
            factor_names=(),
            m=m,
            weights=weights,
            offset=offset,
            max_iterations=max_iterations,
            epsilon=epsilon,
        )
        states: list[SourceWorkingModelState] = []
        trials: list[SourceWorkingTrial] = []

        def capture(
            iteration: Any,
            accepted: Any,
            step: Any,
            lsp: Any,
            n_sp: Any,
            log_phi: Any,
            crit: Any,
            dev: Any,
            devold: Any,
            proposal: Any,
            qrx: Any,
            blocks: Any,
            null_dimension: Any,
            n_obs: Any,
        ) -> None:
            qrx = self._ro.ListVector(qrx)
            state = SourceWorkingModelState(
                self._source_working_blocks(blocks),
                _r_array(qrx.rx2("XX"), dtype=np.float64),
                _r_array(qrx.rx2("Xy"), dtype=np.float64).ravel(),
                float(qrx.rx2("y.norm2")[0]),
                int(n_obs[0]),
                int(null_dimension[0]),
            )
            if states and not _same_source_working_state(states[0], state):
                raise RBridgeError("additive bgam.fitd changed its frozen state")
            if not states:
                states.append(state)
            parameters = _r_array(lsp, dtype=np.float64)
            count = int(n_sp[0])
            previous = _r_array(step, dtype=np.float64)
            if previous.shape != parameters.shape:
                if previous.shape != (1,) or previous[0] != 0.0:
                    raise RBridgeError("unexpected bgam.fitd Nstep shape")
                previous = np.zeros_like(parameters)
            determinant = self._call_internal(
                "ldetS",
                blocks,
                self._to_r_vector(parameters[:count]),
                self._base.rep(False, count),
                np=self._base.ncol(qrx.rx2("XX")),
                root=False,
                repara=False,
            )
            trials.append(
                SourceWorkingTrial(
                    int(iteration[0]),
                    _r_array(accepted, dtype=np.float64),
                    previous,
                    float(dev[0]),
                    float(devold[0]),
                    float(crit[0]),
                    self._source_working_proposal(
                        proposal,
                        parameters[:count],
                        float(log_phi[0]),
                        self._source_fitchol_tolerance(crit),
                        len(parameters),
                    ),
                    _r_array(determinant.rx2("ldet1"), dtype=np.float64),
                    _r_array(determinant.rx2("ldet2"), dtype=np.float64),
                )
            )

        fitter = self._instrumented_bam_fit_chol(
            (
                "iter",
                "lsp0",
                "Nstep",
                "lsp",
                "n.sp",
                "log.phi",
                "crit",
                "dev",
                "devold",
                "prop",
                "qrx",
                "Sl",
                "Mp",
                "nobs",
            ),
            capture,
        )
        fit = fitter(self._ro.Formula(formula), **arguments)
        if not states:
            raise RBridgeError("bgam.fitd made no Sl.fitChol trial")
        return SourceWorkingModelTrace(
            states[0],
            tuple(trials),
            self._bam_discrete_result(fit, smoothing, newdata=None, trial_records=None),
        )

    def trace_bam_regular_working_model(
        self,
        formula: str,
        data: pd.DataFrame,
        *,
        family: str,
        link: str | None = None,
        sp: np.ndarray | None = None,
        factor_names: tuple[str, ...] = (),
        m: int | tuple[int, ...] | None = 30,
        weights: np.ndarray | None = None,
        offset: np.ndarray | None = None,
        max_iterations: int = 200,
        epsilon: float = 1e-7,
        allow_failure: bool = False,
        theta: float | None = None,
        theta_fixed: bool = False,
    ) -> SourceRegularWorkingModelTrace:
        """Replayable pinned non-additive ``bgam.fitd`` state at every trial.

        Every ``Sl.fitChol`` trial reports the iteration's own working model
        (``qrx``), its step-controlled coefficients and ``kk``, and the exact
        proposal with R's ``exp(rho)``. ``sp`` fixes every smoothing
        parameter, so ``rho`` is ``Sl.fitChol``'s ``rho0`` (R's ``log(sp)``).
        ``allow_failure`` returns an R error (for example ``bgam.fitd``'s
        ``"Non-finite deviance"`` stop) as a trace with ``failure`` set
        instead of raising.
        """
        if not isinstance(allow_failure, bool):
            raise ValueError("allow_failure must be a boolean")
        if family == "gaussian" and link in (None, "identity"):
            raise ValueError("additive Gaussian traces use trace_bam_working_model")
        smoothing, arguments = self._bam_discrete_arguments(
            data,
            family=family,
            link=link,
            sp=sp,
            initial_sp=None,
            initial_scale=None,
            scale=None,
            factor_names=factor_names,
            m=m,
            weights=weights,
            offset=offset,
            max_iterations=max_iterations,
            epsilon=epsilon,
            theta=theta,
            theta_fixed=theta_fixed,
        )
        from rpy2 import rinterface

        extended = family == "nb"
        fixed_rho = (
            None
            if sp is None
            else _r_array(
                self._base.log(self._to_r_vector(smoothing)), dtype=np.float64
            )
        )
        blocks_cache: list[tuple[SourceWorkingPenaltyBlock, ...]] = []
        trials: list[SourceRegularWorkingTrial] = []

        def vector(value: Any) -> np.ndarray:
            return _r_array(value, dtype=np.float64).ravel()

        def capture(
            iteration: Any,
            accepted: Any,
            step: Any,
            lsp: Any,
            n_sp: Any,
            log_phi: Any,
            crit: Any,
            dev: Any,
            devold: Any,
            proposal: Any,
            qrx: Any,
            blocks: Any,
            null_dimension: Any,
            n_obs: Any,
            coefficient_trials: Any,
            coefficients: Any,
            *theta_now: Any,
        ) -> None:
            qrx = self._ro.ListVector(qrx)
            if not blocks_cache:
                blocks_cache.append(self._source_working_blocks(blocks))
            parameters = vector(lsp)
            count = int(n_sp[0])
            rho = parameters[:count] if fixed_rho is None else fixed_rho
            previous = vector(step)
            if previous.shape != parameters.shape:
                if previous.shape != (1,) or previous[0] != 0.0:
                    raise RBridgeError("unexpected bgam.fitd Nstep shape")
                previous = np.zeros_like(parameters)
            start = vector(accepted)
            if start.shape != parameters.shape:
                raise RBridgeError("unexpected bgam.fitd lsp0 shape")
            trials.append(
                SourceRegularWorkingTrial(
                    int(iteration[0]),
                    int(coefficient_trials[0]),
                    None
                    if coefficients.typeof == rinterface.RTYPES.NILSXP
                    else vector(coefficients),
                    _r_array(qrx.rx2("R"), dtype=np.float64),
                    vector(qrx.rx2("f")),
                    SourceWorkingModelState(
                        blocks_cache[0],
                        _r_array(qrx.rx2("XX"), dtype=np.float64),
                        vector(qrx.rx2("Xy")),
                        float(qrx.rx2("y.norm2")[0]),
                        int(n_obs[0]),
                        int(null_dimension[0]),
                    ),
                    start,
                    previous,
                    float(dev[0]),
                    float(devold[0]),
                    float(crit[0]),
                    self._source_working_proposal(
                        proposal,
                        rho,
                        float(log_phi[0]),
                        self._source_fitchol_tolerance(crit),
                        len(parameters),
                    ),
                    float(theta_now[0][0]) if theta_now else None,
                )
            )

        ending: list[tuple[int, int, bool, float | None]] = []

        def finish(
            iteration: Any, coefficient_trials: Any, converged: Any, *theta_now: Any
        ) -> None:
            ending.append(
                (
                    int(iteration[0]),
                    int(coefficient_trials[0]),
                    bool(converged[0]),
                    float(theta_now[0][0]) if theta_now else None,
                )
            )

        fitter = self._instrumented_bam_fit_chol(
            (
                "iter",
                "lsp0",
                "Nstep",
                "lsp",
                "n.sp",
                "log.phi",
                "crit",
                "dev",
                "devold",
                "prop",
                "qrx",
                "Sl",
                "Mp",
                "nobs",
                "kk",
                "coef",
                *(("theta",) if extended else ()),
            ),
            capture,
            final_symbols=("iter", "kk", "conv", *(("theta",) if extended else ())),
            final_callback=finish,
        )
        from rpy2.rinterface_lib.embedded import RRuntimeError

        try:
            fit = fitter(self._ro.Formula(formula), **arguments)
        except RRuntimeError as error:
            if not allow_failure:
                raise
            return SourceRegularWorkingModelTrace(
                tuple(trials), None, None, None, None, failure=str(error)
            )
        if not trials or len(ending) != 1:
            raise RBridgeError("bgam.fitd made no Sl.fitChol trial or loop exit")
        final_iteration, final_trials, converged, final_log_theta = ending[0]
        return SourceRegularWorkingModelTrace(
            tuple(trials),
            self._bam_discrete_result(fit, smoothing, newdata=None, trial_records=None),
            final_iteration,
            final_trials,
            converged,
            final_log_theta=final_log_theta,
        )

    def _source_working_blocks(
        self, blocks: Any
    ) -> tuple[SourceWorkingPenaltyBlock, ...]:
        """Translate an ``Sl.setup`` list without changing any stored value."""
        from rpy2 import rinterface

        result = []
        for block in self._ro.ListVector(blocks):
            start = int(block.rx2("start")[0]) - 1
            stop = int(block.rx2("stop")[0])
            transform = _r_array(block.rx2("D"), dtype=np.float64)
            if transform.ndim == 1:
                transform = np.diag(transform)
            roots_r = block.rx2("rS")
            roots = (
                ()
                if roots_r.typeof == rinterface.RTYPES.NILSXP
                else tuple(_r_array(root, dtype=np.float64) for root in roots_r)
            )
            result.append(
                SourceWorkingPenaltyBlock(
                    start,
                    stop,
                    transform,
                    tuple(
                        _r_array(penalty, dtype=np.float64)
                        for penalty in block.rx2("S")
                    ),
                    roots,
                    tuple(
                        _r_array(
                            self._call_internal("pcrossprod", root, trans=True),
                            dtype=np.float64,
                        )
                        for root in (() if not roots else roots_r)
                    ),
                    int(block.rx2("rank")[0]),
                    _r_array(block.rx2("ind"), dtype=bool),
                    bool(block.rx2("repara")[0]),
                )
            )
        return tuple(result)

    def _source_working_proposal(
        self,
        proposal: Any,
        log_smoothing: np.ndarray,
        log_scale: float,
        tolerance: np.ndarray,
        n_parameters: int,
    ) -> SourceWorkingProposal:
        """Translate one ``Sl.fitChol`` list and R's own ``exp(rho)``."""
        proposal = self._ro.ListVector(proposal)
        step = _r_array(proposal.rx2("step"), dtype=np.float64).ravel()
        if step.shape != (n_parameters,):
            if step.shape != (1,) or step[0] != 0.0:
                raise RBridgeError("unexpected Sl.fitChol step shape")
            step = np.zeros(n_parameters)
        return SourceWorkingProposal(
            np.array(log_smoothing, dtype=np.float64),
            _r_array(
                self._base.exp(self._to_r_vector(log_smoothing)), dtype=np.float64
            ),
            float(log_scale),
            float(tolerance[0]),
            _r_array(proposal.rx2("beta"), dtype=np.float64).ravel(),
            _r_array(proposal.rx2("grad"), dtype=np.float64).ravel(),
            _r_array(proposal.rx2("hess"), dtype=np.float64).reshape(
                n_parameters, n_parameters, order="F"
            ),
            step,
            float(proposal.rx2("ldetS")[0]),
            float(proposal.rx2("ldetXXS")[0]),
        )

    def _source_bam_working_statistics(
        self,
        formula: str,
        data: pd.DataFrame,
        weights: np.ndarray,
        offset: np.ndarray,
    ) -> tuple[Any, Any, Any, Any, Any, Any]:
        """Pinned discrete design, ``Sl.setup`` and reparameterized moments."""
        self._require_rpy2()
        base = self._ro.baseenv
        frame = self._to_r_dataframe(data)
        setup = self._mgcv.bam(
            self._ro.Formula(formula), data=frame, discrete=30, fit=False
        )
        width = int(self._base.ncol(setup.rx2("X"))[0])
        design = self._call_internal(
            "Xbd",
            setup.rx2("Xd"),
            self._base.diag(width),
            setup.rx2("kd"),
            setup.rx2("ks"),
            setup.rx2("ts"),
            setup.rx2("dt"),
            setup.rx2("v"),
            setup.rx2("qc"),
            drop=setup.rx2("drop"),
        )
        blocks = self._call_internal("Sl.setup", setup)
        weight = self._to_r_vector(np.asarray(weights, dtype=np.float64))
        response = frame.rx2("y")
        residual = base["-"](
            response, self._to_r_vector(np.asarray(offset, dtype=np.float64))
        )
        weighted_design = base["*"](weight, design)
        gram = self._base.crossprod(design, weighted_design)
        rhs = self._base.as_numeric(
            self._base.crossprod(design, base["*"](weight, residual))
        )
        yy = self._base.sum(base["*"](weight, base["^"](residual, 2)))
        ranks = self._base.c(*(block.rx2("rank") for block in blocks))
        unpenalized = base["-"](self._base.ncol(design), self._base.sum(ranks))
        repara = {"inverse": False, "both.sides": True, "cov": False}
        gram = self._call_internal("Sl.initial.repara", blocks, gram, **repara)
        rhs = self._call_internal("Sl.initial.repara", blocks, rhs, **repara)
        return design, blocks, gram, rhs, yy, unpenalized

    def source_bam_sl_fitchol_proposal(
        self,
        formula: str,
        data: pd.DataFrame,
        sp: np.ndarray,
        weights: np.ndarray,
        offset: np.ndarray,
        *,
        fixed_scale: bool,
    ) -> dict[str, np.ndarray]:
        """Evaluate the pinned discrete design and ``Sl.fitChol`` proposal."""
        base = self._ro.baseenv
        design, blocks, gram, rhs, yy, unpenalized = (
            self._source_bam_working_statistics(formula, data, weights, offset)
        )
        proposal = self._call_internal(
            "Sl.fitChol",
            blocks,
            gram,
            rhs,
            self._base.log(self._to_r_vector(np.array(sp, dtype=np.float64))),
            yy=yy,
            **{"log.phi": -3.0, "phi.fixed": fixed_scale},
            nobs=len(data),
            Mp=unpenalized,
            tol=0,
        )
        coefficients = self._call_internal(
            "Sl.initial.repara",
            blocks,
            proposal.rx2("beta"),
            inverse=True,
            **{"both.sides": False},
            cov=False,
        )
        phi = self._base.exp(self._to_r_vector([-3.0]))
        residual_sum = base["-"](
            yy, self._base.sum(base["*"](proposal.rx2("beta"), rhs))
        )
        criterion = base["+"](
            base["+"](
                base["/"](residual_sum, base["*"](2, phi)),
                base["/"](
                    base["-"](proposal.rx2("ldetXXS"), proposal.rx2("ldetS")),
                    2,
                ),
            ),
            base["/"](
                base["*"](
                    base["-"](len(data), unpenalized),
                    self._base.log(base["*"](base["*"](2, self._base.pi), phi)),
                ),
                2,
            ),
        )
        return {
            "X": _r_array(design),
            "beta": _r_array(coefficients),
            "criterion": _r_array(criterion),
            "grad": _r_array(proposal.rx2("grad")),
            "hess": _r_array(proposal.rx2("hess")),
            "step": _r_array(proposal.rx2("step")),
        }

    def source_bam_sl_fitchol_state(
        self,
        formula: str,
        data: pd.DataFrame,
        log_smoothing: np.ndarray,
        weights: np.ndarray,
        offset: np.ndarray,
        *,
        fixed_scale: bool,
        log_scale: float = -3.0,
    ) -> SourceSlFitCholEvaluation:
        """Pinned ``Sl.fitChol`` with every input it consumed, at exact rho.

        Unlike :meth:`source_bam_sl_fitchol_proposal`, R receives the log
        smoothing parameters themselves, so a caller can feed the returned
        roots, products, moments and R's ``exp(rho)`` into another kernel.
        """
        rho = np.asarray(log_smoothing, dtype=np.float64)
        if rho.ndim != 1 or not len(rho) or not np.all(np.isfinite(rho)):
            raise ValueError("log_smoothing must be a finite non-empty vector")
        if not np.isfinite(log_scale):
            raise ValueError("log_scale must be finite")
        _, blocks, gram, rhs, yy, unpenalized = self._source_bam_working_statistics(
            formula, data, weights, offset
        )
        values = self._to_r_vector(rho)
        proposal = self._call_internal(
            "Sl.fitChol",
            blocks,
            gram,
            rhs,
            values,
            yy=yy,
            **{"log.phi": float(log_scale), "phi.fixed": fixed_scale},
            nobs=len(data),
            Mp=unpenalized,
            tol=0,
        )
        determinant = self._call_internal(
            "ldetS",
            blocks,
            values,
            self._base.rep(False, len(rho)),
            np=self._base.ncol(gram),
            root=False,
            repara=False,
        )
        return SourceSlFitCholEvaluation(
            SourceWorkingModelState(
                self._source_working_blocks(blocks),
                _r_array(gram, dtype=np.float64),
                _r_array(rhs, dtype=np.float64).ravel(),
                float(yy[0]),
                len(data),
                int(unpenalized[0]),
            ),
            self._source_working_proposal(
                proposal,
                rho,
                log_scale,
                np.zeros(1),
                len(rho) + (not fixed_scale),
            ),
            _r_array(determinant.rx2("ldet1"), dtype=np.float64),
            _r_array(determinant.rx2("ldet2"), dtype=np.float64),
        )

    def source_bam_penalty_determinant(
        self, formula: str, data: pd.DataFrame, rho: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Return pinned ``ldetS`` value and derivatives for a bam setup."""
        self._require_rpy2()
        setup = self._mgcv.bam(
            self._ro.Formula(formula),
            data=self._to_r_dataframe(data),
            discrete=30,
            fit=False,
        )
        values = self._to_r_vector(np.array(rho, dtype=np.float64))
        result = self._call_internal(
            "ldetS",
            self._call_internal("Sl.setup", setup),
            values,
            self._base.rep(False, len(rho)),
            np=self._base.ncol(setup.rx2("X")),
            root=False,
            repara=False,
        )
        return {
            "value": _r_array(result.rx2("ldetS")),
            "first": _r_array(result.rx2("ldet1")),
            "second": _r_array(result.rx2("ldet2")),
        }

    def source_bam_penalty_root_products(
        self, formula: str, data: pd.DataFrame
    ) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """Return pinned ``Sl.setup`` roots and ``pcrossprod`` products."""
        self._require_rpy2()
        setup = self._mgcv.bam(
            self._ro.Formula(formula),
            data=self._to_r_dataframe(data),
            discrete=30,
            fit=False,
        )
        first = self._call_internal("Sl.setup", setup)[0]
        return tuple(
            (
                _r_array(root),
                _r_array(self._call_internal("pcrossprod", root, trans=True)),
            )
            for root in first.rx2("rS")
        )

    def source_linear_algebra(self) -> SourceLinearAlgebra:
        """Report the LAPACK and BLAS libraries pinned R is running."""
        self._require_rpy2()
        base = self._ro.baseenv
        lapack = str(base["La_library"]()[0])
        version = str(base["La_version"]()[0])
        blas = str(base["extSoftVersion"]().rx2("BLAS")[0])
        config = None
        if lapack:
            try:
                function = ctypes.CDLL(lapack).openblas_get_config
            except (OSError, AttributeError):
                function = None
            if function is not None:
                function.argtypes = []
                function.restype = ctypes.c_char_p
                raw = function()
                config = None if raw is None else raw.decode()
        return SourceLinearAlgebra(lapack, version, blas, config)

    def source_pivoted_matrix_factor(self, matrix: np.ndarray) -> dict[str, np.ndarray]:
        """Return pinned pivoted ``chol`` fields for an identical R matrix."""
        self._require_rpy2()
        factor = self._ro.baseenv["chol"](self._to_r_matrix(matrix), pivot=True)
        # Copy: views into R vectors dangle once R releases the objects.
        return {
            "C": _r_array(factor, dtype=np.float64),
            "pivot": _r_array(self._base.attr(factor, "pivot")),
            "rank": _r_array(self._base.attr(factor, "rank")),
        }

    def source_pivoted_coupled_factors(
        self, data: pd.DataFrame, rho: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Build pinned tensor projected penalty and pivoted factor."""
        self._require_rpy2()
        base = self._ro.baseenv
        setup = self._mgcv.bam(
            self._ro.Formula('y ~ te(x,z,bs="cr",k=3)'),
            data=self._to_r_dataframe(data),
            discrete=30,
            fit=False,
        )
        blocks = self._call_internal("Sl.setup", setup)
        block = next(block for block in blocks if len(block.rx2("S")) > 1)
        roots = block.rx2("rS")
        log_smoothing = self._to_r_vector(np.asarray(rho, dtype=np.float64))
        penalty = base["+"](
            base["*"](
                self._base.tcrossprod(roots[0]),
                self._base.exp(base["["](log_smoothing, 1)),
            ),
            base["*"](
                self._base.tcrossprod(roots[1]),
                self._base.exp(base["["](log_smoothing, 2)),
            ),
        )
        diagonal = self._base.sqrt(self._base.diag(penalty))
        scaled = base["/"](self._base.t(base["/"](penalty, diagonal)), diagonal)
        factor = base["chol"](scaled, pivot=True)
        return {
            "A": _r_array(scaled, dtype=np.float64),
            "C": _r_array(factor, dtype=np.float64),
            "pivot": _r_array(self._base.attr(factor, "pivot")),
            "rank": _r_array(self._base.attr(factor, "rank")),
        }

    def source_working_penalty_setup(
        self, penalties: tuple[np.ndarray, ...], ranks: tuple[int, ...]
    ) -> tuple[np.ndarray, int, tuple[np.ndarray, ...], tuple[np.ndarray, ...]]:
        """Return pinned ``Sl.setup`` transform, projected penalties and roots."""
        self._require_rpy2()
        size = len(penalties[0])
        base = self._ro.baseenv
        source_penalties = base["list"](
            *(self._to_r_matrix(penalty) for penalty in penalties)
        )
        smooth = self._ro.ListVector(
            {
                "S": source_penalties,
                "rank": self._to_r_vector(np.asarray(ranks, dtype=np.int32)),
                "first.para": 1,
                "last.para": size,
                "repara": True,
            }
        )
        setup = self._ro.ListVector(
            {
                "X": self._base.matrix(0.0, 1, size),
                "n.paraPen": 0,
                "smooth": base["list"](smooth),
            }
        )
        blocks = self._call_internal("Sl.setup", setup)
        if len(blocks) != 1:
            raise RBridgeError("Pinned Sl.setup did not retain one penalty block")
        block = blocks[0]
        transform = _r_array(block.rx2("D"), dtype=np.float64)
        if transform.size == size:
            transform = np.diag(transform)
        rank = int(block.rx2("rank")[0])
        if len(penalties) == 1:
            projected = (_r_array(self._base.diag(block.rx2("ind"))),)
            roots: tuple[np.ndarray, ...] = ()
        else:
            projected = tuple(
                np.pad(_r_array(penalty), ((0, size - rank), (0, size - rank)))
                for penalty in block.rx2("S")
            )
            roots = tuple(_r_array(root) for root in block.rx2("rS"))
        return transform, rank, projected, roots
