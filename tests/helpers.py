"""Shared test helpers.

Public API (used directly by tests):
- SEED, N — constants
- r_available() — import-time R check for @pytest.mark.skipif
- r_tolerance() — tolerance tier by family
- make_smooth_spec() — SmoothSpec factory with many call-site variants

Private API (used by conftest fixtures and complex test-local fixtures):
- _generate_family_data() — single-predictor family data
- _inference_equivalence_cases(), _inference_r_cases() — shared prediction zoos
"""

from __future__ import annotations

import ctypes
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd

from jaxgam.formula.terms import SmoothSpec
from tests.tolerances import LOOSE, MODERATE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SEED = 42
N = 200


def nb_theta_six_row_case(variant: str) -> dict[str, object]:
    """Construct the reviewed high-theta count boundary without saved data."""
    if variant not in ("integer", "fractional_ge_one"):
        raise ValueError("Unknown six-row NB theta variant.")
    return {
        "integer_counts": variant == "integer",
        "log_theta": [float(np.log(1e6))],
        "max_y": 8,
        "mu": [1.2, 2.4, 3.1, 0.8, 2.0, 4.0],
        "weight": [0.8, 1.2, 0.0, 1.0, 0.9, 1.1],
        "y": (
            [0.0, 2.0, 3.0, 4.0, 0.0, 7.0]
            if variant == "integer"
            else [0.0, 2.25, 3.5, 4.75, 0.0, 7.25]
        ),
    }


def nb_optimizer_case_data() -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Recreate the seeded weighted NB model used by the exact outer gates."""
    rng = np.random.default_rng(8803)
    x = np.linspace(-1.0, 1.0, 180)
    offset = 0.12 * np.cos(2.0 * x)
    mu = np.exp(1.0 + 1.1 * np.sin(3.2 * x) + 0.35 * np.cos(6.0 * x) + offset)
    theta = 2.7
    y = rng.negative_binomial(theta, theta / (theta + mu)).astype(float)
    weight = 0.7 + 0.6 * rng.random(len(x))
    weight[::29] = 0.0
    return pd.DataFrame({"x": x, "y": y}), weight, offset


# ---------------------------------------------------------------------------
# R bridge helpers
# ---------------------------------------------------------------------------


def r_available() -> bool:
    """Check if R and mgcv are available with correct versions.

    Safe to call at import time (used in ``@pytest.mark.skipif``).
    """
    try:
        from tests.r_bridge import RBridge

        if not RBridge.available():
            return False
        ok, _ = RBridge.check_versions()
        return ok
    except Exception:
        return False


def r_links_pinned_openblas(r_bridge) -> bool:
    """Whether R's LAPACK is the OpenBLAS 0.3.26 the provider pins.

    The source-LAPACK provider declares scipy-openblas32 0.3.26, the OpenBLAS
    release the pinned Linux R links; results that depend on R's linked BLAS
    are bitwise there. The library is located with the typed
    ``RBridge.linked_lapack`` query (never the platform name) and identified
    by its own ``openblas_get_config`` string, since Debian installs OpenBLAS's
    LAPACK under the unversioned name ``liblapack.so.3``.
    """
    try:
        library = ctypes.CDLL(r_bridge.linked_lapack(), mode=ctypes.RTLD_LOCAL)
        config = library.openblas_get_config
    except (OSError, AttributeError):
        return False
    config.argtypes = []
    config.restype = ctypes.c_char_p
    return (config() or b"").decode().startswith("OpenBLAS 0.3.26 ")


def _openblas_identity(library: ctypes.CDLL, prefix: str) -> str:
    """``openblas_get_config`` and core name of one loaded OpenBLAS."""
    fields = []
    for name in ("openblas_get_config", "openblas_get_corename"):
        try:
            function = getattr(library, prefix + name)
        except AttributeError:
            fields.append(f"{name}: not exported")
            continue
        function.argtypes = []
        function.restype = ctypes.c_char_p
        fields.append((function() or b"").decode())
    return " | ".join(fields)


def platform_report_lines() -> list[str]:
    """CPU, SIMD and OpenBLAS kernel identity for platform-sensitive runs.

    Reports the CPU model and its AVX-512 flags, NumPy's enabled AVX-512
    dispatch, and the OpenBLAS configuration and selected core of the
    provider wheel and of R's linked LAPACK, so a platform-sensitive run
    records which runner type it sampled.
    """
    import platform

    lines = [f"machine: {platform.machine()}"]
    try:
        with open("/proc/cpuinfo", encoding="utf-8") as handle:
            first = handle.read().split("\n\n")[0]
        info = {
            key.strip(): value.strip()
            for key, _, value in (line.partition(":") for line in first.splitlines())
        }
        flags = sorted(
            flag for flag in info.get("flags", "").split() if "avx512" in flag
        )
        lines.append(f"cpu: {info.get('model name', '?')}")
        lines.append(f"avx512 flags: {' '.join(flags) or 'none'}")
    except OSError:
        lines.append("cpu: /proc/cpuinfo unavailable")
    try:
        from numpy._core._multiarray_umath import __cpu_features__ as features
    except ImportError:  # pragma: no cover - NumPy 1.x layout
        from numpy.core._multiarray_umath import __cpu_features__ as features
    enabled = [name for name, on in features.items() if on and "AVX512" in name]
    lines.append(f"numpy avx512 dispatch: {' '.join(enabled) or 'none'}")
    try:
        from jaxgam.smooths import source_lapack

        provider = source_lapack._source_lapack()
        identity = _openblas_identity(provider.library, "scipy_")
        lines.append(f"provider wheel OpenBLAS: {identity}")
    except Exception as error:
        lines.append(f"provider wheel OpenBLAS: unavailable ({error})")
    if r_available():
        try:
            from tests.r_bridge import RBridge

            path = RBridge().linked_lapack()
            library = ctypes.CDLL(path, mode=ctypes.RTLD_LOCAL)
            lines.append(f"R LAPACK OpenBLAS: {_openblas_identity(library, '')}")
        except Exception as error:
            lines.append(f"R LAPACK OpenBLAS: unavailable ({error})")
    else:
        lines.append("R LAPACK OpenBLAS: R unavailable")
    return lines


def r_tolerance(family_name: str):
    """Return tolerance tier for R comparison by family.

    Gaussian: MODERATE (single PIRLS iteration, no compounding).
    GLM families: LOOSE (iterative PIRLS + Newton, differences compound).
    """
    if family_name == "gaussian":
        return MODERATE
    return LOOSE


# ---------------------------------------------------------------------------
# Assertion collection
# ---------------------------------------------------------------------------


class _AssertCollector:
    """Collect assertion failures across related checks."""

    def __init__(self) -> None:
        self.failures: list[str] = []

    def check(self, name: str, fn: Callable[[], None]) -> None:
        """Run one assertion block and retain its failure message."""
        try:
            fn()
        except AssertionError as exc:
            self.failures.append(f"{name}: {exc}")

    def raise_if_any(self, label: str) -> None:
        """Raise one readable assertion if any collected check failed."""
        if self.failures:
            raise AssertionError(
                f"{label} failed:\n  - " + "\n  - ".join(self.failures)
            )


def check_that(condition: bool, message: str) -> None:
    """Boolean assertion usable inside a lambda passed to ``_AssertCollector``.

    Lambdas cannot contain ``assert`` statements, so non-numeric checks
    (``converged``, ``mu > 0``, ``theta is not None``) raise ``AssertionError``
    through this helper instead of ``np.testing.assert_`` (which ruff flags).
    """
    if not condition:
        raise AssertionError(message)


# ---------------------------------------------------------------------------
# SmoothSpec factory
# ---------------------------------------------------------------------------


def make_smooth_spec(
    variables: list[str],
    bs: str = "tp",
    k: int = 10,
    by: str | None = None,
    smooth_type: str = "s",
    **extra_args: object,
) -> SmoothSpec:
    """Create a ``SmoothSpec`` for testing."""
    return SmoothSpec(
        variables=variables,
        bs=bs,
        k=k,
        by=by,
        smooth_type=smooth_type,
        extra_args=dict(extra_args),
    )


# ---------------------------------------------------------------------------
# FittingData construction (used across test modules)
# ---------------------------------------------------------------------------


def _setup_fd(formula: str, data: pd.DataFrame, family):
    """Build FittingData from formula + data.

    Shared helper for tests that need a FittingData without going
    through the full GAM API.
    """
    from jaxgam.fitting.data import FittingData
    from jaxgam.formula.design import ModelSetup
    from jaxgam.formula.parser import parse_formula

    spec = parse_formula(formula)
    setup = ModelSetup.build(spec, data)
    return FittingData.from_setup(setup, family)


# ---------------------------------------------------------------------------
# Private data generators (used by conftest and test-local fixtures)
# ---------------------------------------------------------------------------


def _make_nb_data(
    n: int = 200,
    seed: int = SEED,
    true_theta: float = 2.0,
) -> pd.DataFrame:
    """Generate single-predictor NB count data with known theta.

    Used across NB test modules (fitting, custom_jvp, results).
    """
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 1, n)
    eta = np.sin(2 * np.pi * x) + 0.5
    mu = np.exp(eta)
    y = rng.negative_binomial(
        n=true_theta, p=true_theta / (mu + true_theta), size=n
    ).astype(float)
    return pd.DataFrame({"x": x, "y": y})


def _generate_family_data(family_name: str, n: int | None = None) -> pd.DataFrame:
    """Generate single-predictor synthetic data for a given family.

    Parameters
    ----------
    family_name : str
        One of "gaussian", "binomial", "poisson", "gamma".
    n : int or None
        Sample size.  If ``None``, defaults to 200 (300 for binomial).
    """
    rng = np.random.default_rng(SEED)
    if n is None:
        n = 200 if family_name != "binomial" else 300
    x = rng.uniform(0, 1, n)

    if family_name == "gaussian":
        y = np.sin(2 * np.pi * x) + rng.normal(0, 0.3, n)
    elif family_name == "binomial":
        eta = 2 * np.sin(2 * np.pi * x)
        prob = 1.0 / (1.0 + np.exp(-eta))
        y = rng.binomial(1, prob, n).astype(float)
    elif family_name == "poisson":
        eta = np.sin(2 * np.pi * x) + 0.5
        y = rng.poisson(np.exp(eta)).astype(float)
    elif family_name == "gamma":
        eta = 0.5 * np.sin(2 * np.pi * x) + 1.0
        mu = np.exp(eta)
        y = rng.gamma(5.0, scale=mu / 5.0, size=n)
    elif family_name == "nb":
        eta = np.sin(2 * np.pi * x) + 0.5
        mu = np.exp(eta)
        theta = 2.0
        y = rng.negative_binomial(n=theta, p=theta / (mu + theta), size=n).astype(float)
    else:
        raise ValueError(f"Unknown family: {family_name}")

    return pd.DataFrame({"x": x, "y": y})


# ---------------------------------------------------------------------------
# Shared inference prediction cases
# ---------------------------------------------------------------------------


def _inference_factor_by_case(n: int = 120) -> tuple[Any, ...]:
    """Binomial factor-by case shared by predictor and result-mode gates."""
    rng = np.random.default_rng(SEED)
    levels = ["a", "b"]
    x = rng.uniform(size=n)
    fac_values = rng.choice(levels, size=n)
    eta = np.where(fac_values == "a", np.sin(2 * np.pi * x), 0.5 * x)
    data = pd.DataFrame(
        {
            "x": x,
            "fac": pd.Categorical(fac_values, categories=levels),
            "y": rng.binomial(1, 1.0 / (1.0 + np.exp(-eta))).astype(float),
        }
    )
    newdata = pd.DataFrame(
        {
            "x": rng.uniform(size=30),
            "fac": pd.Categorical(rng.choice(levels, size=30), categories=levels),
        }
    )
    return (
        "y ~ s(x, by=fac, k=5, bs='cr') + fac",
        data,
        newdata,
        "binomial",
        [1.0, 1.0],
        None,
        None,
    )


def _inference_tensor_case(kind: str, n: int = 120) -> tuple[Any, ...]:
    """Gaussian tensor case shared by predictor and result-mode gates."""
    rng = np.random.default_rng(SEED + (1 if kind == "te" else 2))
    x1 = rng.uniform(size=n)
    x2 = rng.uniform(size=n)
    y = np.sin(2 * np.pi * x1) + 0.5 * x2 + rng.normal(scale=0.2, size=n)
    data = pd.DataFrame({"x1": x1, "x2": x2, "y": y})
    newdata = pd.DataFrame({"x1": rng.uniform(size=30), "x2": rng.uniform(size=30)})
    return (
        f"y ~ {kind}(x1, x2, k=4)",
        data,
        newdata,
        "gaussian",
        [1.0, 1.0],
        None,
        None,
    )


def _inference_one_dimensional_case(
    family: str | Any,
    *,
    offset: bool = False,
) -> tuple[Any, ...]:
    """One-dimensional family case shared by inference behavior gates."""
    family_name = family if isinstance(family, str) else family.family_name.lower()
    data = _generate_family_data(family_name, n=120)
    rng = np.random.default_rng(SEED + 100)
    newdata = pd.DataFrame({"x": rng.uniform(size=30)})
    train_offset = np.linspace(0.1, 0.3, len(data)) if offset else None
    predict_offset = np.linspace(0.2, 0.4, len(newdata)) if offset else None
    return (
        "y ~ s(x, k=6, bs='cr')",
        data,
        newdata,
        family,
        [1.0],
        train_offset,
        predict_offset,
    )


def _inference_equivalence_cases() -> dict[str, tuple[Any, ...]]:
    """Internal full/predictor/inference equivalence zoo."""
    from jaxgam.families.standard import Gamma

    return {
        "gaussian-s": _inference_one_dimensional_case("gaussian"),
        "binomial-factor-by": _inference_factor_by_case(),
        "poisson-offset": _inference_one_dimensional_case("poisson", offset=True),
        "negative-binomial": _inference_one_dimensional_case("nb"),
        "gamma-log": _inference_one_dimensional_case(Gamma(link="log")),
        "tensor-te": _inference_tensor_case("te"),
        "tensor-ti": _inference_tensor_case("ti"),
    }


def _inference_r_cases() -> dict[
    str, tuple[str, str, pd.DataFrame, pd.DataFrame, Any, str]
]:
    """Direct-mgcv inference cases supported by the rpy2 bridge."""
    from jaxgam.families.negative_binomial import NegativeBinomial
    from jaxgam.families.standard import Gamma

    rng = np.random.default_rng(SEED)
    n = 180
    x1 = rng.uniform(size=n)
    x2 = rng.uniform(size=n)
    tensor_data = pd.DataFrame(
        {
            "x1": x1,
            "x2": x2,
            "y": np.sin(2 * np.pi * x1) + 0.5 * x2 + rng.normal(0, 0.25, n),
        }
    )
    tensor_new = pd.DataFrame({"x1": rng.uniform(size=40), "x2": rng.uniform(size=40)})
    factor_formula, factor_data, factor_new, *_ = _inference_factor_by_case(n=180)
    one_d_new = pd.DataFrame({"x": rng.uniform(size=40)})

    return {
        "tensor": (
            "y ~ te(x1, x2, k=5)",
            "y ~ te(x1, x2, k=c(5,5))",
            tensor_data,
            tensor_new,
            "gaussian",
            "gaussian",
        ),
        "factor-by": (
            factor_formula,
            factor_formula,
            factor_data,
            factor_new,
            "binomial",
            "binomial",
        ),
        "negative-binomial": (
            "y ~ s(x, k=8, bs='cr')",
            "y ~ s(x, k=8, bs='cr')",
            _generate_family_data("nb", n=180),
            one_d_new,
            NegativeBinomial(),
            "nb",
        ),
        "gamma-log": (
            "y ~ s(x, k=8, bs='cr')",
            "y ~ s(x, k=8, bs='cr')",
            _generate_family_data("gamma", n=180),
            one_d_new,
            Gamma(link="log"),
            "gamma_log",
        ),
    }


# ---------------------------------------------------------------------------
# Per-platform reviewed replay bounds
# ---------------------------------------------------------------------------

# Platform classes for reviewed replay counts. On x86_64 the class is decided
# from the CPU's own flags (AVX-512F), never assumed from the OS or runner.
PLATFORM_CLASSES = ("macos-arm64", "linux-aarch64", "x86_64", "x86_64-avx512")


def _x86_cpu_flags() -> frozenset[str] | None:
    """Lower-case CPU feature flags on x86_64, or ``None`` when unreadable."""
    import subprocess
    import sys

    if sys.platform.startswith("linux"):
        try:
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                for line in handle:
                    if line.startswith("flags"):
                        return frozenset(line.partition(":")[2].split())
        except OSError:
            return None
        return None
    if sys.platform == "darwin":
        try:
            text = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.features", "machdep.cpu.leaf7_features"],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        except (OSError, subprocess.CalledProcessError):
            return None
        return frozenset(token.lower().replace(".", "_") for token in text.split())
    return None


def cpu_platform_class() -> str:
    """Return the running platform's replay class from the CPU's own flags.

    ``macos-arm64`` / ``linux-aarch64`` for 64-bit Arm; on x86_64,
    ``x86_64-avx512`` when the CPU reports AVX-512F (``avx512f`` in
    ``/proc/cpuinfo`` flags, ``AVX512F`` in macOS ``machdep.cpu`` features)
    and ``x86_64`` otherwise. Anything else, or unreadable x86 flags, returns
    a descriptive class that has no reviewed entry.
    """
    import platform
    import sys

    machine = platform.machine().lower()
    if machine in ("arm64", "aarch64"):
        if sys.platform == "darwin":
            return "macos-arm64"
        if sys.platform.startswith("linux"):
            return "linux-aarch64"
        return f"{sys.platform}-{machine}"
    if machine in ("x86_64", "amd64"):
        flags = _x86_cpu_flags()
        if flags is None:
            return "x86_64-unreadable-flags"
        return "x86_64-avx512" if "avx512f" in flags else "x86_64"
    return f"{sys.platform}-{machine}"


# Key of a per-case entry: every platform class without its own entry must
# need no replay (bound 0), exactly as if the case were unlisted for it. Used
# when a case needs a reviewed bound on some classes only.
OTHER_CLASSES_NEED_NONE = "other classes need none"

# Every replay-bound check of the session, printed by tests/conftest.py's
# terminal summary so a run reports its measured counts even when it passes.
REPLAY_COUNT_RECORDS: list[tuple[str, str, str, int, int | None]] = []


def check_replay_bound(
    table_name: str,
    table: dict[Any, dict[str, tuple[int, int]]],
    key: Any,
    measured: int,
) -> None:
    """Assert a measured replay count against the running platform's entry.

    ``table`` maps a case to ``{platform class: (bound, measured count)}``.
    A case absent from the table must need no replay on any platform (bound
    0). A listed case on a platform class without a reviewed entry fails and
    reports the measured count, so the entry can be reviewed from it, unless
    the case lists :data:`OTHER_CLASSES_NEED_NONE`, whose bound must be 0.
    """
    platform_class = cpu_platform_class()
    if key not in table:
        bound: int | None = 0
    else:
        entries = table[key]
        if entries.get(OTHER_CLASSES_NEED_NONE, (0, 0))[0] != 0:
            raise ValueError(f"{table_name}: {OTHER_CLASSES_NEED_NONE} must be 0")
        entry = entries.get(platform_class, entries.get(OTHER_CLASSES_NEED_NONE))
        bound = None if entry is None else entry[0]
    REPLAY_COUNT_RECORDS.append(
        (table_name, repr(key), platform_class, measured, bound)
    )
    if bound is None:
        raise AssertionError(
            f"{table_name}: no reviewed replay bound for {key!r} on platform class "
            f"{platform_class}; measured {measured} replayed decisions"
        )
    if measured > bound:
        raise AssertionError(
            f"{table_name}: {measured} replayed decisions for {key!r} on "
            f"{platform_class}; reviewed bound {bound}"
        )
