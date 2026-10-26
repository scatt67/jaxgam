"""Declared source-LAPACK provider and cubic shrinkage parity gates."""

from __future__ import annotations

import hashlib
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import scipy

from jaxgam.smooths import source_lapack
from jaxgam.smooths.cubic import CubicRegressionSmooth
from tests.helpers import _AssertCollector, r_links_pinned_openblas
from tests.tolerances import MODERATE, STRICT

_ACCELERATE_CUBIC_MODERATE_DIGESTS = {
    (1, 32): "33611ede66607446ae8787994bcffa8138bf0e0e07b749bd4ce9bc0d865accbc",
    (3, 64): "ccab68ff65ffb3defb8290b275f8f958a3ea926359fb59a13572eac518276344",
    (4, 64): "db292f5db932f26bef58f0c4e331506b9df68ba0ad7a097684f60ba3722baa52",
    (5, 65): "1b295cbe443849e1129d416617ccf0eb17fa1b82464864d4b60afdff8c0ab3dc",
    (7, 96): "d995c05bdae128fdd1d3f795dd18cabf56c42c3f37f9c33fc81f37b0d9479c50",
}


def _reviewed_raw_cubic_tolerance(index, knots, raw, shrink):
    """Limit Mac EVR allowance to the five audited source inputs and controls."""
    if not source_lapack.scipy_uses_accelerate_lapack():
        return STRICT
    if scipy.__version__ != "1.17.0" or shrink != 0.1:
        return STRICT
    fingerprint = hashlib.sha256(
        np.asarray(knots, dtype="<f8").tobytes()
        + np.asarray(raw, dtype="<f8").tobytes()
        + np.asarray(shrink, dtype="<f8").tobytes()
    ).hexdigest()
    if fingerprint == _ACCELERATE_CUBIC_MODERATE_DIGESTS.get((index, len(knots))):
        return MODERATE
    return STRICT


def test_workspace_bound_validation_and_reject_before_provider(monkeypatch) -> None:
    with pytest.raises(ValueError, match="integer >= 3"):
        source_lapack.cubic_shrinkage_workspace_bound(True)
    with pytest.raises(ValueError, match="integer >= 3"):
        source_lapack.cubic_shrinkage_workspace_bound(2)
    required = source_lapack.cubic_shrinkage_workspace_bound(5)
    finite = Mock(side_effect=AssertionError("finite mask must not be allocated"))
    provider = Mock(side_effect=AssertionError("provider must not be loaded"))
    monkeypatch.setattr(source_lapack.np, "isfinite", finite)
    monkeypatch.setattr(source_lapack, "_source_lapack", provider)
    with pytest.raises(MemoryError, match=rf"requires {required} workspace bytes"):
        source_lapack.cubic_shrinkage_penalty(
            np.eye(5), available_workspace_bytes=required - 1
        )
    finite.assert_not_called()
    provider.assert_not_called()


def test_accelerate_evr_rejects_unbounded_workspace_before_matrix_copy(
    monkeypatch,
) -> None:
    monkeypatch.setattr(source_lapack, "scipy_uses_accelerate_lapack", lambda: True)
    monkeypatch.setattr(
        source_lapack.linalg.lapack,
        "dsyevr_lwork",
        lambda k, **_kwargs: (65 * k, 10 * k, 0),
    )
    eigen = Mock(side_effect=AssertionError("eigensolve must not allocate"))
    monkeypatch.setattr(source_lapack.linalg, "eigh", eigen)
    with pytest.raises(RuntimeError, match="lwork"):
        source_lapack.cubic_shrinkage_penalty(np.eye(5))
    eigen.assert_not_called()


def test_lanczos_workspace_rejects_before_mask_or_provider(monkeypatch) -> None:
    required = source_lapack.source_lanczos_workspace_bound(7, 4)
    assert required == 8 * (7 * 7 * 7 + 2 * 7 * 4 + 19 * 7 + 35)
    finite = Mock(side_effect=AssertionError("finite mask must not be allocated"))
    provider = Mock(side_effect=AssertionError("provider must not be loaded"))
    monkeypatch.setattr(source_lapack.np, "isfinite", finite)
    monkeypatch.setattr(source_lapack, "_source_lanczos", provider)
    with pytest.raises(MemoryError, match=rf"requires {required} workspace bytes"):
        source_lapack.source_lanczos(
            np.eye(7), 4, available_workspace_bytes=required - 1
        )
    finite.assert_not_called()
    provider.assert_not_called()


@pytest.mark.parametrize(("n", "k"), [(True, 1), (0, 1), (4, 0), (3, 4)])
def test_lanczos_workspace_dimension_validation(n, k) -> None:
    with pytest.raises(ValueError, match="Source Lanczos"):
        source_lapack.source_lanczos_workspace_bound(n, k)


@pytest.mark.parametrize(
    ("value", "name"),
    [
        (np.nan, "lwork"),
        (np.inf, "lwork"),
        (2.5, "lwork"),
        (0.0, "lwork"),
        (float(source_lapack._LAPACK_INT_MAX) + 1.0, "lwork"),
        (129.0, "liwork"),
    ],
)
def test_workspace_query_requires_finite_integral_lp64_value(value, name) -> None:
    cap = 128
    with pytest.raises(RuntimeError, match=name):
        source_lapack._query_length(value, name=name, cap=cap)


def test_windows_provider_uses_sole_package_local_dll(monkeypatch, tmp_path) -> None:
    package = tmp_path / "scipy_openblas32"
    library_dir = package / "lib"
    library_dir.mkdir(parents=True)
    origin = package / "__init__.py"
    origin.touch()
    library = library_dir / "libscipy_openblas.dll"
    library.touch()
    monkeypatch.setattr(
        source_lapack.importlib_util,
        "find_spec",
        lambda _package: SimpleNamespace(origin=str(origin)),
    )
    monkeypatch.setattr(source_lapack.sys, "platform", "win32")
    assert (
        source_lapack._provider_library_path(
            "scipy-openblas32",
            darwin_name="libscipy_openblas.dylib",
            linux_name="libscipy_openblas.so",
        )
        == library.resolve()
    )


def test_declared_provider_version_config_and_lp64_symbols() -> None:
    assert "scipy_openblas32" not in sys.modules
    source_lapack._source_lapack.cache_clear()
    provider = source_lapack._source_lapack()
    assert provider.module.__version__ == "0.3.26.0.4"
    assert provider.module.get_openblas_config().startswith("OpenBLAS 0.3.26 ")
    assert provider.dsyevr_work.restype is source_lapack.ctypes.c_int
    assert provider.dgemm.restype is None
    assert "scipy_openblas32" not in sys.modules
    assert provider.dptsv.restype is source_lapack.ctypes.c_int
    for name in ("dstedc_work", "dgesdd_work"):
        assert getattr(provider, name).restype is source_lapack.ctypes.c_int
    for name in ("ddot", "dnrm2"):
        assert getattr(provider, name).restype is source_lapack.ctypes.c_double
    for name in ("dsymv", "daxpy", "dgemv", "dscal"):
        assert getattr(provider, name).restype is None


def test_natural_cubic_system_solve_matches_dense_and_preserves_inputs() -> None:
    diagonal = np.array([2.0, 3.0, 4.0], dtype=np.float64)
    off_diagonal = np.array([0.25, -0.5], dtype=np.float64)
    rhs = np.array(
        [[1.0, -2.0], [0.5, 3.0], [-1.0, 4.0]],
        dtype=np.float64,
    )
    diagonal_before = diagonal.copy()
    off_diagonal_before = off_diagonal.copy()
    rhs_before = rhs.copy()
    system = np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)

    actual = source_lapack.natural_cubic_system_solve(diagonal, off_diagonal, rhs)

    np.testing.assert_allclose(
        actual, np.linalg.solve(system, rhs), rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_array_equal(diagonal, diagonal_before)
    np.testing.assert_array_equal(off_diagonal, off_diagonal_before)
    np.testing.assert_array_equal(rhs, rhs_before)
    assert actual.flags.f_contiguous


@pytest.mark.parametrize(
    ("diagonal", "off_diagonal", "rhs", "error", "message"),
    [
        (
            np.ones(2, dtype=np.float32),
            np.ones(1),
            np.ones((2, 1)),
            TypeError,
            "diagonal.*float64",
        ),
        (
            np.ones(2),
            np.ones(1, dtype=np.float32),
            np.ones((2, 1)),
            TypeError,
            "off-diagonal.*float64",
        ),
        (
            np.ones(2),
            np.ones(1),
            np.ones((2, 1), dtype=np.float32),
            TypeError,
            "right-hand side.*float64",
        ),
        (np.ones(2), np.ones(2), np.ones((2, 1)), ValueError, "wrong shape"),
        (np.ones(2), np.ones(1), np.ones(2), ValueError, "wrong shape"),
        (
            np.array([1.0, np.nan]),
            np.ones(1),
            np.ones((2, 1)),
            ValueError,
            "finite",
        ),
    ],
)
def test_natural_cubic_system_validation(
    diagonal, off_diagonal, rhs, error, message
) -> None:
    with pytest.raises(error, match=message):
        source_lapack.natural_cubic_system_solve(diagonal, off_diagonal, rhs)


def test_natural_cubic_system_rejects_non_positive_definite() -> None:
    with pytest.raises(np.linalg.LinAlgError, match="not positive definite"):
        source_lapack.natural_cubic_system_solve(
            np.array([1.0, 0.0]),
            np.array([1.0]),
            np.ones((2, 1)),
        )


@pytest.mark.parametrize(
    ("penalty", "message"),
    [
        (np.eye(4, dtype=np.float32), "float64"),
        (np.ones(4), "square"),
        (np.eye(2), "square"),
        (np.array([[1.0, 0.0, 0.0], [0.0, np.nan, 0.0], [0.0, 0.0, 1.0]]), "finite"),
    ],
)
def test_input_validation(penalty, message) -> None:
    error = TypeError if message == "float64" else ValueError
    with pytest.raises(error, match=message):
        source_lapack.cubic_shrinkage_penalty(penalty)


@pytest.mark.parametrize("workspace", [True, -1, 1.5])
def test_workspace_value_validation(workspace) -> None:
    with pytest.raises(ValueError, match="nonnegative integer"):
        source_lapack.cubic_shrinkage_penalty(
            np.eye(3), available_workspace_bytes=workspace
        )


@pytest.mark.parametrize("shrink", [0.0, -0.1, np.inf, np.nan])
def test_shrink_value_validation(shrink) -> None:
    with pytest.raises(ValueError, match="finite and positive"):
        source_lapack.cubic_shrinkage_penalty(np.eye(3), shrink=shrink)


def test_provider_rejects_wrong_version_before_library_lookup(monkeypatch) -> None:
    monkeypatch.setattr(
        source_lapack.importlib_metadata, "version", lambda _package: "0.3.27.0"
    )
    source_lapack._source_lapack.cache_clear()
    with pytest.raises(RuntimeError, match=r"required; found 0\.3\.27\.0"):
        source_lapack._source_lapack()
    source_lapack._source_lapack.cache_clear()


def test_provider_rejects_wrong_openblas_config(monkeypatch) -> None:
    config = Mock(return_value=b"OpenBLAS 0.3.30 DYNAMIC_ARCH")
    library = SimpleNamespace(scipy_openblas_get_config=config)
    monkeypatch.setattr(
        source_lapack.importlib_metadata,
        "version",
        lambda _package: "0.3.26.0.4",
    )
    monkeypatch.setattr(
        source_lapack,
        "_provider_library_path",
        lambda *_args, **_kwargs: source_lapack.Path("/declared/provider"),
    )
    monkeypatch.setattr(source_lapack.ctypes, "CDLL", lambda *_args, **_kwargs: library)
    with pytest.raises(RuntimeError, match=r"unexpected config 'OpenBLAS 0\.3\.30"):
        source_lapack._declared_provider(
            "scipy-openblas32",
            version="0.3.26.0.4",
            prefix="OpenBLAS 0.3.26 ",
            darwin_name="libscipy_openblas.dylib",
            linux_name="libscipy_openblas.so",
            library_name="scipy_openblas",
            config_symbol="scipy_openblas_get_config",
            ilp64=False,
        )


def test_declared_provider_rejects_wrong_integer_abi(monkeypatch) -> None:
    config = Mock(return_value=b"OpenBLAS 0.3.30 USE64BITINT DYNAMIC_ARCH")
    library = SimpleNamespace(scipy_openblas_get_config=config)
    monkeypatch.setattr(
        source_lapack.importlib_metadata,
        "version",
        lambda _package: "0.3.30.0.2",
    )
    monkeypatch.setattr(
        source_lapack,
        "_provider_library_path",
        lambda *_args, **_kwargs: source_lapack.Path("/declared/provider"),
    )
    monkeypatch.setattr(source_lapack.ctypes, "CDLL", lambda *_args, **_kwargs: library)
    with pytest.raises(RuntimeError, match="required LP64 ABI"):
        source_lapack._declared_provider(
            "scipy-openblas32",
            version="0.3.30.0.2",
            prefix="OpenBLAS 0.3.30 ",
            darwin_name="libscipy_openblas.dylib",
            linux_name="libscipy_openblas.so",
            library_name="scipy_openblas",
            config_symbol="scipy_openblas_get_config",
            ilp64=False,
        )


def test_source_lanczos_selected_pairs_satisfy_the_input_operator() -> None:
    assert "scipy_openblas64" not in sys.modules
    source_lapack._source_lanczos.cache_clear()
    provider = source_lapack._source_lanczos()
    assert provider.module.__version__ == "0.3.30.0.2"
    assert provider.module.get_openblas_config().startswith("OpenBLAS 0.3.30 ")
    assert provider.dsyevd_work.restype is source_lapack.ctypes.c_int64
    assert "scipy_openblas64" not in sys.modules
    rng = np.random.default_rng(1022026)
    raw = rng.normal(size=(19, 19))
    matrix = (raw + raw.T) * 0.5
    values, vectors = source_lapack.source_lanczos(matrix, 7)
    np.testing.assert_allclose(
        vectors.T @ vectors, np.eye(7), rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        matrix @ vectors,
        vectors * values[np.newaxis, :],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_cubic_shrinkage_raw_penalties_match_pinned_r_across_block_boundary(
    r_bridge,
) -> None:
    """Exact raw source penalties retain the audited provider-specific gate."""
    if r_bridge.mode != "rpy2":
        pytest.skip("Raw identical-input eigen oracle requires rpy2.")
    rng = np.random.default_rng(92026)
    cases = [np.linspace(-1.0, 1.0, 5)]
    for k in (32, 64, 65, 96):
        cases.append(np.linspace(-2.0, 3.0, k))
        cases.append(np.cumsum(rng.uniform(0.01, 2.0, size=k)) - 4.0)
    collector = _AssertCollector()
    for index, knots in enumerate(cases):
        _, raw = CubicRegressionSmooth._compute_natural_f_and_penalty(knots)
        k = len(knots)
        expected = r_bridge.source_cubic_shrinkage_penalty(raw)
        actual = source_lapack.cubic_shrinkage_penalty(raw)
        tolerance = _reviewed_raw_cubic_tolerance(index, knots, raw, 0.1)
        collector.check(
            f"case{index}/k{k}",
            lambda actual=actual, expected=expected, tolerance=tolerance: (
                np.testing.assert_allclose(
                    actual, expected, rtol=tolerance.rtol, atol=tolerance.atol
                )
            ),
        )
    collector.raise_if_any("source cubic shrinkage")


def test_unreviewed_raw_cubic_inputs_remain_strict(monkeypatch) -> None:
    monkeypatch.setattr(source_lapack, "scipy_uses_accelerate_lapack", lambda: True)
    monkeypatch.setattr(scipy, "__version__", "1.17.0")
    knots = np.linspace(-2.0, 3.0, 32)
    _, raw = CubicRegressionSmooth._compute_natural_f_and_penalty(knots)
    fingerprint = hashlib.sha256(
        np.asarray(knots, dtype="<f8").tobytes()
        + np.asarray(raw, dtype="<f8").tobytes()
        + np.asarray(0.1, dtype="<f8").tobytes()
    ).hexdigest()
    monkeypatch.setitem(_ACCELERATE_CUBIC_MODERATE_DIGESTS, (1, 32), fingerprint)
    assert _reviewed_raw_cubic_tolerance(1, knots, raw, 0.1) is MODERATE
    changed = raw.copy()
    changed[0, 0] = np.nextafter(changed[0, 0], np.inf)
    assert _reviewed_raw_cubic_tolerance(1, knots, changed, 0.1) is STRICT
    assert _reviewed_raw_cubic_tolerance(1, knots, raw, 0.11) is STRICT
    assert _reviewed_raw_cubic_tolerance(2, knots, raw, 0.1) is STRICT


def test_source_pivoted_qr_follows_mgcv_qr_contract_and_preserves_input() -> None:
    """DGEQP3 on the provider: free pivots, zero-based order, R in pivot order."""
    from scipy import linalg

    rng = np.random.default_rng(3107)
    generic = rng.standard_normal((17, 6))
    original = generic.copy()
    upper, pivots = source_lapack.source_pivoted_qr(generic)
    np.testing.assert_array_equal(generic, original)
    assert upper.shape == (6, 6)
    np.testing.assert_array_equal(np.tril(upper, -1), 0.0)
    np.testing.assert_array_equal(pivots, linalg.qr(generic, pivoting=True)[2])
    pivoted = generic[:, pivots]
    np.testing.assert_allclose(
        upper.T @ upper, pivoted.T @ pivoted, rtol=STRICT.rtol, atol=STRICT.atol
    )
    aliased = generic.copy()
    aliased[:, 5] = aliased[:, 2]
    aliased_upper, aliased_pivots = source_lapack.source_pivoted_qr(aliased)
    assert len({2, 5} & set(aliased_pivots[:5].tolist())) == 1
    assert abs(aliased_upper[-1, -1]) <= 1e-12 * abs(aliased_upper[0, 0])
    wide_upper, wide_pivots = source_lapack.source_pivoted_qr(generic.T.copy())
    assert wide_upper.shape == (6, 17)
    assert sorted(wide_pivots.tolist()) == list(range(17))


@pytest.mark.parametrize(
    "matrix",
    [np.ones((3, 2), dtype=np.float32), np.empty((0, 2)), np.array([[np.nan]])],
)
def test_source_pivoted_qr_rejects_invalid_matrices(matrix) -> None:
    with pytest.raises(ValueError, match="Source pivoted QR"):
        source_lapack.source_pivoted_qr(matrix)


def test_source_frobenius_norm_is_one_sequential_column_major_sum() -> None:
    assert source_lapack.source_frobenius_norm(np.array([[3.0], [4.0]])) == 5.0
    assert source_lapack.source_frobenius_norm(np.empty((0, 3))) == 0.0
    values = np.array([[1e16, 1.0], [-1e16, 1.0]])
    # Column-major order: ((1e32 + 1e32) + 1) + 1, sequentially in binary64.
    expected = np.sqrt(((1e32 + 1e32) + 1.0) + 1.0)
    assert source_lapack.source_frobenius_norm(values) == expected


def test_each_provider_library_runs_single_threaded_after_load() -> None:
    """Both wheel libraries are pinned to one OpenBLAS thread when loaded."""
    for loader in (source_lapack._source_lapack, source_lapack._source_lanczos):
        loader.cache_clear()
        provider = loader()
        assert provider.module.num_threads() == 1, provider.module.package
        loader.cache_clear()


_VENDORED_THREADS_SCRIPT = r"""
import ctypes
import json
import sys
from pathlib import Path

import numpy
import scipy
import scipy.linalg  # loads SciPy's vendored BLAS/LAPACK

def vendored():
    found = {}
    for module in (numpy, scipy):
        root = Path(module.__file__).resolve().parent
        candidates = list(root.parent.glob(f"{module.__name__}.libs/*openblas*"))
        candidates += list(root.glob(".dylibs/*openblas*"))
        for path in candidates:
            library = ctypes.CDLL(str(path), ctypes.RTLD_LOCAL)
            for suffix in ("", "64_"):
                name = f"scipy_openblas_get_num_threads{suffix}"
                if hasattr(library, name):
                    getter = getattr(library, name)
                    getter.restype = ctypes.c_int
                    found[f"{module.__name__}:{path.name}"] = getter
    return found

getters = vendored()
before = {name: getter() for name, getter in getters.items()}
import jaxgam.smooths.source_lapack as source_lapack
providers = [source_lapack._source_lapack(), source_lapack._source_lanczos()]
after = {name: getter() for name, getter in getters.items()}
print(json.dumps({
    "jaxgam": source_lapack.__file__,
    "before": before,
    "after": after,
    "providers": [provider.module.num_threads() for provider in providers],
    "lapack": scipy.__config__.CONFIG["Build Dependencies"]["lapack"]["name"],
}))
"""


def test_provider_pin_leaves_numpy_and_scipy_blas_threads_unchanged() -> None:
    """NumPy's and SciPy's vendored BLAS keep the caller's thread count."""
    import json
    import os
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    env = dict(os.environ, OPENBLAS_NUM_THREADS="3", PYTHONPATH=str(root))
    completed = subprocess.run(
        [sys.executable, "-c", _VENDORED_THREADS_SCRIPT],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    assert report["jaxgam"].startswith(str(root)), report["jaxgam"]
    assert report["providers"] == [1, 1]
    assert report["after"] == report["before"]
    if report["lapack"] == "Accelerate":
        # Accelerate-backed NumPy/SciPy vendor no OpenBLAS thread control.
        assert not report["before"]
    else:
        assert report["before"], "no vendored NumPy/SciPy OpenBLAS was found"
        assert set(report["before"].values()) == {3}


_PROVIDER_OUTPUT_SCRIPT = r"""
import hashlib
import json

import numpy as np

import jaxgam.smooths.source_lapack as source_lapack
from jaxgam.formula.rank_coordinates import rank_revealing_coordinate_map
from jaxgam.smooths.source_penalty_coordinates import source_symmetric_eigh


def digest(*values):
    sha = hashlib.sha256()
    for value in values:
        sha.update(np.ascontiguousarray(value).tobytes())
    return sha.hexdigest()


# Elementwise inputs only: no NumPy/SciPy BLAS call precedes the provider.
k = 96
i, j = np.indices((k, k), dtype=np.float64)
penalty = np.exp(-np.abs(i - j) / 7.0) + 0.01 * np.cos(i * j / 11.0)
penalty = 0.5 * (penalty + penalty.T)
n = 150
a, b = np.indices((n, n), dtype=np.float64)
kernel = np.exp(-(((a - b) / 23.0) ** 2))
rows = np.arange(240, dtype=np.float64)[:, None]
columns = np.arange(9, dtype=np.float64)[None, :]
design = np.sin(0.37 * rows * (columns + 1.0)) + 0.1 * columns
design[:, 8] = design[:, 7]  # an exactly aliased pair, as in x + s(x, bs='gp')
root = np.zeros((5, 9))
root[np.arange(5), np.arange(5)] = 1.0 + np.arange(5)
values, vectors = source_lapack.source_lanczos(kernel, 20)
eig_values, eig_vectors = source_symmetric_eigh(penalty)
coordinate_map = rank_revealing_coordinate_map(design, root, retained_rank=8)
report = {
    "threads": [
        source_lapack._source_lapack().module.num_threads(),
        source_lapack._source_lanczos().module.num_threads(),
    ],
    "cubic_shrinkage": digest(source_lapack.cubic_shrinkage_penalty(penalty)),
    "natural_cubic": digest(
        source_lapack.natural_cubic_system_solve(
            4.0 + np.arange(40.0), np.ones(39), np.cos(np.indices((40, 3))[0] * 1.0)
        )
    ),
    "lanczos": digest(values, vectors),
    "symmetric_eigen": digest(eig_values, eig_vectors),
    "rank_gauge": digest(coordinate_map.keep, coordinate_map.drop),
    "drop": coordinate_map.drop.tolist(),
}
print(json.dumps(report))
"""


def _provider_outputs_with_threads(threads: str) -> dict:
    import json
    import os
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    env = dict(
        os.environ,
        OPENBLAS_NUM_THREADS=threads,
        OMP_NUM_THREADS=threads,
        PYTHONPATH=str(root),
    )
    completed = subprocess.run(
        [sys.executable, "-c", _PROVIDER_OUTPUT_SCRIPT],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    assert completed.returncode == 0, completed.stderr
    return json.loads(completed.stdout.strip().splitlines()[-1])


def test_provider_outputs_do_not_depend_on_caller_openblas_threads() -> None:
    """Cubic shrinkage, Lanczos and the rank gauge ignore OPENBLAS_NUM_THREADS."""
    single = _provider_outputs_with_threads("1")
    many = _provider_outputs_with_threads("8")
    assert single["threads"] == many["threads"] == [1, 1]
    assert many == single


def test_thread_pin_rejects_missing_controls_and_an_unaccepted_setting() -> None:
    with pytest.raises(RuntimeError, match="thread-count controls"):
        source_lapack._pin_single_thread(SimpleNamespace(), "scipy-openblas32", "")
    library = SimpleNamespace(
        scipy_openblas_set_num_threads64_=Mock(),
        scipy_openblas_get_num_threads64_=Mock(return_value=4),
    )
    with pytest.raises(RuntimeError, match="single-thread setting"):
        source_lapack._pin_single_thread(library, "scipy-openblas64", "64_")
    library.scipy_openblas_set_num_threads64_.assert_called_once_with(1)


def _rlanczos_kernel(case: str) -> tuple[np.ndarray, int]:
    """Symmetric input matrix and Ritz count of one Rlanczos parity case.

    The Gaussian-process cases are smooth.construct.gp's knot kernels at its
    retained rank; the thin-plate kernel on clustered knots is indefinite
    with a dominant negative eigenvalue, so Rlanczos retains pairs from both
    ends of the spectrum. All are built from runtime-generated data.
    """
    from jaxgam.smooths.gaussian_process import GaussianProcessSmooth
    from jaxgam.smooths.tprs import tps_semi_kernel
    from tests.helpers import make_smooth_spec

    rng = np.random.default_rng(102626)
    clustered = np.sort(np.r_[rng.normal(-1, 0.05, 30), rng.normal(1.2, 0.08, 31)])
    if case == "tprs/clustered":
        return tps_semi_kernel(np.abs(clustered[:, None] - clustered), 2, 1), 4
    data, k, extra = {
        "gp/uniform": ({"x": rng.uniform(0, 1, 300)}, -1, {}),
        "gp/clustered": ({"x": clustered}, 8, {"kernel": "matern_5_2"}),
        "gp/planar": (
            {"x": rng.uniform(0, 1, 120), "z": rng.uniform(0, 1, 120)},
            30,
            {},
        ),
    }[case]
    smooth = GaussianProcessSmooth(make_smooth_spec(list(data), bs="gp", k=k, **extra))
    smooth.setup(data)
    E, _ = smooth._gp_E(smooth._knt, smooth._knt, resolved_rho=smooth._resolved_rho)
    return E, smooth.rank


_RLANCZOS_CASES = ("gp/uniform", "gp/clustered", "gp/planar", "tprs/clustered")


def test_rlanczos_matches_pinned_slanczos(r_bridge) -> None:
    """The exact route ports mat.c::Rlanczos and its BLAS/LAPACK calls.

    Against an R that links the pinned OpenBLAS 0.3.26, every Ritz pair is
    bitwise identical. R linked to another BLAS (macOS Accelerate) reorders
    ``dsymv``/``ddot``; there the Ritz values match at STRICT, while its Ritz
    vectors are BLAS-dependent beyond STRICT for ill-conditioned kernels
    (measured on Accelerate R: 9.6e-9 for the clustered Matern kernel, 5.0e-12
    for the planar one) and are not compared. The Gaussian-process raw-basis
    gates compare the resulting bases against that R at STRICT.
    """
    if r_bridge.mode != "rpy2":
        pytest.skip("Identical-input slanczos oracle requires rpy2.")
    pinned = r_links_pinned_openblas(r_bridge)
    tol = np.finfo(float).eps ** 0.5
    collector = _AssertCollector()
    for case in _RLANCZOS_CASES:
        E, k = _rlanczos_kernel(case)
        values, vectors = source_lapack.source_lanczos(
            E, k, tol=tol, tprs_tridiagonal=True
        )
        expected = r_bridge.source_slanczos(E, k, tol)
        collector.check(
            f"{case}/values",
            lambda values=values, expected=expected: np.testing.assert_allclose(
                values, expected["values"], rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
        if pinned:
            collector.check(
                f"{case}/bitwise",
                lambda values=values, vectors=vectors, expected=expected: (
                    np.testing.assert_array_equal(values, expected["values"]),
                    np.testing.assert_array_equal(vectors, expected["vectors"]),
                ),
            )
        collector.check(
            f"{case}/orthonormal Ritz vectors",
            lambda vectors=vectors, k=k: np.testing.assert_allclose(
                vectors.T @ vectors, np.eye(k), rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    collector.raise_if_any("source Rlanczos")


def test_rlanczos_keeps_leading_pairs_without_convergence(monkeypatch) -> None:
    """A run that never certifies convergence keeps Rlanczos's m=k, lm=0."""
    E, _ = _rlanczos_kernel("tprs/clustered")
    original = source_lapack.source_trisymeig

    def uncertified(diagonal, off_diagonal):
        values, vectors = original(diagonal, off_diagonal)
        vectors = vectors.copy()
        vectors[-1, :] = 1.0  # every Ritz error bound is then |beta_j| > 0
        return values, vectors

    monkeypatch.setattr(source_lapack, "source_trisymeig", uncertified)
    values, vectors = source_lapack.source_lanczos(
        E, 4, tol=np.finfo(float).tiny, tprs_tridiagonal=True
    )
    complete = np.sort(np.linalg.eigvalsh(E))[::-1]
    # The leading algebraic pairs, not the largest magnitudes (E has a large
    # negative eigenvalue that the converged selection would retain).
    assert np.abs(complete[-1]) > np.abs(complete[3])
    np.testing.assert_allclose(values, complete[:4], rtol=STRICT.rtol, atol=STRICT.atol)
    assert vectors.shape == (len(E), 4)


def test_source_trisymeig_returns_descending_tridiagonal_pairs() -> None:
    diagonal = np.array([2.0, -1.0, 3.0, 0.5])
    off_diagonal = np.array([0.3, -0.7, 0.2])
    values, vectors = source_lapack.source_trisymeig(diagonal, off_diagonal)
    T = np.diag(diagonal) + np.diag(off_diagonal, 1) + np.diag(off_diagonal, -1)
    assert np.all(np.diff(values) <= 0)
    np.testing.assert_allclose(
        values, np.sort(np.linalg.eigvalsh(T))[::-1], rtol=STRICT.rtol, atol=STRICT.atol
    )
    np.testing.assert_allclose(
        T @ vectors, vectors * values, rtol=STRICT.rtol, atol=STRICT.atol
    )


def test_source_project_rows_is_one_dgemv_per_row() -> None:
    rng = np.random.default_rng(37)
    rows = rng.standard_normal((7, 5))
    UZ = rng.standard_normal((5, 4))
    np.testing.assert_allclose(
        source_lapack.source_project_rows(rows, UZ),
        rows @ UZ,
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )
    with pytest.raises(ValueError, match="UZ row dimension"):
        source_lapack.source_project_rows(rows.T, UZ)
