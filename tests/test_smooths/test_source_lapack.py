"""Declared source-LAPACK provider and cubic shrinkage parity gates."""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from jaxgam.smooths import source_lapack
from jaxgam.smooths.cubic import CubicRegressionSmooth
from tests.helpers import _AssertCollector
from tests.tolerances import STRICT


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
    """Original and blocked-DSYTRD sizes retain source ordering at STRICT."""
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
        collector.check(
            f"case{index}/k{k}",
            lambda actual=actual, expected=expected: np.testing.assert_allclose(
                actual, expected, rtol=STRICT.rtol, atol=STRICT.atol
            ),
        )
    collector.raise_if_any("source cubic shrinkage")
