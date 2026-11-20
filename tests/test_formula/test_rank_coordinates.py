"""Phase-1 import boundary tests for global rank-coordinate construction."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from jaxgam.formula.rank_coordinates import IdentifiableCoordinateMap

# Outcome can depend on the CPU's SIMD maths or BLAS kernel family.
pytestmark = pytest.mark.platform_sensitive


def test_identifiable_coordinate_map_reconstructs_cpu_outputs() -> None:
    """The Phase-1 map owns the public coefficient/covariance reconstruction."""
    coordinate_map = IdentifiableCoordinateMap(
        basis=np.eye(3)[:, (0, 2)],
        fitting_to_public=np.array(
            (
                (0.0, 1.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 0.0, 1.0),
            )
        ),
        keep=np.array((0, 2)),
        drop=np.array((1,)),
    )
    np.testing.assert_allclose(
        coordinate_map.reconstruct_coefficients(np.array((2.0, -3.0))),
        np.array((0.0, 2.0, -3.0)),
    )
    np.testing.assert_allclose(
        coordinate_map.reconstruct_covariance(np.diag((4.0, 9.0))),
        np.diag((0.0, 4.0, 9.0)),
    )
    assert coordinate_map.n_coef == 3
    with pytest.raises(ValueError, match="coefficient"):
        coordinate_map.reconstruct_coefficients(np.ones(3))
    with pytest.raises(ValueError, match="covariance"):
        coordinate_map.reconstruct_covariance(np.eye(3))


def test_global_rank_coordinate_helpers_import_without_jax() -> None:
    """Load the true CPU module without the package's Phase-2 entry point.

    ``jaxgam.__init__`` intentionally configures JAX for public API imports.
    The subprocess installs namespace packages first, then imports the actual
    Phase-1 helper and its dependencies directly. This proves the helper does
    not introduce a JAX import of its own.
    """
    source_root = Path(__file__).resolve().parents[2] / "jaxgam"
    script = f"""
import sys
import types
from pathlib import Path

root = Path({str(source_root)!r})
for name, path in (
    ("jaxgam", root),
    ("jaxgam.formula", root / "formula"),
):
    package = types.ModuleType(name)
    package.__path__ = [str(path)]
    sys.modules[name] = package

from jaxgam.formula.rank_coordinates import global_rank_coordinate_map

assert callable(global_rank_coordinate_map)
assert not any(name == "jax" or name.startswith("jax.") for name in sys.modules)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
