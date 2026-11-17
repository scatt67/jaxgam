"""Process-level warning for JAX LAPACK kernels running on Apple Accelerate.

On Apple silicon the default NumPy and SciPy wheels link Apple Accelerate,
and JAX's CPU LAPACK kernels take their function pointers from SciPy.  When
several LAPACK calls run concurrently inside one compiled computation the
process heap can be corrupted.  No JAX or XLA setting prevents it; the
OpenBLAS builds of NumPy and SciPy do.  :func:`warn_if_accelerate_lapack` runs
once from ``jaxgam/__init__.py`` after the JAX configuration.
"""

from __future__ import annotations

import os
import platform
import sys
import warnings

import jax

from jaxgam.smooths.source_lapack import accelerate_lapack_in_process

ACCELERATE_OPT_OUT = "JAXGAM_ALLOW_ACCELERATE"

_warned = False


def _accelerate_message() -> str:
    import numpy
    import scipy

    versions = f"numpy=={numpy.__version__} scipy=={scipy.__version__}"
    return (
        "NumPy/SciPy on this Apple silicon Mac use Apple Accelerate, and JAX's "
        "CPU LAPACK kernels run on it. Concurrent LAPACK calls inside one "
        "compiled computation can corrupt memory: the process may abort, and "
        "occasionally results are wrong with no error. This was observed "
        "fitting models with about 100 smooths (about 1,000 coefficients); "
        "none was seen with 50 smooths or fewer. Linux is unaffected. Fix: "
        "install the OpenBLAS builds of NumPy and SciPy, with uv: "
        "`uv pip install --reinstall --no-deps --python-platform "
        f"aarch64-apple-darwin {versions}`, or with pip: `pip download "
        "--only-binary=:all: --no-deps --platform macosx_12_0_arm64 "
        f"{versions} -d wheels && pip install --force-reinstall --no-deps "
        f"wheels/*.whl`. Set {ACCELERATE_OPT_OUT}=1 to silence this warning."
    )


def accelerate_warning_applies() -> bool:
    """Whether this process runs JAX CPU LAPACK on Accelerate on Apple silicon."""
    if os.environ.get(ACCELERATE_OPT_OUT) == "1":
        return False
    if sys.platform != "darwin" or platform.machine() != "arm64":
        return False
    if not accelerate_lapack_in_process():
        return False
    # Only now touch the backend: elsewhere importing jaxgam never initializes
    # a JAX backend.
    return jax.default_backend() == "cpu"


def warn_if_accelerate_lapack() -> bool:
    """Warn once per process; return whether a warning was issued now."""
    global _warned
    if _warned or not accelerate_warning_applies():
        return False
    _warned = True
    warnings.warn(_accelerate_message(), RuntimeWarning, stacklevel=2)
    return True
