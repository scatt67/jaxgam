"""PyMGCV: Python port of R's mgcv for Generalized Additive Models."""

import os
import pathlib

import jax

# ---------------------------------------------------------------------------
# JAX configuration — must run before any module-level JIT compilation.
# ---------------------------------------------------------------------------

jax.config.update("jax_enable_x64", True)

# Persistent compilation cache: caches compiled XLA executables to disk so
# that subsequent Python sessions skip the ~700ms XLA compilation step.
# Tracing (~270ms) still runs each cold start, but this cuts total first-fit
# time from ~1s to ~300ms.
#
# Override: set JAX_COMPILATION_CACHE_DIR env var, or call
#   jax.config.update("jax_compilation_cache_dir", "/your/path")
# before importing pymgcv.
#
# Disable: set PYMGCV_NO_COMPILATION_CACHE=1.
if not os.environ.get("PYMGCV_NO_COMPILATION_CACHE"):
    _cache_dir = os.environ.get("JAX_COMPILATION_CACHE_DIR")
    if not _cache_dir:
        _cache_dir = str(pathlib.Path.home() / ".cache" / "pymgcv" / "jax")
        jax.config.update("jax_compilation_cache_dir", _cache_dir)
    # Cache everything, including fast compiles — the small XLA programs
    # (element-wise ops) are cheap to store and add up.
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)

from pymgcv.api import GAM  # noqa: E402

__all__ = ["GAM"]
