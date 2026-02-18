"""Phase boundary import guard.

Phase 1 modules (formula/, smooths/, penalties/) must NOT trigger a JAX import.
This is a load-bearing architectural constraint — see AGENTS.md.
"""

import importlib
import sys

import pytest

PHASE1_MODULES = [
    "pymgcv.formula",
    "pymgcv.formula.parser",
    "pymgcv.formula.terms",
    "pymgcv.formula.design",
    "pymgcv.smooths",
    "pymgcv.smooths.base",
    "pymgcv.smooths.tprs",
    "pymgcv.smooths.cubic",
    "pymgcv.smooths.tensor",
    "pymgcv.smooths.registry",
    "pymgcv.penalties",
    "pymgcv.penalties.penalty",
    "pymgcv.smooths.constraints",
]


@pytest.mark.parametrize("module_name", PHASE1_MODULES)
def test_phase1_import_does_not_trigger_jax(module_name: str) -> None:
    """Importing a Phase 1 module must not cause jax to be imported."""
    # Remove jax and the target module from sys.modules so we get a clean import
    modules_to_remove = [
        key
        for key in sys.modules
        if key == "jax" or key.startswith("jax.") or key.startswith("pymgcv.")
    ]
    saved = {key: sys.modules.pop(key) for key in modules_to_remove}

    try:
        importlib.import_module(module_name)
        assert "jax" not in sys.modules, (
            f"Importing {module_name} triggered a jax import. "
            f"Phase 1 modules must not depend on JAX."
        )
    finally:
        # Restore original sys.modules state
        for key in list(sys.modules):
            if key.startswith("pymgcv."):
                sys.modules.pop(key, None)
        sys.modules.update(saved)
