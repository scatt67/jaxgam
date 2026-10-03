"""Validation of the public, frozen execution resource policy."""

from dataclasses import FrozenInstanceError

import pytest

from jaxgam import FitControl


@pytest.mark.parametrize(
    "name", ["batch_rows", "memory_budget_bytes", "output_budget_bytes"]
)
@pytest.mark.parametrize("value", [True, 0, -1, 1.5, None])
def test_resource_limits_require_positive_integers(name, value) -> None:
    with pytest.raises(ValueError, match=name):
        FitControl(**{name: value})


def test_uncertainty_and_compression_validation_and_frozen_defaults() -> None:
    with pytest.raises(ValueError, match="uncertainty"):
        FitControl(uncertainty="approximate")
    with pytest.raises(ValueError, match="linear_solver"):
        FitControl(linear_solver="automatic")
    for value in (1, None, "yes"):
        with pytest.raises(ValueError, match="boolean"):
            FitControl(gaussian_compression=value)
    for uncertainty in ("none", "fisher", "covariance"):
        control = FitControl(uncertainty=uncertainty, batch_rows=1)
        assert control.execution == "dense"
        assert control.gaussian_compression is False
        assert control.linear_solver == "cholesky"
        assert control.batch_rows == 1
        with pytest.raises(FrozenInstanceError):
            control.batch_rows = 2
