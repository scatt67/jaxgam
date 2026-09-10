"""Validation of the public, frozen execution resource policy."""

from dataclasses import FrozenInstanceError

import pytest

from jaxgam import EFSControl, FitControl


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


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("outer_limit", True),
        ("outer_limit", 1.5),
        ("pirls_max_iter", "20"),
        ("history_limit", 0),
        ("log_lambda_max", True),
        ("log_lambda_max", "15"),
        ("score_tolerance", False),
        ("score_tolerance", -1.0),
        ("pirls_tolerance", "1e-7"),
        ("pirls_tolerance", 0.0),
    ],
)
def test_efs_control_rejects_invalid_typed_values(name: str, value) -> None:
    with pytest.raises(ValueError, match=name):
        EFSControl(**{name: value})


def test_fit_control_composes_one_frozen_efs_control_route() -> None:
    efs = EFSControl(outer_limit=7, history_limit=4)
    control = FitControl(efs=efs)
    assert control.efs is efs
    assert FitControl().efs == EFSControl()
    with pytest.raises(ValueError, match="EFSControl"):
        FitControl(efs=None)  # type: ignore[arg-type]
    with pytest.raises(FrozenInstanceError):
        efs.outer_limit = 8
