"""Discrete, working-model, family and tensor modes of the scaling harness."""

import sys
from pathlib import Path

import numpy as np
import pytest

from scripts import benchmark_scaling
from scripts.benchmark_scaling import ScalingCase


def _case(n_rows: int = 19, **kwargs: object) -> ScalingCase:
    kwargs.setdefault("result_mode", "prediction")
    return ScalingCase(n_rows, 5, **kwargs)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"execution": "discrete", "result_mode": "full"}, "result_mode"),
        ({"execution": "discrete-wm", "result_mode": "inference"}, "result_mode"),
        ({"execution": "stream", "family": "poisson"}, "requires execution"),
        ({"execution": "efs", "family": "nb"}, "requires execution"),
        ({"execution": "dense", "family": "poisson"}, "requires execution"),
        ({"execution": "discrete", "family": "binomial"}, "family must be"),
        ({"execution": "discrete", "model": "spline"}, "model must be"),
        ({"execution": "stream-exact", "model": "tensor"}, "tensor"),
        ({"execution": "stream-qr", "model": "tensor"}, "tensor"),
        ({"execution": "efs", "model": "tensor"}, "tensor"),
        ({"execution": "discrete", "model": "tensor", "smooth_count": 2}, "tensor"),
        ({"execution": "stream-qr", "discretization_m": 30}, "discretization_m"),
        ({"execution": "discrete", "discretization_m": 1}, "discretization_m"),
        ({"execution": "discrete", "discretization_m": True}, "discretization_m"),
        ({"execution": "discrete", "discretization_m": 3.5}, "discretization_m"),
    ],
)
def test_discrete_family_and_model_cases_validate(
    kwargs: dict[str, object], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _case(**kwargs)


@pytest.mark.parametrize(
    "execution", ["dense", "stream", "stream-qr", "stream-exact", "efs"]
)
def test_original_gaussian_additive_contracts_and_inputs_are_unchanged(
    execution: str,
) -> None:
    case = _case(7, execution=execution)
    route = benchmark_scaling._route_contract(case)
    provenance = benchmark_scaling._input_provenance(
        benchmark_scaling._make_data(case), case
    )

    assert not {"family", "model", "theta", "discretization_m"} & set(route)
    assert provenance["generator"] == (
        "poisson_sine" if execution == "efs" else "gaussian_sine"
    )
    if execution == "stream-qr":
        assert route == {
            "requested_execution": "stream-qr",
            "smoothing_parameter_order": ["s(x)"],
            "fixed_smoothing_parameter_values": [1.0],
            "adapter": "public",
            "strategy": "fixed-sp streamed pivoted-QR PIRLS",
            "result_mode": "prediction",
        }


def test_discrete_route_contracts_name_strategy_grid_and_smoothing() -> None:
    fixed = benchmark_scaling._route_contract(
        _case(execution="discrete", smooth_count=2)
    )
    selected = benchmark_scaling._route_contract(
        _case(execution="discrete-wm", smooth_count=2, discretization_m=30)
    )
    nb = benchmark_scaling._route_contract(_case(execution="discrete-wm", family="nb"))
    tensor = benchmark_scaling._route_contract(
        _case(execution="discrete", model="tensor")
    )
    exact_nb = benchmark_scaling._route_contract(
        _case(execution="stream-exact", family="nb")
    )

    assert fixed["adapter"] == selected["adapter"] == "public"
    assert fixed["result_mode"] == selected["result_mode"] == "prediction"
    assert "fixed-sp continuous-discrete" in fixed["strategy"]
    assert "working-model REML" in selected["strategy"]
    assert fixed["fixed_smoothing_parameter_values"] == [1.0, 1.0]
    assert selected["fixed_smoothing_parameter_values"] is None
    assert fixed["discretization_m"] is None
    assert "default grid" in fixed["discretization_m_semantics"]
    assert selected["discretization_m"] == 30
    assert "per-margin" in selected["discretization_m_semantics"]
    assert nb["family"] == "nb"
    assert "estimated from start" in nb["theta"]
    assert "NB/log, estimated theta" in exact_nb["strategy"]
    assert "Gaussian" not in exact_nb["strategy"]
    assert tensor["model"] == "tensor"
    assert tensor["family"] == "gaussian"
    assert tensor["fixed_smoothing_parameter_values"] == [1.0, 1.0]
    assert tensor["smoothing_parameter_order"] == [
        "te(x0,x1):margin x0",
        "te(x0,x1):margin x1",
    ]


def test_family_and_tensor_inputs_are_shared_by_every_supporting_route() -> None:
    def frame_digest(case: ScalingCase) -> str:
        data = benchmark_scaling._make_data(case)
        return benchmark_scaling._input_provenance(data, case)["frame_sha256"]

    gaussian = {
        frame_digest(_case(execution=execution, smooth_count=2))
        for execution in ("stream", "stream-exact", "discrete", "discrete-wm")
    }
    poisson = {
        frame_digest(_case(execution="efs", smooth_count=2)),
        *(
            frame_digest(_case(execution=execution, family="poisson", smooth_count=2))
            for execution in ("dense-reml", "stream-qr", "discrete", "discrete-wm")
        ),
    }
    nb_cases = [
        _case(41, execution=execution, family="nb", smooth_count=2)
        for execution in ("stream-exact", "discrete-wm")
    ]
    tensor_cases = [
        _case(execution=execution, model="tensor")
        for execution in ("dense", "dense-reml", "discrete", "discrete-wm")
    ]

    assert len(gaussian) == len(poisson) == 1
    assert gaussian != poisson
    assert len({frame_digest(case) for case in nb_cases}) == 1
    nb_data = benchmark_scaling._make_data(nb_cases[0])
    assert np.all(nb_data.y.to_numpy() >= 0)
    assert np.array_equal(nb_data.y.to_numpy(), np.round(nb_data.y.to_numpy()))
    assert benchmark_scaling._input_provenance(nb_data, nb_cases[0])["generator"] == (
        "nb_additive_sine"
    )
    assert len({frame_digest(case) for case in tensor_cases}) == 1
    tensor = tensor_cases[0]
    assert benchmark_scaling._formula(tensor) == "y ~ te(x0, x1, k=5, bs='cr')"
    assert benchmark_scaling._fixed_smoothing_parameters(tensor) == [1.0, 1.0]
    assert list(benchmark_scaling._make_data(tensor).columns) == ["x0", "x1", "y"]
    assert (
        benchmark_scaling._input_provenance(
            benchmark_scaling._make_data(tensor), tensor
        )["generator"]
        == "gaussian_tensor_sine"
    )


def test_discrete_control_and_family_objects_follow_the_case() -> None:
    from jaxgam.families.negative_binomial import NegativeBinomial

    fixed = benchmark_scaling._discrete_control(
        _case(execution="discrete", batch_rows=13)
    )
    selected = benchmark_scaling._discrete_control(
        _case(execution="discrete-wm", discretization_m=25)
    )
    nb = benchmark_scaling._family_object(_case(execution="discrete", family="nb"))

    assert fixed.execution == selected.execution == "discrete"
    assert fixed.smoothing_strategy == "exact_reml"
    assert selected.smoothing_strategy == "working_model_reml"
    assert fixed.batch_rows == 13
    assert fixed.uncertainty == "fisher"
    assert fixed.discretization.m is None
    assert selected.discretization.m == 25
    assert isinstance(nb, NegativeBinomial)
    assert nb.n_theta > 0
    assert benchmark_scaling._family_object(_case(execution="efs")) == "poisson"
    assert benchmark_scaling._family_object(_case(execution="discrete")) == "gaussian"


@pytest.mark.parametrize("execution", ["discrete", "discrete-wm"])
def test_tiny_discrete_routes_record_setup_and_public_fit_accounting(
    execution: str,
    minimal_git_checkout: Path,
) -> None:
    record = benchmark_scaling.run_isolated(
        _case(60, execution=execution, batch_rows=13, repeats=2, smooth_count=2),
        minimal_git_checkout,
    )
    setup = record["setup"]
    fit = record["fit"]
    accounting = fit["result"]["public_fit_accounting"]
    scans = fit["result"]["source_scans"]

    assert record["within_run_consistency"]["valid"]
    assert setup["model"]["basis_fingerprint_kind"] == (
        "discrete_compact_basis_fingerprint"
    )
    assert setup["model"]["frozen_contract"]["unpenalized_rank_deficit"] is None
    assert setup["source_scans"]["setup_delta"] > 0
    assert "discrete preparation" in setup["source_scans"]["status"]
    assert setup["discrete_setup"]["setup_retained_table_bytes"] > 0
    assert setup["discrete_setup"]["descriptor_device_peak_bytes"] > 0
    compact_rows = setup["discrete_setup"]["compact_rows_per_group"]
    assert len(compact_rows) >= 2
    assert all(0 < rows <= 60 for rows in compact_rows)
    assert fit["result"]["execution_route"] == "discrete"
    assert fit["result"]["converged"]
    assert accounting["smoothing_strategy"] == (
        "fixed" if execution == "discrete" else "working_model_reml"
    )
    assert accounting["discretization_strategy"] == "mgcv_range_grid"
    assert accounting["source_scans"] >= 2
    # The end-to-end public fit repeats the setup scans the setup child made.
    assert scans["cold_fit_delta"] == (
        setup["source_scans"]["setup_delta"] + accounting["source_scans"]
    )
    assert scans["warm_fit_deltas"] == [scans["cold_fit_delta"]] * 2
    assert accounting["batches_scanned"] == accounting["source_scans"] * 5
    assert 0 < fit["result"]["known_workspace_bytes"] < accounting["fit_peak_bytes"]
    assert "discrete-route" in fit["result"]["n_iter_semantics"]
    assert (
        fit["result"]["inner_iterations"]
        == (accounting["working_model_proposal_evaluations"])
    )
    assert fit["timings"]["cold_compile_and_fit_s"] > 0
    assert fit["timings"]["warm_fit_median_s"] > 0
    assert fit["timings"]["prediction_se_s"] is not None
    assert fit["route"]["discretization_m"] is None


def test_tiny_count_family_routes_record_family_and_theta(
    minimal_git_checkout: Path,
) -> None:
    # The test image has no .git; the harness fails closed without source
    # provenance, so isolated children run from a committed minimal checkout.
    poisson = benchmark_scaling.run_isolated(
        _case(80, execution="discrete", family="poisson", batch_rows=29),
        minimal_git_checkout,
    )["fit"]
    nb = benchmark_scaling.run_isolated(
        _case(80, execution="discrete-wm", family="nb", batch_rows=29),
        minimal_git_checkout,
    )["fit"]
    tensor_setup = benchmark_scaling.run_isolated(
        _case(60, execution="discrete-wm", model="tensor", discretization_m=12),
        minimal_git_checkout,
    )["setup"]

    assert poisson["route"]["family"] == "poisson"
    assert poisson["input"]["generator"] == "poisson_sine"
    assert poisson["result"]["public_fit_accounting"]["theta"] is None
    assert nb["route"]["family"] == "nb"
    assert nb["result"]["public_fit_accounting"]["theta"] > 0
    assert nb["result"]["public_fit_accounting"]["smoothing_strategy"] == (
        "working_model_reml"
    )
    assert tensor_setup["model"]["n_penalties"] == 2
    assert tensor_setup["model"]["n_coef"] == 25
    assert tensor_setup["formula"] == "y ~ te(x0, x1, k=5, bs='cr')"


def test_cli_threads_family_model_and_grid_into_the_case(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[ScalingCase] = []

    def run(case: ScalingCase, _root: Path) -> dict:
        seen.append(case)
        return {"case": "recorded"}

    monkeypatch.setattr(benchmark_scaling, "run_isolated", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_scaling.py",
            "--n",
            "40",
            "--k",
            "6",
            "--execution",
            "discrete-wm",
            "--family",
            "nb",
            "--discretization-m",
            "40",
            "--smooth-count",
            "2",
            "--result-mode",
            "prediction",
        ],
    )
    benchmark_scaling.main()

    (case,) = seen
    assert case.execution == "discrete-wm"
    assert case.family == "nb"
    assert case.model == "additive"
    assert case.discretization_m == 40
    assert '"case": "recorded"' in capsys.readouterr().out
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_scaling.py",
            "--n",
            "40",
            "--execution",
            "dense-reml",
            "--model",
            "tensor",
            "--result-mode",
            "prediction",
        ],
    )
    benchmark_scaling.main()
    assert seen[-1].model == "tensor"
    assert seen[-1].execution == "dense-reml"


_KERNEL_MODELS = ("tp", "gp", "shared", "ti")


@pytest.mark.parametrize("model", _KERNEL_MODELS)
@pytest.mark.parametrize(
    "kwargs",
    [
        {"execution": "stream-qr"},
        {"execution": "stream-exact"},
        {"execution": "stream-gaussian-qr"},
        {"execution": "efs"},
        {"execution": "discrete", "smooth_count": 2},
    ],
)
def test_kernel_and_shared_models_reject_streamed_efs_and_counts(
    model: str, kwargs: dict[str, object]
) -> None:
    with pytest.raises(ValueError, match=f"model='{model}'"):
        _case(model=model, **kwargs)


def test_kernel_and_shared_models_fix_formula_order_and_inputs() -> None:
    expected = {
        "tp": ("y ~ s(x0, x1, k=5, bs='tp')", ["s(x0,x1)"], ["x0", "x1", "y"]),
        "gp": ("y ~ s(x0, k=5, bs='gp')", ["s(x0)"], ["x0", "y"]),
        "shared": (
            "y ~ s(x0, k=5, bs='cr') + te(x0, x1, k=5, bs='cr')",
            ["s(x0)", "te(x0,x1):margin x0", "te(x0,x1):margin x1"],
            ["x0", "x1", "y"],
        ),
        "ti": (
            "y ~ s(x0, k=5, bs='cr') + s(x1, k=5, bs='cr') + ti(x0, x1, k=5, bs='cr')",
            ["s(x0)", "s(x1)", "ti(x0,x1):margin x0", "ti(x0,x1):margin x1"],
            ["x0", "x1", "y"],
        ),
    }
    for model, (formula, order, columns) in expected.items():
        dense = _case(execution="dense-reml", model=model)
        discrete = _case(execution="discrete", model=model)
        data = benchmark_scaling._make_data(dense)
        assert benchmark_scaling._formula(dense) == formula
        assert benchmark_scaling._smoothing_parameter_order(dense) == order
        assert benchmark_scaling._fixed_smoothing_parameters(discrete) == (
            [1.0] * len(order)
        )
        assert list(data.columns) == columns
        assert np.all(np.isfinite(data.y))
        assert benchmark_scaling._input_provenance(data, dense)["generator"] == (
            f"gaussian_{model}_sine"
        )
        # Every route that supports the model fits the same frame.
        np.testing.assert_array_equal(
            benchmark_scaling._make_data(discrete).to_numpy(), data.to_numpy()
        )
        assert benchmark_scaling._route_contract(discrete)["model"] == model
    poisson = _case(execution="discrete", model="gp", family="poisson")
    assert np.all(benchmark_scaling._make_data(poisson).y >= 0)


@pytest.mark.parametrize("model", _KERNEL_MODELS)
def test_kernel_and_shared_model_penalties_match_their_sp_order(model: str) -> None:
    """The dense setup has one penalty per named fixed smoothing parameter."""
    case = _case(60, execution="dense-reml", model=model)
    record = benchmark_scaling.run_setup(case, Path.cwd())

    assert record["formula"] == benchmark_scaling._formula(case)
    assert record["model"]["n_penalties"] == len(
        benchmark_scaling._smoothing_parameter_order(case)
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"execution": "stream-qr", "discrete_table_mib": 256},
        {"execution": "dense-reml", "discrete_table_mib": 256},
        {"execution": "discrete", "discrete_table_mib": 0},
        {"execution": "discrete", "discrete_table_mib": True},
        {"execution": "discrete-wm", "discrete_table_mib": 2.5},
    ],
)
def test_discrete_table_budget_is_validated(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="discrete_table_mib"):
        _case(**kwargs)


def test_discrete_table_budget_reaches_the_control_and_contract() -> None:
    from jaxgam.data.discretize import DiscretizationBudget

    default = _case(execution="discrete-wm")
    raised = _case(execution="discrete-wm", discrete_table_mib=256)

    assert (
        benchmark_scaling._discrete_control(default).discretization.budget
        == DiscretizationBudget()
    )
    budget = benchmark_scaling._discrete_control(raised).discretization.budget
    assert budget.max_table_bytes == budget.max_batch_bytes == 256 << 20
    assert budget.max_selector_bytes == DiscretizationBudget().max_selector_bytes
    assert "discrete_table_and_batch_budget_mib" not in (
        benchmark_scaling._route_contract(default)
    )
    assert (
        benchmark_scaling._route_contract(raised)["discrete_table_and_batch_budget_mib"]
        == 256
    )


def test_cli_threads_the_discrete_table_budget(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    seen: list[ScalingCase] = []

    def run(case: ScalingCase, _root: Path) -> dict:
        seen.append(case)
        return {"case": "recorded"}

    monkeypatch.setattr(benchmark_scaling, "run_isolated", run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_scaling.py",
            "--execution",
            "discrete-wm",
            "--model",
            "tp",
            "--discrete-table-mib",
            "512",
            "--result-mode",
            "prediction",
        ],
    )
    benchmark_scaling.main()

    assert seen[-1].discrete_table_mib == 512
    assert seen[-1].model == "tp"
    assert "recorded" in capsys.readouterr().out
