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


def _child(case: ScalingCase, checkout: Path, mode: str) -> dict:
    """Run one isolated setup or fit child against a committed checkout.

    The test image has no ``.git`` and the harness's provenance is
    fail-closed, so real-route children run from the committed minimal
    checkout fixture, as the other isolated harness tests do.
    """
    return benchmark_scaling._run_child(case, checkout, mode)


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
def test_kernel_and_shared_model_penalties_match_their_sp_order(
    model: str, minimal_git_checkout: Path
) -> None:
    """The dense setup has one penalty per named fixed smoothing parameter."""
    case = _case(60, execution="dense-reml", model=model)
    record = _child(case, minimal_git_checkout, "setup")

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


@pytest.mark.parametrize(
    "kwargs",
    [
        {"execution": "dense-reml", "memory_budget_mib": 1024},
        {"execution": "efs", "memory_budget_mib": 1024},
        {"execution": "stream-qr", "memory_budget_mib": 0},
        {"execution": "discrete", "memory_budget_mib": False},
        {"execution": "stream-exact", "memory_budget_mib": 1.5},
    ],
)
def test_memory_budget_is_validated(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="memory_budget_mib"):
        _case(**kwargs)


@pytest.mark.parametrize(
    "execution",
    ["stream", "stream-qr", "stream-exact", "stream-gaussian-qr", "discrete-wm"],
)
def test_memory_budget_reaches_every_streamed_and_discrete_fit_control(
    monkeypatch: pytest.MonkeyPatch, execution: str
) -> None:
    import jaxgam

    seen: list[object] = []

    class _FakeGAM:
        def __init__(self, *_args: object, control: object, **_kwargs: object):
            seen.append(control)

        def fit(self, *_args: object, **_kwargs: object) -> str:
            return "fitted"

    monkeypatch.setattr(jaxgam, "GAM", _FakeGAM)
    default = _case(execution=execution)
    raised = _case(execution=execution, memory_budget_mib=1536)
    data = benchmark_scaling._make_data(default)

    assert benchmark_scaling._build_fit(default, data).run() == "fitted"
    assert benchmark_scaling._build_fit(raised, data).run() == "fitted"
    library_default = jaxgam.FitControl().memory_budget_bytes
    assert seen[0].memory_budget_bytes == library_default
    assert seen[1].memory_budget_bytes == 1536 << 20
    assert "memory_budget_mib" not in benchmark_scaling._route_contract(default)
    assert benchmark_scaling._route_contract(raised)["memory_budget_mib"] == 1536


def test_streamed_setup_uses_the_case_memory_budget(minimal_git_checkout: Path) -> None:
    record = _child(
        _case(40, execution="stream-qr", batch_rows=16, memory_budget_mib=640),
        minimal_git_checkout,
        "setup",
    )

    assert record["source_scans"]["setup_memory_budget_bytes"] == 640 << 20


@pytest.mark.parametrize(
    "kwargs",
    [
        {"execution": "discrete", "model": "re"},
        {"execution": "discrete", "levels": 10},
        {"execution": "discrete", "model": "re", "levels": 1},
        {"execution": "discrete", "model": "re", "levels": True},
    ],
)
def test_random_effect_levels_are_validated(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError, match="levels"):
        _case(**kwargs)


@pytest.mark.parametrize("execution", ["stream-qr", "stream-exact", "efs"])
def test_random_effect_model_rejects_streamed_and_efs_routes(execution: str) -> None:
    with pytest.raises(ValueError, match="model='re'"):
        _case(execution=execution, model="re", levels=10)


def test_random_effect_model_declares_every_level_and_hashes_codes() -> None:
    import pandas as pd

    case = _case(30, execution="discrete-wm", model="re", levels=50)
    data = benchmark_scaling._make_data(case)

    assert benchmark_scaling._formula(case) == (
        "y ~ s(x0, k=5, bs='cr') + s(g, bs='re')"
    )
    assert benchmark_scaling._smoothing_parameter_order(case) == ["s(x0)", "s(g)"]
    assert list(data.columns) == ["x0", "g", "y"]
    assert isinstance(data.g.dtype, pd.CategoricalDtype)
    assert list(data.g.cat.categories) == [f"g{level}" for level in range(50)]
    assert data.g.nunique() < 50  # unobserved levels stay declared
    provenance = benchmark_scaling._input_provenance(data, case)
    assert provenance == benchmark_scaling._input_provenance(
        benchmark_scaling._make_data(case), case
    )
    relabelled = data.assign(g=data.g.cat.rename_categories(lambda c: c + "x"))
    assert (
        benchmark_scaling._input_provenance(relabelled, case)["frame_sha256"]
        != provenance["frame_sha256"]
    )
    source = benchmark_scaling._row_source(data, case)
    batch = next(iter(source.scan(64)))
    assert isinstance(batch.columns["g"].dtype, pd.CategoricalDtype)
    assert benchmark_scaling._route_contract(case)["levels"] == 50
    assert benchmark_scaling._discrete_control(case).discretization.factor_names == (
        "g",
    )
    assert (
        benchmark_scaling._discrete_control(
            _case(execution="discrete-wm")
        ).discretization.factor_names
        == ()
    )


@pytest.mark.parametrize("execution", ["dense-reml", "discrete-wm"])
def test_random_effect_model_fits_on_dense_and_discrete_routes(
    execution: str, minimal_git_checkout: Path
) -> None:
    case = _case(400, execution=execution, model="re", levels=40)
    setup = _child(case, minimal_git_checkout, "setup")
    fit = _child(case, minimal_git_checkout, "fit")

    assert setup["valid"]
    assert fit["valid"]
    assert setup["model"]["n_coef"] == fit["result"]["n_coef"] == 1 + 4 + 40
    assert setup["model"]["n_penalties"] == 2
    assert setup["input"] == fit["input"]


def test_cli_threads_random_effect_levels(
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
            "re",
            "--levels",
            "1000",
            "--result-mode",
            "prediction",
        ],
    )
    benchmark_scaling.main()

    assert seen[-1].levels == 1000
    assert "recorded" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"execution": "operator", "family": "nb"}, "fixed theta"),
        ({"execution": "operator-efs", "model": "re-only", "levels": 300}, "poisson"),
        ({"execution": "operator", "result_mode": "full"}, "result_mode"),
        ({"execution": "operator", "discrete_table_mib": 8}, "discrete_table_mib"),
        ({"execution": "operator", "model": "re-only"}, "levels"),
    ],
)
def test_operator_cases_validate(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _case(**kwargs)


def test_operator_routes_use_public_pcg_controls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jaxgam

    seen: list[dict[str, object]] = []

    class _FakeGAM:
        def __init__(self, formula: str, **kwargs: object):
            seen.append({"formula": formula, **kwargs})

        def fit(self, *_args: object, **_kwargs: object) -> str:
            return "fitted"

    monkeypatch.setattr(jaxgam, "GAM", _FakeGAM)
    fixed = _case(execution="operator", model="re", levels=20, memory_budget_mib=64)
    efs = _case(execution="operator-efs", model="re-only", levels=300, family="poisson")
    for case in (fixed, efs):
        plan = benchmark_scaling._build_fit(case, benchmark_scaling._make_data(case))
        assert plan.run() == "fitted"
        assert plan.source is not None

    fixed_call, efs_call = seen
    assert fixed_call["sp"] == [1.0, 1.0]
    assert fixed_call["control"].execution == "operator"
    assert fixed_call["control"].linear_solver == "pcg"
    assert fixed_call["control"].uncertainty == "none"
    assert fixed_call["control"].memory_budget_bytes == 64 << 20
    assert efs_call["formula"] == "y ~ s(g, bs='re')"
    assert efs_call["optimizer"] == "efs"
    assert "sp" not in efs_call
    assert (
        benchmark_scaling._route_contract(efs)["fixed_smoothing_parameter_values"]
        is None
    )


def test_validity_accepts_only_the_route_documented_missing_fields() -> None:
    from types import SimpleNamespace

    result = SimpleNamespace(
        coefficients=np.ones(3),
        smoothing_params=np.ones(1),
        score=None,
        deviance=1.0,
        scale=None,
        edf_total=None,
        converged=True,
        convergence_info="converged",
    )
    omitted = benchmark_scaling._omitted_fields(_case(execution="operator"))

    accepted = benchmark_scaling._fit_validity(result, omitted)
    assert accepted["valid"]
    assert accepted["finite_fields"]["score"] is None
    rejected = benchmark_scaling._fit_validity(result)
    assert not rejected["valid"]
    assert rejected["finite_fields"]["score"] is False
    efs = _case(execution="operator-efs", family="poisson", model="re-only", levels=300)
    assert benchmark_scaling._omitted_fields(efs) == frozenset()


def test_operator_fixed_sp_setup_and_fit_records(minimal_git_checkout: Path) -> None:
    case = _case(240, execution="operator", model="re", levels=60, batch_rows=64)
    setup = _child(case, minimal_git_checkout, "setup")
    fit = _child(case, minimal_git_checkout, "fit")

    assert setup["valid"]
    assert fit["valid"]
    assert setup["model"]["n_coef"] == fit["result"]["n_coef"] == 1 + 4 + 60
    assert setup["model"]["basis_fingerprint_kind"] == "operator_prepared_model_digest"
    ledger = setup["operator_setup"]
    assert ledger["random_effect_coefficients"] == [60]
    assert ledger["preparation_budget_bytes"] == 512 << 20
    assert "never formed p x p" in ledger["penalty_contract_status"]
    assert setup["source_scans"]["setup_delta"] > 0
    assert setup["source_scans"]["setup_max_batch_rows"] <= 64
    result = fit["result"]
    assert result["score"] is None
    assert result["edf_total"] is None
    assert result["cold_fit_validity"]["finite_fields"]["score"] is None
    assert (
        result["inner_iterations"]
        == (result["operator_diagnostics"]["krylov_iterations"])
    )
    assert result["operator_diagnostics"]["peak_bound_bytes"] > 0
    assert fit["timings"]["prediction_se_s"] is None
    assert "operator" in fit["timings"]["prediction_se_status"]
    assert result["source_scans"]["cold_fit_delta"] > 0


def test_operator_efs_setup_reserves_an_eighth_of_the_budget(
    minimal_git_checkout: Path,
) -> None:
    case = _case(
        514,
        execution="operator-efs",
        model="re-only",
        levels=257,
        family="poisson",
        memory_budget_mib=256,
    )
    setup = _child(case, minimal_git_checkout, "setup")

    assert setup["valid"]
    assert setup["operator_setup"]["preparation_budget_bytes"] == (256 << 20) // 8
    assert setup["model"]["n_coef"] == 1 + 257


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"execution": "dense-reml", "stress": "tails"}, "stress must be"),
        ({"execution": "dense-reml", "stress": "concurvity"}, "two smooths"),
        (
            {"execution": "dense-reml", "model": "tensor", "stress": "concurvity"},
            "additive",
        ),
        (
            {"execution": "dense-reml", "family": "poisson", "stress": "near-perfect"},
            "Gaussian",
        ),
        ({"execution": "efs", "stress": "near-perfect"}, "Gaussian"),
    ],
)
def test_stress_cases_validate(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _case(**kwargs)


def test_stress_generators_change_only_their_own_part_of_the_input() -> None:
    base_case = _case(400, execution="dense-reml", smooth_count=2)
    base = benchmark_scaling._make_data(base_case)

    def stressed(stress: str, **kwargs: object):
        case = _case(
            400, execution="dense-reml", smooth_count=2, stress=stress, **kwargs
        )
        return case, benchmark_scaling._make_data(case)

    case, concurvity = stressed("concurvity")
    np.testing.assert_array_equal(concurvity.x0, base.x0)
    assert np.max(np.abs(concurvity.x1 - concurvity.x0)) < 0.1
    assert benchmark_scaling._input_provenance(concurvity, case)["generator"] == (
        "gaussian_additive_sine_concurvity"
    )

    _, zero = stressed("zero-weights")
    np.testing.assert_array_equal(zero.y, base.y)
    assert 0.05 < np.mean(zero.w == 0.0) < 0.15
    assert set(np.unique(zero.w)) == {0.0, 1.0}

    _, extreme = stressed("extreme-weights")
    assert extreme.w.min() < 1e-3
    assert extreme.w.max() > 1e3
    heavy = extreme.w > 100
    assert np.std((extreme.y - base.y)[heavy]) < np.std(base.y)

    _, offset = stressed("offset")
    np.testing.assert_allclose(offset.y - offset.off, base.y)
    assert np.all((offset.off >= np.log(0.5)) & (offset.off <= np.log(2.0)))

    _, poisson_offset = stressed("offset", family="poisson")
    assert np.all(poisson_offset.y >= 0)
    assert "off" in poisson_offset
    _, nb_offset = stressed("offset", family="nb")
    assert np.all(nb_offset.y >= 0)
    assert not np.array_equal(nb_offset.y, poisson_offset.y)
    plain_counts = benchmark_scaling._make_data(
        _case(400, execution="dense-reml", smooth_count=2, family="poisson")
    )
    _, weighted_counts = stressed("extreme-weights", family="poisson")
    np.testing.assert_array_equal(weighted_counts.y, plain_counts.y)
    assert weighted_counts.w.max() > 1e3

    _, perfect = stressed("near-perfect")
    signal = 0.4 * np.sin(2 * np.pi * perfect.x0) + 0.4 * np.sin(4 * np.pi * perfect.x1)
    assert np.max(np.abs(perfect.y - signal)) < 1e-5
    assert np.std(base.y - signal) > 0.1
    assert list(perfect.columns) == ["x0", "x1", "y"]


def test_weights_and_offsets_reach_sources_and_dense_fits(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import jaxgam

    calls: list[dict[str, object]] = []

    class _FakeGAM:
        def __init__(self, *_args: object, **_kwargs: object):
            pass

        def fit(self, _data: object, **kwargs: object) -> str:
            calls.append(kwargs)
            return "fitted"

    monkeypatch.setattr(jaxgam, "GAM", _FakeGAM)
    for execution in ("dense", "dense-reml", "efs"):
        case = _case(
            50,
            execution=execution,
            smooth_count=2,
            stress="offset",
            result_mode="prediction",
        )
        data = benchmark_scaling._make_data(case)
        assert benchmark_scaling._build_fit(case, data).run() == "fitted"
        np.testing.assert_array_equal(calls[-1]["offset"], data.off.to_numpy())
        assert "weights" not in calls[-1]

    case = _case(60, execution="stream-qr", smooth_count=2, stress="zero-weights")
    data = benchmark_scaling._make_data(case)
    source = benchmark_scaling._row_source(data, case)
    batch = next(iter(source.scan(64)))
    np.testing.assert_array_equal(batch.weight, data.w.to_numpy())
    assert set(batch.columns) == {"x0", "x1"}
    assert benchmark_scaling._route_contract(case)["stress"] == "zero-weights"


@pytest.mark.parametrize(
    ("execution", "stress"),
    [("dense-reml", "offset"), ("stream-qr", "zero-weights")],
)
def test_stress_cases_fit_through_the_public_routes(
    execution: str, stress: str, minimal_git_checkout: Path
) -> None:
    case = _case(300, execution=execution, smooth_count=2, stress=stress)
    record = _child(case, minimal_git_checkout, "fit")

    assert record["valid"]
    assert record["route"]["stress"] == stress


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"execution": "dense-reml", "nb_theta": 5.0}, "nb_theta"),
        ({"execution": "dense-reml", "family": "nb", "nb_theta": 0.0}, "nb_theta"),
        ({"execution": "dense-reml", "family": "nb", "nb_theta": True}, "nb_theta"),
        ({"execution": "dense-reml", "outlier_fraction": 0.01}, "outlier_fraction"),
        (
            {"execution": "dense-reml", "family": "poisson", "outlier_fraction": 0.6},
            "outlier_fraction",
        ),
        ({"execution": "efs", "outlier_fraction": 0.01}, "outlier_fraction"),
    ],
)
def test_count_tail_options_validate(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _case(**kwargs)


def test_count_tail_generators_keep_the_default_draw() -> None:
    base_case = _case(2000, execution="dense-reml", family="nb", smooth_count=2)
    base = benchmark_scaling._make_data(base_case)
    explicit = benchmark_scaling._make_data(
        _case(2000, execution="dense-reml", family="nb", smooth_count=2, nb_theta=3.0)
    )
    np.testing.assert_array_equal(explicit.y, base.y)

    near_poisson_case = _case(
        2000, execution="dense-reml", family="nb", smooth_count=2, nb_theta=1e4
    )
    near_poisson = benchmark_scaling._make_data(near_poisson_case)
    assert np.var(near_poisson.y) < np.var(base.y)
    contract = benchmark_scaling._route_contract(near_poisson_case)
    assert contract["theta"] == "estimated from start 1.0 (data drawn at 10000.0)"
    assert (
        benchmark_scaling._input_provenance(near_poisson, near_poisson_case)[
            "generator"
        ]
        == "nb_additive_sine_theta10000"
    )

    outlier_case = _case(
        2000, execution="dense-reml", family="nb", smooth_count=2, outlier_fraction=0.01
    )
    outliers = benchmark_scaling._make_data(outlier_case)
    changed = outliers.y.to_numpy() != base.y.to_numpy()
    assert 5 <= changed.sum() <= 40
    assert np.all(outliers.y[changed] >= 1000)
    assert base.y.max() < 1000
    assert benchmark_scaling._route_contract(outlier_case)["outlier_fraction"] == 0.01


def test_count_tail_cases_fit_and_reach_the_cli(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    minimal_git_checkout: Path,
) -> None:
    record = _child(
        _case(
            400,
            execution="dense-reml",
            family="nb",
            smooth_count=2,
            outlier_fraction=0.02,
        ),
        minimal_git_checkout,
        "fit",
    )
    assert record["valid"]

    seen: list[ScalingCase] = []
    monkeypatch.setattr(
        benchmark_scaling, "run_isolated", lambda case, _root: seen.append(case) or {}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark_scaling.py",
            "--execution",
            "stream-exact",
            "--family",
            "nb",
            "--nb-theta",
            "500",
            "--outlier-fraction",
            "0.001",
            "--result-mode",
            "prediction",
        ],
    )
    benchmark_scaling.main()
    assert seen[-1].nb_theta == 500.0
    assert seen[-1].outlier_fraction == 0.001
    capsys.readouterr()


def test_three_dimensional_thin_plate_model(minimal_git_checkout: Path) -> None:
    case = _case(60, execution="dense-reml", model="tp3")
    data = benchmark_scaling._make_data(case)

    assert benchmark_scaling._formula(case) == "y ~ s(x0, x1, x2, k=5, bs='tp')"
    assert benchmark_scaling._smoothing_parameter_order(case) == ["s(x0,x1,x2)"]
    assert list(data.columns) == ["x0", "x1", "x2", "y"]
    assert benchmark_scaling._input_provenance(data, case)["generator"] == (
        "gaussian_tp3_sine"
    )
    with pytest.raises(ValueError, match="model='tp3'"):
        _case(execution="stream-qr", model="tp3")
    record = _child(case, minimal_git_checkout, "setup")
    assert record["valid"]
    assert record["model"]["n_penalties"] == 1
    # mgcv's default order for d=3 is m=3 (2m > d+1); its null space has
    # C(5, 3) = 10 columns, so k=5 is raised to 11 (10 after centring, plus
    # the intercept).
    assert record["model"]["n_coef"] == 11
