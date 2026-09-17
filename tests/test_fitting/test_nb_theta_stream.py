"""Batch theta derivatives, padding, explicit trial/cache isolation and JIT."""

import hashlib
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgam.families.negative_binomial import NegativeBinomial
from jaxgam.fitting.efs_theta import conditional_theta_nll
from jaxgam.fitting.nb_theta_stream import (
    nb_conditional_theta_batch,
    nb_conditional_theta_step,
)
from tests.tolerances import MODERATE, STRICT


@pytest.mark.parametrize("link", ["log", "identity", "sqrt"])
@pytest.mark.parametrize("fractional", [False, True])
def test_batch_global_derivatives_match_frozen_dense_objective_and_jit(
    link, fractional
):
    family = NegativeBinomial(theta=0.7, link=link)
    y = jnp.asarray([0.0, 2.0, 3.0, 4.0, 0.0, 7.0])
    if fractional:
        y = y.at[1:].add(jnp.asarray([0.25, 0.5, 0.75, 0.0, 0.25]))
        y = y.at[4].set(0.0)
    mu = jnp.asarray([1.2, 2.4, 3.1, 0.8, 2.0, 4.0])
    eta = jnp.log(mu) if link == "log" else (mu if link == "identity" else jnp.sqrt(mu))
    weight = jnp.asarray([0.8, 1.2, 0.0, 1.0, 0.9, 1.1])
    integer = not fractional
    indices = y.astype(jnp.int64) if integer else jnp.zeros_like(y, dtype=jnp.int64)
    compiled = jax.jit(
        nb_conditional_theta_batch,
        static_argnames=("family", "max_y", "integer_counts"),
    )
    for theta in (0.1, 2.7, 1e6, 0.1):
        log_theta = jnp.asarray([np.log(theta)])

        def objective(t):
            return conditional_theta_nll(
                t, eta, y, weight, indices, family, max_y=8, integer_counts=integer
            )

        expected = np.array(
            [
                jax.jit(objective)(log_theta),
                jax.jit(jax.grad(objective))(log_theta)[0],
                jax.jit(jax.hessian(objective))(log_theta)[0, 0],
            ]
        )
        results = []
        for where in (slice(0, 2), slice(2, 5), slice(5, 6)):
            result = compiled(
                log_theta,
                eta[where],
                y[where],
                weight[where],
                jnp.ones_like(y[where], dtype=bool),
                family,
                max_y=8,
                integer_counts=integer,
            )
            assert result.admissible
            results.append(np.array(result[:3]))
        actual = np.sum(results, axis=0)
        # Exact six-row fixture only; see the checked-in numerical review.
        # Its stable fixed-mean derivatives agree with 65-digit arithmetic,
        # while unchanged dense nonlog AD cancels near theta=1e6.
        objective_tolerance = (
            MODERATE
            if fractional and theta == 1e6 and link in ("identity", "sqrt")
            else STRICT
        )
        np.testing.assert_allclose(
            actual[0],
            expected[0],
            rtol=objective_tolerance.rtol,
            atol=objective_tolerance.atol,
        )
        if theta == 1e6:
            canonical = json.dumps(
                {
                    "mu": np.asarray(mu).tolist(),
                    "y": np.asarray(y).tolist(),
                    "weight": np.asarray(weight).tolist(),
                    "log_theta": np.asarray(log_theta).tolist(),
                    "max_y": 8,
                    "integer_counts": integer,
                },
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
            expected_hash = (
                "82c7333e5f8e0e862875afc4257e25abdd9aff3f5d2ab2ed656ea44e73a8d6a3"
                if fractional
                else "da0bc32e17c97b4decc43fa523c00315fd4f975a0fa73c7fadbdc0f3e5e16fcc"
            )
            assert hashlib.sha256(canonical).hexdigest() == expected_hash
        derivative_tolerance = (
            MODERATE if theta == 1e6 and link in ("identity", "sqrt") else STRICT
        )
        np.testing.assert_allclose(
            actual[1:],
            expected[1:],
            rtol=derivative_tolerance.rtol,
            atol=derivative_tolerance.atol,
        )


def test_padding_empty_and_zero_prior_blocks_are_neutral_under_jit(capsys):
    family = NegativeBinomial(theta=0.7, link="identity")
    compiled = jax.jit(
        nb_conditional_theta_batch,
        static_argnames=("family", "max_y", "integer_counts"),
    )
    for n in (0, 3):
        result = compiled(
            jnp.zeros(1),
            jnp.full(n, jnp.nan),
            jnp.full(n, jnp.nan),
            jnp.full(n, jnp.nan),
            jnp.zeros(n, dtype=bool),
            family,
            max_y=0,
            integer_counts=True,
            debug=jnp.asarray(True),
        )
        result.nll.block_until_ready()
        np.testing.assert_array_equal(np.asarray(result[:3]), np.zeros(3))
        assert result.admissible
    assert "NB theta batch nll=" in capsys.readouterr().out


@pytest.mark.parametrize(
    "defect",
    ["theta", "eta", "weight", "negative", "fractional", "capacity", "integer"],
)
def test_dynamic_domain_and_metadata_defects_fail_closed(defect):
    theta, eta, y, weight = (
        jnp.zeros(1),
        jnp.ones(2),
        jnp.array([2.0, 3.0]),
        jnp.ones(2),
    )
    if defect == "theta":
        theta = theta.at[0].set(jnp.inf)
    if defect == "eta":
        eta = eta.at[0].set(-1.0)
    if defect == "weight":
        weight = weight.at[0].set(-1.0)
    if defect == "negative":
        y = y.at[0].set(-1.0)
    if defect == "fractional":
        y = y.at[0].set(0.5)
    if defect == "capacity":
        y = y.at[0].set(11.0)
    if defect == "integer":
        y = y.at[0].set(2.5)
    result = jax.jit(
        nb_conditional_theta_batch,
        static_argnames=("family", "max_y", "integer_counts"),
    )(
        theta,
        eta,
        y,
        weight,
        jnp.ones(2, dtype=bool),
        NegativeBinomial(link="identity"),
        max_y=10,
        integer_counts=True,
    )
    assert not result.admissible


@pytest.mark.parametrize(
    ("gradient", "hessian", "expected", "valid"),
    [
        (3.0, 2.0, -1.5, True),
        (3.0, -2.0, -1.5, True),
        (30.0, 2.0, -4.0, True),
        (3.0, 0.0, -3.0, False),
        (3.0, np.nan, -3.0, False),
        (np.inf, 2.0, -4.0, False),
    ],
)
def test_source_scalar_proposal_repairs_curvature_limits_step_and_rejects_failure(
    gradient, hessian, expected, valid
):
    step, usable = jax.jit(nb_conditional_theta_step)(gradient, hessian)
    assert step == expected
    assert bool(usable) == valid


@pytest.mark.parametrize(
    "defect", ["mask-shape", "mask-dtype", "dtype", "fixed", "theta-shape", "capacity"]
)
def test_static_batch_contract_rejects_invalid_shapes_types_and_parameters(defect):
    theta, eta, y, weight, valid = (
        jnp.zeros(1),
        jnp.ones(2),
        jnp.ones(2),
        jnp.ones(2),
        jnp.ones(2, dtype=bool),
    )
    family, max_y = NegativeBinomial(), 2
    if defect == "mask-shape":
        valid = valid[:1]
    if defect == "mask-dtype":
        valid = valid.astype(int)
    if defect == "dtype":
        y = y.astype(jnp.float32)
    if defect == "fixed":
        family = NegativeBinomial(fixed=True)
    if defect == "theta-shape":
        theta = jnp.zeros(2)
    if defect == "capacity":
        max_y = -1
    with pytest.raises(
        (ValueError, TypeError), match=r"mask|float64|estimated|theta|integer"
    ):
        nb_conditional_theta_batch(
            theta, eta, y, weight, valid, family, max_y=max_y, integer_counts=True
        )


@pytest.mark.parametrize("B", [16, 64])
def test_compiled_theta_batch_memory_and_output_are_recorded(B, caplog):
    import json
    import logging

    caplog.set_level(logging.INFO)
    family = NegativeBinomial(link="identity")
    theta = jnp.asarray([np.log(0.7)])
    eta, y, weight, valid = (
        jnp.linspace(2.0, 3.0, B),
        jnp.arange(B, dtype=float) % 7,
        jnp.ones(B),
        jnp.ones(B, dtype=bool),
    )
    compiled = (
        jax.jit(
            nb_conditional_theta_batch,
            static_argnames=("family", "max_y", "integer_counts"),
        )
        .lower(theta, eta, y, weight, valid, family, max_y=64, integer_counts=True)
        .compile()
    )
    result = compiled(theta, eta, y, weight, valid)
    result.nll.block_until_ready()
    assert result.admissible
    memory = compiled.memory_analysis()
    assert memory.output_size_in_bytes <= 64
    logging.getLogger(__name__).info(
        "NB_THETA_COMPILED_MEMORY %s",
        json.dumps(
            {
                "B": B,
                "global_count_capacity": 64,
                "dtype": "float64",
                "jax": jax.__version__,
                "backend": jax.default_backend(),
                "argument_bytes": memory.argument_size_in_bytes,
                "output_bytes": memory.output_size_in_bytes,
                "temporary_bytes": memory.temp_size_in_bytes,
                "alias_bytes": memory.alias_size_in_bytes,
            },
            sort_keys=True,
        ),
    )
    assert "NB_THETA_COMPILED_MEMORY" in caplog.text


@pytest.mark.parametrize("link", ["log", "identity", "sqrt"])
@pytest.mark.parametrize("variant", ["integer", "fractional_ge_one"])
def test_fixed_mean_theta_derivatives_match_65_digit_reference_strict(link, variant):
    from decimal import Decimal, localcontext
    from pathlib import Path

    fixture = json.loads(
        (
            Path(__file__).parents[1]
            / "fixtures"
            / ("efs52_nb_theta_six_row_" + variant + ".json")
        ).read_text()
    )
    mu, y, weight = [np.asarray(fixture[key]) for key in ("mu", "y", "weight")]
    log_theta = np.asarray(fixture["log_theta"])
    with localcontext() as context:
        context.prec = 65
        theta = Decimal.from_float(float(np.exp(log_theta[0])))
        gradient = hessian = Decimal(0)

        def psi(a):
            return (
                a.ln()
                - 1 / (2 * a)
                - 1 / (12 * a**2)
                + 1 / (120 * a**4)
                - 1 / (252 * a**6)
                + 1 / (240 * a**8)
            )

        def trigamma(a):
            return (
                1 / a
                + 1 / (2 * a**2)
                + 1 / (6 * a**3)
                - 1 / (30 * a**5)
                + 1 / (42 * a**7)
                - 1 / (30 * a**9)
            )

        for yi, mi, wi in zip(y, mu, weight, strict=True):
            a, b, c = [Decimal.from_float(float(value)) for value in (yi, mi, wi)]
            if variant == "integer":
                difference = -sum((1 / (theta + k) for k in range(int(a))), Decimal(0))
                derivative = sum(
                    (1 / (theta + k) ** 2 for k in range(int(a))), Decimal(0)
                )
            else:
                difference = psi(theta) - psi(theta + a)
                derivative = trigamma(theta) - trigamma(theta + a)
            row_gradient = theta * (
                difference + (1 + b / theta).ln() + (a - b) / (theta + b)
            )
            row_hessian = row_gradient + theta**2 * (
                derivative - b / (theta * (theta + b)) - (a - b) / (theta + b) ** 2
            )
            gradient += c * row_gradient
            hessian += c * row_hessian
    eta = np.log(mu) if link == "log" else mu if link == "identity" else np.sqrt(mu)
    result = jax.jit(
        nb_conditional_theta_batch,
        static_argnames=("family", "max_y", "integer_counts"),
    )(
        jnp.asarray(log_theta),
        jnp.asarray(eta),
        jnp.asarray(y),
        jnp.asarray(weight),
        jnp.ones_like(y, dtype=bool),
        NegativeBinomial(theta=0.7, link=link),
        max_y=fixture["max_y"],
        integer_counts=fixture["integer_counts"],
    )
    assert result.admissible
    np.testing.assert_allclose(
        [result.gradient, result.hessian],
        [float(gradient), float(hessian)],
        rtol=STRICT.rtol,
        atol=STRICT.atol,
    )


def test_saved_source_evidence_keeps_failed_early_trajectories_and_binary_values():
    from pathlib import Path

    directory = Path(__file__).parents[1] / "fixtures"
    report = json.loads(
        (directory / "efs52_nb_theta_numerical_review.json").read_text()
    )
    binary_path = directory / "efs52_nb_theta_review_R_binary.npz"
    assert (
        hashlib.sha256(binary_path.read_bytes()).hexdigest()
        == report["binary_npz_sha256"]
    )
    with np.load(binary_path, allow_pickle=False) as binary:
        for record in report["records"]:
            prefix = record["variant"] + "_" + record["link"]
            for field in ("R_initial", "R_final"):
                np.testing.assert_array_equal(
                    binary[prefix + "_" + field], record[field]
                )
            np.testing.assert_array_equal(
                binary[prefix + "_R_path"], record["R_theta_path"]
            )
            actual = np.asarray(record["stable_final"]["theta_path"])
            reference = np.asarray(record["R_theta_path"])
            assert len(actual) == len(reference)
            bound = MODERATE.atol + MODERATE.rtol * np.abs(reference)
            assert np.any(np.abs(actual - reference) > bound)
