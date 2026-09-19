"""Genuine fixed-sp regular release systems and extreme source prior weights."""

import hashlib
import inspect
import json
from dataclasses import replace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxgam.control import FitControl
from jaxgam.data.source import DataFrameRowSource
from jaxgam.execution.regular_stream import fit_regular_streamed_pirls
from jaxgam.execution.stream import StreamPIRLSControl
from jaxgam.families.standard import Binomial, Gamma, Gaussian, Poisson
from jaxgam.fitting.data import PreparedFittingMetadata
from jaxgam.formula.design_provider import StreamDesign
from jaxgam.formula.parser import parse_formula
from jaxgam.formula.prepare import prepare_model
from jaxgam.penalties.structure import (
    DiagonalPenalty,
    IdentityTransform,
    PenaltyBlock,
    PenaltyStructure,
)
from jaxgam.results import GAMPredictionResult
from tests.helpers import _AssertCollector
from tests.r_bridge import RBridge
from tests.test_execution.test_regular_starts import _LINKS, _case
from tests.tolerances import MODERATE, STRICT

# Exact reviewed input/model/control configurations, not family release policy.
_APPROVED_DIGESTS = {
    (Poisson, "identity"): (
        "e7bcb1ad84794e5ededa7859254b4b4bb2f38f7f93a146237a9d0a99575c28ec"
    ),
    (Binomial, "log"): (
        "a264744a1adf9473d6277b14e8d6571814ec107443a3c1380d4d6f0233f13e67"
    ),
}


def _fixture_digest(
    family_class, link, X, y, weight, offset, start, structure, rho, prepared, control
):
    # Fourteen significant digits avoid incidental libm last-bit differences
    # across pinned image architectures. Exact original arrays and rho are
    # separately checked against the unchanged fixture and natural-sp formula.
    def canonical(values):
        a = np.asarray(values)
        return {
            "shape": list(a.shape),
            "values": [format(float(x), ".14g") for x in a.ravel()],
        }

    description = {
        "fixture_source_sha256": hashlib.sha256(
            inspect.getsource(_case).encode()
        ).hexdigest(),
        "family": family_class.__name__,
        "link": link,
        "formula": "y~x",
        "inputs": [canonical(a) for a in (X, y, weight, offset, start)],
        "penalty": canonical(structure.materialize(np.zeros(1))),
        "rho": canonical(rho),
        "basis_fingerprint": prepared.basis_fingerprint,
        "penalty_blocks": [
            {
                "start": b.start,
                "stop": b.stop,
                "sp_indices": list(b.sp_indices),
                "ranks": list(b.ranks),
                "transform": canonical(b.transform.dense()),
            }
            for b in structure.blocks
        ],
        "penalty_rank": 1,
        "penalty_null_dim": 1,
        "control": {
            "batch_rows": control.batch_rows,
            "tol": control.tol,
            "max_iter": control.max_iter,
            "solver": control.solver_policy,
        },
        "R": "4.5.2",
        "mgcv": "1.9-3",
    }
    return hashlib.sha256(
        json.dumps(description, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _rank_one_preparation(source, family):
    """Freeze a slope penalty on the existing public p2 geometry.

    This layer gate supplies an explicit CPU prepared system, rather than
    introducing public parametric-penalty syntax. D is identity, and the
    intercept is the declared penalty null space in both implementations.
    """
    prepared = prepare_model(parse_formula("y~x"), source, family=family)
    assert prepared.n_coef == 2
    structure = PenaltyStructure(
        n_coef=2,
        blocks=(
            PenaltyBlock(
                start=1,
                stop=2,
                sp_indices=(0,),
                local_penalties=(DiagonalPenalty(np.ones(1)),),
                transform=IdentityTransform(1),
                ranks=(1,),
            ),
        ),
    )
    rho = np.log(np.array([0.35]))
    rho.setflags(write=False)
    fitting = replace(
        prepared.fitting,
        penalty_structure=structure,
        log_lambda_init=rho,
        total_penalty_rank=1,
        total_penalty_null_dim=1,
    )
    fingerprint = hashlib.sha256(
        (prepared.basis_fingerprint + ":rank1-public-slope-penalty").encode()
    ).hexdigest()
    return replace(
        prepared, penalties=structure, fitting=fitting, basis_fingerprint=fingerprint
    ), rho


def _source_reference(
    family,
    link,
    X,
    y,
    weight,
    offset,
    start,
    *,
    derivatives=False,
    rho=None,
    score_phi=0.7,
):
    return RBridge(mode="rpy2").regular_gam_fit3_penalized(
        family.family_name,
        link,
        X,
        np.asarray(y),
        weight,
        offset,
        start,
        np.atleast_1d(np.log(0.35) if rho is None else rho),
        score_phi=score_phi,
        derivatives=derivatives,
    )


def _check_penalized_fit(family_class, link, *, extreme=False):
    family, data, weight, offset, start = _case(family_class, link)
    if extreme:
        # Ten decades on each side of one, including a real neutral prior.
        # Permute them so the two dominant rows are not adjacent in X.
        rng = np.random.default_rng(73640)
        weight = rng.permutation(np.geomspace(1e-10, 1e10, len(weight)))
        weight[3] = 0.0
    source = DataFrameRowSource(data, response="y", weights=weight, offset=offset)
    prepared, rho = _rank_one_preparation(source, family)
    fit_control = StreamPIRLSControl(
        batch_rows=17, tol=1e-7, max_iter=200, solver_policy="qr"
    )
    X = np.column_stack((np.ones(len(data)), data.x))
    inputs = tuple(np.array(a, copy=True) for a in (X, data.y, weight, offset, start))
    approved_boundary = not extreme and (family_class, link) in _APPROVED_DIGESTS
    if approved_boundary:
        (
            _,
            original_data,
            original_weight,
            original_offset,
            original_start,
        ) = _case(family_class, link)
        for observed, original in zip(
            (X, data.y, weight, offset, start),
            (
                np.column_stack((np.ones(len(original_data)), original_data.x)),
                original_data.y,
                original_weight,
                original_offset,
                original_start,
            ),
            strict=True,
        ):
            np.testing.assert_array_equal(observed, original)
        np.testing.assert_array_equal(rho, np.log(np.array([0.35])))
        assert (
            _fixture_digest(
                family_class,
                link,
                X,
                data.y,
                weight,
                offset,
                start,
                prepared.fitting.penalty_structure,
                rho,
                prepared,
                fit_control,
            )
            == _APPROVED_DIGESTS[(family_class, link)]
        )
    fit_tolerance = MODERATE if approved_boundary else STRICT
    score_tolerance = (
        MODERATE if approved_boundary and family_class is Binomial else STRICT
    )
    # Native AMD64 exceeds STRICT only for EDF on the immutable reviewed
    # Binomial/log digest. Four source/trajectory corrections and the literal
    # gdi.c scalar reduction are recorded in the Sep-26 numerical review.
    edf_tolerance = (
        MODERATE if approved_boundary and family_class is Binomial else STRICT
    )
    oracle = _source_reference(family, link, X, data.y, weight, offset, start)
    reference, null, covariance = (
        oracle["reference"],
        oracle["null"],
        oracle["covariance"],
    )
    source_status = oracle["status"]
    assert source_status[0] == 1.0
    if approved_boundary:
        np.testing.assert_array_equal(source_status, [1.0, 4.0])
        np.testing.assert_array_equal(oracle["candidate_valid"], [1.0])
    reference_mu = np.asarray(family.link.inverse(X @ reference[:2] + offset))
    reference_se = np.sqrt(reference[3] * np.einsum("ij,jk,ik->i", X, covariance, X))
    checks = _AssertCollector()
    for B in (1, 17, 200) if extreme else (17,):
        result = fit_regular_streamed_pirls(
            StreamDesign(prepared, source),
            family,
            rho,
            maximum_bytes=10000000,
            score_scale=1.0 if family.scale_known else 0.7,
            initial_coefficients=start,
            control=replace(fit_control, batch_rows=B),
        )
        state = result.state
        assert state.converged
        if approved_boundary:
            assert B == 17
            assert state.n_iter == 4
            assert result.final_refit_accepted
            assert result.source_score.candidate_valid
        assert result.initial_coefficients_present
        np.testing.assert_array_equal(np.asarray(state.log_lambda), rho)
        metadata = PreparedFittingMetadata.from_prepared(prepared, family)
        assert metadata.n_penalties == metadata.total_penalty_rank == 1
        prediction = GAMPredictionResult._from_stream_fit(
            stream_state=state,
            prepared=prepared,
            metadata=metadata,
            family=family,
            formula="y~x",
            method="REML",
            control=FitControl(),
        )
        actual_covariance = np.asarray(
            state.fisher_coefficient_factor.hessian_inverse(jnp.eye(2))
        )
        actual_mu = np.asarray(
            family.link.inverse(X @ np.asarray(state.coefficients) + offset)
        )
        actual_se = np.sqrt(
            float(state.scale) * np.einsum("ij,jk,ik->i", X, actual_covariance, X)
        )
        normalized_y = jnp.asarray(
            family.execution_initial_response(data.y.to_numpy(), weight)
        )
        X_jax = jnp.asarray(X)
        offset_jax = jnp.asarray(offset)
        weight_jax = jnp.asarray(weight)

        def source_deviance(
            beta,
            X_value=X_jax,
            offset_value=offset_jax,
            y_value=normalized_y,
            weight_value=weight_jax,
        ):
            mu = family.link.inverse(X_value @ beta + offset_value)
            return jnp.sum(
                family.deviance_derivative_contributions(y_value, mu, weight_value)
            )

        source_jacobian = 0.5 * jax.hessian(source_deviance)(
            jnp.asarray(result.information_coefficients)
        ) + jnp.diag(jnp.asarray([0.0, 0.35]))
        source_inverse = result.source_coefficient_factor.hessian_inverse(jnp.eye(2))
        np.testing.assert_allclose(
            source_jacobian @ source_inverse,
            np.eye(2),
            rtol=STRICT.rtol,
            atol=STRICT.atol,
        )
        assert not result.source_solve_coefficients.flags.writeable
        source_gradient = 0.5 * jax.grad(source_deviance)(
            jnp.asarray(result.information_coefficients)
        ) + jnp.diag(jnp.asarray([0.0, 0.35])) @ jnp.asarray(
            result.information_coefficients
        )
        scaled_source_gradient = float(
            jnp.max(jnp.abs(source_gradient))
            / (1.0 + abs(result.source_score.stopping_penalized_deviance))
        )
        assert scaled_source_gradient <= fit_control.tol
        for field, observed, expected, tolerance in (
            ("beta", np.asarray(state.coefficients), reference[:2], fit_tolerance),
            (
                "deviance/scale",
                np.r_[state.deviance, state.scale],
                reference[2:4],
                STRICT,
            ),
            ("EDF", state.edf, reference[4], edf_tolerance),
            ("REML", prediction.score, reference[5], score_tolerance),
            ("mean", actual_mu, reference_mu, fit_tolerance),
            ("null", result.null_coefficients, null, STRICT),
            ("Fisher covariance", actual_covariance, covariance, fit_tolerance),
            ("link SE", actual_se, reference_se, fit_tolerance),
        ):
            checks.check(
                f"B{B}/{field}",
                lambda a=observed, b=expected, t=tolerance: np.testing.assert_allclose(
                    a, b, rtol=t.rtol, atol=t.atol
                ),
            )
        assert result.source_score.score_phi == (1.0 if family.scale_known else 0.7)
        assert result.source_score.reported_phi == float(state.scale)
        assert result.source_score.penalized_deviance == float(state.penalized_deviance)
        assert np.isfinite(np.asarray(state.penalized_deviance))
        for original, current in zip(
            inputs, (X, data.y, weight, offset, start), strict=True
        ):
            np.testing.assert_array_equal(current, original)
    checks.raise_if_any(f"Penalized {family.family_name}/{link}, extreme={extreme}")


@pytest.mark.usefixtures("r_bridge")
@pytest.mark.parametrize("family_class", [Gaussian, Gamma, Poisson, Binomial])
@pytest.mark.parametrize("link", _LINKS)
def test_all_32_regular_cells_use_a_nonempty_fixed_sp_penalty(family_class, link):
    _check_penalized_fit(family_class, link)


@pytest.mark.usefixtures("r_bridge")
@pytest.mark.parametrize(
    ("family_class", "link"),
    [(Poisson, "log"), (Binomial, "probit"), (Gamma, "identity"), (Gaussian, "log")],
)
def test_real_penalized_regular_fits_with_extreme_prior_weights(family_class, link):
    _check_penalized_fit(family_class, link, extreme=True)
