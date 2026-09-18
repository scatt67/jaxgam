"""Genuine fixed-sp regular release systems and extreme source prior weights."""

import hashlib
import inspect
import json
import subprocess
from dataclasses import replace

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


def _source_reference(tmp_path, family, link, X, y, weight, offset, start):
    for name, values in (
        ("X", X),
        ("y", y),
        ("w", weight),
        ("off", offset),
        ("start", start),
    ):
        np.savetxt(tmp_path / name, values, fmt="%.17g")
    script = r"""
library(mgcv)
stopifnot(getRversion()=="4.5.2",packageVersion("mgcv")=="1.9.3")
a <- commandArgs(TRUE);d <- a[1]
read <- function(name) as.matrix(read.table(file.path(d,name)))
X <- read("X");y <- c(read("y"));w <- c(read("w"));off <- c(read("off"))
fam <- get(a[2],asNamespace("stats"))(link=a[3]);fam <- mgcv:::fix.family(fam)
fam <- mgcv:::fix.family.link(fam);fam <- mgcv:::fix.family.var(fam)
fam <- mgcv:::fix.family.ls(fam);family <- fam;nobs <- length(y);weights <- w
# Initialize before the unweighted natural-order null projection, retaining
# real zero-prior rows. The first eta still comes from the supplied start.
eval(fam$initialize)
null <- qr.coef(qr(X),rep(fam$linkfun(mean(y)),length(y)));null[is.na(null)] <- 0
known <- a[2] %in% c("binomial","poisson")
# S=diag(0,1): put the unique penalized direction first for gam.reparam,
# exactly as gam.fit3's UrS/U1 contract requires. Eb is the unscaled root.
E <- matrix(c(0,1),1,2);U1 <- matrix(c(0,1,1,0),2,2)
UrS <- list(matrix(1,1,1));sp <- log(.35)
if (!known) sp <- c(sp,log(.7))
diag <- new.env(parent=emptyenv())
walk <- function(e) {
 if (is.call(e) && identical(e[[1]],as.name("<-")) &&
     identical(e[[2]],as.name("oo")) && is.call(e[[3]]) &&
     identical(e[[3]][[1]],as.name(".C")) &&
     identical(e[[3]][[2]],as.name("C_gdi1"))) {
  return(bquote({
   oo <- .(e[[3]])
   candidate_eta <- drop(x%*%oo$beta+offset)
   diag$valid <- valideta(candidate_eta)&&validmu(linkinv(candidate_eta))
   oo
  }))
 }
 if (!is.call(e)) return(e)
 as.call(lapply(as.list(e),walk))
}
source_clone <- mgcv:::gam.fit3;body(source_clone) <- walk(body(source_clone))
environment(source_clone) <- list2env(list(diag=diag),
 parent=environment(mgcv:::gam.fit3))
fit <- source_clone(x=X,y=y,sp=sp,Eb=E,UrS=UrS,weights=w,offset=off,
 U1=U1,Mp=1,family=fam,control=gam.control(epsilon=1e-7,maxit=200),deriv=0,
 scale=if (known) 1 else 0,scoreType="REML",null.coef=null,start=c(read("start")))
writeBin(as.double(c(fit$converged,fit$iter)),file.path(d,"status"),size=8,endian="little")
stopifnot(fit$converged)
writeBin(as.double(diag$valid),file.path(d,"candidate_valid"),size=8,endian="little")
writeBin(as.double(c(fit$coefficients,fit$deviance,if (known) 1 else fit$scale.est,
 fit$trA,fit$REML,null,as.vector(tcrossprod(fit$rV)))),
 file.path(d,"reference"),size=8,endian="little")
"""
    completed = subprocess.run(
        [
            "Rscript",
            "-e",
            script,
            str(tmp_path),
            family.family_name,
            "1/mu^2" if link == "inverse_squared" else link,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert completed.returncode == 0, (
        f"Pinned penalized oracle failed: {completed.stderr}"
    )
    raw = np.fromfile(tmp_path / "reference", dtype="<f8")
    return raw[:6], raw[6:8], raw[8:].reshape(2, 2, order="F")


def _check_penalized_fit(tmp_path, family_class, link, *, extreme=False):
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
    reference, null, covariance = _source_reference(
        tmp_path, family, link, X, data.y, weight, offset, start
    )
    source_status = np.fromfile(tmp_path / "status", dtype="<f8")
    assert source_status[0] == 1.0
    if approved_boundary:
        np.testing.assert_array_equal(source_status, [1.0, 4.0])
        np.testing.assert_array_equal(
            np.fromfile(tmp_path / "candidate_valid", dtype="<f8"), [1.0]
        )
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
        for field, observed, expected, tolerance in (
            ("beta", np.asarray(state.coefficients), reference[:2], fit_tolerance),
            (
                "deviance/scale/EDF",
                np.r_[state.deviance, state.scale, state.edf],
                reference[2:5],
                STRICT,
            ),
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
def test_all_32_regular_cells_use_a_nonempty_fixed_sp_penalty(
    tmp_path, family_class, link
):
    _check_penalized_fit(tmp_path, family_class, link)


@pytest.mark.usefixtures("r_bridge")
@pytest.mark.parametrize(
    ("family_class", "link"),
    [(Poisson, "log"), (Binomial, "probit"), (Gamma, "identity"), (Gaussian, "log")],
)
def test_real_penalized_regular_fits_with_extreme_prior_weights(
    tmp_path, family_class, link
):
    _check_penalized_fit(tmp_path, family_class, link, extreme=True)
