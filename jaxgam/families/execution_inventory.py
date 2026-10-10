"""Checked-in evidence inventory for registered family/link execution cells.

This module is deliberately descriptive.  It is not consulted by fit drivers:
release routing must use family-owned capabilities plus an explicitly reviewed
policy, never an incidental family-name allowlist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

InventoryStatus = Literal["dense_r_parity_partial", "dense_generic_unverified"]
RConstructorStatus = Literal["accepted", "rejected"]
EFSCellStatus = Literal[
    "internal_pinned_parity",
    "internal_pinned_parity_with_named_boundary",
    "implementation_missing",
    "r_rejected",
]
EvidenceScope = Literal[
    "fixed_sp_pirls_coefficients_deviance_mu_moderate",
    "free_reml_score_edf_loose",
    "free_reml_deviance_theta_loose",
    "dense_invariants_only",
    "unverified",
]


@dataclass(frozen=True)
class FamilyExecutionInventoryEntry:
    """One constructor-accepted registered family/link/parameter-mode cell."""

    family: str
    link: str
    parameter_mode: Literal["none", "fixed_theta", "estimated_theta"]
    default_link: bool
    mathematical_canonical: bool
    legacy_route_is_canonical: bool
    r_constructor_status: RConstructorStatus
    r_advertised: bool
    r_constructor_link: str
    r_start_validity_notes: str
    efs_status: EFSCellStatus
    numerical_boundaries: tuple[str, ...]
    dense_status: InventoryStatus
    evidence_scope: EvidenceScope
    stream_status: str
    evidence: tuple[str, ...]


_LINKS = (
    "identity",
    "log",
    "logit",
    "inverse",
    "probit",
    "cloglog",
    "sqrt",
    "inverse_squared",
)

# These are the link sets advertised by the pinned family constructors.  The
# base R constructors subsequently call ``make.link`` for a character link
# outside this set, so all eight registered links are *constructed* even when
# they are not an advertised family contract.  ``inverse_squared`` maps to
# R's spelling ``1/mu^2``. NB differs: its
# constructor explicitly rejects a non-listed link (R/efam.r:160-168).
_R_ADVERTISED_LINKS = {
    "gaussian": frozenset({"identity", "log", "inverse"}),
    "binomial": frozenset({"logit", "probit", "cloglog", "log"}),
    "poisson": frozenset({"log", "identity", "sqrt"}),
    "gamma": frozenset({"inverse", "log", "identity"}),
    "nb": frozenset({"log", "identity", "sqrt"}),
}
_R_CONSTRUCTOR_LINK = {link: link for link in _LINKS}
_R_CONSTRUCTOR_LINK["inverse_squared"] = "1/mu^2"
_CANONICAL = {
    "gaussian": "identity",
    "binomial": "logit",
    "poisson": "log",
    "gamma": "inverse",
}
_DEFAULT_LINK = {
    "gaussian": "identity",
    "binomial": "logit",
    "poisson": "log",
    "gamma": "inverse",
    "nb": "log",
}

# These are deliberately notes, not an executable predicate or a statement of
# every R-supported data domain.  ``family$initialize`` is allowed to depend
# on prior weights, binomial trial encoding and a supplied start.  The pinned
# sources are the authority for those details (stats/R/family.R and
# mgcv/R/efam.r); EFS fixtures record the narrower input profile actually
# exercised below.
_R_START_VALIDITY_NOTES = {
    "gaussian": "NULL start comes from stats::gaussian()$initialize (mu <- y)",
    "binomial": "initialize depends on binomial response/trial encoding and weights",
    "poisson": "NULL start comes from stats::poisson()$initialize",
    "gamma": "NULL start requires the positive Gamma initialization path",
    "nb": "mgcv::nb() uses its count-family initialization path",
}


def _eta_domain(family: str, link: str) -> str:
    if link == "identity":
        if family == "gaussian":
            return "finite eta"
        if family == "binomial":
            return "0 < eta < 1"
        return "eta > 0"
    if link == "log":
        return "eta < 0" if family == "binomial" else "finite eta"
    if link in {"logit", "probit", "cloglog"}:
        return "finite eta; link inverse must remain finite and valid"
    if link == "inverse":
        if family == "binomial":
            return "eta > 1"
        return "finite eta != 0" if family == "gaussian" else "eta > 0"
    if link == "sqrt":
        if family == "binomial":
            return "0 < eta < 1"
        return "finite eta > 0"
    assert link == "inverse_squared"
    return "eta > 1" if family == "binomial" else "finite eta > 0"


def _r_constructor_status(family: str, link: str) -> RConstructorStatus:
    if family == "nb":
        return "accepted" if link in _R_ADVERTISED_LINKS[family] else "rejected"
    return "accepted"


def _r_start_validity_notes(family: str, link: str) -> str:
    if _r_constructor_status(family, link) == "rejected":
        return "not applicable: pinned R constructor rejects this registered link"
    start = _R_START_VALIDITY_NOTES[family]
    if family == "gaussian" and link == "log":
        start = "Gaussian/log NULL start needs positive response for finite linkfun(y)"
    elif family == "gaussian" and link == "inverse":
        start = (
            "Gaussian/inverse NULL start needs nonzero response for finite linkfun(y)"
        )
    elif family == "gaussian" and link == "sqrt":
        start = (
            "Gaussian/sqrt NULL start needs nonnegative response for finite linkfun(y)"
        )
    elif family == "gaussian" and link == "inverse_squared":
        start = (
            "Gaussian/1/mu^2 NULL start needs nonzero response; negative y gives "
            "finite positive eta"
        )
    return f"{start}; selected eta note: {_eta_domain(family, link)}"


def _efs_cell_status(
    family: str,
    link: str,
    parameter_mode: Literal["none", "fixed_theta", "estimated_theta"],
) -> EFSCellStatus:
    del parameter_mode
    if _r_constructor_status(family, link) == "rejected":
        return "r_rejected"
    return "implementation_missing"


def _numerical_boundaries(
    family: str,
    link: str,
    parameter_mode: Literal["none", "fixed_theta", "estimated_theta"],
) -> tuple[str, ...]:
    del family, link, parameter_mode
    return ()


_DENSE_EVIDENCE: dict[tuple[str, str, str], tuple[EvidenceScope, tuple[str, ...]]] = {
    ("gaussian", "identity", "none"): (
        "fixed_sp_pirls_coefficients_deviance_mu_moderate",
        ("tests/test_fitting/test_pirls.py::TestVsR::test_vs_r_gaussian",),
    ),
    ("binomial", "logit", "none"): (
        "fixed_sp_pirls_coefficients_deviance_mu_moderate",
        ("tests/test_fitting/test_pirls.py::TestVsR::test_vs_r_binomial",),
    ),
    ("poisson", "log", "none"): (
        "fixed_sp_pirls_coefficients_deviance_mu_moderate",
        ("tests/test_fitting/test_pirls.py::TestVsR::test_vs_r_poisson",),
    ),
    ("gamma", "inverse", "none"): (
        "fixed_sp_pirls_coefficients_deviance_mu_moderate",
        ("tests/test_fitting/test_pirls.py::TestVsR::test_vs_r_gamma",),
    ),
    ("gamma", "log", "none"): (
        "free_reml_score_edf_loose",
        (
            "tests/test_fitting/test_pirls.py::"
            "TestNonCanonicalNewtonHessian::test_gamma_log_free_reml_matches_r",
        ),
    ),
    ("nb", "log", "estimated_theta"): (
        "free_reml_deviance_theta_loose",
        (
            "tests/test_fitting/test_nb_fitting.py::TestNBNonCanonicalLink::test_nb_noncanonical_link_matches_r",
        ),
    ),
    ("nb", "identity", "estimated_theta"): (
        "free_reml_deviance_theta_loose",
        (
            "tests/test_fitting/test_nb_fitting.py::TestNBNonCanonicalLink::test_nb_noncanonical_link_matches_r",
        ),
    ),
    ("nb", "sqrt", "estimated_theta"): (
        "free_reml_deviance_theta_loose",
        (
            "tests/test_fitting/test_nb_fitting.py::TestNBNonCanonicalLink::test_nb_noncanonical_link_matches_r",
        ),
    ),
    ("nb", "log", "fixed_theta"): (
        "dense_invariants_only",
        (
            "tests/test_fitting/test_nb_fitting.py::TestNBFixedTheta::test_converges",
            "tests/test_fitting/test_nb_fitting.py::TestNBHardGateInvariants::test_deviance_non_negative",
        ),
    ),
}


def _entry(
    family: str,
    link: str,
    parameter_mode: Literal["none", "fixed_theta", "estimated_theta"],
) -> FamilyExecutionInventoryEntry:
    key = (family, link, parameter_mode)
    evidence_scope, evidence = _DENSE_EVIDENCE.get(
        key,
        (
            "unverified",
            ("constructor accepts every registered Link via ExponentialFamily",),
        ),
    )
    stream = (
        "canonical_fixed_sp_current"
        if key
        in {
            ("gaussian", "identity", "none"),
            ("binomial", "logit", "none"),
            ("poisson", "log", "none"),
        }
        else "not_yet_routed"
    )
    return FamilyExecutionInventoryEntry(
        family=family,
        link=link,
        parameter_mode=parameter_mode,
        default_link=link == _DEFAULT_LINK[family],
        mathematical_canonical=family != "nb" and link == _CANONICAL[family],
        # The current NB class leaves canonical_link_cls unset, so this is
        # false even for its default log link.  This records legacy behavior;
        # it is not a mathematical-information claim.
        legacy_route_is_canonical=family != "nb" and link == _CANONICAL[family],
        r_constructor_status=_r_constructor_status(family, link),
        r_advertised=link in _R_ADVERTISED_LINKS[family],
        r_constructor_link=_R_CONSTRUCTOR_LINK[link],
        r_start_validity_notes=_r_start_validity_notes(family, link),
        efs_status=_efs_cell_status(family, link, parameter_mode),
        numerical_boundaries=_numerical_boundaries(family, link, parameter_mode),
        dense_status=(
            "dense_r_parity_partial"
            if evidence_scope not in {"dense_invariants_only", "unverified"}
            else "dense_generic_unverified"
        ),
        evidence_scope=evidence_scope,
        stream_status=stream,
        evidence=(
            *evidence,
            *(("pinned mgcv 1.9-3 R/efam.r:160-168",) if family == "nb" else ()),
        ),
    )


FAMILY_EXECUTION_INVENTORY: tuple[FamilyExecutionInventoryEntry, ...] = tuple(
    _entry(family, link, "none")
    for family in ("gaussian", "binomial", "poisson", "gamma")
    for link in _LINKS
) + tuple(
    _entry("nb", link, mode)
    for link in _LINKS
    for mode in ("fixed_theta", "estimated_theta")
)
