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
    elif family == "poisson" and link in {"logit", "probit", "cloglog"}:
        start = (
            "Poisson bounded-link NULL start uses y + 0.1; an integer y=1 "
            "leaves the inverse-link domain, so that profile needs an explicit "
            "valid coefficient start"
        )
    return f"{start}; selected eta note: {_eta_domain(family, link)}"


def _efs_cell_status(
    family: str,
    link: str,
    parameter_mode: Literal["none", "fixed_theta", "estimated_theta"],
) -> EFSCellStatus:
    if _r_constructor_status(family, link) == "rejected":
        return "r_rejected"
    if (family, link, parameter_mode) == ("nb", "log", "estimated_theta"):
        return "internal_pinned_parity_with_named_boundary"
    return "internal_pinned_parity"


def _numerical_boundaries(
    family: str,
    link: str,
    parameter_mode: Literal["none", "fixed_theta", "estimated_theta"],
) -> tuple[str, ...]:
    boundaries: list[str] = []
    if (family, link, parameter_mode) == ("nb", "log", "estimated_theta"):
        boundaries.append("near_poisson_selected_theta_loose_exception")
    if family == "binomial" and link == "log":
        boundaries.append("initial_alpha_near_zero_resolution_unresolved")
    if family == "poisson" and link in {"logit", "probit", "cloglog"}:
        boundaries.append("integer_one_response_requires_explicit_valid_start")
    if family == "nb" and link in {"identity", "sqrt"}:
        boundaries.extend(
            (
                "signed_observed_curvature_uses_EFS_only_policy",
                "selected_fit_moderate_after_four_correction_passes",
            )
        )
    return tuple(boundaries)


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

# ``internal_pinned_parity`` is intentionally narrower than constructor
# acceptance. These identifiers exercise EFS under the input profile documented
# below; they do not certify every family/link input admitted by R.
_EFS_EVIDENCE: dict[tuple[str, str, str], tuple[str, ...]] = {
    ("gaussian", "identity", "none"): (
        "tests/test_execution/test_efs.py::test_unknown_scale_gaussian_efs_keeps_score_phi_separate_from_fletcher",
    ),
    ("gaussian", "log", "none"): (
        "tests/test_execution/test_efs_regular_links.py::test_advertised_regular_link_efs_matches_pinned_r_strict",
        "tests/test_fitting/test_efs_regular_pirls.py::test_final_gdi1_provenance_matches_live_pinned_r",
    ),
    ("gaussian", "inverse", "none"): (
        "tests/test_execution/test_efs_regular_links.py::test_advertised_regular_link_efs_matches_pinned_r_strict",
    ),
    ("binomial", "logit", "none"): (
        "tests/test_execution/test_efs.py::test_known_scale_efs_matches_pinned_r_from_matched_initial_state",
    ),
    ("binomial", "probit", "none"): (
        "tests/test_execution/test_efs_regular_links.py::test_advertised_regular_link_efs_matches_pinned_r_strict",
    ),
    ("binomial", "cloglog", "none"): (
        "tests/test_execution/test_efs_regular_links.py::test_advertised_regular_link_efs_matches_pinned_r_strict",
    ),
    ("binomial", "log", "none"): (
        "tests/test_execution/test_efs_regular_links.py::test_advertised_regular_link_efs_matches_pinned_r_strict",
    ),
    ("poisson", "log", "none"): (
        "tests/test_execution/test_efs.py::test_known_scale_efs_matches_pinned_r_from_matched_initial_state",
        "tests/test_execution/test_efs.py::test_coupled_efs_statistics_and_fit_match_pinned_r",
    ),
    ("poisson", "identity", "none"): (
        "tests/test_execution/test_efs_regular_links.py::test_advertised_regular_link_efs_matches_pinned_r_strict",
        "tests/test_fitting/test_efs_regular_pirls.py::test_invalid_gdi1_candidate_returns_pre_gdi1_feasible_state",
    ),
    ("poisson", "sqrt", "none"): (
        "tests/test_execution/test_efs_regular_links.py::test_advertised_regular_link_efs_matches_pinned_r_strict",
    ),
    ("gamma", "inverse", "none"): (
        "tests/test_execution/test_efs.py::test_unknown_scale_gamma_efs_matches_pinned_r",
    ),
    ("gamma", "log", "none"): (
        "tests/test_execution/test_efs.py::test_unknown_scale_gamma_efs_matches_pinned_r",
    ),
    ("gamma", "identity", "none"): (
        "tests/test_execution/test_efs_regular_links.py::test_advertised_regular_link_efs_matches_pinned_r_strict",
    ),
    ("nb", "log", "fixed_theta"): (
        "tests/test_execution/test_efs.py::test_fixed_theta_nb_log_efs_matches_pinned_r_with_real_weights_and_offsets",
    ),
    ("nb", "log", "estimated_theta"): (
        "tests/test_execution/test_efs.py::test_estimated_nb_efs_controller_matches_pinned_matched_start_trace",
    ),
}

_NB_NONCANONICAL_SELECTED_GATE = (
    "tests/test_execution/test_efs_nb_links.py::"
    "test_nb_noncanonical_efs_matched_start_default_controller_profile_matches_pinned_r"
)
_NB_CONDITIONAL_STRICT_GATE = (
    "tests/test_fitting/test_efs_theta.py::"
    "test_conditional_theta_nonlog_links_match_pinned_r_strict"
)
for _link in ("identity", "sqrt"):
    _EFS_EVIDENCE[("nb", _link, "fixed_theta")] = (_NB_NONCANONICAL_SELECTED_GATE,)
    _EFS_EVIDENCE[("nb", _link, "estimated_theta")] = (
        _NB_NONCANONICAL_SELECTED_GATE,
        _NB_CONDITIONAL_STRICT_GATE,
    )

_NB_REJECTED_LINK_GATE = (
    "tests/test_execution/test_efs.py::"
    "test_known_scale_efs_rejects_links_rejected_by_pinned_nb"
)
for _link in ("logit", "inverse", "probit", "cloglog", "inverse_squared"):
    for _mode in ("fixed_theta", "estimated_theta"):
        _EFS_EVIDENCE[("nb", _link, _mode)] = (_NB_REJECTED_LINK_GATE,)

_CONSTRUCTOR_EXTENSION_GATE = (
    "tests/test_execution/test_efs_regular_links.py::"
    "test_constructor_extension_regular_link_efs_matches_pinned_r_strict"
)
for _key in (
    *(
        ("gaussian", link, "none")
        for link in ("logit", "probit", "cloglog", "sqrt", "inverse_squared")
    ),
    *(
        ("binomial", link, "none")
        for link in ("identity", "inverse", "sqrt", "inverse_squared")
    ),
    *(
        ("gamma", link, "none")
        for link in ("logit", "probit", "cloglog", "sqrt", "inverse_squared")
    ),
):
    _EFS_EVIDENCE[_key] = (_CONSTRUCTOR_EXTENSION_GATE,)

_POISSON_EXPLICIT_START_GATE = (
    "tests/test_execution/test_efs_regular_links.py::"
    "test_nonadvertised_poisson_efs_valid_explicit_start_matches_pinned_r_strict"
)
for _link in ("logit", "inverse", "probit", "cloglog", "inverse_squared"):
    _key = ("poisson", _link, "none")
    _EFS_EVIDENCE[_key] = (_POISSON_EXPLICIT_START_GATE,)
for _link in ("logit", "probit", "cloglog"):
    _key = ("poisson", _link, "none")
    _EFS_EVIDENCE[_key] = (
        *_EFS_EVIDENCE[_key],
        "tests/test_execution/test_efs_regular_links.py::"
        "test_bounded_poisson_efs_default_start_failure_matches_pinned_r",
    )

_EFS_FIXTURE_PROFILE = (
    "EFS evidence profile: dense identifiable design, positive prior weights and "
    "working weights within clipping bounds, and the named basis/scale fixture"
)


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
            *_EFS_EVIDENCE.get(key, ()),
            *(
                (_EFS_FIXTURE_PROFILE,)
                if _efs_cell_status(family, link, parameter_mode).startswith(
                    "internal_pinned_parity"
                )
                else ()
            ),
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
