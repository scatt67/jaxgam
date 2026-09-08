"""Checked-in evidence inventory for registered family/link execution cells.

This module is deliberately descriptive.  It is not consulted by fit drivers:
release routing must use family-owned capabilities plus an explicitly reviewed
policy, never an incidental family-name allowlist.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

InventoryStatus = Literal["dense_r_parity_partial", "dense_generic_unverified"]
RLinkStatus = Literal["allowed", "not_advertised_by_r"]
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
    r_link_status: RLinkStatus
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

# Pinned base-R family constructors own the standard-family allowed link sets.
# NB's set is explicitly checked in mgcv 1.9-3 R/efam.r:160-168.
_R_LINKS = {
    "gaussian": frozenset({"identity", "log", "inverse"}),
    "binomial": frozenset({"logit", "probit", "cloglog", "log"}),
    "poisson": frozenset({"log", "identity", "sqrt"}),
    "gamma": frozenset({"inverse", "log", "identity"}),
    "nb": frozenset({"log", "identity", "sqrt"}),
}
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
        ("tests/test_fitting/test_nb_fitting.py::TestNBNonCanonicalLink",),
    ),
    ("nb", "identity", "estimated_theta"): (
        "free_reml_deviance_theta_loose",
        ("tests/test_fitting/test_nb_fitting.py::TestNBNonCanonicalLink",),
    ),
    ("nb", "sqrt", "estimated_theta"): (
        "free_reml_deviance_theta_loose",
        ("tests/test_fitting/test_nb_fitting.py::TestNBNonCanonicalLink",),
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
            "dense_invariants_only"
            if family == "nb" and parameter_mode == "fixed_theta"
            else "unverified",
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
        r_link_status=(
            "allowed" if link in _R_LINKS[family] else "not_advertised_by_r"
        ),
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
