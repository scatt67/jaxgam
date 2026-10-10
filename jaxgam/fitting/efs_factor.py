"""Exact EFS contractions using reviewed coefficient-factor actions.

Penalty roots remain local and one RHS block is padded to fitting coordinates
at a time. This kernel does not reconstruct information/covariance matrices,
change the dense Cholesky entry point, differentiate a factorization, or fit
coefficients. The provider owns the matching Fisher-state provenance.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from jaxgam.fitting.efs import (
    _RHS_COLUMN_BUDGET,
    EFSStatistics,
    EFSStatisticsPlan,
    _determinant_derivative,
)
from jaxgam.fitting.signed_qr import SignedQRCoefficientFactor
from jaxgam.fitting.state import (
    CholeskyCoefficientFactor,
    CoefficientFactor,
    PivotedQRCoefficientFactor,
)

jax.config.update("jax_enable_x64", True)


def efs_factor_statistics(
    plan: EFSStatisticsPlan,
    beta: jax.Array,
    fisher_factor: CoefficientFactor,
    rho: jax.Array,
    *,
    debug: bool = False,
) -> EFSStatistics:
    """Compute source d/t/q in the factor's fitting/rank coordinates.

    ``t = ||B^-T root||²`` uses the same reviewed factor action as prediction
    uncertainty. A truncated identifiable factor projects original-coordinate
    RHS into its retained subspace explicitly; this algebra alone does not
    authorize a new rank-deficient score domain. Providers must separately
    establish penalty/rank/source attribution and score admissibility.

    Maximum visible RHS allocation is p*min(32,r) for one root block, and
    rank*min(32,r) for its solved block, without an m-by-p-by-p stack. Native
    solve scratch is a separate compiled-memory observation.
    """
    if beta.shape != (plan.n_coef,) or rho.shape != (plan.n_penalties,):
        raise ValueError("matching EFS coefficient/parameter shapes required")
    if fisher_factor.n_coef != plan.n_coef:
        raise ValueError("EFS Fisher factor uses different fitting coordinates")
    if beta.dtype != jnp.float64 or rho.dtype != jnp.float64:
        raise TypeError("EFS factor statistics require float64 parameters")
    for leaf in jax.tree_util.tree_leaves(fisher_factor):
        if jnp.issubdtype(leaf.dtype, jnp.floating) and leaf.dtype != jnp.float64:
            raise TypeError("EFS Fisher factor requires float64 numeric leaves")
    coordinate_valid = jnp.asarray(True)
    if isinstance(fisher_factor, CholeskyCoefficientFactor):
        if fisher_factor.lower.shape != (plan.n_coef, plan.n_coef):
            raise ValueError("square matching Cholesky factor required")
    else:
        qr_factor = (
            fisher_factor.absolute_factor
            if isinstance(fisher_factor, SignedQRCoefficientFactor)
            else fisher_factor
        )
        if not isinstance(qr_factor, PivotedQRCoefficientFactor):
            raise TypeError("reviewed EFS coefficient factor required")
        rank = qr_factor.rank
        if (
            rank > plan.n_coef
            or qr_factor.R.shape != (rank, rank)
            or qr_factor.pivots.shape != (rank,)
            or qr_factor.keep.shape != (rank,)
        ):
            raise ValueError(
                "matching QR rank/pivot/retained-coordinate shapes required"
            )
        if not jnp.issubdtype(
            qr_factor.pivots.dtype, jnp.integer
        ) or not jnp.issubdtype(qr_factor.keep.dtype, jnp.integer):
            raise TypeError("QR coordinate indices must have integer dtype")
        ordered_keep = jnp.sort(qr_factor.keep)
        coordinate_valid = (
            (rank > 0)
            & jnp.all(jnp.sort(qr_factor.pivots) == jnp.arange(rank))
            & jnp.all((qr_factor.keep >= 0) & (qr_factor.keep < plan.n_coef))
            & jnp.all(jnp.diff(ordered_keep) > 0)
        )
        if isinstance(fisher_factor, SignedQRCoefficientFactor) and (
            fisher_factor.vectors.shape != (rank, rank)
            or fisher_factor.correction.shape != (rank,)
        ):
            raise ValueError("matching signed QR factor shapes required")
    d, valid = _determinant_derivative(plan, rho)
    t = jnp.zeros((plan.n_penalties,), dtype=beta.dtype)
    q = jnp.zeros((plan.n_penalties,), dtype=beta.dtype)
    actions_finite = jnp.asarray(True)
    for root in plan.roots:
        if (
            not 0 <= root.start <= root.stop <= plan.n_coef
            or not 0 <= root.sp_index < plan.n_penalties
        ):
            raise ValueError("EFS root uses different fitting/parameter coordinates")
        if root.values.dtype != jnp.float64:
            raise TypeError("EFS root values require float64")
        if root.kind == "dense":
            if root.values.ndim != 2 or root.values.shape[0] != root.stop - root.start:
                raise ValueError("matching local dense EFS root shape required")
        elif root.kind in {"identity", "diagonal"}:
            if (
                len(set(root.indices)) != len(root.indices)
                or any(i < 0 or i >= root.stop - root.start for i in root.indices)
                or (root.kind == "identity" and root.values.shape != ())
                or (
                    root.kind == "diagonal"
                    and root.values.shape != (len(root.indices),)
                )
            ):
                raise ValueError("matching local indexed EFS root shape required")
        else:
            raise ValueError("reviewed EFS root kind required")
        local_beta = beta[root.start : root.stop]
        if root.kind == "dense":
            projected_beta = root.values.T @ local_beta
            n_columns = root.values.shape[1]
        else:
            selected = jnp.asarray(root.indices, dtype=jnp.int32)
            projected_beta = root.values * local_beta[selected]
            n_columns = len(root.indices)
        q = q.at[root.sp_index].add(jnp.vdot(projected_beta, projected_beta).real)
        for start in range(0, n_columns, _RHS_COLUMN_BUDGET):
            stop = min(start + _RHS_COLUMN_BUDGET, n_columns)
            rhs = jnp.zeros((plan.n_coef, stop - start), dtype=beta.dtype)
            if root.kind == "dense":
                rhs = rhs.at[root.start : root.stop, :].set(root.values[:, start:stop])
            else:
                rows = root.start + jnp.asarray(
                    root.indices[start:stop], dtype=jnp.int32
                )
                values = (
                    root.values if root.kind == "identity" else root.values[start:stop]
                )
                rhs = rhs.at[rows, jnp.arange(stop - start)].set(values)
            solved = fisher_factor.root_transpose_inverse(rhs)
            actions_finite = actions_finite & jnp.all(jnp.isfinite(solved))
            t = t.at[root.sp_index].add(jnp.vdot(solved, solved).real)
    factor_finite = jnp.isfinite(fisher_factor.logdet_hessian())
    for leaf in jax.tree_util.tree_leaves(fisher_factor):
        factor_finite = factor_finite & jnp.all(jnp.isfinite(leaf))
    input_valid = (
        jnp.all(jnp.isfinite(beta))
        & jnp.all(jnp.isfinite(rho))
        & factor_finite
        & coordinate_valid
        & actions_finite
        & jnp.all(jnp.isfinite(d))
        & jnp.all(jnp.isfinite(t))
        & jnp.all(jnp.isfinite(q))
    )
    result = EFSStatistics(d, t, q, valid, input_valid)

    def report(_):
        jax.debug.print(
            "EFS factor d={d} t={t} q={q} valid={v}", d=d, t=t, q=q, v=input_valid
        )

    jax.lax.cond(jnp.asarray(debug), report, lambda _: None, operand=None)
    return result
