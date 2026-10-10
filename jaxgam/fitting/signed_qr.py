"""JIT coefficient-space actions for the signed ``pls_fit1`` correction.

The absolute-weight augmented QR root is corrected by ``I - 2 Qneg.T Qneg``.
Zero correction eigenvalues permit a coefficient pseudoinverse, but do not
define an admissible likelihood determinant. Indefinite corrections are
rejected by the host solver before this factor is constructed.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import jax
import jax.numpy as jnp

if TYPE_CHECKING:
    from jaxgam.fitting.state import PivotedQRCoefficientFactor


def _scale_rows(scale: jax.Array, value: jax.Array) -> jax.Array:
    return scale * value if value.ndim == 1 else scale[:, None] * value


@jax.tree_util.register_pytree_node_class
@dataclass(frozen=True)
class SignedQRCoefficientFactor:
    """Retained-coordinate signed factor with explicit score admissibility.

    ``H = P R.T V diag(correction) V.T R P.T``, embedded by ``keep``.
    Root actions use ``B = sqrt(correction) V.T R P.T``. At a zero
    eigenvalue they return the source generalized inverse in QR coordinates;
    ``score_admissible`` is false and ``logdet_hessian`` is minus infinity.
    """

    absolute_factor: PivotedQRCoefficientFactor
    vectors: jax.Array
    correction: jax.Array

    def tree_flatten(self):
        return (self.absolute_factor, self.vectors, self.correction), None

    @classmethod
    def tree_unflatten(cls, _auxiliary, children) -> SignedQRCoefficientFactor:
        return cls(*children)

    @property
    def n_coef(self) -> int:
        return self.absolute_factor.n_coef

    @property
    def rank(self) -> int:
        return self.absolute_factor.rank

    @property
    def score_admissible(self) -> jax.Array:
        return jnp.all(self.correction > 0.0)

    def root_inverse(self, rhs_rows: jax.Array) -> jax.Array:
        """Apply the source root inverse action with zero modes suppressed."""
        value = jnp.asarray(rhs_rows)
        if value.ndim not in (1, 2) or value.shape[0] != self.rank:
            raise ValueError("signed QR row RHS does not match retained factor rank")
        inverse_sqrt = jnp.where(
            self.correction > 0.0,
            jax.lax.rsqrt(jnp.where(self.correction > 0.0, self.correction, 1.0)),
            0.0,
        )
        return self.absolute_factor.root_inverse(
            self.vectors @ _scale_rows(inverse_sqrt, value)
        )

    def root_transpose_inverse(self, rhs_original: jax.Array) -> jax.Array:
        """Apply the transpose source root inverse to ``(p, ...)``."""
        value = self.vectors.T @ self.absolute_factor.root_transpose_inverse(
            rhs_original
        )
        inverse_sqrt = jnp.where(
            self.correction > 0.0,
            jax.lax.rsqrt(jnp.where(self.correction > 0.0, self.correction, 1.0)),
            0.0,
        )
        return _scale_rows(inverse_sqrt, value)

    def hessian_inverse(self, rhs_original: jax.Array) -> jax.Array:
        """Apply the source signed coefficient solve, including zero modes."""
        return self.root_inverse(self.root_transpose_inverse(rhs_original))

    def logdet_hessian(self) -> jax.Array:
        """Return the signed determinant only for a positive correction."""
        return jnp.where(
            self.score_admissible,
            self.absolute_factor.logdet_hessian() + jnp.sum(jnp.log(self.correction)),
            -jnp.inf,
        )
