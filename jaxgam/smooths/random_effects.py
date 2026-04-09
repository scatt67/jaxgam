"""Dense random effects smooth (bs="re").

Treats random effects as penalized smooth terms with an identity penalty.
The model matrix is equivalent to R's ``model.matrix(~v1:v2:...:vN - 1)``
and the penalty is ``I_k`` (ridge), making it full rank with
``null_space_dim = 0``.

This module is Phase 1 (NumPy only, no JAX imports).

Design doc reference: docs/dense_random_effects/design.md
R source reference: R/smooth.r smooth.construct.re.smooth.spec()
"""

from __future__ import annotations

from typing import Any

import numpy as np
import numpy.typing as npt
import pandas as pd

from jaxgam.formula.terms import SmoothSpec
from jaxgam.penalties.penalty import Penalty
from jaxgam.smooths.base import Smooth
from jaxgam.smooths.utils import get_col, get_factor_levels, is_factor


class RandomEffectSmooth(Smooth):
    """Dense random effects smooth (bs="re").

    For ``s(v1, v2, ..., bs="re")``, constructs the model matrix equivalent
    to ``model.matrix(~v1:v2:...:vN - 1)`` with an identity penalty.

    The ``k`` argument from the spec is ignored -- the basis dimension is
    always ``ncol(X)`` (number of interaction levels).

    Parameters
    ----------
    spec : SmoothSpec
        Smooth term specification from the formula parser.
    """

    def __init__(self, spec: SmoothSpec) -> None:
        super().__init__(spec)
        self.side_constrain = False
        self._noterp = True
        self._random = True
        self._has_centering_constraint = False

        # Populated by setup()
        self._levels: dict[str, list[Any]] | None = None
        self._is_factor: dict[str, bool] | None = None
        self._X: npt.NDArray[np.floating] | None = None
        self._S: npt.NDArray[np.floating] | None = None

    def setup(self, data: dict[str, npt.NDArray[np.floating]]) -> None:
        """Construct RE basis from data.

        1. Determine which variables are factors, store levels.
        2. Build interaction model matrix (~v1:v2:...:vN - 1).
        3. Set penalty = normalized identity.

        Parameters
        ----------
        data : dict[str, np.ndarray]
            Mapping from variable names to data arrays.
        """
        variables = self.spec.variables

        # Detect factors and store levels
        self._is_factor = {}
        self._levels = {}
        for var in variables:
            col = get_col(data, var)
            if is_factor(col):
                self._is_factor[var] = True
                self._levels[var] = get_factor_levels(col)
            else:
                self._is_factor[var] = False

        # Build interaction model matrix
        X = self._build_interaction_matrix(data)

        self._X = X
        k = X.shape[1]
        self.n_coefs = k
        self.null_space_dim = 0
        self.rank = k

        # Penalty = identity, then normalize
        S = np.eye(k)
        [S], self._s_scale = self._smoothcon_normalize(X, [S])
        self._S = S

        self._is_setup = True

    def _build_interaction_matrix(
        self, data: dict[str, npt.NDArray[np.floating]]
    ) -> npt.NDArray[np.floating]:
        """Build ``~v1:v2:...:vN - 1`` interaction matrix.

        Row-wise Kronecker product of per-variable encodings:
        - Factor: one-hot indicator (n, L)
        - Numeric: column vector (n, 1)

        Column ordering matches R's ``model.matrix()``: first variable
        varies fastest.

        Parameters
        ----------
        data : dict[str, np.ndarray]
            Data containing all smooth variables.

        Returns
        -------
        np.ndarray
            Interaction model matrix, shape ``(n, k)``.
        """
        variables = self.spec.variables
        col0 = get_col(data, variables[0])
        n = len(col0)

        # Start with column of ones
        result = np.ones((n, 1))

        for var in variables:
            col = get_col(data, var)
            if self._is_factor[var]:
                levels = self._levels[var]
                term = self._encode_factor(col, levels)
            else:
                term = np.asarray(col, dtype=float).reshape(n, 1)

            # Interaction: row-wise Kronecker product
            # R's model.matrix() orders with earlier variables varying
            # fastest, so the new term's index is the outer (slow) index.
            n_old = result.shape[1]
            n_new = term.shape[1]
            new_result = np.empty((n, n_old * n_new))
            for j in range(n_new):
                for i in range(n_old):
                    new_result[:, j * n_old + i] = result[:, i] * term[:, j]
            result = new_result

        return result

    @staticmethod
    def _encode_factor(
        col: pd.Series | npt.NDArray,
        levels: list[Any],
    ) -> npt.NDArray[np.floating]:
        """One-hot encode a factor column against known levels.

        Observations with levels not in ``levels`` get NaN rows (handled
        downstream by zeroing non-finite values in ``predict_matrix``).

        Parameters
        ----------
        col : pd.Series or np.ndarray
            Factor column.
        levels : list
            Ordered levels to encode against.

        Returns
        -------
        np.ndarray
            Indicator matrix, shape ``(n, len(levels))``.
        """
        n = len(col)
        indicator = np.full((n, len(levels)), np.nan)
        col_arr = np.asarray(col)
        for j, lev in enumerate(levels):
            indicator[col_arr == lev, j] = 1.0

        # Rows where all columns are NaN have unseen levels; leave as NaN.
        # Rows that matched at least one level: fill remaining NaN with 0.
        matched = np.any(~np.isnan(indicator), axis=1)
        indicator[matched] = np.where(
            np.isnan(indicator[matched]), 0.0, indicator[matched]
        )
        return indicator

    def build_design_matrix(
        self, data: dict[str, npt.NDArray[np.floating]]
    ) -> npt.NDArray[np.floating]:
        """Build the design matrix for the given data.

        Parameters
        ----------
        data : dict[str, np.ndarray]
            Mapping from variable names to data arrays.

        Returns
        -------
        np.ndarray
            Design matrix, shape ``(n, n_coefs)``.
        """
        self._require_setup()
        return self.predict_matrix(data)

    def predict_matrix(
        self, new_data: dict[str, npt.NDArray[np.floating]]
    ) -> npt.NDArray[np.floating]:
        """Build the prediction matrix for new data.

        Unseen factor levels produce NaN in the indicator encoding,
        which are zeroed out so that new levels contribute zero to
        the linear predictor. Matches R's ``Predict.matrix.random.effect``.

        Parameters
        ----------
        new_data : dict[str, np.ndarray]
            Mapping from variable names to new data arrays.

        Returns
        -------
        np.ndarray
            Prediction matrix, shape ``(n_new, n_coefs)``.
        """
        self._require_setup()
        X = self._build_interaction_matrix(new_data)
        X[~np.isfinite(X)] = 0.0
        return X

    def build_penalty_matrices(self) -> list[Penalty]:
        """Build the penalty matrices for this smooth.

        Returns a single identity penalty (after smoothCon normalization)
        with full rank and zero null space dimension.

        Returns
        -------
        list[Penalty]
            Single-element list with the identity penalty.
        """
        self._require_setup()
        return [Penalty(self._S, rank=self.rank, null_space_dim=0)]

    def __repr__(self) -> str:
        vars_str = ",".join(self.spec.variables)
        return (
            f"RandomEffectSmooth(variables=[{vars_str}], "
            f"n_coefs={self.n_coefs}, rank={self.rank})"
        )
