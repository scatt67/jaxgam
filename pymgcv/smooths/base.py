"""Abstract Smooth base class for all smooth term types.

Defines the interface that every smooth (tp, ts, cr, cs, cc, tensor)
must implement. This module is Phase 1 (NumPy only, no JAX imports).

Design doc reference: docs/design.md Section 5.1
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt

from pymgcv.formula.terms import SmoothSpec
from pymgcv.penalties.penalty import Penalty


class Smooth(ABC):
    """Abstract base class for smooth terms.

    A smooth encapsulates the basis construction, penalty construction,
    and prediction matrix for one smooth term in a GAM formula.

    Parameters
    ----------
    spec : SmoothSpec
        Smooth term specification from the formula parser.
    """

    def __init__(self, spec: SmoothSpec) -> None:
        self.spec = spec
        self.n_coefs: int = 0
        self.null_space_dim: int = 0
        self.rank: int = 0

    @abstractmethod
    def setup(self, data: dict[str, npt.NDArray[np.floating]]) -> None:
        """Construct the smooth basis from data.

        This is the main construction method. After calling setup(),
        the smooth is ready to produce design and penalty matrices.

        Parameters
        ----------
        data : dict[str, np.ndarray]
            Mapping from variable names to data arrays. Must contain
            all variables referenced by ``self.spec.variables``.
        """

    @abstractmethod
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

    @abstractmethod
    def build_penalty_matrices(self) -> list[Penalty]:
        """Build the penalty matrices for this smooth.

        Returns
        -------
        list[Penalty]
            One or more penalty matrices. Most smooths have one;
            tensor products have one per marginal.
        """

    @abstractmethod
    def predict_matrix(
        self, new_data: dict[str, npt.NDArray[np.floating]]
    ) -> npt.NDArray[np.floating]:
        """Build the prediction matrix for new data.

        Parameters
        ----------
        new_data : dict[str, np.ndarray]
            Mapping from variable names to new data arrays.

        Returns
        -------
        np.ndarray
            Prediction matrix, shape ``(n_new, n_coefs)``.
        """
