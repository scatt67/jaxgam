"""Sum-to-zero and identifiability constraints via CoefficientMap.

Implements the constraint pipeline that R's mgcv applies during model setup:

1. **Per-smooth centering constraint** (``apply_sum_to_zero``): Absorbs
   the sum-to-zero constraint via QR decomposition, reducing each smooth
   by 1 column. Equivalent to R's ``smoothCon(absorb.cons=TRUE)``.

2. **Inter-term identifiability** (``gam_side``): When variable names
   repeat across smooths (e.g. ``s(x1) + te(x1,x2)``), detects and
   removes additionally confounded columns. Port of R's ``gam.side()``.

3. **CoefficientMap**: Immutable record of all constraints, used by
   predict/summary in Phase 3.

This module is Phase 1 (NumPy + SciPy only, no JAX imports).

Design doc reference: docs/design.md Section 5.10
R source reference: R/mgcv.r gam.side(), fixDependence(), augment.smX()
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt
from scipy import linalg

from jaxgam.smooths.by_variable import FactorBySmooth, NumericBySmooth
from jaxgam.smooths.tensor import TensorInteractionSmooth

if TYPE_CHECKING:
    from jaxgam.smooths.base import Smooth

    # Duck-typed union for smooth-like objects; FactorBySmooth and
    # NumericBySmooth share the same interface as Smooth but do not
    # inherit from it (see by_variable.py module docstring).
    SmoothLike = Smooth | FactorBySmooth | NumericBySmooth


# ---------------------------------------------------------------------------
# TermBlock
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TermBlock:
    """One term's constraint info in the model matrix.

    Records the position and reparameterization for one term (parametric
    or smooth) in the constrained model matrix.

    Parameters
    ----------
    label : str
        Term label (e.g. ``"s(x1)"``, ``"parametric"``).
    col_start : int
        Starting column in the constrained model matrix.
    n_coefs : int
        Number of coefficients after constraints.
    n_coefs_raw : int
        Number of coefficients before constraints.
    term_type : str
        ``"parametric"`` or ``"smooth"``.
    smooth : Smooth or FactorBySmooth or NumericBySmooth or None
        Reference to the smooth object (None for parametric terms).
    penalty_indices : tuple[int, ...]
        Indices into the global penalty list.
    Z_centering : np.ndarray or None
        Constraint absorption matrix from centering, shape ``(k, k-1)``.
    del_index : tuple[int, ...]
        Column indices deleted by ``gam_side``.
    """

    label: str
    col_start: int
    n_coefs: int
    n_coefs_raw: int
    term_type: str  # "parametric" | "smooth"
    smooth: SmoothLike | None = None
    penalty_indices: tuple[int, ...] = ()
    Z_centering: npt.NDArray[np.floating] | None = field(default=None, repr=False)
    del_index: tuple[int, ...] = ()


# ---------------------------------------------------------------------------
# CoefficientMap
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CoefficientMap:
    """Global immutable mapping from model coefficients to term structure.

    Created once during model setup (Phase 1). Used by ``predict()``,
    ``summary()``, and all post-estimation code (Phase 3).

    All constraint pipeline operations — centering absorption, dependence
    detection, and inter-term identifiability — are exposed as static or
    class methods on this class.

    Parameters
    ----------
    terms : tuple[TermBlock, ...]
        One entry per model term (parametric + smooths).
    total_coefs : int
        Total constrained coefficient count.
    total_coefs_raw : int
        Total raw (pre-constraint) coefficient count.
    has_intercept : bool
        Whether the model includes an intercept.
    """

    terms: tuple[TermBlock, ...]
    total_coefs: int
    total_coefs_raw: int
    has_intercept: bool

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def get_term(self, label: str) -> TermBlock:
        """Look up a term by label.

        Parameters
        ----------
        label : str
            Term label (e.g. ``"s(x1)"``, ``"intercept"``).

        Returns
        -------
        TermBlock
            The matching term block.

        Raises
        ------
        KeyError
            If no term matches the label.
        """
        for t in self.terms:
            if t.label == label:
                return t
        raise KeyError(f"No term '{label}'")

    def term_slice(self, label: str) -> slice:
        """Return slice for a term's columns in constrained space.

        Parameters
        ----------
        label : str
            Term label.

        Returns
        -------
        slice
            Column slice in the constrained model matrix.
        """
        t = self.get_term(label)
        return slice(t.col_start, t.col_start + t.n_coefs)

    # ------------------------------------------------------------------
    # Coefficient space transformations
    # ------------------------------------------------------------------

    def constrained_to_full(
        self, beta_c: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Map constrained coefficients to full (raw) space.

        For each term: insert zeros at ``del_index`` positions, then
        multiply by ``Z_centering`` if present.

        Parameters
        ----------
        beta_c : np.ndarray, shape ``(total_coefs,)``
            Constrained coefficient vector.

        Returns
        -------
        np.ndarray, shape ``(total_coefs_raw,)``
            Full coefficient vector in raw basis space.
        """
        beta_raw_parts: list[npt.NDArray[np.floating]] = []

        for term in self.terms:
            beta_term = beta_c[term.col_start : term.col_start + term.n_coefs]

            # Undo gam_side deletion: insert zeros at deleted positions
            if term.del_index:
                n_after_centering = term.n_coefs + len(term.del_index)
                beta_expanded = np.zeros(n_after_centering)
                keep = [j for j in range(n_after_centering) if j not in term.del_index]
                beta_expanded[keep] = beta_term
                beta_term = beta_expanded

            # Undo centering: multiply by Z
            if term.Z_centering is not None:
                beta_term = term.Z_centering @ beta_term

            beta_raw_parts.append(beta_term)

        return np.concatenate(beta_raw_parts)

    def full_to_constrained(
        self, beta_raw: npt.NDArray[np.floating]
    ) -> npt.NDArray[np.floating]:
        """Map full (raw) coefficients to constrained space.

        For each term: multiply by ``Z.T``, then delete at ``del_index``.

        Parameters
        ----------
        beta_raw : np.ndarray, shape ``(total_coefs_raw,)``
            Full coefficient vector in raw basis space.

        Returns
        -------
        np.ndarray, shape ``(total_coefs,)``
            Constrained coefficient vector.
        """
        beta_c_parts: list[npt.NDArray[np.floating]] = []
        raw_offset = 0

        for term in self.terms:
            beta_term = beta_raw[raw_offset : raw_offset + term.n_coefs_raw]
            raw_offset += term.n_coefs_raw

            # Apply centering: Z.T @ beta
            if term.Z_centering is not None:
                beta_term = term.Z_centering.T @ beta_term

            # Apply gam_side deletion
            if term.del_index:
                keep = [j for j in range(len(beta_term)) if j not in term.del_index]
                beta_term = beta_term[keep]

            beta_c_parts.append(beta_term)

        return np.concatenate(beta_c_parts)

    def transform_X(
        self, X_raw_block: npt.NDArray[np.floating], term_label: str
    ) -> npt.NDArray[np.floating]:
        """Transform a raw design matrix block to constrained space.

        Parameters
        ----------
        X_raw_block : np.ndarray, shape ``(n, n_coefs_raw)``
            Raw design matrix block for one term.
        term_label : str
            Term label.

        Returns
        -------
        np.ndarray
            Constrained design matrix block.
        """
        term = self.get_term(term_label)

        X = X_raw_block
        if term.Z_centering is not None:
            X = X @ term.Z_centering

        if term.del_index:
            keep = [j for j in range(X.shape[1]) if j not in term.del_index]
            X = X[:, keep]

        return X

    def transform_S(
        self, S_raw: npt.NDArray[np.floating], term_label: str
    ) -> npt.NDArray[np.floating]:
        """Transform a raw penalty matrix to constrained space.

        Parameters
        ----------
        S_raw : np.ndarray
            Raw penalty matrix.
        term_label : str
            Term label.

        Returns
        -------
        np.ndarray
            Constrained penalty matrix.
        """
        term = self.get_term(term_label)

        S = S_raw
        if term.Z_centering is not None:
            S = term.Z_centering.T @ S @ term.Z_centering
            S = 0.5 * (S + S.T)

        if term.del_index:
            keep = [j for j in range(S.shape[0]) if j not in term.del_index]
            S = S[np.ix_(keep, keep)]

        return S

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def build(
        cls,
        smooths: list[SmoothLike],
        X_smooth_blocks: list[npt.NDArray[np.floating]],
        S_smooth_blocks: list[list[npt.NDArray[np.floating]]],
        has_intercept: bool = True,
        n_parametric: int = 0,
        X_parametric: npt.NDArray[np.floating] | None = None,
        apply_centering: bool = True,
        apply_side: bool = True,
        tol: float = np.finfo(float).eps ** 0.5,
    ) -> tuple[
        CoefficientMap,
        list[npt.NDArray[np.floating]],
        list[list[npt.NDArray[np.floating]]],
    ]:
        """Orchestrate constraint pipeline and build CoefficientMap.

        Ties together centering, ``gam_side``, and ``CoefficientMap``
        construction. Called by design matrix assembly (Task 1.10).

        Parameters
        ----------
        smooths : list
            Smooth objects.
        X_smooth_blocks : list[np.ndarray]
            Raw design matrices for each smooth.
        S_smooth_blocks : list[list[np.ndarray]]
            Raw penalty matrices for each smooth.
        has_intercept : bool
            Whether the model has an intercept.
        n_parametric : int
            Number of parametric columns (including intercept).
        X_parametric : np.ndarray or None
            Parametric design matrix (for intercept detection in gam_side).
        apply_centering : bool
            Whether to apply sum-to-zero centering constraints.
        apply_side : bool
            Whether to apply gam_side identifiability constraints.
        tol : float
            Tolerance for dependence detection.

        Returns
        -------
        coef_map : CoefficientMap
            The immutable coefficient map.
        X_constrained : list[np.ndarray]
            Constrained smooth design matrices.
        S_constrained : list[list[np.ndarray]]
            Constrained smooth penalty matrices.
        """
        m = len(smooths)

        # Work on copies
        X_blocks = [X.copy() for X in X_smooth_blocks]
        S_blocks = [[S.copy() for S in Ss] for Ss in S_smooth_blocks]

        # Track centering Z matrices per smooth
        Z_centering_list: list[npt.NDArray[np.floating] | None] = [None] * m
        raw_n_coefs: list[int] = [X.shape[1] for X in X_blocks]

        # Apply sum-to-zero centering constraints
        if apply_centering:
            for i, sm in enumerate(smooths):
                if isinstance(sm, NumericBySmooth) and not getattr(
                    sm, "has_centering_constraint", True
                ):
                    continue

                # ti() already absorbs marginal constraints during construction;
                # applying centering again would incorrectly remove a column.
                base_smooth = getattr(sm, "base_smooth", sm)
                if isinstance(base_smooth, TensorInteractionSmooth):
                    continue

                if isinstance(sm, FactorBySmooth):
                    X_c, S_c, Z = cls.apply_sum_to_zero_factor_by(
                        X_blocks[i],
                        S_blocks[i],
                        sm.n_levels,
                        sm.k_per_level,
                        base_constraint=getattr(sm, "_base_constraint", None),
                    )
                else:
                    X_c, S_c, Z = cls.apply_sum_to_zero(X_blocks[i], S_blocks[i])

                X_blocks[i] = X_c
                S_blocks[i] = S_c
                Z_centering_list[i] = Z

        # Apply gam_side identifiability constraints
        del_indices: list[tuple[int, ...] | None] = [None] * m
        if apply_side:
            del_indices = cls.gam_side(
                smooths, X_blocks, S_blocks, X_parametric, tol=tol
            )

        # Build TermBlock for each term
        term_blocks: list[TermBlock] = []
        col_offset = 0
        penalty_offset = 0

        # Parametric term (intercept + linear)
        if n_parametric > 0:
            term_blocks.append(
                TermBlock(
                    label="parametric",
                    col_start=col_offset,
                    n_coefs=n_parametric,
                    n_coefs_raw=n_parametric,
                    term_type="parametric",
                )
            )
            col_offset += n_parametric

        # Smooth terms
        for i, sm in enumerate(smooths):
            n_penalties_i = len(S_blocks[i])
            penalty_indices = tuple(
                range(penalty_offset, penalty_offset + n_penalties_i)
            )
            penalty_offset += n_penalties_i

            label = cls.smooth_label(sm)
            n_coefs_constrained = X_blocks[i].shape[1]

            term_blocks.append(
                TermBlock(
                    label=label,
                    col_start=col_offset,
                    n_coefs=n_coefs_constrained,
                    n_coefs_raw=raw_n_coefs[i],
                    term_type="smooth",
                    smooth=sm,
                    penalty_indices=penalty_indices,
                    Z_centering=Z_centering_list[i],
                    del_index=(del_indices[i] if del_indices[i] is not None else ()),
                )
            )
            col_offset += n_coefs_constrained

        total_raw = n_parametric + sum(raw_n_coefs)

        coef_map = cls(
            terms=tuple(term_blocks),
            total_coefs=col_offset,
            total_coefs_raw=total_raw,
            has_intercept=has_intercept,
        )

        return coef_map, X_blocks, S_blocks

    # ------------------------------------------------------------------
    # Centering constraints
    # ------------------------------------------------------------------

    @staticmethod
    def apply_sum_to_zero(
        X: npt.NDArray[np.floating],
        S_list: list[npt.NDArray[np.floating]],
    ) -> tuple[
        npt.NDArray[np.floating],
        list[npt.NDArray[np.floating]],
        npt.NDArray[np.floating],
    ]:
        """Absorb sum-to-zero constraint via QR decomposition.

        Port of R's ``smoothCon`` centering absorption. The constraint
        vector is ``C = sum(X, axis=0)`` (R uses ``colSums``). The
        constraint is absorbed by projecting into the null space of C
        via QR.

        Parameters
        ----------
        X : np.ndarray, shape ``(n, k)``
            Design matrix for one smooth.
        S_list : list[np.ndarray]
            Penalty matrices, each shape ``(k, k)``.

        Returns
        -------
        X_c : np.ndarray, shape ``(n, k-1)``
            Constrained design matrix.
        S_c_list : list[np.ndarray]
            Constrained penalty matrices, each shape ``(k-1, k-1)``.
        Z : np.ndarray, shape ``(k, k-1)``
            Constraint absorption matrix (null space of C).
        """
        C = X.sum(axis=0)  # (k,) — matches R's colSums and tensor.py

        # QR of the constraint vector
        Q, _ = np.linalg.qr(C[:, np.newaxis], mode="complete")  # (k, k)
        Z = Q[:, 1:]  # (k, k-1) null space of C

        # Apply constraint
        X_c = X @ Z  # (n, k-1)
        S_c_list = []
        for S in S_list:
            S_c = Z.T @ S @ Z
            S_c = 0.5 * (S_c + S_c.T)  # symmetrize
            S_c_list.append(S_c)

        return X_c, S_c_list, Z

    @staticmethod
    def apply_sum_to_zero_factor_by(
        X: npt.NDArray[np.floating],
        S_list: list[npt.NDArray[np.floating]],
        n_levels: int,
        k_per_level: int,
        base_constraint: npt.NDArray[np.floating] | None = None,
    ) -> tuple[
        npt.NDArray[np.floating],
        list[npt.NDArray[np.floating]],
        npt.NDArray[np.floating],
    ]:
        """Apply centering constraint to each level's block.

        For ``FactorBySmooth``, R computes the constraint from the base
        smooth's full-data design matrix (before indicator multiplication),
        so all levels share the **same** Z matrix.  When ``base_constraint``
        is provided, we replicate that behaviour.

        Parameters
        ----------
        X : np.ndarray, shape ``(n, n_levels * k_per_level)``
            Block-structured design matrix.
        S_list : list[np.ndarray]
            Penalty matrices, each shape
            ``(n_levels * k_per_level, n_levels * k_per_level)``.
        n_levels : int
            Number of factor levels.
        k_per_level : int
            Basis dimension per level.
        base_constraint : np.ndarray or None
            ``colSums(X_base)`` from the base smooth evaluated on ALL data
            (before indicator multiplication).  When provided, all levels
            share the same constraint direction, matching R's ``smoothCon``.

        Returns
        -------
        X_c : np.ndarray
            Constrained design matrix.
        S_c_list : list[np.ndarray]
            Constrained penalty matrices.
        Z : np.ndarray
            Block-diagonal constraint absorption matrix.
        """
        # Compute the shared constraint direction from the base smooth
        if base_constraint is not None:
            C = base_constraint
        else:
            # Fallback: use first level's block (backwards compat)
            C = X[:, :k_per_level].sum(axis=0)

        Q, _ = np.linalg.qr(C[:, np.newaxis], mode="complete")
        Z_one = Q[:, 1:]  # same Z for all levels

        Z = linalg.block_diag(*([Z_one] * n_levels))

        X_c = X @ Z
        S_c_list = []
        for S in S_list:
            S_c = Z.T @ S @ Z
            S_c = 0.5 * (S_c + S_c.T)
            S_c_list.append(S_c)

        return X_c, S_c_list, Z

    # ------------------------------------------------------------------
    # Identifiability detection
    # ------------------------------------------------------------------

    @staticmethod
    def fix_dependence(
        X1: npt.NDArray[np.floating],
        X2: npt.NDArray[np.floating],
        tol: float = np.finfo(float).eps ** 0.5,
        rank_def: int = 0,
    ) -> list[int] | None:
        """Find columns of X2 linearly dependent on X1.

        Port of R's ``fixDependence`` (mgcv.r:502-533). Uses pivoted QR
        factorizations to detect linear dependence.

        Parameters
        ----------
        X1 : np.ndarray, shape ``(n, p1)``
            Reference design matrix.
        X2 : np.ndarray, shape ``(n, p2)``
            Design matrix to check for dependence on X1.
        tol : float
            Tolerance for detecting near-zero R diagonal elements.
        rank_def : int
            Known degree of rank deficiency. If > 0, used instead of tol.

        Returns
        -------
        list[int] or None
            0-based column indices of X2 that are dependent on X1,
            or None if X2 is fully independent of X1.
        """
        if X1.shape[0] != X2.shape[0]:
            raise ValueError(
                f"X1 and X2 must have the same number of rows, "
                f"got {X1.shape[0]} and {X2.shape[0]}."
            )

        r = X1.shape[1]

        # Pivoted QR of X1 (single full QR, matching R's qr() call)
        Q1, R1, _ = linalg.qr(X1, pivoting=True)
        R11 = abs(R1[0, 0])

        # Project X2 into residual space of X1
        QtX2 = Q1.T @ X2
        QtX2_resid = QtX2[r:, :]  # shape (n-r, p2)

        # Pivoted QR of residual
        _, R2, piv2 = linalg.qr(QtX2_resid, pivoting=True, mode="economic")

        r_dim = min(QtX2_resid.shape)

        # Scan R2 diagonal from bottom-right
        r0 = r_dim
        if rank_def > 0 and rank_def <= r_dim:
            r0 = r_dim - rank_def
        else:
            while r0 > 0:
                block = R2[r0 - 1 : r_dim, r0 - 1 : r_dim]
                if np.mean(np.abs(block)) >= R11 * tol:
                    break
                r0 -= 1

        if r0 >= r_dim:
            return None

        ind = [int(piv2[j]) for j in range(r0, r_dim)]

        if len(ind) == 0:
            return None
        return ind

    @staticmethod
    def gam_side(
        smooths: list[SmoothLike],
        X_blocks: list[npt.NDArray[np.floating]],
        S_blocks: list[list[npt.NDArray[np.floating]]],
        X_parametric: npt.NDArray[np.floating] | None = None,
        tol: float = np.finfo(float).eps ** 0.5,
        with_pen: bool = True,
    ) -> list[tuple[int, ...] | None]:
        """Detect and remove inter-term identifiability confounds.

        Port of R's ``gam.side`` (mgcv.r:564-728). Processes smooths
        from low to high dimension, detecting when higher-dimensional
        smooths contain columns linearly dependent on lower-dimensional
        ones.

        Parameters
        ----------
        smooths : list
            Smooth objects (Smooth, FactorBySmooth, or NumericBySmooth).
        X_blocks : list[np.ndarray]
            Design matrix for each smooth (already centering-constrained).
        S_blocks : list[list[np.ndarray]]
            Penalty matrices for each smooth.
        X_parametric : np.ndarray or None
            Parametric model matrix (for intercept detection).
        tol : float
            Tolerance for dependence detection.
        with_pen : bool
            Whether to augment with sqrt-penalty for detection.

        Returns
        -------
        list[tuple[int, ...] | None]
            For each smooth, tuple of 0-based column indices to delete,
            or None if no columns deleted.
        """
        m = len(smooths)
        if m == 0:
            return []

        # Collect variable names per smooth and max dimension
        v_names_all: list[str] = []
        max_dim = 1
        smooth_vn: list[list[str]] = []

        for sm in smooths:
            vn = CoefficientMap._smooth_variable_names(sm)
            smooth_vn.append(vn)
            v_names_all.extend(vn)
            dim_i = CoefficientMap._smooth_dim(sm)
            if dim_i > max_dim:
                max_dim = dim_i

        # Early return if all variable names unique (no nesting possible)
        if len(v_names_all) == len(set(v_names_all)):
            return [None] * m

        # Detect intercept in parametric matrix
        intercept = False
        if X_parametric is not None and X_parametric.shape[1] > 0:
            col_sds = np.std(X_parametric, axis=0)
            if np.any(col_sds < np.finfo(float).eps ** 0.75):
                intercept = True
            else:
                f = np.ones(X_parametric.shape[0])
                Q_p, _ = np.linalg.qr(X_parametric, mode="reduced")
                ff = Q_p @ (Q_p.T @ f)
                if np.max(np.abs(ff - f)) < np.finfo(float).eps ** 0.75:
                    intercept = True

        # Build index: for each unique variable name, which smooths use it
        unique_vars = list(dict.fromkeys(v_names_all))
        sm_id: dict[str, list[int]] = {v: [] for v in unique_vars}

        for d in range(1, max_dim + 1):
            for i, sm in enumerate(smooths):
                dim_i = CoefficientMap._smooth_dim(sm)
                if dim_i == d and CoefficientMap._smooth_side_constrain(sm):
                    for vn in smooth_vn[i]:
                        sm_id[vn].append(i)

        # Setup for augmented matrices if with_pen
        n_obs = X_blocks[0].shape[0] if X_blocks else 0

        if with_pen:
            k_start = 0
            p_inds: list[npt.NDArray[np.integer]] = []
            for i in range(m):
                k_i = X_blocks[i].shape[1]
                p_inds.append(np.arange(k_start, k_start + k_i))
                k_start += k_i
            total_np = k_start

            Xa_cache: dict[int, npt.NDArray[np.floating]] = {}

        # Process smooths from low to high dimension
        del_indices: list[tuple[int, ...] | None] = [None] * m

        for d in range(1, max_dim + 1):
            for i in range(m):
                dim_i = CoefficientMap._smooth_dim(smooths[i])
                if dim_i != d or not CoefficientMap._smooth_side_constrain(smooths[i]):
                    continue

                # Collect X columns from lower-dimensional smooths
                # sharing variables
                if with_pen:
                    if intercept:
                        X1 = np.concatenate(
                            [
                                np.ones((n_obs, 1)),
                                np.zeros((total_np, 1)),
                            ],
                            axis=0,
                        )
                    else:
                        X1 = np.zeros((n_obs + total_np, 0))
                else:
                    X1 = np.ones((n_obs, 1)) if intercept else np.zeros((n_obs, 0))

                x1_components: set[int] = set()

                for vn in smooth_vn[i]:
                    b = sm_id[vn]
                    try:
                        k_pos = b.index(i)
                    except ValueError:
                        continue
                    for lower_pos in range(k_pos):
                        lower_idx = b[lower_pos]
                        if lower_idx in x1_components:
                            continue
                        x1_components.add(lower_idx)

                        if with_pen:
                            if lower_idx not in Xa_cache:
                                Xa_cache[lower_idx] = CoefficientMap._augment_smooth_x(
                                    X_blocks[lower_idx],
                                    S_blocks[lower_idx],
                                    p_inds[lower_idx],
                                    total_np,
                                    n_obs,
                                )
                            Xa = Xa_cache[lower_idx]
                            X1 = np.column_stack([X1, Xa]) if X1.shape[1] > 0 else Xa
                        else:
                            Xb = X_blocks[lower_idx]
                            X1 = np.column_stack([X1, Xb]) if X1.shape[1] > 0 else Xb

                # Check for dependence
                n_intercept_cols = 1 if intercept else 0
                if X1.shape[1] <= n_intercept_cols:
                    continue

                if with_pen:
                    if i not in Xa_cache:
                        Xa_cache[i] = CoefficientMap._augment_smooth_x(
                            X_blocks[i],
                            S_blocks[i],
                            p_inds[i],
                            total_np,
                            n_obs,
                        )
                    ind = CoefficientMap.fix_dependence(X1, Xa_cache[i], tol=tol)
                else:
                    ind = CoefficientMap.fix_dependence(X1, X_blocks[i], tol=tol)

                if ind is not None:
                    del_indices[i] = tuple(ind)

                    # Apply deletions to X_blocks and S_blocks
                    keep = [j for j in range(X_blocks[i].shape[1]) if j not in ind]
                    X_blocks[i] = X_blocks[i][:, keep]
                    for s_idx, _S_mat in enumerate(S_blocks[i]):
                        S_blocks[i][s_idx] = _S_mat[np.ix_(keep, keep)]

                    # Recompute ALL parameter indices after deletion because
                    # later augmented matrices depend on the global offset.
                    if with_pen:
                        k_start = 0
                        for j in range(m):
                            k_j = X_blocks[j].shape[1]
                            p_inds[j] = np.arange(k_start, k_start + k_j)
                            k_start += k_j
                        total_np = k_start
                        Xa_cache.pop(i, None)

        return del_indices

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _mroot(
        S: npt.NDArray[np.floating], rank: int | None = None
    ) -> npt.NDArray[np.floating]:
        """Matrix square root via eigendecomposition.

        Equivalent to R's ``mroot()``. Returns matrix R such that
        ``R.T @ R ~ S`` (up to numerical rank).

        Parameters
        ----------
        S : np.ndarray, shape ``(k, k)``
            Symmetric PSD matrix.
        rank : int or None
            Expected rank. If None, uses all positive eigenvalues.

        Returns
        -------
        np.ndarray, shape ``(k, rank)``
            Matrix square root.
        """
        eigvals, eigvecs = linalg.eigh(S)

        if rank is None:
            tol = np.max(np.abs(eigvals)) * max(S.shape[0], 1) * np.finfo(float).eps
            pos = eigvals > tol
        else:
            idx = np.argsort(eigvals)[::-1]
            pos = np.zeros(len(eigvals), dtype=bool)
            pos[idx[:rank]] = True

        rS = eigvecs[:, pos] @ np.diag(np.sqrt(np.maximum(eigvals[pos], 0.0)))
        return rS  # (k, rank)

    @staticmethod
    def _augment_smooth_x(
        X: npt.NDArray[np.floating],
        S_list: list[npt.NDArray[np.floating]],
        p_ind: npt.NDArray[np.integer],
        total_np: int,
        n_obs: int,
    ) -> npt.NDArray[np.floating]:
        """Create augmented model matrix with sqrt-penalty rows.

        Port of R's ``augment.smX`` (mgcv.r:536-562). Stacks scaled
        sqrt-penalty below X at the smooth's parameter positions.

        Parameters
        ----------
        X : np.ndarray, shape ``(n, k)``
            Design matrix for one smooth.
        S_list : list[np.ndarray]
            Penalty matrices, each shape ``(k, k)``.
        p_ind : np.ndarray
            Parameter indices for this smooth in the global model
            (0-based), shape ``(k,)``.
        total_np : int
            Total number of penalized parameters.
        n_obs : int
            Number of observations (rows in original X).

        Returns
        -------
        np.ndarray, shape ``(n_obs + total_np, k)``
            Augmented design matrix.
        """
        k = X.shape[1]
        n_aug = n_obs + total_np

        if len(S_list) == 0:
            X_aug = np.zeros((n_aug, k))
            X_aug[:n_obs, :] = X
            return X_aug

        # Compute scaled penalty sum St
        # sqrmaX computed once from first penalty, reused for all
        # (matching R's augment.smX)
        ind0 = np.where(np.mean(np.abs(S_list[0]), axis=0) != 0)[0]
        if len(ind0) > 0:
            sqrmaX = np.mean(np.abs(X[:, ind0])) ** 2
            alpha = sqrmaX / np.mean(np.abs(S_list[0][np.ix_(ind0, ind0)]))
        else:
            sqrmaX = 1.0
            alpha = 1.0
        St = S_list[0] * alpha

        for i in range(1, len(S_list)):
            ind_i = np.where(np.mean(np.abs(S_list[i]), axis=0) != 0)[0]
            if len(ind_i) > 0:
                alpha_i = sqrmaX / np.mean(np.abs(S_list[i][np.ix_(ind_i, ind_i)]))
            else:
                alpha_i = 1.0
            St = St + S_list[i] * alpha_i

        # Matrix square root
        rS = CoefficientMap._mroot(St, rank=k)

        # Build augmented matrix
        X_aug = np.zeros((n_aug, k))
        X_aug[:n_obs, :] = X
        X_aug[n_obs + p_ind, :] = rS.T[: len(p_ind), :]

        return X_aug

    @staticmethod
    def _smooth_variable_names(
        sm: SmoothLike,
    ) -> list[str]:
        """Get variable names for a smooth, including by-variable suffixes.

        Matches R's naming: variable names are augmented with by-variable
        and by-level suffixes to distinguish different by-groups.

        Parameters
        ----------
        sm : Smooth or FactorBySmooth or NumericBySmooth
            The smooth to inspect.

        Returns
        -------
        list[str]
            Variable names, with by-variable suffix if applicable.
        """
        if isinstance(sm, (FactorBySmooth, NumericBySmooth)):
            return [v + sm.by_variable for v in sm.spec.variables]
        return list(sm.spec.variables)

    @staticmethod
    def _smooth_dim(
        sm: SmoothLike,
    ) -> int:
        """Get the dimension (number of covariates) of a smooth.

        Parameters
        ----------
        sm : Smooth or FactorBySmooth or NumericBySmooth
            The smooth to inspect.

        Returns
        -------
        int
            Number of covariates.
        """
        return len(sm.spec.variables)

    @staticmethod
    def _smooth_side_constrain(
        sm: SmoothLike,
    ) -> bool:
        """Get ``side_constrain`` flag for a smooth.

        Parameters
        ----------
        sm : Smooth or FactorBySmooth or NumericBySmooth
            The smooth to inspect.

        Returns
        -------
        bool
            Whether this smooth participates in side constraints.
        """
        if isinstance(sm, (FactorBySmooth, NumericBySmooth)):
            return getattr(sm.base_smooth, "side_constrain", True)
        return getattr(sm, "side_constrain", True)

    @staticmethod
    def smooth_label(
        sm: SmoothLike,
    ) -> str:
        """Get a human-readable label for a smooth.

        Parameters
        ----------
        sm : Smooth or FactorBySmooth or NumericBySmooth
            The smooth to label.

        Returns
        -------
        str
            Label string (e.g. ``"s(x1)"``, ``"te(x1,x2)"``).
        """
        if isinstance(sm, FactorBySmooth):
            vars_str = ",".join(sm.spec.variables)
            return f"{sm.spec.smooth_type}({vars_str},by={sm.by_variable})"
        elif isinstance(sm, NumericBySmooth):
            return sm.label
        vars_str = ",".join(sm.spec.variables)
        return f"{sm.spec.smooth_type}({vars_str})"
