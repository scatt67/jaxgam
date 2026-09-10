"""CPU packed-reflector QR reducer; no PIRLS or public routing."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy import linalg
from scipy.linalg import lapack

from jaxgam.fitting.qr import qr_root_inverse

LocalPenaltyRoot = tuple[slice, np.ndarray]


def _qt(packed: np.ndarray, tau: np.ndarray, z: np.ndarray) -> np.ndarray:
    """Apply the packed thin-Q transpose, including empty reflector cases."""
    if len(z) == 0 or len(tau) == 0:
        return np.array(z, dtype=float, copy=True)
    p = packed[:, : len(tau)]
    rhs = np.asfortranarray(z[:, None], dtype=float)
    f = lapack.get_lapack_funcs("ormqr", (p, rhs))
    _, work, info = f("L", "T", p, tau, rhs, lwork=-1)
    if info or work[0] < 1:
        raise np.linalg.LinAlgError("ormqr workspace failure")
    out, _, info = f("L", "T", p, tau, rhs, lwork=int(work[0]))
    if info:
        raise np.linalg.LinAlgError("ormqr failure")
    return out[:, 0]


@dataclass(frozen=True)
class PositiveQRState:
    R: np.ndarray
    qtz: np.ndarray
    residual_tail: float
    n_data_rows: int
    pivots: np.ndarray

    def __post_init__(self) -> None:
        for name in ("R", "qtz", "pivots"):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    @property
    def n_coef(self) -> int:
        return self.R.shape[1]


@dataclass(frozen=True)
class QRFactorSolve:
    R: np.ndarray
    pivots: np.ndarray
    coefficients: np.ndarray
    data_residual_tail: float
    original_n_coef: int
    keep: np.ndarray

    def __post_init__(self) -> None:
        for name in ("R", "pivots", "coefficients", "keep"):
            value = np.array(getattr(self, name), copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)

    def project(self, vector: np.ndarray) -> np.ndarray:
        """Select retained entries from a ``(p, ...)`` fitting-coordinate array."""
        value = np.asarray(vector)
        if value.ndim < 1 or value.shape[0] != self.original_n_coef:
            raise ValueError("vector is not in the factor's original coordinates")
        return np.array(value[self.keep, ...], copy=True)

    def reconstruct(self, reduced: np.ndarray) -> np.ndarray:
        """Embed a ``(rank, ...)`` array into ``(original_p, ...)`` coordinates."""
        value = np.asarray(reduced)
        if value.ndim < 1 or value.shape[0] != len(self.keep):
            raise ValueError("reduced vector does not match factor subspace")
        result = np.zeros((self.original_n_coef, *value.shape[1:]))
        result[self.keep, ...] = value
        return result


def _factor(
    A: np.ndarray, z: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    (packed, tau), R, piv = linalg.qr(A, mode="raw", pivoting=True)
    qtz = _qt(packed, tau, z)
    rows = min(A.shape)
    return (
        np.asarray(R[:rows]),
        qtz[:rows],
        float(qtz[rows:] @ qtz[rows:]),
        np.asarray(piv),
    )


def qr_update(
    state: PositiveQRState | None,
    weighted_X: np.ndarray,
    weighted_z: np.ndarray,
    *,
    data_rows: bool = True,
    n_coef: int | None = None,
) -> PositiveQRState:
    """Merge one non-negative-curvature row block into packed QR state.

    ``weighted_X`` has shape ``(rows, p)`` and ``weighted_z`` has shape
    ``(rows,)``.  Empty blocks are neutral.  When no preceding state exists,
    their otherwise-unknown coefficient dimension must be supplied as
    ``n_coef``.
    """
    A, z = np.asarray(weighted_X, float), np.asarray(weighted_z, float)
    if (
        A.ndim != 2
        or z.ndim != 1
        or len(A) != len(z)
        or not np.all(np.isfinite(A))
        or not np.all(np.isfinite(z))
    ):
        raise ValueError("finite matching weighted QR rows required")
    if n_coef is not None and (
        not isinstance(n_coef, (int, np.integer))
        or isinstance(n_coef, (bool, np.bool_))
        or n_coef != A.shape[1]
    ):
        raise ValueError("n_coef must match the weighted QR coefficient dimension")
    if len(A) == 0:
        if state is not None:
            return state
        if n_coef is None:
            raise ValueError("empty QR update requires matching n_coef")
        return PositiveQRState(
            np.empty((0, n_coef)), np.empty(0), 0.0, 0, np.arange(n_coef)
        )
    if state is None:
        R, f, tail, piv = _factor(A, z)
        return PositiveQRState(
            R, f, tail if data_rows else 0.0, len(A) if data_rows else 0, piv
        )
    if A.shape[1] != state.n_coef:
        raise ValueError("coefficient dimension changed")
    U = np.empty_like(state.R)
    U[:, state.pivots] = state.R
    R, f, tail, piv = _factor(np.vstack((U, A)), np.r_[state.qtz, z])
    return PositiveQRState(
        R,
        f,
        state.residual_tail + (tail if data_rows else 0.0),
        state.n_data_rows + (len(A) if data_rows else 0),
        piv,
    )


def reduce_positive_qr(
    batches: Iterable[tuple[np.ndarray, np.ndarray]], *, n_coef: int | None = None
) -> PositiveQRState:
    """Reduce replayed positive-curvature batches without retaining row arrays."""
    state = None
    for A, z in batches:
        if len(A) or state is not None:
            state = qr_update(state, A, z, n_coef=n_coef)
        elif n_coef is not None:
            state = qr_update(None, A, z, n_coef=n_coef)
        else:
            A_array, z_array = np.asarray(A, float), np.asarray(z, float)
            if (
                A_array.ndim != 2
                or z_array.ndim != 1
                or len(A_array) != len(z_array)
                or not np.all(np.isfinite(A_array))
                or not np.all(np.isfinite(z_array))
            ):
                raise ValueError("finite matching weighted QR rows required")
    if state is None:
        if (
            n_coef is None
            or not isinstance(n_coef, (int, np.integer))
            or isinstance(n_coef, (bool, np.bool_))
            or n_coef < 0
        ):
            raise ValueError("all-zero batches require their coefficient dimension")
        return PositiveQRState(
            np.empty((0, n_coef)), np.empty(0), 0.0, 0, np.arange(n_coef)
        )
    return state


def _validated_roots(
    state: PositiveQRState, roots: Iterable[LocalPenaltyRoot]
) -> tuple[LocalPenaltyRoot, ...]:
    """Validate bounded local roots before either rank check or augmentation."""
    checked: list[LocalPenaltyRoot] = []
    for where, root in roots:
        root = np.asarray(root, float)
        if not isinstance(where, slice) or where.step not in (None, 1):
            raise ValueError("local penalty roots require a unit coefficient slice")
        if (
            not isinstance(where.start, (int, np.integer))
            or isinstance(where.start, (bool, np.bool_))
            or not isinstance(where.stop, (int, np.integer))
            or isinstance(where.stop, (bool, np.bool_))
        ):
            raise ValueError("local penalty root bounds must be non-bool integers")
        if where.start < 0 or where.stop < where.start or where.stop > state.n_coef:
            raise ValueError("local penalty root slice is outside fitting coordinates")
        if (
            root.ndim != 2
            or root.shape[1] != where.stop - where.start
            or not np.all(np.isfinite(root))
        ):
            raise ValueError("invalid local penalty root")
        checked.append((where, root))
    return tuple(checked)


def _unpivot(R: np.ndarray, pivots: np.ndarray) -> np.ndarray:
    output = np.empty_like(R)
    output[:, pivots] = R
    return output


def _balanced_qr(
    data_R: np.ndarray, roots: Iterable[LocalPenaltyRoot], n_coef: int
) -> PositiveQRState | None:
    """Build the bounded, normalized ``pls_fit1`` rank system.

    This is deliberately separate from the actual lambda-scaled augmented QR.
    It follows ``pls_fit1``'s ``100 * eps * R_cond`` triangular condition
    check, rather than an SVD or square-root-epsilon rank cutoff.  Thus useful
    condition-1e8 directions remain identifiable while the common null space
    is removed before actual penalty scaling.
    """
    # ``pls_fit1`` first reduces each source independently.  Preserve that
    # bounded representation, then normalize the aggregate penalty factor
    # once (Es), rather than changing the relative contribution of individual
    # local roots by normalizing them separately.
    penalty_state = None
    for where, root in roots:
        if np.linalg.norm(root):
            embedded = np.zeros((len(root), n_coef))
            embedded[:, where] = root
            penalty_state = qr_update(
                penalty_state, embedded, np.zeros(len(root)), data_rows=False
            )

    balanced = None
    if data_R.size and np.linalg.norm(data_R):
        balanced = qr_update(
            None,
            data_R / np.linalg.norm(data_R),
            np.zeros(len(data_R)),
            data_rows=False,
        )
    if penalty_state is not None and np.linalg.norm(penalty_state.R):
        penalty_R = _unpivot(penalty_state.R, penalty_state.pivots)
        balanced = qr_update(
            balanced,
            penalty_R / np.linalg.norm(penalty_R),
            np.zeros(len(penalty_R)),
            data_rows=False,
        )
    return balanced


def _structural_rank(
    data_R: np.ndarray, roots: Iterable[LocalPenaltyRoot], n_coef: int
) -> int:
    """Return the rank of a pre-augmentation balanced data/root system."""
    balanced = _balanced_qr(data_R, roots, n_coef)
    if balanced is None or balanced.R.size == 0:
        return 0
    return _triangular_rank(balanced.R)


def _triangular_rank(R: np.ndarray) -> int:
    """Apply the pinned ``pls_fit1`` condition policy to a pivoted QR factor."""
    rank = min(R.shape)
    rank_tol = 100.0 * np.finfo(float).eps
    while rank:
        condition = _mgcv_r_cond(R[:rank, :rank])
        if np.isfinite(condition) and rank_tol * condition <= 1.0:
            break
        rank -= 1
    return rank


def _mgcv_r_cond(R: np.ndarray) -> float:
    """Port ``src/gdi.c::R_cond``'s triangular condition estimator exactly."""
    size = R.shape[1]
    if R.shape[0] < size:
        return np.inf
    p = np.zeros(size)
    y = np.empty(size)
    y_inf = 0.0
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        for column in range(size - 1, -1, -1):
            yp = (1.0 - p[column]) / R[column, column]
            ym = (-1.0 - p[column]) / R[column, column]
            pp = p[:column] + R[:column, column] * yp
            pm = p[:column] + R[:column, column] * ym
            if abs(yp) + np.sum(abs(pp)) >= abs(ym) + np.sum(abs(pm)):
                y[column] = yp
                p[:column] = pp
            else:
                y[column] = ym
                p[:column] = pm
            y_inf = max(y_inf, abs(y[column]))
        R_inf = max((np.sum(abs(R[row, row:size])) for row in range(size)), default=0.0)
    return float(R_inf * y_inf)


def solve_augmented_qr(
    state: PositiveQRState,
    roots: Iterable[LocalPenaltyRoot],
    *,
    identifiable_subspace: Sequence[int] | np.ndarray | None = None,
    balanced_roots: Iterable[LocalPenaltyRoot] | None = None,
) -> QRFactorSolve:
    """Solve actual local-root augmentation with explicit joint-alias policy.

    ``roots`` are the actual, possibly lambda-scaled local penalty roots for
    the solve. ``balanced_roots`` must be the corresponding unweighted or
    otherwise scale-independent roots from fitting metadata; only those roots
    establish the data/penalty common-null rank.  If omitted, ``roots`` are
    used as a fixed-system compatibility fallback, which is not suitable for
    free-smoothing streaming integration because zero/tiny lambda can hide a
    structurally identifiable direction.

    Returned coefficients are in *fitting* coordinates. If the balanced
    system is deficient, ``identifiable_subspace`` must name the fixed
    original-coordinate columns retained by CoefficientMap; omitted
    coordinates are returned exactly zero. An actual lambda-scaled factor that
    is ill-conditioned raises a distinct error and is never classified as a
    joint alias.
    """
    roots = _validated_roots(state, roots)
    rank_roots = roots
    if balanced_roots is not None:
        rank_roots = _validated_roots(state, balanced_roots)
        if len(rank_roots) != len(roots) or any(
            actual_where != balanced_where or actual_root.shape != balanced_root.shape
            for (actual_where, actual_root), (balanced_where, balanced_root) in zip(
                roots, rank_roots, strict=True
            )
        ):
            raise ValueError(
                "balanced_roots must match actual local-root slices and shapes"
            )
    data_R = _unpivot(state.R, state.pivots)
    balanced = _balanced_qr(data_R, rank_roots, state.n_coef)
    structural_rank = (
        0 if balanced is None or balanced.R.size == 0 else _triangular_rank(balanced.R)
    )
    raw = None if identifiable_subspace is None else np.asarray(identifiable_subspace)
    if raw is not None and (raw.ndim != 1 or (raw.size and raw.dtype.kind not in "iu")):
        raise ValueError("identifiable_subspace indices must be integer, not bool")
    if (
        structural_rank == state.n_coef
        and raw is not None
        and not np.array_equal(raw, np.arange(state.n_coef))
    ):
        raise ValueError("full-rank QR accepts only the identity subspace")
    if structural_rank < state.n_coef and identifiable_subspace is None:
        raise np.linalg.LinAlgError(
            "joint data/penalty alias requires an explicit identifiable subspace"
        )
    current = state
    for where, root in roots:
        E = np.zeros((len(root), state.n_coef))
        E[:, where] = root
        current = qr_update(current, E, np.zeros(len(root)), data_rows=False)
    if structural_rank < state.n_coef:
        keep = np.array(raw, dtype=int, copy=True)
        if (
            keep.ndim != 1
            or len(keep) != structural_rank
            or len(np.unique(keep)) != len(keep)
            or np.any((keep < 0) | (keep >= state.n_coef))
        ):
            raise ValueError(
                "identifiable_subspace must be unique original-coordinate "
                "joint-rank columns"
            )
        if structural_rank == 0:
            return QRFactorSolve(
                np.empty((0, 0)),
                np.empty(0, dtype=int),
                np.zeros(state.n_coef),
                state.residual_tail,
                state.n_coef,
                keep,
            )
        assert balanced is not None
        balanced_A = _unpivot(balanced.R, balanced.pivots)[:, keep]
        _, balanced_R, _ = linalg.qr(balanced_A, mode="economic", pivoting=True)
        if balanced_R.shape[0] < len(keep) or _triangular_rank(balanced_R) < len(keep):
            raise np.linalg.LinAlgError(
                "declared identifiable subspace is not structurally full rank"
            )
        A = _unpivot(current.R, current.pivots)[:, keep]
        q, R, piv = linalg.qr(A, mode="economic", pivoting=True)
        qtz = (
            q.T @ np.pad(current.qtz, (0, max(0, len(A) - len(current.qtz))))[: len(A)]
        )
        if R.shape[0] < len(keep) or _triangular_rank(R) < len(keep):
            raise np.linalg.LinAlgError(
                "actual augmented QR is numerically ill-conditioned after "
                "structural identification"
            )
        reduced = np.asarray(qr_root_inverse(R[: len(keep)], qtz[: len(keep)], piv))
        beta = np.zeros(state.n_coef)
        beta[keep] = reduced
        return QRFactorSolve(
            R[: len(keep)], piv, beta, state.residual_tail, state.n_coef, keep
        )
    R = current.R[: state.n_coef]
    qtz = np.pad(current.qtz, (0, max(0, state.n_coef - len(current.qtz))))[
        : state.n_coef
    ]
    if _triangular_rank(R) < state.n_coef:
        raise np.linalg.LinAlgError(
            "actual augmented QR is numerically ill-conditioned after "
            "structural identification"
        )
    return QRFactorSolve(
        R,
        current.pivots,
        np.asarray(qr_root_inverse(R, qtz, current.pivots)),
        state.residual_tail,
        state.n_coef,
        np.arange(state.n_coef),
    )
