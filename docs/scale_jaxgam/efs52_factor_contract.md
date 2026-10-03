# Exact factor contractions for streamed EFS

This bounded EFS5.2 component supplies exact manageable-p penalty contractions.
It does not enable public streamed EFS or conditional theta updates. The accepted
provider extraction remains unchanged.

The pinned mgcv 1.9-3 `R/gam.fit4.r:efsudr` (lines 821–954) projects coefficients
and the covariance root into the penalty range. For each penalty root it computes
`bSb = sum((Yb %*% UrS)^2)` and `trVS = sum((t(rV) %*% Y %*% UrS)^2)`; the
penalty determinant derivative comes from `ldetS1`. `efs_factor_statistics`
retains the reviewed determinant calculation and local prepared roots, computes
q with the existing operation order, and replaces only the dense triangular
solve with the reviewed coefficient factor's `root_transpose_inverse` action.
Neither information nor covariance matrices are reconstructed.

Cholesky, pivoted QR, and signed QR factors use their existing fitting coordinate
and rank conventions. Static mismatched dimensions/dtypes reject before execution.
Dynamic pivot permutations, unique retained indices, finite factors and finite
actions are checked under JIT. Empty rank is invalid. A reduced positive rank
computes a projection into the supplied retained subspace; this alone does not
authorize a rank-deficient fit or score. The provider must establish the actual
identifiable space, penalty rank, Fisher-state provenance, and rho/phi/theta/source
and basis attribution before returning a valid trial. Signed factors must come
from the reviewed solver; constructing arbitrary nonorthogonal signed vectors is
outside that factor contract.

Plans can be prepared from `PreparedFittingMetadata`, which contains no X, y,
weights or offsets. Its CPU root preparation body is unchanged. The original
`efs_statistics` and dense controller continue unchanged.

Each root uses at most 32 RHS columns per block, including a smaller final block.
One padded p-by-columns RHS and one rank-by-columns action output are visible at
a time; q uses only the local root projection. The plan's local roots and the
supplied factor are input storage. There is no m-by-p-by-p factor or covariance
stack. Native triangular solve scratch, compiler scheduling and device allocation
are separate measurements, rather than inferred from these source-level bounds.
A reproducible compiled-memory report records arguments, outputs, temporary and
alias bytes for explicit shapes; it is not an RSS or performance claim.

Owning tests compare Cholesky results byte for byte with the unchanged dense
kernel under the same JIT execution mode, and compare pivoted/signed contractions
at STRICT. A pinned RBridge oracle independently evaluates `gam.reparam` and
covariance-root contractions for coupled noncommuting penalties. Projection,
invalid coordinates, root attribution, trial isolation, RHS tail blocks and JIT
failure flags have separate layer gates. Broad final-model parity remains owned
by the validation matrix and subsequent streamed-provider components.

The first local check compared compiled factor output with an eager dense output:
it passed STRICT but differed by one final bit in a coupled determinant derivative.
The byte check now compares both kernels under JIT, matching the production mode;
no production arithmetic or tolerance was changed.

Compiled-memory checks use full-rank pivoted QR with disjoint singleton
identity roots, including final blocks smaller than 32. They report compiler
arguments, outputs, temporaries and aliases separately from the visible
RHS/action bound. Runtime RSS and allocator peaks require separate measurement;
compiler temporaries alone establish no general device-memory or speed claim.
