# Gaussian global initialization correction

This bounded component implements the Gaussian start installed by pinned
mgcv 1.9-3 `fix.family`, called by public `estimate.gam` before fitting
(`R/mgcv.r:1914`; `R/gam.fit3.r:2552–2555`). For log links the source uses
`pmax(y, .01 * sd(y))`; for inverse links it uses
`y + (y == 0) * .01 * sd(y)`. This permits a few negative log-link responses
or zero inverse-link responses when the whole model has an admissible null
anchor. The unpatched stats initializer remains the default primitive.

Gaussian owns an additional three-scalar count/mean/centered-sum-of-squares
summary for these two links. Batch kernels and merges are JIT-compatible;
Chan merges produce one global unweighted sample SD. Every real row,
including a zero-prior-weight row, contributes. Empty batches and invalid
padding contribute nothing. No response array is retained across batches.
Other Gaussian links retain the existing four-scalar likelihood summary.

The regular controller finalizes this metadata before validating and
replaying each patched start. Its null anchor still uses the distinct global
unweighted mean of the initialized response. The existing source scans and
workspace ledger stay in force; the extra persistent summary is three
float64 scalars. Noncanonical Gaussian reporting uses the source Fletcher
reduction, with observed-information score and Fisher covariance remaining
separate. The existing dense regular EFS null-start helper uses the same
family metadata; its supplied-coefficient start branch is unchanged.
The EFS incoming smoothing initializer also evaluates the patched mustart
from this global summary. Its regular-family working-weight gate permits
individual zero weights, as pinned `initial.spg` does, while rejecting
negative/nonfinite prior or working weights, nonfinite weighted design
statistics and a globally uninformative design. The NB observed/Fisher
selection and positive-weight admissibility remain unchanged. An owning
actual `initial.spg` oracle retains the zero-prior row and checks global
all-zero and invalid-prior rejection.
The incoming unknown-scale helper permits neutral zero priors when reducing
source `get.null.coef` deviance and requires positive global prior support.
Full EFS admission is narrower: only the validated Gaussian/log and
Gaussian/inverse noncanonical source loops admit individual zero priors.
Every positive weight keeps the existing lower/upper bounds. Canonical and
NB routes retain their previous fitting guards; this component does not
claim zero-prior support for those clipped PIRLS routes.

Owning gates compare actual pinned `fix.family(gaussian(...))` initialization
with CPU and JIT summaries across batch partitions and empty/padded batches.
Actual public `gam` fits include negative log responses or zero inverse
responses, unequal weights, a real zero weight and an offset. At a shared
trial scale, coefficients, means, deviance, reported Fletcher scale, EDF,
REML, covariance and prediction SE must pass repository STRICT across three
batch caps. The score uses the actual source `reml.scale`; `sig2` is its
distinct reported Fletcher scale. The initial failed diagnostic that used
`sig2` in the score is preserved, rather than relaxing that assertion.
The tests also own dense EFS patched starts, preserved supplied starts,
summary shape checks and the source singleton-SD domain failure.
Two validation-matrix gates require these same newly admitted responses to
finish through public `GAM(..., optimizer="efs")` and pinned
`gam(..., optimizer="efs")`, including smoothing, coefficients, fitted values,
deviance, score, EDF, covariance, null deviance and reported Fletcher scale.

This does not implement the outer trial-scale optimizer or finish the full
regular-family/link, no-intercept, control, resource and public release
requirements of PR7.3/PR7.4. The accepted signed recovery component remains
a separate parent commit.
