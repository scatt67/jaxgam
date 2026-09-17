# Shared Gaussian source initialization correction

Pinned mgcv 1.9-3 public `estimate.gam` applies `fix.family` before fitting
(`R/mgcv.r:1914`; `R/gam.fit3.r:2552–2555`). Gaussian/log uses
`mustart <- pmax(y, .01 * sd(y))`; Gaussian/inverse uses
`mustart <- y + (y == 0) * sd(y) * .01`. Their global unweighted sample SD
includes every real response, including observations with zero prior weight.

Gaussian owns three additional scalar count/mean/centered-square-sum leaves
for these two links. CPU/JIT batch summaries merge global response moments
without retaining rows. Finalized metadata supplies the source-patched
mustart; calls without metadata retain the unpatched initializer. Empty and
padded batches are neutral. Other Gaussian links retain the four-leaf summary.
The null coefficient anchor still uses the distinct unweighted response mean.

Dense EFS uses this metadata in its initial working state and incoming
smoothing initializer. Source `initial.spg` accepts neutral zero working
weights; `get.null.coef` uses zero priors neutrally in its null deviance and
retains the original row denominator in `null.scale / 10`. Global informative
support, finite statistics and nonnegative priors remain required. Full-fit
zero-prior admission is limited to the owning-tested Gaussian/log and inverse
noncanonical source loops. Positive priors retain the existing lower/upper
bounds; canonical and NB fitting guards retain their prior behavior.

The validation matrix owns actual default public EFS fits with the preserved
negative log-link responses or zero inverse-link responses, unequal priors,
a real zero-prior row and an offset. Coefficients, fitted means, deviance,
REML, smoothing, EDF, covariance, null deviance and Fletcher scale start at
repository STRICT. Layer tests compare actual `fix.family`, `initial.spg`
and `get.null.coef` output and cover CPU/JIT/batch-global summaries, invalid
priors, global zero support, singleton SD and supplied-coefficient starts.
Reported Fletcher `sig2` remains distinct from the score's trial scale.

This publication slice is mechanically split from reviewed component
`15a3fdf0db93de47a813119f870a0f28978c2d2f`. Its shared production hunks are
preserved exactly. The unpublished regular controller, signed recovery
hooks and controller-dependent common-trial-scale test remain in that later
milestone. Binomial source materialization/JVP changes from `4a9aa12` are
excluded. This correction does not complete PR7.3/PR7.4 regular streaming.
