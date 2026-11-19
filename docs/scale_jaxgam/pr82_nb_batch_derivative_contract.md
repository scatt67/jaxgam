# PR8.2 Negative Binomial batch derivative contract

This component adds the row-local derivative kernel and the exact fixed-state
streamed REML host for Negative Binomial theta. It does not add an optimizer
or substitute the EFS conditional-theta update for joint REML.

The source mapping is mgcv 1.9-3 `R/gam.fit4.r` lines 561–628,
`R/efam.r` lines 206–273, and `src/gdi.c`'s IFT derivative assembly. At the
selected coefficient state, the batch exports derivatives of the observed
information `X' (Deta2 / 2) X`, deviance, and saturated likelihood with
respect to fitting-coordinate beta and explicit `log_theta`. It separately
exports

```text
partial_log_theta g = X' (Detath / 2)
```

for the coefficient stationarity equation

```text
g = X' (Deta / 2) + S beta = 0.
```

The existing dense custom JVP writes the same equation multiplied by two as
`X' Deta + 2 S beta = 0`. The streamed host must solve its adjoint with
`X' (Deta2 / 2) X + S` and contract the exported vector as
`-a' partial_log_theta g`; mixing the two conventions would introduce an
incorrect factor of two.

Theta is always an explicit array leaf. The family and execution context are
static configuration and lineage checks; kernels never read mutable family
theta. Estimated and fixed family modes use identical batch statistics. The
estimated host retains the documented joint order `[rho, log_theta]`; the
fixed host consumes the beta cotangent and omits the returned theta coordinate.
Negative Binomial fixes `phi=1`, so no placeholder scale coordinate is
returned.

The accepted count-prefix saturated-likelihood primitive is reused with the
global `max_y`, integer-count mode, and per-row count indices passed
explicitly. Padding is neutral, invalid real rows fail closed, and outputs are
only two coefficient vectors, one theta scalar, two counts, and a validity
flag. A compiled `n=128, p=17` gate charges temporary storage against eight
batch-by-coefficient arrays plus sixteen coefficient-square arrays; no batch
or reverse-mode tape is retained by the result.

Validation covers all supported links (`log`, `identity`, `sqrt`), fixed and
estimated modes, theta values 0.1, 2.7 and 1e6, fractional responses, a count
tail through 1000, exact pinned-R `Dd`/`ls` contractions, independent dense AD,
padded batch reduction, dynamic-theta cache isolation, and an independent
five-point coefficient-refit finite difference. All numerical assertions
start and remain at repository `STRICT` tolerance.

## Fixed-state host assembly

`evaluate_nb_stream_reml` accepts `[rho, log_theta]` for an estimated family
and `rho` for a fixed-theta family. Every call reconverges the coefficient
subproblem at the requested immutable theta with
`fit_nb_streamed_pirls(..., estimate_theta=False)`. The conditional EFS theta
controller is never called. A warm start supplies only coefficients from a
compatible converged trial; score statistics and theta parameters are always
recomputed from the replayable source.

The host preserves `gam.fit4`/`gdi2` score provenance. Raw deviance and the
observed information belong to the reported coefficient state. The penalty
and its direct smoothing derivative use the separately retained final source
solve candidate. The observed signed-QR factor supplies the determinant,
inverse, and adjoint solve. After reducing the batch cotangents, the assembled
theta derivative is

```text
partial score / partial log_theta
  - adjoint' partial_log_theta g.
```

Pinned R returns extended-family coordinates before smoothing coordinates;
the oracle gate explicitly converts its `[log_theta, rho]` result to the
JaxGAM `[rho, log_theta]` contract. Fixed-theta R drops the theta derivative,
as does the host.

The memory preflight first rejects the base controller and O(Bp+p²) adjoint
ledger. It then performs one bounded scalar count-summary scan before any
prefix allocation. Every fixed/trial coefficient fit uses the planned prefix
again for its final saturated-likelihood score, so the shared NB controller
charges four live float64 tables plus 512 bytes of fixed aligned-buffer
headroom even when conditional theta is disabled. The compatibility score
kernel chooses its integer path per batch, so this charge also applies when a
different batch makes the global source fractional. Conditional theta retains
its distinct sixteen-table value/gradient/Hessian peak when the global count
plan is integral. The planned host derivative uses that global plan, shares
the controller's four-table phase maximum, and separately charges its
O(Bp+p²) arrays and retained source-solve coefficient copy. Large-count tests
prove an insufficient budget fails after only the bounded summary and before
coefficient working scans, score dispatch, or derivative dispatch. The
summary and derivative scans are included in the returned source and batch
counts.

Host validation adds all three links against live pinned `gam.fit4` for both
fixed and free theta, the existing dense joint-theta custom JVP, and full
five-point reconverged finite differences for both rho and log theta. It also
checks trial-theta/family isolation, fixed-theta coordinate omission, warm
starts, factor residuals, scan counts, and prospective large-count memory.
Every numerical comparison remains `STRICT`.
