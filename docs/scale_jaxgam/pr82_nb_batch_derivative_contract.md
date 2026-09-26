# PR8.2 Negative Binomial batch derivative contract

This component adds the row-local derivative kernel needed by a future exact
streamed REML host for estimated Negative Binomial theta. It does not add an
optimizer or substitute the EFS conditional-theta update for joint REML.

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
