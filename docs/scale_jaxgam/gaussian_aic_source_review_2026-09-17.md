# Gaussian prior-weight AIC source correction

This change owns only `Gaussian.aic`, a CPU post-estimation family method. It
preserves stats R 4.5.2 `gaussian()$aic` (family.R lines 278–280): the logarithmic
weight term includes every prior weight. With positive deviance, any real zero
prior therefore yields positive infinity. With zero deviance and a zero prior,
the source expression yields NaN. Perfect fits with strictly positive priors
retain negative infinity. These are raw source diagnostics, not accepted-model
convergence flags.

The preserved reproducer is y=[0,1], mu=[.1,.9], wt=[0,1]. Pinned R returns +Inf;
the old method discarded the zero prior and returned -2.9208806002773837. The
correction includes all weights and suppresses only NumPy's expected arithmetic
warnings. It does not clip weights, replace source diagnostics, or change the
passed-scale convention: Gaussian family AIC uses deviance/n, irrespective of
that argument.

This differs intentionally from pinned mgcv 1.9-3 `fix.family.ls` in
R/gam.fit3.r lines 2501–2504: saturated Gaussian likelihood counts positive
priors and includes only their logarithms. That fitting likelihood remains
unchanged and finite for mixed positive/zero prior rows. Owning CPU/JIT tests
compare its value and first two scale derivatives to the actual pinned source,
separately from raw CPU AIC boundary tests.

This is independent of the frozen Gaussian global-start publication and the
unpublished regular controller. It makes no full family release, result-level
AIC, or stream admission claim. All new numerical assertions start at STRICT;
no tolerance exception is requested.

Validation: exact pinned image `gaussian-aic-source-r2-20260917` passed all
253 owning tests (96.44s, terminal exit 0, no skips/xfails); standard-family
statement coverage is 97.13%. Lint passes. The first revision retained an
empty-vector Python division-by-zero bug, now corrected with NumPy division
to preserve the source NaN. Its failure log and the earlier insufficient
coverage scope remain separate evidence; no assertion was weakened.
