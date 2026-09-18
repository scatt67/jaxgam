# Regular penalized release matrix: reviewed cancellation decision

The first pinned release gate preserves the original32 datasets/starts and uses an actual public slope penalty S=diag(0,1), lambda=.35. At B17,30cells passSTRICT; all4extreme-prior policies passSTRICT at B1/17/200. Exactly the two previously preserved Poisson/identity and Binomial/log seeded inputs failSTRICT under this nonempty penalty. No assertion or production bytes have changed.

The main reviewer explicitly approved this field-specific decision after reading the complete correction functions/counters and raw errors. The earlier no-penalty approval was not applied automatically.

Candidate1 is the already implemented source-good row correction, executed on the new nonempty-penalty full fit (the same real zero-prior row is omitted from working QR and retained in the public NULL projection). It is inherited reviewed production, not a newly written r9 edit. Candidates2–4 are distinct source-aligned solver corrections executed in the real full controller. Each nonempty-penalty final-QR override ran7times per full fit; literal-QtZ and source-order triangular overrides also ran7times when selected. Their complete functions and source-clone code are archived in the raw log. Source/Eta/penalty/input rows are unchanged.

| Fixture | Actual correction | Max beta abs | REML abs | Mean abs | Fisher covariance abs | Link SE abs | Source/Python iterations | STRICT restored |
|---|---|---:|---:|---:|---:|---:|---|---|
| poisson/identity | source_good_rows | 2.876272e-10 | 1.43387524e-10 | 2.11864415e-10 | 4.80669172e-12 | 2.29791464e-11 | 4/4 | No |
| poisson/identity | source_final_qr | 2.876272e-10 | 1.43387524e-10 | 2.11864415e-10 | 4.80669172e-12 | 2.29791464e-11 | 4/4 | No |
| poisson/identity | preserve_literal_qtz | 2.87627089e-10 | 1.43387524e-10 | 2.11864193e-10 | 4.80669541e-12 | 2.29792574e-11 | 4/4 | No |
| poisson/identity | source_triangular_order | 2.87627089e-10 | 1.43387524e-10 | 2.11864193e-10 | 4.80669541e-12 | 2.29792574e-11 | 4/4 | No |
| binomial/log | source_good_rows | 3.02982106e-09 | 5.37986722e-09 | 2.68429345e-09 | 2.18769239e-10 | 5.30959415e-10 | 4/4 | No |
| binomial/log | source_final_qr | 3.02982106e-09 | 5.37986722e-09 | 2.68429345e-09 | 2.18769239e-10 | 5.30959415e-10 | 4/4 | No |
| binomial/log | preserve_literal_qtz | 3.04629777e-09 | 6.84919854e-09 | 2.601383e-09 | 4.21828947e-10 | 1.33988814e-09 | 4/4 | No |
| binomial/log | source_triangular_order | 3.04629777e-09 | 6.84919854e-09 | 2.601383e-09 | 4.21828947e-10 | 1.33988814e-09 | 4/4 | No |

Source final candidates and Python final candidates are admissible in every measured case; both truly converge in4iterations at identical default epsilon1e-7/maxit200. Actual source-clone stopping penalized deviance and Python stopping penalized deviance agree exactly at baseline (candidate3/4 Binomial differ1.42e-14). The source raw pre-gdi deviance and candidate solve penalty remain separately attributed; their measured errors are archived individually. No raw weight sign or alpha replacement is altered.

Baseline production Poisson REML error1.43387524e-10 passesSTRICT; beta2.876272e-10, mean2.118644e-10, covariance4.806692e-12, SE2.297915e-11 fail selected STRICT assertions. Baseline Binomial beta3.029821e-9 and REML5.379867e-9 failSTRICT; mean2.684293e-9, covariance2.187692e-10, SE5.309594e-10 also fail. Baseline deviance, known scale, EDF and NULL anchor passSTRICT for both fixtures. Baseline SE maximum relative residues are1.7546e-10/3.3077e-9.

Approved exact scope: repository MODERATE only for beta, fitted means, Fisher covariance and link SE on these unchanged original83-row two fixtures with rank1 slope penalty lambda=.35 at B17; additionally REML for Binomial/log. Keep deviance, known scale, EDF, NULL anchor and Poisson REML STRICT. All other30cells and all extreme-prior gates remainSTRICT. Require source/Python true4iteration convergence, valid finite final candidates and unchanged input copies. No production policy, source recovery or guard changes are proposed; no LOOSE.

Numerical-pass evidence: `/private/tmp/jaxgam-pr73-regular-release-r9a-corrections-r2.py/.log/.json`, terminal0. The original JSON-bool diagnostic harness failure is preserved in r9a-corrections.log and is not counted as a correction attempt. The first owning gate remains immutable:2failed34passed95.29s,terminal1, r9a-owning.log. All176image Python files match the host; all175prior r8h files remain byte-identical. Image sha256:387f5c7d1cb56492fd9199deb78d419261b36382e09938683a7f4e58d836b965.


## Immutable scope guard

The checked-in gate checks the original case generator source, every input vector, model/basis fingerprint, local penalty layout/transform, unscaled penalty, rho and actual controller settings against immutable digests. Arrays/rho also match the untouched generator and natural lambda=.35 formula exactly. The digest serializes floating values at14significant digits to avoid incidental libm last-bit differences across pinned image architectures; this remains tighter than repositorySTRICT and does not change the actual shared R/Python inputs. The exact digests are Poisson/identity `e7bcb1ad84794e5ededa7859254b4b4bb2f38f7f93a146237a9d0a99575c28ec` and Binomial/log `a264744a1adf9473d6277b14e8d6571814ec107443a3c1380d4d6f0233f13e67`. No acceptance decision is applied to a changed digest, a different batch cap, or an extreme-prior case. Actual source pre-fallback and Python final candidates must both be valid; both must truly converge in4iterations.

This slice changes only owning tests and this numerical record. All175previous r8h production/test/script Python files remain byte-identical. Public all-family integration is PR7.6; joint smoothing/phi is PR8.1. They are separate follow-ups, not evidence that these specific regular/signed gates failed. Main-agent review owns whole milestone acceptance and publication composition.


## Final pinned evidence

The final owning file passes36tests in85.13s, original launcher exit0, with no skips/xfails/warnings. All32real rank1-penalty cells are exercised atB17; four distinct known/unknown-scale extreme-prior fits are exercised atB1/17/200. Image `jaxgam-test:pr73-regular-release-r9c-20260917`, immutable ID sha256:1dcea55aa2f8c3ea4f40eb4b571bb4dfebac7d6fe7d305b64a315ff00d0597a4. All176production/test/script Python hashes match the host; all175previous r8h Python files remain byte-identical. Lint passes.

The previously reviewed owning controller/kernel coverage remains the unchanged-code evidence:179passed with93.19%combined coverage (controller91%,CPUinitializer90%,shared kernel97%), plus separately accepted signed/null modules above80%. The r9slice adds no production module or arithmetic. Final log `/private/tmp/jaxgam-pr73-regular-release-r9c-owning.log`; lint/build/imagefiles use the same r9c prefix.
