# Dense EFS benchmark restart handoff, 2026-09-17

This is the historical pause snapshot. Work subsequently resumed: the real-R
skip guard and metadata validation are complete, and main-agent review accepted
explicit MODERATE selection only for the exact default smoke. See
`efs_benchmark_validation_2026-09-17.md` for final evidence and numerical scope.
The statuses and commands below describe the earlier pause revision.

Work is paused at the user's request. No representative timing run has started,
no commit has been created, and no GitHub publication is authorized for this
benchmark. Published public EFS #61 already passed native CI; the benchmark is
a separate unfinished requirement.

## Checkout and WIP

- Branch: `codex/efs-benchmark-resume-20260917`.
- Worktree: `/private/tmp/jaxgam-efs-benchmark-resume-20260917`.
- Base/current HEAD: `951ca4bef7be67e065d6ed5fe2a798c6f9f39f5f`.
- Changes: `Makefile`, new `scripts/benchmark_efs.py`, new
  `tests/test_scripts/test_benchmark_efs.py`, new numerical-review document
  `docs/scale_jaxgam/efs_benchmark_validation_2026-09-17.md`, and this handoff.
- Revision manifest:
  `/private/tmp/jaxgam-efs-benchmark-pause-manifest-20260917.json`.

The prior Sept 10 temporary review checkout was incomplete on restart. This
checkout restored the archived root-reviewed patch from
`docs/scale_jaxgam/restart_artifacts_2026-09-10/efs-benchmark-root-review.patch`
in the main repository. Preserve root's TemporaryFile stream-draining repair
and bounded 1 MiB stdout/stderr regression.

Implemented corrections reject nonfinite/unconverged fits (including every
timed cold/warm fit), derive starts from saved roundtrip CSV input, hash-check
worker input, and separately report STRICT/MODERATE. The CLI requires valid
fits and STRICT EFS/R parity. Newton-versus-EFS residuals are labeled as a
different-optimizer observation. Recent metadata edits record Newton controls,
the composed FitControl, and a reason when container CPU model is unavailable.
No production fitting modules have changed.

## Executed evidence and revision boundaries

The tested r2 image is `jaxgam-test:efs-benchmark-resume-20260917-r2`, exact ID
`sha256:8da633b0fea321ab9f1ecc955d60560239ad242729343339098cdbad20439408`.
Its benchmark script SHA256 is
`6b6cbdae50e5dda84d5d9bf14c7976c3b466e4576cdb03ed2289ab59e443dcbb`.

- Pinned JAX-first owning-file coverage: 20 passed in 13.25 seconds, pytest
  status 0, process exit 0, script coverage 89% (302 statements, 34 missed).
  Raw log: `/private/tmp/jaxgam-efs-benchmark-resume-pinned-coverage-r2.log`.
  Unified session 60734 is terminal; do not poll or restart it.
- `make lint` passed before the latest metadata additions. Build/lint session
  16590 is terminal exit 0. Current metadata additions and their assertions
  have **not** been rerun through lint/tests/image build.
- Saved-input pinned smoke: terminal session 59722 exited 2 as expected.
  Raw log: `/private/tmp/jaxgam-efs-benchmark-smoke-20260917-r2.log`.
  Artifact directory:
  `/private/tmp/jaxgam-efs-benchmark-smoke-20260917-r2/` contains input.csv,
  config.json, report.json. Stopped container
  `jaxgam-efs-benchmark-smoke-20260917-r2` retains the same output.
- Five diagnostic passes: terminal session 98554 exited 0. Raw log:
  `/private/tmp/jaxgam-efs-benchmark-numerical-review-20260917.log`.
  Reproducible external diagnostic script and JSON:
  `/Users/shanecatts/Documents/GitHub/jaxgam/scratch/efs-benchmark-numerical-review-20260917/review.py`
  and `numerical-review.json`; input/config/report are saved beside them.

Pinned versions are R 4.5.2 / mgcv 1.9-3, source commit
`fb7e8e718377513e78ba6c6bf7e60757fc6a32a9`. The source sections read on restart
were pinned `R/gam.fit4.r` efsudr lines 821-940 and `R/mgcv.r` initialization
lines 1998-2035. Diagnostic timings occurred during other validation and are
not representative performance evidence.

The current script hash differs from the tested image because of those
metadata additions. Current hashes are authoritative in the pause manifest.
Do not describe the r2 coverage result as validating the final current source.

## Numerical blocker and local-test issue

The original smoke remains unchanged: seed 20260910, n=128, p=13, m=3,
cubic k=5; CSV SHA256
`8a49df584d593df11ed6ffc44e66ec72ab37f2352ab37b353e1c42c4232a26b9`.
All timed fits converge and are finite. Default EFS/R outer counts match at
34/34 and scale is exactly 1. STRICT fails: maximum absolute beta error
8.5088e-8, means 3.4200e-7, smoothing 3.5371e-5, EDF 4.0398e-7, deviance
2.1089e-6, score 2.2683e-8. MODERATE diagnostic passes. No new tolerance
exception is applied or accepted.

Five failed correction/diagnostic passes are recorded in the numerical-review
document: saved-input roundtrip correction; independent RBridge extraction
(exact match to R worker); direct controller versus public adapter (exact
match); tighter score tolerance 0.001; tighter PIRLS tolerance 1e-10. The latter
two control profiles still fail STRICT. This rules out serialization and
public result conversion as the observed cause, without resolving the full
algorithmic discrepancy.

Root identified an unresolved test-local compatibility issue:
`test_direct_saved_input_workers_and_original_strict_boundary` invokes actual
R without an `r_available`/version skip, so absent or wrong-version R would
fail instead of skip. Add the standard pinned R availability guard on resume.
`test_r_failed_fit_is_reported` mocks subprocess.run and requires no R; it
should continue testing the failure path locally. Preserve the real-R smoke
and its expected STRICT failure.

## Resume steps

1. Read this handoff, the numerical-review document, main AGENTS, and root's
   restart handoff. Preserve the checkout and root-reviewed pipe regression.
2. Fix the actual-R test availability guard. Run `make lint` and rebuild the
   pinned image from this checkout. Revalidate the full owning test file using
   JAX-first coverage, not direct pytest --cov, which previously aborted 134.
   The launcher must initialize `jax.devices()` before coverage.start and
   pytest imports, run `tests/test_scripts/test_benchmark_efs.py`, emit both
   pytest/process terminal status, and report script coverage. Save raw output
   with direct redirection or pipefail.
3. Match final source/test hashes to the exact image and make a coherent local
   commit for root review. No push or helper PR.
4. Obtain a quiet-window signal from root before representative timing.
   Root's PR11 Docker container 24855dfde3dd was still live at pause; root will
   also confirm Gaussian has no tests in flight. Do not infer that old process
   state remains current on resume.
5. During that quiet window, sequentially execute the representative saved-data
   benchmark in the final pinned image, with explicit source/image environment
   metadata. Keep terminal exit 2 if STRICT fails; preserve report and fit
   validity rather than changing tolerance to make the CLI green. Record
   host/VM settings and clearly unavailable metrics. No general speedup claim.

The benchmark command is:

```sh
uv run python scripts/benchmark_efs.py --rows 1200 --smooths 10 \
  --basis-dimension 6 --seed 20260910 --warm-repeats 3 --threads 1 \
  --output-dir /tmp/efs-artifact
```

Docker bind mounts under /private/tmp in this environment may resolve inside
the VM rather than the host. Use the shared repository scratch path or a named
container plus docker cp to retain outputs. Never delete user-global caches.
