---
name: plan-review
description: Review working-tree changes against one unit of work — the single section of a referenced implementation_plan.md that carries an AGENT REVIEW marker — plus its sibling design.md. Runs two inline review passes, engineering (YAGNI/SOLID/DRY/PEP/complexity, test-cheating detection) and statistical (mgcv/GAM math correctness vs the R source), then verifies and reports findings. Use when asked to review an implementation against its design/plan, review the unstaged work for a commit, or check that an agent's implementation is faithful to the docs.
---

# Plan Task Review

Reviews an in-progress implementation against its authoritative design doc and
the specific implementation-plan task it claims to deliver. Two review passes,
run inline by you — no subagents — then a verification step before reporting.

## 1. Resolve the target — one plan file, one marked task

The unit of review is **the single plan section carrying the `AGENT REVIEW`
marker** — not the whole plan, not the whole branch.

`$ARGUMENTS` is normally a path to an implementation plan, however the user
writes it: `docs/production_api/implementation_plan.md`, an `@`-mention of the
same file, or a bare feature name (`production_api` → `docs/<name>/implementation_plan.md`).
An optional trailing task id (`production_api E`, `… implementation_plan.md B1`)
selects a section directly and overrides marker search.

If `$ARGUMENTS` is empty, search the docs tree for the marker and use the file
it lands in:

```bash
grep -rn 'AGENT REVIEW' docs/ --include='*.md'
```

### Selecting the section by task id

A task id in `$ARGUMENTS` names the section directly and **overrides marker
search** — the marker never has to exist. Match it against the plan's headings
as a whole word after `## Commit`:

```bash
grep -n '^## ' docs/<feature>/implementation_plan.md
```

`A` matches `## Commit A — …`; `B1` matches `## Commit B1 — …`. Ids that are a
prefix of more than one heading are **ambiguous, not a best guess** — a plan
containing both `Commit B0` and `Commit B1` makes a bare `B` ambiguous, so list
both headings and ask. Same if the id matches nothing.

Once matched, take the section as the range from that heading to the next
`^## ` (or EOF), exactly as below.

### Locate the marked section

The marker may sit on the section heading itself or on any line inside the
section body. Find it, then walk back to the enclosing `##` heading:

```bash
PLAN=docs/<feature>/implementation_plan.md
grep -n 'AGENT REVIEW' "$PLAN"     # marker line(s)
grep -n '^## '          "$PLAN"     # section boundaries
```

The section under review runs from the last `^## ` heading at or before the
marker line, up to the next `^## ` heading (or EOF). Read exactly that range
with `Read(file_path=$PLAN, offset=<start>, limit=<end-start>)` — plan files
run well over a thousand lines, so do not read the whole file.

Then resolve, and **state all four in your opening message** so the user can
correct the scope before you start reading code:

| | |
|---|---|
| Plan file | `docs/<feature>/implementation_plan.md` |
| Section | the `## Commit X — …` heading you extracted |
| Line range | `<start>–<end>` |
| Design doc | `docs/<feature>/design.md` — the sibling of the plan file |

### When the marker is missing or ambiguous

- **No marker and no task id given:** ask the user which section to review.
  Do not guess — the marker is the entire scope contract for this review.
- **Multiple markers:** if they fall inside one section, that is one target,
  proceed. If they span sections, list the candidate headings and ask which
  single unit to review. Do not review several at once; run the skill again
  per unit.
- **A task id was given but no such section exists:** list the `## ` headings
  in the plan and ask.
- **Marker sits above the first `## ` heading** (in front matter or the
  Overview preamble): treat that as ambiguous and ask, rather than reviewing
  the whole document.

## 2. Read the docs before the code

Read, in this order, and do not skip:

1. The marked section from step 1, in full — its deliverables, its explicit
   non-goals, its stated test requirements. This is the scope contract.
2. The plan's `## Overview`, `## Hard Allow-List — Must Not Regress`, and
   `## Definition of Done` sections — these set invariants the diff must not
   break even though they sit outside the marked section. Locate them from the
   `grep -n '^## '` output and read those ranges only.
3. Every `design.md` section the marked section cites (`design §5.1`, `§10.3`,
   …). Read the cited sections, not the whole file.
4. `docs/design.md` sections relevant to the touched subsystem — use the
   topic→section table in `CLAUDE.md` to pick them.

Note anything the plan marks **deferred**, **out of scope**, or **removed in
round N**. Code that implements a removed or deferred item is a finding, not a
bonus.

## 3. Collect the diff

```bash
git status --porcelain
git diff --stat && git diff
git ls-files --others --exclude-standard   # new files: read each in full
```

Default scope is the **unstaged** working tree plus untracked files. If
`git diff --cached --stat` is non-empty, say so explicitly and ask whether to
include staged changes — never silently review a different scope than the user
asked for.

## 4. Mechanical test-integrity pre-pass

Run this before the review passes. The hits are grounded evidence — every one
is a line the implementing agent touched in a way that can weaken a test, so
each needs an explicit verdict in the engineering pass.

```bash
git diff -U0 -- tests/ | grep -nE '^[+-].*(rtol|atol|tol=|approx|places=|decimal=|delta=)'
git diff -U0 -- tests/ | grep -nE '^\+.*(skip|xfail|skipif|pytest\.mark\.slow|return  *#|pass  *#)'
git diff -U0 -- tests/ | grep -nE '^-.*(assert|np\.testing|check_that)'
git diff -- tests/ | grep -nE '^\+.*(seed|SEED|random_state)'
```

Read `tests/tolerances.py` to get the current frozen tolerance classes. Any
numeric tolerance in a test that is not an attribute read off one of those
frozen classes is a finding — including "just this once" literals, and
including a `ToleranceClass(...)` constructed inline at a call site. A test
switching to a looser class than it used before is a finding unless the diff
justifies it with a measured gap.

## 5. Review inline, in two passes

**Do this yourself. Do not spawn subagents** — not for the review, not for
"gathering context" first. You have already read the plan section, the design
sections, and the diff; a subagent would start cold and re-derive all of it,
and its findings would come back as claims you then have to re-verify against
the same files. Reviewing inline is both cheaper and more accurate here.

Run two passes over the same diff, in this order, each driven by its reference
file. Read the reference file at the start of its pass and work the checklist:

1. **Engineering pass** → `references/engineering-reviewer.md`
   Faithfulness to the marked section, YAGNI, SOLID, DRY (and over-DRY),
   complexity, PEP/idiom, project convention and phase discipline, test
   integrity, edges and failure modes.
2. **Statistical pass** → `references/statistical-reviewer.md`
   The mathematics against the mgcv R source: estimator correctness, basis and
   penalty construction, numerical soundness, the §18.1 hard gates, and whether
   the tests actually pin the statistics.

Keep the passes genuinely separate — do not collapse them into one sweep. They
ask different questions of the same lines, and the statistical pass requires
reading R source that the engineering pass has no reason to open. A helper the
engineering pass wants deleted as over-abstraction is often the one the
statistical pass finds load-bearing for numerical stability; you only see that
tension if both passes actually ran.

Throughout: you are **read-only**. Inspect and report; do not edit, do not fix,
do not commit. If the user wants fixes, that is a separate request after they
have seen the findings.

Hold each pass to the scope resolved in step 1. Diff hunks belonging to a
different plan section get a one-line note, not a finding. Judge the code
against the marked section's own text — its deliverables, its non-goals, its
stated test requirements — not against a paraphrase of it.

## 6. Verify before reporting

Your own first-pass findings are hypotheses, not facts. Before a finding goes
in the report:

- Open the cited file at the cited line and confirm the code says what you
  think it says. Never report something you inferred from a diff hunk without
  reading it in context — a hunk hides the guard clause twenty lines up.
- Drop anything about code outside the diff, or about an item the plan
  explicitly defers or scopes out.
- Drop style nits that match surrounding project convention — consistency with
  the codebase beats abstract purity.
- Where a finding is cheap to test, test it: `make test-local`, or
  `uv run pytest <file> -x -q` for the touched tests. A confirmed failure is
  worth more than a paragraph of reasoning.
- Where the two passes conflict, resolve it against the design doc and the R
  source and report the resolution, not both claims.

## 7. Report

Report in the conversation, most severe first. For each surviving finding:
file:line link, one-sentence defect, the concrete failure it causes, and which
doc section or R source it contradicts. Group as:

- **Blocking** — wrong math/statistics, unfaithful to the design, breaks a
  hard-gate invariant, or a test that cannot fail.
- **Should fix** — SOLID/DRY/YAGNI violations, dead abstraction, complexity,
  PEP/convention drift.
- **Note** — minor, or an observation worth the user's judgment.

End with a one-line verdict on whether the diff delivers the task section as
specified. State plainly what you could not verify. If nothing survives
verification, say so — do not pad the report.
