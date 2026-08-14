# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

> **Data paths**: the Cradle HAZ inputs live in `inputs/Cradle_HAZ_Element/`. The `InfluenceMatrix/`
> directory no longer exists — older docs and archived `run_log*.txt` files still reference it, but
> those are historical records, not runnable paths.

```powershell
# V4 (RECOMMENDED) — unified pipeline, global scope (≡ V2 default)
python scripts/fatigue_unified_pipeline.py `
  --ir-strs  inputs/Cradle_HAZ_Element/Xpeng_Target_Stress.txt `
  --spc-strs inputs/Cradle_HAZ_Element/Xpeng_Unit_Load_Stress.txt `
  --ir-max-subcase 1000001 --ir-min-subcase 1000002 `
  --fit-scope global --phase3-mode irls `
  --max-active-groups 6 --auto-mandatory-top-k 2 `
  --critical-elems 10058616,10014072 `
  --target-ranking 10058616,10014072 `
  --output-dir outputs/unified_global

# V4 — critical-row scope (≡ V3)
python scripts/fatigue_unified_pipeline.py `
  --ir-strs  inputs/Cradle_HAZ_Element/Xpeng_Target_Stress.txt `
  --spc-strs inputs/Cradle_HAZ_Element/Xpeng_Unit_Load_Stress.txt `
  --ir-max-subcase 1000001 --ir-min-subcase 1000002 `
  --fit-scope critical --phase3-mode slsqp `
  --max-active-groups 4 --auto-mandatory-top-k 2 `
  --critical-elems 10058616,10014072 `
  --target-ranking 10058616,10014072 `
  --output-dir outputs/unified_critical

# V2 — frozen reference implementation (global-fit + Phase 3 refinement)
python scripts/fatigue_lasso_pipeline.py `
  --ir-strs  inputs/Cradle_HAZ_Element/Xpeng_Target_Stress.txt `
  --spc-strs inputs/Cradle_HAZ_Element/Xpeng_Unit_Load_Stress.txt `
  --ir-max-subcase 1000001 --ir-min-subcase 1000002 `
  --max-active-groups 6 --auto-mandatory-top-k 2 `
  --critical-elems 10058616,10014072 `
  --target-ranking 10058616,10014072 `
  --output-dir outputs/cradle_run

# V3 — frozen reference implementation (critical-ranked local-fit)
python scripts/fatigue_ranking_pipeline.py `
  --ir-strs  inputs/Cradle_HAZ_Element/Xpeng_Target_Stress.txt `
  --spc-strs inputs/Cradle_HAZ_Element/Xpeng_Unit_Load_Stress.txt `
  --ir-max-subcase 1000001 --ir-min-subcase 1000002 `
  --max-active-groups 4 --auto-mandatory-top-k 2 `
  --critical-elems 10058616,10014072 `
  --target-ranking 10058616,10014072 `
  --output-dir outputs/ranking_run

# Compare two stress tensors visually (Coupon .strs files are no longer in the repo —
# point these at your own target / FE-verified pair)
python scripts/compare_stress_tensors.py `
  --target   <target-stress.strs> `
  --verified <verified-stress.strs>

# Syntax check (no external data needed)
python -c "import ast; [ast.parse(open(f, encoding='utf-8').read()) for f in ['scripts/fatigue_lasso_pipeline.py','scripts/fatigue_ranking_pipeline.py','scripts/fatigue_unified_pipeline.py']]; print('OK')"
```

**Fast regression pair** (~20 s each) for any change to `fatigue_lasso_pipeline.py`, which V3 and V4
both import: run `--max-active-groups 6 --auto-mandatory-top-k 6 --critical-elems 10058616
--target-ranking 10058616` in both `--irls-mode` values and compare against
`outputs/cradle_v2_g6_mk6_{ranking,residual}` — expect rank 24, `relative_error_range`
0.23888124699690558, groups `[1,5,7,9,10,11]`. Do **not** compare `alpha` (~1e-6 run-to-run jitter, see
`docs/fatigue_lasso_method.md` §6.8). Avoid `--auto-mandatory-top-k 2` with `--max-active-groups 6`
for regressions — BCD then searches free groups and takes ~45 min. Full checklist: method doc §9.1.

> **Note**: `scripts/ir_lasso_pipeline.py` (legacy single-subcase Coupon pipeline) was removed from the
> repo in commit `1739a03`. There is currently no runnable Coupon smoke-test — all remaining
> pipelines require a MAX **and** MIN subcase pair (they fit stress *range*, not absolute stress). See
> `docs/fatigue_lasso_method.md` Appendix A for the historical Coupon workflow this replaced.

## Architecture

Four scripts share the same STRS parser, solvers and matrix-building helpers — all defined exactly once
in `fatigue_lasso_pipeline.py`, which the others import. **They must stay in the same directory**
(sibling import with a `sys.path` fallback).

```
fatigue_lasso_pipeline.py     — V2 + THE SHARED LIBRARY. Every parser, solver and writer lives here.
                                 As a pipeline: global-fit Phase 1 + Phase 3 ranking refinement
                                 (residual IRLS or ranking SLSQP). FROZEN — behaviour must not change,
                                 so archived runs stay reproducible and V4 has a validation baseline.
fatigue_ranking_pipeline.py   — V3: Phase 1 fitted on CRITICAL ELEMENT ROWS ONLY, Phase 3 always
                                 local-fit SLSQP. FROZEN. Imports helpers from V2.
fatigue_unified_pipeline.py   — V4: BOTH of the above behind two switches. All new work starts here.
                                 Imports everything from V2; defines only run_phase1(),
                                 write_unified_report() and run_unified_pipeline().
compare_stress_tensors.py     — visualisation/validation utility, no optimisation
```

**The core insight**: V2 and V3 are the same four-phase method at two values of one parameter — the
**fit row set** R, i.e. which rows of H enter the least-squares objective (V2: all m; V3: the critical
elements' 6·n_crit). V4 exposes this as `--fit-scope {global,critical}` × `--phase3-mode
{irls,slsqp,none}`. Full derivation: `docs/fatigue_lasso_method.md` §6.

Relatedly, `--critical-weight` is the *continuous* version of the same knob: `--fit-scope critical` is
its w → ∞ limit, and V2's residual IRLS is an ill-conditioned iterative approach to that limit (this is
the root cause of the documented weight-saturation failure). See method doc §6.7 before proposing
anything that raises `--critical-weight` to force a ranking.

### Data Flow (V2 — `fatigue_lasso_pipeline.py`)

```
STRS files
  └─ parse_subcases()          → Dict[subcase_id, SubcaseData]
  └─ build_vector_with_order() → target vectors σ_max, σ_min  (m,)
  └─ build_matrix()            → H  (m × n), ordered by sorted(subcase_ids)

Phase 0  compute_group_influence()  → Frobenius norm per (elem, group)
         → mandatory_feature_mask   (n,) bool

Phase 1  solve_bcd() or find_alpha_for_k_groups()
         → active_mask_fixed        LOCKED after Phase 1, never changes

Phase 3  IRLS loop: _weighted_ols_fixed_support() only (no LASSO re-run)
         → f_range  (n,)

Phase 2  solve_phase2_mean()
         → f_mean   (n,)

Recovery f_max = f_mean + f_range/2
         f_min = f_mean - f_range/2
```

`active_mask_fixed` being immutable throughout Phase 3 is the key V2 invariant. Any change that re-runs group selection inside the IRLS loop will re-introduce the 139% error-explosion bug from V1.

### Data Flow (V3 — `fatigue_ranking_pipeline.py`)

Same shape as V2, with two changes: Phase 1 is fitted on `H[crit_row_mask]` /
`delta_sigma[crit_row_mask]` (critical elements' rows only, not all m rows), and Phase 3 is always
`solve_local_ranking()` — SLSQP minimizing the **local** (critical-rows-only) residual subject to
ranking constraints, instead of V2's IRLS weight-boosting or global-fit SLSQP. `active_mask_fixed` is
still locked after Phase 1 and never reopened — the same invariant, same reason. Phase 0 and Phase 2
are the literal same functions as V2 (imported, not reimplemented).

### Data Flow (V4 — `fatigue_unified_pipeline.py`)

Same shape as V2, generalised by one row mask. `R = fit_row_set(fit_scope)`:

```
R = all m rows                       if --fit-scope global    (≡ V2)
R = _make_crit_row_mask(...)         if --fit-scope critical  (≡ V3)

Phase 0  run_phase0()                       → mandatory_feature_mask   (identical, scope-free)

Phase 1  run_phase1(..., fit_row_mask=R)    → solve_bcd / find_alpha_for_k_groups /
                                               _grid_search_alpha on H[R], Δσ[R], W[R]
         → active_mask_fixed                  LOCKED after Phase 1, never changes

Phase 3  --phase3-mode irls  → solve_residual_irls(..., fit_row_mask=R)
         --phase3-mode slsqp → solve_ranking_constrained(..., fit_row_mask=R)
         --phase3-mode none  → f_range = Phase 1 coefficients
         → f_range  (n,)                      ranking ALWAYS evaluated over all elements

Phase 2  solve_phase2_mean()                → f_mean  (n,)   (always global, scope-independent)

Recovery f_max = f_mean + f_range/2
         f_min = f_mean - f_range/2
```

The `active_mask_fixed` invariant is unchanged and applies identically in every mode. `run_phase0()`,
`solve_residual_irls()` and `solve_ranking_constrained()` are shared with V2: **every parameter added
for V4 defaults to the V2 behaviour**, so V2's call sites pass none of them and its output is
bit-identical (verified against archived runs). Keep it that way when editing them.

V4 writes the **union** of the V2 and V3 metadata keys to `run_log*.txt` (`relative_error_range` *and*
`local_fit_error_range`/`global_error_range`; both `final_relative_error` and `final_local_fit_error`),
so readers written against either predecessor keep working. Don't drop keys from that union.

## STRS File Format and Parsing

OptiStruct STRS text files have this structure:

```
$SUBCASE <id>
$ELEMENT STRESS(PLATE) [REAL]
--------  (separator line, ignored)
<elem_id>  <VON>  <XX1>  <XX2>  <YY1>  <YY2>  <XY1>  <XY2>
```

Column mapping (0-indexed parts after split): `parts[0]`=elem_id, `parts[1]`=VON (ignored), `parts[2]`=XX1, `parts[3]`=XX2, `parts[4]`=YY1, `parts[5]`=YY2, `parts[6]`=XY1, `parts[7]`=XY2.

The canonical component order everywhere in the code is: `["XX1", "YY1", "XY1", "XX2", "YY2", "XY2"]` (top-face first, then bottom-face — note XX2/YY2 come from `parts[3]`/`parts[5]`, not sequentially from the file).

## H Matrix and Group Structure

`build_matrix()` sorts subcases by subcase ID before stacking columns. The Cradle HAZ SPC file uses the pattern `1,2,3 | 7,8,9 | 13,14,15 | ...` — 15 groups of 3 (FX, FY, FZ per application point). H is therefore `109 524 × 45`.

A **force group** is always 3 consecutive columns in H (since columns are sorted by subcase ID). Group index g occupies columns `[3g, 3g+1, 3g+2]`. All group-level logic (LASSO penalty, BCD, influence scores) relies on this alignment — do not reorder columns or change the sort key.

## Key Files

| Path | Role |
|------|------|
| `scripts/fatigue_unified_pipeline.py` | **V4 unified pipeline — all new work starts here** |
| `scripts/fatigue_lasso_pipeline.py` | V2 pipeline **and the shared library** (parsers, solvers, writers). Frozen behaviour; edit only in ways that preserve V2's output |
| `scripts/fatigue_ranking_pipeline.py` | V3 pipeline (critical-ranked local-fit); frozen; imports shared helpers from `fatigue_lasso_pipeline.py` |
| `docs/fatigue_lasso_method.md` | Single developer-facing reference: math, algorithm, CLI, and diagnostics for V1 (historical, Appendix A), V2, V3, and V4 (§6) |
| `docs/manual_summary.md` | SAE paper draft (method write-up, not implementation) |
| `docs/outputs_summary.md` | Index of every archived run in `outputs/` with commands and results |
| `inputs/Cradle_HAZ_Element/Xpeng_Target_Stress_Damage.csv` | Ground-truth fatigue ranking (element 10058616 = rank 1) |

## Dependencies

`numpy`, `scipy`, `scikit-learn`, `group-lasso`, `tqdm`, `matplotlib` (compare script only).
Environment: Python 3.12. `scipy` is required by all three pipelines for the SLSQP ranking solve.

## Known Issues

- `outputs/cradle_v3_g4_mk2` is the only archived run where the ranking constraint was satisfied
  (rank 1), and **no code currently in the repo reproduces it** — HEAD produces rank 21. Verified
  against a pristine checkout, so it predates the V4 merge. Do not use it as a regression target or
  cite it as validated. Details and the leading hypothesis: `docs/fatigue_lasso_method.md` §6.9.
- Phase 1 `alpha` carries ~1e-6 run-to-run jitter from the `group_lasso` solver; the recovered force
  vectors are stable. Never use `alpha` as a run-equality check (§6.8).

## Fatigue Analysis Documentation Protocol

When you perform analysis in the fatigue/structural simulation or ML domain and reach a non-obvious conclusion — including:
- Quantitative findings about stress data, damage rankings, or influence scores
- ML/optimisation insights (convergence behaviour, hyperparameter sensitivity, failure modes)
- Root-cause analysis with supporting evidence
- Validation results that confirm or contradict a prior assumption

**You must write those findings into `docs/fatigue_lasso_method.md`** immediately after the analysis, under the most relevant existing section, or under a new dated subsection (e.g., `### 2026-05-27 — IRLS Convergence Finding`). Follow the existing document style: tables for data comparisons, LaTeX-style math for formulas, and a brief "Why it matters" sentence.

Do NOT leave insights only in the conversation — they must be persisted to the method document to survive context resets.
