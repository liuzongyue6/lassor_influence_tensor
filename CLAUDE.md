# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```powershell
# Cradle HAZ fatigue pipeline — V2, global-fit + ranking refinement (18 254 elements, ~minutes)
python scripts/fatigue_lasso_pipeline.py `
  --ir-strs  InfluenceMatrix/Cradle_HAZ_Element/Xpeng_Target_Stress.txt `
  --spc-strs InfluenceMatrix/Cradle_HAZ_Element/Xpeng_Unit_Load_Stress.txt `
  --ir-max-subcase 1000001 --ir-min-subcase 1000002 `
  --max-active-groups 6 --auto-mandatory-top-k 2 `
  --critical-elems 10058616,10014072 `
  --target-ranking 10058616,10014072 `
  --output-dir outputs/cradle_run

# Cradle HAZ fatigue pipeline — V3, critical-ranked local-fit (use when V2's global-fit
# group selection starves the critical elements — see docs/fatigue_lasso_method.md §5)
python scripts/fatigue_ranking_pipeline.py `
  --ir-strs  InfluenceMatrix/Cradle_HAZ_Element/Xpeng_Target_Stress.txt `
  --spc-strs InfluenceMatrix/Cradle_HAZ_Element/Xpeng_Unit_Load_Stress.txt `
  --ir-max-subcase 1000001 --ir-min-subcase 1000002 `
  --max-active-groups 4 --auto-mandatory-top-k 2 `
  --critical-elems 10058616,10014072 `
  --target-ranking 10058616,10014072 `
  --output-dir outputs/ranking_run

# Compare two stress tensors visually
python scripts/compare_stress_tensors.py `
  --target   InfluenceMatrix/Coupon/LE5Quad4_Inertia_Relief_Target_Stress.txt `
  --verified InfluenceMatrix/Coupon/LE5Quad4_SPC_Verified_Stress.strs

# Syntax check (no external data needed)
python -c "import ast; ast.parse(open('scripts/fatigue_lasso_pipeline.py').read()); print('OK')"
```

> **Note**: `scripts/ir_lasso_pipeline.py` (legacy single-subcase Coupon pipeline) was removed from the
> repo in commit `1739a03`. There is currently no runnable Coupon smoke-test — both remaining
> pipelines require a MAX **and** MIN subcase pair (they fit stress *range*, not absolute stress). See
> `docs/fatigue_lasso_method.md` Appendix A for the historical Coupon workflow this replaced.

## Architecture

Three scripts share the same STRS parser and matrix-building helpers (all defined in
`fatigue_lasso_pipeline.py`; `fatigue_ranking_pipeline.py` imports them):

```
fatigue_lasso_pipeline.py     — V2: dual-subcase fatigue pipeline, global-fit Phase 1 + Phase 3
                                 ranking refinement (residual IRLS or ranking SLSQP mode).
                                 Production default; all new work starts here.
fatigue_ranking_pipeline.py   — V3: dual-subcase, Phase 1 fitted on CRITICAL ELEMENT ROWS ONLY,
                                 Phase 3 is always local-fit SLSQP. Use when V2's global-fit group
                                 selection can't cover the critical elements (basis insufficiency).
compare_stress_tensors.py     — visualisation/validation utility, no optimisation
```

Full V2-vs-V3 comparison table and guidance on which to use: `docs/fatigue_lasso_method.md` §5.

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
| `scripts/fatigue_lasso_pipeline.py` | V2 production pipeline (global-fit); all new work starts here unless V3 is specifically needed |
| `scripts/fatigue_ranking_pipeline.py` | V3 pipeline (critical-ranked local-fit); imports shared helpers from `fatigue_lasso_pipeline.py` |
| `docs/fatigue_lasso_method.md` | Single developer-facing reference: math, algorithm, CLI, and diagnostics for V1 (historical, Appendix A), V2, and V3 |
| `InfluenceMatrix/Cradle_HAZ_Element/Xpeng_Target_Stress_Damage.csv` | Ground-truth fatigue ranking (element 10058616 = rank 1) |

## Dependencies

`numpy`, `scikit-learn`, `group-lasso`, `tqdm`, `matplotlib` (compare script only). Environment: Python 3.12, miniforge3.

## Fatigue Analysis Documentation Protocol

When you perform analysis in the fatigue/structural simulation or ML domain and reach a non-obvious conclusion — including:
- Quantitative findings about stress data, damage rankings, or influence scores
- ML/optimisation insights (convergence behaviour, hyperparameter sensitivity, failure modes)
- Root-cause analysis with supporting evidence
- Validation results that confirm or contradict a prior assumption

**You must write those findings into `docs/fatigue_lasso_method.md`** immediately after the analysis, under the most relevant existing section, or under a new dated subsection (e.g., `### 2026-05-27 — IRLS Convergence Finding`). Follow the existing document style: tables for data comparisons, LaTeX-style math for formulas, and a brief "Why it matters" sentence.

Do NOT leave insights only in the conversation — they must be persisted to the method document to survive context resets.
