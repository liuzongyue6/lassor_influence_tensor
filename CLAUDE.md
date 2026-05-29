# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```powershell
# Coupon smoke-test (24 elements, fast)
python scripts/ir_lasso_pipeline.py --mode stress `
  --ir-strs InfluenceMatrix/Coupon/LE5Quad4_Inertia_Relief_Target_Stress.txt `
  --spc-strs InfluenceMatrix/Coupon/LE5Quad4_SPC_Unit_Load_Stress.txt

# Cradle HAZ fatigue pipeline (18 254 elements, ~minutes)
python scripts/fatigue_lasso_pipeline.py `
  --ir-strs  InfluenceMatrix/Cradle_HAZ_Element/Xpeng_Target_Stress.txt `
  --spc-strs InfluenceMatrix/Cradle_HAZ_Element/Xpeng_Unit_Load_Stress.txt `
  --ir-max-subcase 1000001 --ir-min-subcase 1000002 `
  --max-active-groups 6 --auto-mandatory-top-k 2 `
  --critical-elems 10058616,10014072 `
  --target-ranking 10058616,10014072 `
  --output-dir outputs/cradle_run

# Compare two stress tensors visually
python scripts/compare_stress_tensors.py `
  --target   InfluenceMatrix/Coupon/LE5Quad4_Inertia_Relief_Target_Stress.txt `
  --verified InfluenceMatrix/Coupon/LE5Quad4_SPC_Verified_Stress.strs

# Syntax check (no external data needed)
python -c "import ast; ast.parse(open('scripts/fatigue_lasso_pipeline.py').read()); print('OK')"
```


## Architecture

Two independent pipelines share the same STRS parser and matrix-building helpers:

```
ir_lasso_pipeline.py          — single-subcase LASSO (original, Coupon / simple cases)
fatigue_lasso_pipeline.py     — dual-subcase fatigue pipeline (Cradle HAZ, production use)
compare_stress_tensors.py     — visualisation utility, no optimisation
```

### Data Flow (fatigue pipeline)

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
| `scripts/fatigue_lasso_pipeline.py` | Production pipeline; all new work goes here |
| `scripts/ir_lasso_pipeline.py` | Legacy single-subcase pipeline; keep backward-compatible |
| `docs/fatigue_lasso_method.md` | Full mathematical write-up (paper draft) |
| `docs/ir_lasso_workflow.md` | Original workflow spec for the Coupon case |
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
