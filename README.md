# Inertia Relief Influence Tensor Workflow

This repository builds a stress/strain influence matrix from OptiStruct .strs/.strn outputs, solves for a **sparse set of load groups** using Group LASSO or BCD, and reproduces a target stress tensor (from a dense multi-load source such as an IR event) with a user-specified sparsity budget.

Two active pipelines cover the dual-subcase (MAX + MIN) fatigue use case, sharing one STRS
parser/matrix-builder (`fatigue_lasso_pipeline.py`):

- **V2 — `scripts/fatigue_lasso_pipeline.py`**: production default. Group LASSO fits the stress
  *range* Δσ globally across all elements, then an optional Phase 3 refines damage ranking for
  user-specified critical elements without reopening group selection.
- **V3 — `scripts/fatigue_ranking_pipeline.py`**: critical-ranked local-fit variant. Group LASSO fits
  Δσ using *only* the critical elements' rows, so group selection is forced to prioritize covering
  those elements instead of the whole component. Use this when V2 reports basis insufficiency (its
  group selection can't excite the critical elements even though its global fit looks fine).

Full math, algorithm pseudocode, CLI reference, and V2-vs-V3 guidance:
[docs/fatigue_lasso_method.md](docs/fatigue_lasso_method.md).

## Environment Setup

Requires **Python 3.12** (developed against miniforge3/conda-forge).

```powershell
# 1) Create and activate an environment (conda/miniforge recommended)
conda create -n lassor python=3.12 -y
conda activate lassor

# 2) Install dependencies
pip install numpy scipy scikit-learn group-lasso tqdm matplotlib
```

| Package | Used by | Purpose |
|---|---|---|
| `numpy` | all scripts | matrix/vector math |
| `scipy` | `fatigue_lasso_pipeline.py`, `fatigue_ranking_pipeline.py`, `compare_stress_tensors.py` | SLSQP constrained optimization (`scipy.optimize.minimize`), KDE/correlation stats |
| `scikit-learn` | `fatigue_lasso_pipeline.py` | `LinearRegression`/`Ridge` for OLS refinement steps |
| `group-lasso` | `fatigue_lasso_pipeline.py` | `GroupLasso` solver for Phase 1 group selection |
| `tqdm` | `fatigue_lasso_pipeline.py` | progress bars during BCD/alpha search |
| `matplotlib` | `compare_stress_tensors.py` only | scatter/error plots |

No `requirements.txt`/`environment.yml` is checked in yet; the `pip install` line above is the
authoritative dependency list (kept in sync with `CLAUDE.md`'s Dependencies section).

Verify the install with a syntax-only smoke test (no data files needed):

```powershell
python -c "import ast; ast.parse(open('scripts/fatigue_lasso_pipeline.py').read()); print('OK')"
```

To confirm the third-party imports actually resolve:

```powershell
python -c "import numpy, scipy, sklearn, group_lasso, tqdm, matplotlib; print('deps OK')"
```

## Project Layout

- InfluenceMatrix/Coupon/LE5Quad4_Inertia_Relief_Target_Stress.strs
- InfluenceMatrix/Coupon/LE5Quad4_SPC_Unit_Load_Stress.strs
- InfluenceMatrix/Coupon/LE5Quad4_SPC_Verified_Stress.strs
- inputs/Cradle_HAZ_Element/Xpeng_Target_Stress.txt
- inputs/Cradle_HAZ_Element/Xpeng_Unit_Load_Stress.txt
- scripts/fatigue_lasso_pipeline.py (V2, production default)
- scripts/fatigue_ranking_pipeline.py (V3, critical-ranked local-fit)
- scripts/compare_stress_tensors.py
- outputs/

> `scripts/ir_lasso_pipeline.py` (legacy single-subcase Coupon LASSO) was removed from the repo in
> commit `1739a03`; there is currently no runnable single-subcase Coupon smoke-test. See
> [docs/fatigue_lasso_method.md](docs/fatigue_lasso_method.md) Appendix A for the historical workflow.

## Quick Start (Cradle HAZ — V2, production default)

```powershell
python scripts/fatigue_lasso_pipeline.py `
  --ir-strs  inputs/Cradle_HAZ_Element/Xpeng_Target_Stress.txt `
  --spc-strs inputs/Cradle_HAZ_Element/Xpeng_Unit_Load_Stress.txt `
  --ir-max-subcase 1000001 --ir-min-subcase 1000002 `
  --max-active-groups 6 --auto-mandatory-top-k 2 `
  --critical-elems 10058616,10014072 `
  --target-ranking 10058616,10014072 `
  --output-dir outputs/cradle_run
```

Key outputs (in `--output-dir`): `stress_f_max_<ts>.csv` / `stress_f_min_<ts>.csv` (block-cycle load
vectors), `stress_ranking_check_<ts>.csv` (predicted vs. target damage rank), `report_<ts>.md`,
`run_log_<ts>.txt`.

## Quick Start (Cradle HAZ — V3, critical-ranked local-fit)

Use this when V2's `ranking_diagnostics.basis_insufficient_warning` fires — i.e. its globally-fitted
group selection can't cover the critical elements no matter how much Phase 3 boosts their weight:

```powershell
python scripts/fatigue_ranking_pipeline.py `
  --ir-strs  inputs/Cradle_HAZ_Element/Xpeng_Target_Stress.txt `
  --spc-strs inputs/Cradle_HAZ_Element/Xpeng_Unit_Load_Stress.txt `
  --ir-max-subcase 1000001 --ir-min-subcase 1000002 `
  --max-active-groups 4 --auto-mandatory-top-k 2 `
  --critical-elems 10058616,10014072 `
  --target-ranking 10058616,10014072 `
  --max-ranking-iter 20 `
  --output-dir outputs/ranking_run
```

Same output files as V2, minus the full-H/target dumps. The `report_<ts>.md` separates
`local_fit_error_range` (optimized, at critical elements only) from `global_error_range`
(informational only — V3 does not optimize whole-component fit). See
[docs/fatigue_lasso_method.md](docs/fatigue_lasso_method.md) §5.8 for when to pick V2 vs. V3.

## CLI Parameter Reference

Both pipelines share the same input files and STRS parser; flags marked **required** have no default.

### V2 — `scripts/fatigue_lasso_pipeline.py`

```powershell
python scripts/fatigue_lasso_pipeline.py --ir-strs <path> --spc-strs <path> [options]
```

| Flag | Type | Default | Description |
|---|---|---|---|
| `--ir-strs` | str | **required** | Target stress file containing MAX and MIN subcases |
| `--spc-strs` | str | **required** | Unit load stress file (one subcase per force direction) |
| `--ir-max-subcase` | int | `1000001` | Subcase ID for σ_max (peak) |
| `--ir-min-subcase` | int | `1000002` | Subcase ID for σ_min (valley) |
| `--scale-h` | float | `1000.0` | Scale factor applied to influence matrix columns (unit load magnitude) |
| `--max-active-groups` | int | `None` | Max number of active force groups (FX/FY/FZ triplets); triggers binary search for α. Omit for no limit (falls back to `--alpha-grid` search) |
| `--critical-elems` | str (CSV) | `""` | Element IDs to up-weight, e.g. `10058616,10014072` |
| `--critical-weight` | float | `10.0` | Weight multiplier applied to critical element residuals |
| `--target-ranking` | str (CSV) | `""` | Ordered element IDs: position 0 → global rank 1, position 1 → rank 2, … Activates Phase-3 IRLS ranking refinement |
| `--alpha-grid` | str (CSV floats) | `0.1,1,10,100,1000` | α candidates for grid search (only used when `--max-active-groups` is not set) |
| `--alpha-lo` | float | `0.01` | Binary-search lower bound for α |
| `--alpha-hi` | float | `100000.0` | Binary-search upper bound for α |
| `--auto-mandatory-top-k` | int | `2` | Phase 0: top-K force groups by influence per critical element to always activate. `0` disables Phase 0 |
| `--mandatory-groups` | str (CSV ints) | `""` | Phase 0: additional 0-indexed group indices to always activate, e.g. `0,3,7` |
| `--max-ranking-iter` | int | `10` | Max iterations for Phase-3 ranking refinement |
| `--gamma` | float | `2.0` | (`residual` mode only) IRLS weight amplification factor per rank-gap unit |
| `--irls-mode` | `residual` \| `ranking` | `residual` | `residual`: IRLS weight boosting (can saturate at 1e8, collapsing global fit). `ranking`: SLSQP constrained OLS — global-fit objective with hard `VM_crit ≥ VM_competitor` constraints, no weight saturation |
| `--ranking-margin` | float | `0.0` | (`ranking` mode only) required margin, Pa: `VM_crit − VM_competitor ≥ margin` |
| `--feasibility-check` / `--no-feasibility-check` | flag | `True` | Run unconstrained OLS before Phase 1 to log the achievable rank upper bound |
| `--standardize` / `--no-standardize` | flag | `True` | Standardize columns before Group LASSO |
| `--output-dir` | str | `outputs` | Output directory |
| `--timestamp` / `--no-timestamp` | flag | `True` | Suffix output filenames with a timestamp |

### V3 — `scripts/fatigue_ranking_pipeline.py`

```powershell
python scripts/fatigue_ranking_pipeline.py --ir-strs <path> --spc-strs <path> [options]
```

Same shared flags as V2 (`--ir-strs`, `--spc-strs`, `--ir-max-subcase`, `--ir-min-subcase`, `--scale-h`,
`--max-active-groups`, `--critical-elems`, `--critical-weight`, `--target-ranking`,
`--auto-mandatory-top-k`, `--mandatory-groups`, `--alpha-lo`, `--alpha-hi`, `--max-ranking-iter`,
`--ranking-margin`, `--feasibility-check`/`--no-feasibility-check`, `--standardize`/`--no-standardize`,
`--output-dir`, `--timestamp`/`--no-timestamp`) with these differences:

| Flag | Type | Default | Description |
|---|---|---|---|
| `--max-active-groups` | int | `6` | (V3 default differs from V2's unlimited default) |
| `--max-ranking-iter` | int | `20` | (V3 default differs from V2's `10`; counts SLSQP iterations, not IRLS iterations) |
| `--max-blockers-per-iter` | int | `200` | Max blocker constraints per SLSQP call (worst violators first) |

V3 has **no** `--alpha-grid`, `--gamma`, or `--irls-mode` — Phase 1 always fits critical-element rows
only, and Phase 3 is always the local-fit SLSQP ranking solve (see
[docs/fatigue_lasso_method.md](docs/fatigue_lasso_method.md) §5 for why).

### Run log

Every run writes `run_log<_ts>.txt` (JSON) to `--output-dir`, capturing the exact command line, all
resolved CLI args, the git commit the run was executed at, elapsed wall-clock time, SHA-256 hashes of
the two input files, the full metadata dict, and the complete console trace (`[INFO]`/`[WARNING]`/
iteration lines) — enough to reproduce or debug a run without re-executing it.

## Compare Target vs Verified Stress

This compares the Target Stress Tensor and the Verified Stress Tensor and generates plots.

```
python scripts/compare_stress_tensors.py \
  --target InfluenceMatrix/Coupon/LE5Quad4_Inertia_Relief_Target_Stress.strs \
  --verified InfluenceMatrix/Coupon/LE5Quad4_SPC_Verified_Stress.strs
```

Outputs:
- outputs/compare_stats_<timestamp>.csv
- outputs/compare_scatter_<timestamp>.png
- outputs/compare_component_error_<timestamp>.png

## How The Pipeline Works

> **V2 — global-fit path** (18 254 elements, MAX+MIN subcases, ranking enforcement as a Phase-3
> refinement): `fatigue_lasso_pipeline.py`.
> **V3 — critical-ranked local-fit path** (same data, but Phase 1 itself is restricted to critical
> elements' rows): `fatigue_ranking_pipeline.py`.
> **Legacy V1 / Coupon path** (single subcase, 24 elements, plain elementwise LASSO): historical only,
> script removed — see [`docs/fatigue_lasso_method.md`](docs/fatigue_lasso_method.md) Appendix A.
>
> Full math and algorithm pseudocode for all three: [`docs/fatigue_lasso_method.md`](docs/fatigue_lasso_method.md).

1) Parse .strs/.strn files by SUBCASE and element ID (aligned by element ID order).
2) Flatten per-element components into a vector using a fixed order.
3) Build $H$ (candidate load library) and $\Delta\sigma_{target} = \sigma_{max} - \sigma_{min}$ (stress-range target — fatigue damage is driven by cyclic range, not absolute stress).
4) V2: Group LASSO / BCD on $\Delta\sigma_{target}$ using all rows, with a sparsity budget on active force groups. V3: the same Group LASSO / BCD machinery, but restricted to the critical elements' rows only.
5) Reconstruct $f_{max}$/$f_{min}$ from the recovered $f_{range}$/$f_{mean}$ and validate with a full FE verification run (`compare_stress_tensors.py`).

## Common Causes Of Mismatch (Target vs Verified)

- Too few candidate load subcases: the target field is not in the span of $H$.
- Over-regularization: $\alpha$ is too large, forcing too-sparse loads.
- Physical mismatch: different boundary conditions, material, or units.
- Component dominance: large components dominate the fit without weighting.

## Recommended Improvements

1) Run a least-squares baseline (no L1) to check whether $H$ can span the target.
2) Increase candidate load points (more SUBCASEs).
3) Apply component weighting to balance XX/YY/XY.
4) Use SVD/PCA to quantify target projection error on the $H$ column space.

## Claude Code Analysis Documentation

Claude Code follows a standing protocol in this project: whenever it performs fatigue simulation analysis, ML analysis (LASSO/IRLS behaviour, influence scores, ranking convergence), or derives non-obvious insights from the data, it **persists those findings to [`docs/fatigue_lasso_method.md`](docs/fatigue_lasso_method.md)** in a dated subsection.

This means `fatigue_lasso_method.md` serves as both a technical reference and a living research log. Check it for the most current understanding of why algorithmic decisions were made.

## Notes

- All parsing uses only .strs/.strn files.
- SUBCASE IDs are the only load identifiers.
- By default, all outputs include a timestamp suffix; use --no-timestamp to keep legacy names.
- For the full method reference — V2 (production), V3 (critical-ranked local-fit), and the legacy
  Coupon workflow (historical, Appendix A) — see [docs/fatigue_lasso_method.md](docs/fatigue_lasso_method.md).
