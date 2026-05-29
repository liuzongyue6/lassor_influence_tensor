# Inertia Relief Influence Tensor Workflow

This repository builds a stress/strain influence matrix from OptiStruct .strs/.strn outputs, solves for a **sparse set of load groups** using Group LASSO or BCD, and reproduces a target stress tensor (from a dense multi-load source such as an IR event) with a user-specified sparsity budget.

## Project Layout

- InfluenceMatrix/Coupon/LE5Quad4_Inertia_Relief_Target_Stress.strs
- InfluenceMatrix/Coupon/LE5Quad4_SPC_Unit_Load_Stress.strs
- InfluenceMatrix/Coupon/LE5Quad4_SPC_Verified_Stress.strs
- scripts/ir_lasso_pipeline.py
- scripts/compare_stress_tensors.py
- outputs/

## Quick Start (Coupon Stress)

Run the LASSO pipeline on the Coupon files:

```
python scripts/ir_lasso_pipeline.py --mode stress \
  --ir-strs InfluenceMatrix/Coupon/LE5Quad4_Inertia_Relief_Target_Stress.strs \
  --spc-strs InfluenceMatrix/Coupon/LE5Quad4_SPC_Unit_Load_Stress.strs
```

Key outputs:
- outputs/stress_result_<timestamp>.csv: solved force per SUBCASE
- outputs/stress_E_target_<timestamp>.csv: flattened target vector
- outputs/stress_H_<timestamp>.csv: influence matrix (unit-load columns)
- outputs/report_<timestamp>.md: fit metrics

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

> **Simple / Coupon path** (single subcase, 24 elements): steps 1–5 below use `ir_lasso_pipeline.py`.  
> **Production / Fatigue path** (18 254 elements, MAX+MIN subcases, ranking enforcement): see `fatigue_lasso_pipeline.py` and [`docs/fatigue_lasso_method.md`](docs/fatigue_lasso_method.md).

1) Parse .strs/.strn files by SUBCASE and element ID (aligned by element ID order).
2) Flatten per-element components into a vector using a fixed order.
3) Build $H$ (candidate load library) and $\sigma_{target}$ (target stress from dense loads).
4) Solve $\min_F \frac{1}{2n}\|HF - \sigma_{target}\|_2^2 + \alpha\|F\|_1$ using LassoCV.
5) Reconstruct loads using SUBCASE IDs and validate with a verification run.

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
- For the Coupon case workflow, see [docs/ir_lasso_workflow.md](docs/ir_lasso_workflow.md).
- For the production fatigue pipeline, see [docs/fatigue_lasso_method.md](docs/fatigue_lasso_method.md).
