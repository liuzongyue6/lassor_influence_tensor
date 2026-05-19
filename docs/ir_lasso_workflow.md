# Inertia Relief Influence Tensor: STRS/STRN + LASSO Workflow

This document defines the full, reproducible workflow to parse OptiStruct .strs/.strn files, build the influence matrix $H$ and target vector $E_{target}$, run LASSO to recover sparse load channels, and export traceable outputs. Only .strs and .strn files are used. Subcase IDs are the only load identifiers.

## 1) Inputs And Scope

### Required files
- InfluenceMatrix/le5quad4_IR/LE5Quad4_IR.strs
- InfluenceMatrix/le5quad4_IR/LE5Quad4_IR.strn
- InfluenceMatrix/le5quad4_SPC/LE5Quad4_SPC.strs
- InfluenceMatrix/le5quad4_SPC/LE5Quad4_SPC.strn

### Assumptions
- The IR files contain a single target subcase (default: the first $SUBCASE$ in file).
- The SPC files contain multiple $SUBCASE$ sections, each corresponding to a unit load case.
- Each $SUBCASE$ contains 24 CQUAD4 plates with 6 stress/strain components per element: XX1, XX2, YY1, YY2, XY1, XY2.
- The vector layout is element-major with component order:
  1. XX1
  2. YY1
  3. XY1
  4. XX2
  5. YY2
  6. XY2

If you want a different ordering, change it in the script config section.

## 2) Parsing Rules (STRS/STRN)

The parser reads each file as plain text and uses these anchors:
- Subcase start: a line matching $SUBCASE <id>
- Block start: $ELEMENT STRESS(PLATE) [REAL] or $ELEMENT STRAIN(PLATE) [REAL]
- Data rows: lines beginning with a plate ID followed by 7 numeric columns (VON, XX1, XX2, YY1, YY2, XY1, XY2)

For each $SUBCASE, the parser extracts:
- Plate ID
- XX1, XX2, YY1, YY2, XY1, XY2

The VON column is ignored.

## 3) Vector Assembly

### Target Vector $E_{target}$
- Source: IR .strs or IR .strn
- Subcase: the first $SUBCASE$ (default)
- Size: $24 \times 6 = 144$ entries
- Layout: element-major, using the component order above
- Element order: canonical from the IR target subcase, used to align all SPC subcases

### Influence Matrix $H$
- Source: SPC .strs or SPC .strn
- Columns: each $SUBCASE$ becomes one column
- Element order: enforced to match the target subcase element IDs
- Size: $144 \times N_{subcase}$
- Scaling: divide each column by 1000 (unit load 1000 N)
- Subcase IDs are kept as column labels

## 4) Optimization Model (LASSO)

We solve the sparse recovery problem:

$$
\min_{F} \frac{1}{2n}\|H F - E_{target}\|_2^2 + \alpha \|F\|_1
$$

- LassoCV is used to select $\alpha$ automatically
- fit_intercept=False to preserve physical meaning
- No automatic normalization is performed unless enabled in the script

## 5) Outputs And Traceability

The script writes all outputs to outputs/:
- stress_H_<timestamp>.csv, strain_H_<timestamp>.csv
- stress_E_target_<timestamp>.csv, strain_E_target_<timestamp>.csv
- stress_result_<timestamp>.csv, strain_result_<timestamp>.csv
- run_log_<timestamp>.txt (file hashes, dimensions, alpha grid, selected alpha, fit metrics)
- report_<timestamp>.md (summary table, nonzero forces, error metrics)

All outputs include subcase IDs so you can map them manually to loads later.

## 6) Execution

From the workspace root:

```
python scripts/ir_lasso_pipeline.py --mode both
```

Optional flags:
- --mode stress|strain|both
- --ir-subcase 1
- --scale-h 1000
- --alpha-grid 1e-6,1e-5,1e-4,1e-3,1e-2,1e-1
- --max-iter 100000
- --tol 1e-6
- --no-timestamp

## 7) Validation Checklist

- Each $SUBCASE should contain exactly 24 plates.
- H shape is (144, Nsubcase) for both stress and strain.
- E_target shape is (144,) for both stress and strain.
- Residual norm and relative error are reported in report.md.

## 8) Modify Points

- Component order: scripts/ir_lasso_pipeline.py -> COMPONENT_ORDER
- Output directory: scripts/ir_lasso_pipeline.py -> DEFAULT_OUTPUT_DIR
- LASSO settings: scripts/ir_lasso_pipeline.py -> solve_lasso
