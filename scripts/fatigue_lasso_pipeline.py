"""
Fatigue-Aware Joint Group LASSO for Stress Range Matching
==========================================================

Three-phase algorithm:
  Phase 1 – Weighted Group LASSO on Δσ = σ_max − σ_min  (fatigue driver)
  Phase 2 – Restricted OLS on σ_mean = (σ_max + σ_min)/2 (R-ratio)
  Phase 3 – IRLS ranking refinement for critical elements  (optional)

Recovery
--------
  f_max = f_mean + f_range / 2
  f_min = f_mean − f_range / 2

Usage
-----
  python scripts/fatigue_lasso_pipeline.py \\
      --ir-strs  InfluenceMatrix/.../target.strs \\
      --spc-strs InfluenceMatrix/.../unit.strs   \\
      --ir-max-subcase 1000001 --ir-min-subcase 1000002 \\
      --max-active-groups 6 \\
      --target-ranking 10058616,10014072,10067924
"""

import argparse
import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

try:
    from sklearn.linear_model import LinearRegression, Ridge
except ImportError as exc:
    raise SystemExit("scikit-learn is required: pip install scikit-learn") from exc

try:
    from group_lasso import GroupLasso
except ImportError as exc:
    raise SystemExit("group-lasso is required: pip install group-lasso") from exc

COMPONENT_ORDER: List[str] = ["XX1", "YY1", "XY1", "XX2", "YY2", "XY2"]
GROUP_SIZE = 3          # FX, FY, FZ per force application point
DEFAULT_OUTPUT_DIR = "outputs"


# ---------------------------------------------------------------------------
# Logging helpers
# ---------------------------------------------------------------------------

def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log(msg: str) -> None:
    """Print msg with HH:MM:SS timestamp prefix; handles embedded newlines."""
    for line in msg.split("\n"):
        if line:
            print(f"[{_ts()}] {line}", flush=True)
        else:
            print(flush=True)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class SubcaseData:
    subcase_id: int
    elem_ids: List[int]
    values: Dict[int, Dict[str, float]]


# ---------------------------------------------------------------------------
# I/O helpers  (logic mirrors ir_lasso_pipeline.py for compatibility)
# ---------------------------------------------------------------------------

def read_text(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as fh:
        return fh.readlines()


def sha256_file(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def parse_subcases(lines: List[str], expect_kind: str = "STRESS") -> Dict[int, SubcaseData]:
    """Parse OptiStruct STRS/STRN text file into SubcaseData dict."""
    subcase_pattern = re.compile(r"\$SUBCASE\s+(\d+)")
    data_row_pattern = re.compile(r"^\s*\d+\s+[-+0-9.Ee]+\s+[-+0-9.Ee]+")

    current_subcase: Optional[int] = None
    in_block = False
    subcases: Dict[int, SubcaseData] = {}

    for line in lines:
        m = subcase_pattern.search(line)
        if m:
            sid = int(m.group(1))
            current_subcase = sid
            if sid not in subcases:
                subcases[sid] = SubcaseData(sid, [], {})
            in_block = False
            continue

        if f"$ELEMENT {expect_kind}(PLATE)" in line:
            in_block = True
            continue

        if in_block and line.strip().startswith("--------"):
            continue

        if in_block and data_row_pattern.match(line):
            parts = line.split()
            eid = int(parts[0])
            vals: Dict[str, float] = {
                "XX1": float(parts[2]),
                "XX2": float(parts[3]),
                "YY1": float(parts[4]),
                "YY2": float(parts[5]),
                "XY1": float(parts[6]),
                "XY2": float(parts[7]),
            }
            if current_subcase is None:
                raise ValueError("Data row before any $SUBCASE header.")
            if eid not in subcases[current_subcase].values:
                subcases[current_subcase].elem_ids.append(eid)
            subcases[current_subcase].values[eid] = vals
            continue

        if in_block and line.strip() == "":
            in_block = False

    return subcases


def build_vector_with_order(subcase: SubcaseData, elem_ids: List[int]) -> np.ndarray:
    if not elem_ids:
        raise ValueError("Element list is empty.")
    vec = []
    for eid in elem_ids:
        if eid not in subcase.values:
            raise ValueError(f"Subcase {subcase.subcase_id} missing element {eid}.")
        ev = subcase.values[eid]
        for comp in COMPONENT_ORDER:
            vec.append(ev[comp])
    return np.asarray(vec, dtype=float)


def build_matrix(
    subcases: Dict[int, SubcaseData],
    scale: float,
    elem_ids: List[int],
) -> Tuple[np.ndarray, List[int]]:
    """Build influence matrix H (m × n_subcases), one column per unit load case."""
    ordered_ids = sorted(subcases.keys())
    cols = [build_vector_with_order(subcases[sid], elem_ids) / scale for sid in ordered_ids]
    return np.column_stack(cols), ordered_ids


def validate_elem_ids(ref: List[int], cand: List[int], subcase_id: int) -> None:
    rs, cs = set(ref), set(cand)
    if rs == cs:
        return
    missing = sorted(rs - cs)
    extra   = sorted(cs - rs)
    raise ValueError(
        f"Element mismatch in subcase {subcase_id}. "
        f"Missing: {missing[:10]}{'...' if len(missing) > 10 else ''} "
        f"Extra: {extra[:10]}{'...' if len(extra) > 10 else ''}"
    )


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


# ---------------------------------------------------------------------------
# CSV writers
# ---------------------------------------------------------------------------

def write_csv_vector(path: str, vector: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["index", "value"])
        for i, v in enumerate(vector, 1):
            w.writerow([i, f"{v:.10e}"])


def write_csv_matrix(path: str, matrix: np.ndarray, subcase_ids: List[int]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["index"] + [f"SUBCASE_{s}" for s in subcase_ids])
        for i in range(matrix.shape[0]):
            w.writerow([i + 1] + [f"{v:.10e}" for v in matrix[i]])


def write_force_csv(path: str, subcase_ids: List[int], coef: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["subcase_id", "force_value"])
        for sid, val in zip(subcase_ids, coef):
            w.writerow([sid, f"{val:.10e}"])


def write_ranking_csv(
    path: str,
    elem_ids: List[int],
    damage_proxy: np.ndarray,
    ranks: np.ndarray,
    target_ranking: List[int],
) -> None:
    desired = {eid: p for p, eid in enumerate(target_ranking, start=1)}
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["element_id", "damage_proxy", "predicted_rank", "desired_rank", "is_critical"])
        for rank_pos in range(1, len(elem_ids) + 1):
            idx = int(np.where(ranks == rank_pos)[0][0])
            eid = elem_ids[idx]
            w.writerow([
                eid,
                f"{damage_proxy[idx]:.6e}",
                rank_pos,
                desired.get(eid, ""),
                str(eid in desired),
            ])


def write_log(path: str, payload: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, indent=2, default=str))


def write_report(path: str, meta: dict) -> None:
    rd = meta.get("ranking_diagnostics")
    lines = [
        "# Fatigue Group LASSO Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Configuration",
        f"- MAX subcase: `{meta['ir_max_subcase']}`",
        f"- MIN subcase: `{meta['ir_min_subcase']}`",
        f"- Elements: {meta['n_elem']}",
        f"- Force groups available: {meta['n_groups']}",
        f"- Active force groups: {meta['active_groups']}",
        f"- Max active groups requested: {meta['max_active_groups_requested']}",
        f"- Auto mandatory top-K: {meta.get('auto_mandatory_top_k', 0)}",
        f"- Mandatory groups selected: {meta.get('mandatory_groups_selected', [])}",
        f"- Critical elements: {meta['critical_elem_ids']}",
        f"- Target ranking: {meta['target_ranking']}",
        f"- Critical weight: {meta['critical_weight']}",
        "",
        "## Quality Metrics",
        f"- Alpha (regularization): {meta['alpha']:.6e}",
        f"- Relative error Δσ (range):  {meta['relative_error_range']:.4%}",
        f"- Relative error σ_max:        {meta['relative_error_max']:.4%}",
        f"- Relative error σ_min:        {meta['relative_error_min']:.4%}",
        f"- Relative error σ_mean:       {meta.get('relative_error_mean', float('nan')):.4%}",
        "",
    ]

    if rd:
        lines += [
            "## Ranking Results",
            f"- Ranking satisfied: **{rd['ranking_satisfied']}**",
            f"- Constraints satisfied: {rd['satisfied_count']} / {rd['total_critical']}",
            f"- Best iteration: {rd['best_iteration']}",
            f"- Iterations run: {rd['iterations_run']}",
            f"- Δσ relative error at best state: {rd['final_relative_error']:.4%}",
            "",
            "### Per-Element",
            "| Element | Desired rank | Achieved rank | Damage proxy |",
            "|---------|-------------|---------------|-------------|",
        ]
        for row in rd["ranking_table"]:
            satisfied_mark = "✓" if row["achieved_rank"] == row["desired_rank"] else "✗"
            lines.append(
                f"| {row['elem_id']} | {row['desired_rank']} "
                f"| {row['achieved_rank']} {satisfied_mark} "
                f"| {row['damage_proxy']:.3e} |"
            )
        lines.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Core math helpers
# ---------------------------------------------------------------------------

def _safe_scale(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    scale = np.std(values, axis=0)
    return np.where(scale < eps, 1.0, scale)


def _make_groups(n_features: int, group_size: int) -> np.ndarray:
    """Build group label array: [1,1,1, 2,2,2, ..., G,G,G, G+1,...] for remainders."""
    n_full = n_features // group_size
    groups = np.repeat(np.arange(1, n_full + 1), group_size)
    remainder = n_features % group_size
    if remainder:
        groups = np.concatenate([groups, np.full(remainder, n_full + 1)])
    return groups


def _group_active_from_features(active_features: np.ndarray, group_size: int) -> np.ndarray:
    """Expand active-feature mask to whole-group granularity.

    If ANY feature in a group is active, all features of that group are marked active.
    Handles non-divisible n gracefully via the last (possibly smaller) group.
    """
    n = len(active_features)
    result = np.zeros(n, dtype=bool)
    n_full = n // group_size
    for g in range(n_full):
        s, e = g * group_size, (g + 1) * group_size
        if active_features[s:e].any():
            result[s:e] = True
    if n > n_full * group_size:
        s = n_full * group_size
        if active_features[s:].any():
            result[s:] = True
    return result


def _count_active_groups(active_mask: np.ndarray, group_size: int) -> int:
    """Count how many force groups have at least one non-zero feature."""
    n = len(active_mask)
    n_full = n // group_size
    count = sum(
        active_mask[g * group_size : (g + 1) * group_size].any()
        for g in range(n_full)
    )
    if n > n_full * group_size and active_mask[n_full * group_size :].any():
        count += 1
    return count


def build_element_weights(
    elem_ids: List[int],
    n_comp: int,
    critical_elem_ids: List[int],
    critical_weight: float,
) -> np.ndarray:
    """Build per-component weight vector.  Critical elements get weight = critical_weight."""
    weights = np.ones(len(elem_ids) * n_comp, dtype=float)
    critical_set = set(critical_elem_ids)
    for idx, eid in enumerate(elem_ids):
        if eid in critical_set:
            weights[idx * n_comp : (idx + 1) * n_comp] = critical_weight
    return weights


def compute_von_mises_range(
    delta_sigma: np.ndarray,
    n_elem: int,
    comp_order: List[str] = COMPONENT_ORDER,
) -> np.ndarray:
    """Compute per-element Von Mises equivalent of stress range.

    For each plate element with top (1) and bottom (2) faces:
        VM = sqrt(ΔXX² + ΔYY² − ΔXX·ΔYY + 3·ΔXY²)
    Returns max(VM_face1, VM_face2) per element as damage proxy.
    """
    n_comp = len(comp_order)
    ds = delta_sigma.reshape(n_elem, n_comp)
    ci = {c: i for i, c in enumerate(comp_order)}

    def _vm(dxx: np.ndarray, dyy: np.ndarray, dxy: np.ndarray) -> np.ndarray:
        return np.sqrt(np.maximum(dxx**2 + dyy**2 - dxx * dyy + 3.0 * dxy**2, 0.0))

    vm1 = _vm(ds[:, ci["XX1"]], ds[:, ci["YY1"]], ds[:, ci["XY1"]])
    vm2 = _vm(ds[:, ci["XX2"]], ds[:, ci["YY2"]], ds[:, ci["XY2"]])
    return np.maximum(vm1, vm2)


def _element_ranks(s_e: np.ndarray) -> np.ndarray:
    """1-indexed ranks: rank 1 = highest damage proxy."""
    n = len(s_e)
    order = np.argsort(-s_e)
    ranks = np.empty(n, dtype=int)
    ranks[order] = np.arange(1, n + 1)
    return ranks


def _weighted_ols_fixed_support(
    H: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    """Weighted OLS on fixed active columns — used in Phase 3 IRLS.

    Does NOT re-run group selection.  active_mask is treated as immutable so that
    IRLS iterations only adjust force magnitudes, never the group structure.
    """
    if not np.any(active_mask):
        return np.zeros(H.shape[1])
    H_active = H[:, active_mask]
    H_w = H_active * weights[:, np.newaxis]
    t_w  = target  * weights
    ridge = Ridge(alpha=1e-8, fit_intercept=False)
    ridge.fit(H_w, t_w)
    coef_full = np.zeros(H.shape[1])
    coef_full[active_mask] = ridge.coef_
    return coef_full


def compute_group_influence(
    H: np.ndarray,
    critical_elem_ids: List[int],
    elem_ids: List[int],
    group_size: int,
    n_comp: int,
) -> Dict[int, np.ndarray]:
    """Per-group Frobenius norm of H rows belonging to each critical element.

    Returns dict: elem_id → array of shape (n_groups,).
    High scores identify which force groups can physically excite that element.
    """
    n_features = H.shape[1]
    n_groups   = n_features // group_size
    elem_idx   = {eid: i for i, eid in enumerate(elem_ids)}
    influence: Dict[int, np.ndarray] = {}
    for eid in critical_elem_ids:
        if eid not in elem_idx:
            continue
        cidx  = elem_idx[eid]
        row_s = cidx * n_comp
        row_e = row_s + n_comp
        H_elem = H[row_s:row_e, :]
        scores = np.array([
            np.linalg.norm(H_elem[:, g * group_size : (g + 1) * group_size], "fro")
            for g in range(n_groups)
        ])
        influence[eid] = scores
    return influence


def write_influence_csv(
    path: str,
    influence_dict: Dict[int, np.ndarray],
    n_groups: int,
    subcase_ids: Optional[List[int]] = None,
) -> None:
    """Write per-group influence scores for each critical element to CSV."""
    headers = ["critical_element"]
    for g in range(n_groups):
        label = f"group_{g}"
        if subcase_ids and g * 3 < len(subcase_ids):
            label += f"(sc{subcase_ids[g * 3]})"
        headers.append(label)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(headers)
        for eid, scores in sorted(influence_dict.items()):
            w.writerow([eid] + [f"{s:.6e}" for s in scores])


# ---------------------------------------------------------------------------
# Weighted Group LASSO solver  (Phase 1 core)
# ---------------------------------------------------------------------------

def solve_group_lasso_weighted(
    H: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    group_size: int,
    alpha: float,
    standardize: bool,
) -> dict:
    """Weighted group LASSO followed by relaxed (unpenalized) OLS on active groups.

    Applies diagonal weight W = diag(weights) by transforming:
        H_w = W · H,   t_w = W · target
    then solves the standard group LASSO on the transformed system.
    After selecting active groups, refits with plain OLS to remove LASSO shrinkage.
    """
    H_w = H * weights[:, np.newaxis]
    t_w = target * weights

    if standardize:
        feat_scale = _safe_scale(H_w)
        tgt_scale  = float(_safe_scale(t_w))
        H_ws = H_w / feat_scale
        t_ws = t_w / tgt_scale
    else:
        feat_scale = np.ones(H_w.shape[1], dtype=float)
        tgt_scale  = 1.0
        H_ws = H_w
        t_ws = t_w

    groups = _make_groups(H.shape[1], group_size)

    gl = GroupLasso(
        groups=groups,
        group_reg=alpha,
        l1_reg=0.0,
        fit_intercept=False,
        scale_reg="inverse_group_size",
        supress_warning=True,
        tol=1e-4,
    )
    gl.fit(H_ws, t_ws.reshape(-1, 1))
    coef_scaled = gl.coef_.flatten()

    active_mask = np.abs(coef_scaled) > 1e-10

    coef_scaled_relaxed = np.zeros_like(coef_scaled)
    if np.any(active_mask):
        ols = LinearRegression(fit_intercept=False)
        ols.fit(H_ws[:, active_mask], t_ws)
        coef_scaled_relaxed[active_mask] = ols.coef_

    coef = coef_scaled_relaxed * tgt_scale / feat_scale

    # Residual is on the original (unweighted) problem
    residual = H @ coef - target

    return {
        "coef": coef,
        "alpha": alpha,
        "active_mask": active_mask,
        "active_groups": _count_active_groups(active_mask, group_size),
        "nonzero": int(active_mask.sum()),
        "residual_norm": float(np.linalg.norm(residual)),
        "relative_error": float(np.linalg.norm(residual) / (np.linalg.norm(target) + 1e-12)),
        "feat_scale": feat_scale,
        "tgt_scale": tgt_scale,
    }


def find_alpha_for_k_groups(
    H: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    group_size: int,
    max_active_groups: int,
    alpha_lo: float,
    alpha_hi: float,
    standardize: bool,
    max_bisect: int = 25,
) -> Tuple[float, dict]:
    """Binary search (log scale) for the smallest alpha that yields ≤ max_active_groups.

    Returns the best (lowest-error) solution satisfying the group count constraint.
    """
    result_lo = solve_group_lasso_weighted(H, target, weights, group_size, alpha_lo, standardize)
    if _count_active_groups(result_lo["active_mask"], group_size) <= max_active_groups:
        return alpha_lo, result_lo

    lo_log = np.log(alpha_lo)
    hi_log = np.log(alpha_hi)
    best_alpha = alpha_hi
    best_result = solve_group_lasso_weighted(H, target, weights, group_size, alpha_hi, standardize)

    for _ in range(max_bisect):
        if hi_log - lo_log < 1e-6:
            break
        mid_log = (lo_log + hi_log) / 2.0
        mid = float(np.exp(mid_log))
        r = solve_group_lasso_weighted(H, target, weights, group_size, mid, standardize)
        n_active = _count_active_groups(r["active_mask"], group_size)
        if n_active <= max_active_groups:
            best_alpha = mid
            best_result = r
            hi_log = mid_log
        else:
            lo_log = mid_log

    return best_alpha, best_result


def _grid_search_alpha(
    H: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    group_size: int,
    alpha_grid: List[float],
    standardize: bool,
) -> Tuple[float, dict]:
    """Grid search: return alpha with lowest relative error."""
    best_result: Optional[dict] = None
    best_err = float("inf")
    best_alpha = alpha_grid[0]
    for a in tqdm(alpha_grid, desc="Alpha grid search"):
        r = solve_group_lasso_weighted(H, target, weights, group_size, a, standardize)
        if r["relative_error"] < best_err:
            best_err = r["relative_error"]
            best_result = r
            best_alpha = a
    assert best_result is not None
    return best_alpha, best_result


# ---------------------------------------------------------------------------
# Phase 2 – restricted OLS for mean stress
# ---------------------------------------------------------------------------

def solve_phase2_mean(
    H: np.ndarray,
    sigma_mean: np.ndarray,
    weights: np.ndarray,
    active_mask: np.ndarray,
) -> np.ndarray:
    """Weighted restricted OLS: fit σ_mean using only the Phase-1 active groups.

    active_mask is expanded to group granularity before use, so partial groups
    that were activated by Phase 1 are fully included here.
    """
    group_active_mask = _group_active_from_features(active_mask, GROUP_SIZE)
    if not np.any(group_active_mask):
        return np.zeros(H.shape[1])

    H_active = H[:, group_active_mask]
    H_w = H_active * weights[:, np.newaxis]
    t_w = sigma_mean * weights

    ridge = Ridge(alpha=1e-8, fit_intercept=False)
    ridge.fit(H_w, t_w)

    coef_full = np.zeros(H.shape[1])
    coef_full[group_active_mask] = ridge.coef_
    return coef_full


# ---------------------------------------------------------------------------
# Phase 1 BCD solver (mandatory + free groups)
# ---------------------------------------------------------------------------

def solve_bcd(
    H: np.ndarray,
    target: np.ndarray,
    weights: np.ndarray,
    group_size: int,
    mandatory_feature_mask: np.ndarray,
    max_free_groups: int,
    alpha_lo: float,
    alpha_hi: float,
    standardize: bool,
    max_bcd_iter: int = 5,
) -> dict:
    """Block Coordinate Descent: mandatory groups (OLS) + free groups (group LASSO).

    Alternates until convergence:
      f_mand = WeightedOLS(H_mand, target − H_free @ f_free)
      f_free = GroupLasso(H_free, target − H_mand @ f_mand, ≤ max_free_groups)

    Final step: joint OLS on all selected columns to remove LASSO shrinkage.
    max_free_groups = 0 means only mandatory groups (pure OLS, no LASSO step).
    """
    H_mand = H[:, mandatory_feature_mask]
    free_mask = ~mandatory_feature_mask
    H_free_all = H[:, free_mask]
    free_indices = np.where(free_mask)[0]

    f_mand = np.zeros(H_mand.shape[1])
    f_free = np.zeros(H_free_all.shape[1])
    free_active_local = np.zeros(H_free_all.shape[1], dtype=bool)
    last_free_alpha = alpha_lo

    for _ in range(max_bcd_iter):
        # OLS step for mandatory groups
        res_mand = target - H_free_all @ f_free
        ridge_mand = Ridge(alpha=1e-8, fit_intercept=False)
        ridge_mand.fit(H_mand * weights[:, np.newaxis], res_mand * weights)
        f_mand = ridge_mand.coef_

        # Group LASSO step for free groups
        if max_free_groups > 0 and H_free_all.shape[1] > 0:
            res_free = target - H_mand @ f_mand
            last_free_alpha, free_result = find_alpha_for_k_groups(
                H_free_all, res_free, weights, group_size,
                max_free_groups, alpha_lo, alpha_hi, standardize,
            )
            f_free = free_result["coef"]
            free_active_local = free_result["active_mask"]

    # Build combined active mask in original feature space
    all_active = mandatory_feature_mask.copy()
    for local_i, global_i in enumerate(free_indices):
        if free_active_local[local_i]:
            all_active[global_i] = True

    # Joint OLS on all selected columns (removes LASSO shrinkage bias)
    coef_full = _weighted_ols_fixed_support(H, target, weights, all_active)
    residual  = H @ coef_full - target

    return {
        "coef":           coef_full,
        "active_mask":    all_active,
        "active_groups":  _count_active_groups(all_active, group_size),
        "nonzero":        int(all_active.sum()),
        "residual_norm":  float(np.linalg.norm(residual)),
        "relative_error": float(np.linalg.norm(residual) / (np.linalg.norm(target) + 1e-12)),
        "alpha":          last_free_alpha,
    }


# ---------------------------------------------------------------------------
# Phase 1 + Phase 3 combined: IRLS ranking loop
# ---------------------------------------------------------------------------

def solve_phase1_with_ranking(
    H: np.ndarray,
    delta_sigma: np.ndarray,
    weights_init: np.ndarray,
    elem_ids: List[int],
    group_size: int,
    alpha_grid: List[float],
    max_active_groups: Optional[int],
    alpha_lo: float,
    alpha_hi: float,
    standardize: bool,
    target_ranking: List[int],
    mandatory_feature_mask: Optional[np.ndarray] = None,
    gamma: float = 2.0,
    max_iter: int = 10,
) -> Tuple[np.ndarray, np.ndarray, float, dict]:
    """Phase 1 (group LASSO / BCD on Δσ) + optional Phase 3 (fixed-support IRLS).

    V2 design: Phase 1 runs ONCE to fix the active group support.  Phase 3 IRLS
    only adjusts force magnitudes via weighted OLS on that fixed support — it never
    re-runs group selection, preventing catastrophic error jumps.

    If mandatory_feature_mask is provided (from Phase 0), Phase 1 uses BCD so
    the mandatory groups are guaranteed to be active regardless of global fit.

    Returns
    -------
    f_range      : force range vector (n,)
    active_mask  : boolean feature mask (group-expanded) from best IRLS state
    alpha_used   : regularization alpha from Phase 1
    diagnostics  : dict with ranking results (empty when target_ranking=[])
    """
    weights  = weights_init.copy()
    n_comp   = len(COMPONENT_ORDER)
    n_elem   = len(elem_ids)
    elem_idx = {eid: i for i, eid in enumerate(elem_ids)}
    K        = len(target_ranking)

    # ── Phase 1 helper (runs once) ────────────────────────────────────────
    def _run_phase1(w: np.ndarray) -> Tuple[float, dict]:
        if mandatory_feature_mask is not None and mandatory_feature_mask.any():
            n_mand = _count_active_groups(mandatory_feature_mask, group_size)
            free_budget = (
                max(0, max_active_groups - n_mand)
                if max_active_groups is not None
                else (H.shape[1] // group_size)
            )
            r = solve_bcd(
                H, delta_sigma, w, group_size,
                mandatory_feature_mask, free_budget,
                alpha_lo, alpha_hi, standardize,
            )
            return r.get("alpha") or alpha_lo, r
        if max_active_groups is not None:
            return find_alpha_for_k_groups(
                H, delta_sigma, w, group_size,
                max_active_groups, alpha_lo, alpha_hi, standardize,
            )
        return _grid_search_alpha(H, delta_sigma, w, group_size, alpha_grid, standardize)

    # ── Phase 1 only (no ranking) ──────────────────────────────────────────
    if K == 0:
        alpha, result = _run_phase1(weights)
        return result["coef"], result["active_mask"], alpha, {}

    # ── Phase 1: one-shot group selection ─────────────────────────────────
    _log("[INFO] Phase 1: selecting active groups (runs once)...")
    alpha_used, p1_result = _run_phase1(weights)

    # Expand to whole-group granularity and LOCK — never changes in Phase 3
    active_mask_fixed = _group_active_from_features(p1_result["active_mask"], group_size)
    n_active_fixed    = _count_active_groups(active_mask_fixed, group_size)
    p1_active_indices = [
        g for g in range(len(active_mask_fixed) // group_size)
        if active_mask_fixed[g * group_size : (g + 1) * group_size].any()
    ]
    _log(
        f"[INFO] Phase 1 locked {n_active_fixed} groups "
        f"(indices: {p1_active_indices}) "
        f"| initial range error: {p1_result['relative_error']:.3%} "
        f"| alpha: {alpha_used:.4e}"
    )

    # Per-critical-element weight multipliers (updated independently)
    per_elem_mult: Dict[int, float] = {
        eid: float(weights_init[elem_idx[eid] * n_comp])
        for eid in target_ranking
        if eid in elem_idx
    }

    best_satisfied = -1
    best_state: Optional[dict] = None
    no_improve_count = 0
    history: List[dict] = []

    # ── Phase 3: IRLS — magnitude tuning on fixed support ─────────────────
    for k in tqdm(range(max_iter), desc="IRLS ranking iterations"):
        # Weighted OLS on FIXED support — no new group selection
        f_range    = _weighted_ols_fixed_support(H, delta_sigma, weights, active_mask_fixed)
        residual   = H @ f_range - delta_sigma
        rel_err    = float(np.linalg.norm(residual) / (np.linalg.norm(delta_sigma) + 1e-12))
        delta_pred = H @ f_range
        s_e        = compute_von_mises_range(delta_pred, n_elem, COMPONENT_ORDER)
        ranks      = _element_ranks(s_e)

        # Check ranking satisfaction
        n_satisfied = 0
        needs_boost: List[Tuple[int, int, int]] = []
        for p, c in enumerate(target_ranking, start=1):
            if c not in elem_idx:
                continue
            actual_rank = int(ranks[elem_idx[c]])
            if actual_rank == p:
                n_satisfied += 1
            elif actual_rank > p:
                needs_boost.append((c, elem_idx[c], actual_rank - p))

        details = [
            {
                "elem":   c,
                "desired": p,
                "actual":  int(ranks[elem_idx[c]]) if c in elem_idx else -1,
                "proxy":   float(s_e[elem_idx[c]])  if c in elem_idx else 0.0,
            }
            for p, c in enumerate(target_ranking, start=1)
        ]
        history.append({"iter": k, "n_satisfied": n_satisfied, "rel_err": rel_err, "details": details})
        _log(
            f"[Iter {k:2d}] satisfied {n_satisfied}/{K} "
            f"| range_err={rel_err:.3%} "
            f"| active_groups={n_active_fixed}"
        )
        for d in details:
            status = "OK" if d["actual"] == d["desired"] else "--"
            w_val  = per_elem_mult.get(d["elem"], 1.0)
            print(
                f"         [{status}] elem={d['elem']:>10d}  "
                f"desired={d['desired']:>3d}  actual={d['actual']:>5d}  "
                f"proxy={d['proxy']:.3e}  weight={w_val:.2e}",
                flush=True,
            )

        # Track best state
        if n_satisfied > best_satisfied:
            best_satisfied = n_satisfied
            best_state = {
                "f_range":    f_range.copy(),
                "active_mask": active_mask_fixed.copy(),
                "alpha":      alpha_used,
                "iter":       k,
                "s_e":        s_e.copy(),
                "ranks":      ranks.copy(),
                "rel_err":    rel_err,
            }
            no_improve_count = 0
        else:
            no_improve_count += 1

        if n_satisfied == K:
            _log(f"[INFO] All {K} ranking constraints satisfied at iteration {k}.")
            break
        if no_improve_count >= 3:
            _log(f"[INFO] No improvement for 3 consecutive iterations. Best: {best_satisfied}/{K}.")
            break

        # IRLS weight update — each critical element grows independently by γ^gap
        # Use log-space arithmetic to prevent float64 overflow for large gaps
        # (e.g. gap=2317 with gamma=2.0 → 2.0^2317 >> 1.8e308)
        _LOG_CAP = np.log(1e8)
        for c, cidx, gap in needs_boost:
            old_w = per_elem_mult[c]
            new_log = np.log(max(per_elem_mult[c], 1e-300)) + gap * np.log(max(gamma, 1e-300))
            per_elem_mult[c] = float(np.exp(min(new_log, _LOG_CAP)))
            weights[cidx * n_comp : (cidx + 1) * n_comp] = per_elem_mult[c]
            print(
                f"         [^] elem={c:>10d}  gap={gap}  "
                f"weight {old_w:.2e} -> {per_elem_mult[c]:.2e}",
                flush=True,
            )

    assert best_state is not None
    final_f      = best_state["f_range"]
    final_mask   = best_state["active_mask"]
    final_alpha  = best_state["alpha"]
    s_e_final    = best_state["s_e"]
    ranks_final  = best_state["ranks"]

    ranking_table = [
        {
            "elem_id":      c,
            "desired_rank": p,
            "achieved_rank": int(ranks_final[elem_idx[c]]) if c in elem_idx else -1,
            "damage_proxy":  float(s_e_final[elem_idx[c]]) if c in elem_idx else 0.0,
        }
        for p, c in enumerate(target_ranking, start=1)
    ]

    residual = H @ final_f - delta_sigma
    diag: dict = {
        "ranking_satisfied":   best_satisfied == K,
        "satisfied_count":     best_satisfied,
        "total_critical":      K,
        "best_iteration":      best_state["iter"],
        "iterations_run":      len(history),
        "final_relative_error": float(np.linalg.norm(residual) / (np.linalg.norm(delta_sigma) + 1e-12)),
        "ranking_table":       ranking_table,
        "history":             history,
    }

    return final_f, final_mask, final_alpha, diag


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_fatigue_pipeline(
    ir_path: str,
    spc_path: str,
    output_dir: str,
    ir_max_subcase: int,
    ir_min_subcase: int,
    scale: float,
    max_active_groups: Optional[int],
    critical_elem_ids: List[int],
    critical_weight: float,
    target_ranking: List[int],
    alpha_grid: List[float],
    alpha_lo: float,
    alpha_hi: float,
    standardize: bool,
    output_suffix: str,
    max_ranking_iter: int = 10,
    gamma: float = 2.0,
    auto_mandatory_top_k: int = 2,
    mandatory_groups: Optional[List[int]] = None,
) -> dict:
    """Full fatigue LASSO pipeline.

    Phase 0: Influence-guided mandatory group pre-selection (guarantees coverage)
    Phase 1: Weighted Group LASSO / BCD on Δσ = σ_max − σ_min
    Phase 2: Restricted OLS on σ_mean on Phase-1 active groups
    Phase 3: Fixed-support IRLS ranking refinement (only if target_ranking given)
    """
    if mandatory_groups is None:
        mandatory_groups = []

    _log("[INFO] Parsing stress files (this may take a moment for large models)...")
    ir_lines  = read_text(ir_path)
    spc_lines = read_text(spc_path)
    _log(f"[INFO] Read {len(ir_lines)} lines (target) + {len(spc_lines)} lines (unit loads)")

    ir_subcases  = parse_subcases(ir_lines,  "STRESS")
    spc_subcases = parse_subcases(spc_lines, "STRESS")

    for sid, label in [(ir_max_subcase, "MAX"), (ir_min_subcase, "MIN")]:
        if sid not in ir_subcases:
            avail = ", ".join(str(s) for s in sorted(ir_subcases))
            raise ValueError(f"IR {label} subcase {sid} not found. Available: {avail}")

    ref_ids = ir_subcases[ir_max_subcase].elem_ids
    validate_elem_ids(ref_ids, ir_subcases[ir_min_subcase].elem_ids, ir_min_subcase)
    for sid, sc in spc_subcases.items():
        validate_elem_ids(ref_ids, sc.elem_ids, sid)

    n_elem = len(ref_ids)
    _log(f"[INFO] Elements: {n_elem} | SPC subcases: {len(spc_subcases)} | H scale: {scale}")

    _log("[INFO] Building stress vectors and influence matrix...")
    sigma_max   = build_vector_with_order(ir_subcases[ir_max_subcase], ref_ids)
    sigma_min   = build_vector_with_order(ir_subcases[ir_min_subcase], ref_ids)
    delta_sigma = sigma_max - sigma_min
    sigma_mean  = (sigma_max + sigma_min) / 2.0

    H, subcase_ids = build_matrix(spc_subcases, scale, ref_ids)
    n_groups = H.shape[1] // GROUP_SIZE
    h_mem_mb = H.nbytes / (1024 ** 2)
    _log(f"[INFO] H shape: {H.shape} ({h_mem_mb:.1f} MB) | force groups: {n_groups}")
    _log(
        f"[INFO] delta_sigma stats: "
        f"min={delta_sigma.min():.3e}  max={delta_sigma.max():.3e}  "
        f"norm={np.linalg.norm(delta_sigma):.3e}"
    )

    # Union of all critical + ranking elements for initial weight assignment
    all_critical = list(dict.fromkeys(critical_elem_ids + target_ranking))
    weights_init = build_element_weights(ref_ids, len(COMPONENT_ORDER), all_critical, critical_weight)

    # ── Phase 0: influence-guided mandatory group selection ────────────────
    mandatory_feature_mask = np.zeros(H.shape[1], dtype=bool)
    influence_dict: Dict[int, np.ndarray] = {}
    selected_mandatory_groups: List[int] = []

    if all_critical and (auto_mandatory_top_k > 0 or mandatory_groups):
        _log("[INFO] Phase 0: Computing group influence on critical elements...")
        influence_dict = compute_group_influence(
            H, all_critical, ref_ids, GROUP_SIZE, len(COMPONENT_ORDER)
        )
        if auto_mandatory_top_k > 0:
            for eid, scores in sorted(influence_dict.items()):
                top_k_idx  = np.argsort(-scores)[:auto_mandatory_top_k]
                top_scores = scores[top_k_idx]
                _log(
                    f"  Influence top-{auto_mandatory_top_k} for elem {eid}: "
                    f"groups {top_k_idx.tolist()} "
                    f"(Frobenius: {[f'{v:.3e}' for v in top_scores]})"
                )
        mandatory_group_set: set = set(mandatory_groups)  # user-specified indices
        if auto_mandatory_top_k > 0:
            for scores in influence_dict.values():
                top_k = np.argsort(-scores)[:auto_mandatory_top_k]
                mandatory_group_set.update(top_k.tolist())
        # Clamp to valid range
        mandatory_group_set = {g for g in mandatory_group_set if 0 <= g < n_groups}
        selected_mandatory_groups = sorted(mandatory_group_set)
        for g in selected_mandatory_groups:
            s = g * GROUP_SIZE
            e = min((g + 1) * GROUP_SIZE, H.shape[1])
            mandatory_feature_mask[s:e] = True
        n_mand = len(selected_mandatory_groups)
        mand_sc_labels = [
            str(subcase_ids[g * 3]) if g * 3 < len(subcase_ids) else f"g{g}"
            for g in selected_mandatory_groups
        ]
        _log(
            f"[INFO] Mandatory groups ({n_mand}): {selected_mandatory_groups} "
            f"(lead subcases: {mand_sc_labels})"
        )
        ensure_dir(output_dir)
        write_influence_csv(
            os.path.join(output_dir, f"stress_group_influence{output_suffix}.csv"),
            influence_dict, n_groups, subcase_ids,
        )

    # ── Phase 1 (+ Phase 3 if ranking requested) ──────────────────────────
    _log("[INFO] Phase 1/3: Group LASSO + IRLS on stress range (delta_sigma)...")
    f_range, active_mask, alpha_used, diag_ranking = solve_phase1_with_ranking(
        H=H,
        delta_sigma=delta_sigma,
        weights_init=weights_init,
        elem_ids=ref_ids,
        group_size=GROUP_SIZE,
        alpha_grid=alpha_grid,
        max_active_groups=max_active_groups,
        alpha_lo=alpha_lo,
        alpha_hi=alpha_hi,
        standardize=standardize,
        target_ranking=target_ranking,
        mandatory_feature_mask=mandatory_feature_mask if mandatory_feature_mask.any() else None,
        gamma=gamma,
        max_iter=max_ranking_iter,
    )

    active_groups = _count_active_groups(active_mask, GROUP_SIZE)
    active_g_indices = [
        g for g in range(n_groups) if active_mask[g * GROUP_SIZE : (g + 1) * GROUP_SIZE].any()
    ]
    active_g_sc = [
        str(subcase_ids[g * 3]) if g * 3 < len(subcase_ids) else f"g{g}"
        for g in active_g_indices
    ]
    _log(
        f"[INFO] Active groups: {active_groups}/{n_groups} "
        f"(indices: {active_g_indices}, lead subcases: {active_g_sc}) "
        f"| nonzero features: {int(active_mask.sum())}"
    )

    # ── Phase 2 ───────────────────────────────────────────────────────────
    _log("[INFO] Phase 2: Restricted OLS on mean stress (sigma_mean)...")
    f_mean = solve_phase2_mean(H, sigma_mean, weights_init, active_mask)
    err_mean_ols = float(
        np.linalg.norm(H @ f_mean - sigma_mean) / (np.linalg.norm(sigma_mean) + 1e-12)
    )
    _log(f"[INFO] Phase 2 mean stress error: {err_mean_ols:.3%}")

    # ── Recover f_max, f_min ──────────────────────────────────────────────
    f_max = f_mean + f_range / 2.0
    f_min = f_mean - f_range / 2.0

    # ── Quality metrics ───────────────────────────────────────────────────
    delta_pred  = H @ f_range
    err_range   = float(np.linalg.norm(delta_pred - delta_sigma) / (np.linalg.norm(delta_sigma) + 1e-12))
    err_max     = float(np.linalg.norm(H @ f_max - sigma_max)    / (np.linalg.norm(sigma_max)   + 1e-12))
    err_min     = float(np.linalg.norm(H @ f_min - sigma_min)    / (np.linalg.norm(sigma_min)   + 1e-12))
    _log(f"[SUCCESS] range_err: {err_range:.3%} | max_err: {err_max:.3%} | min_err: {err_min:.3%}")

    # ── Damage proxy ranking ──────────────────────────────────────────────
    s_e   = compute_von_mises_range(delta_pred, n_elem, COMPONENT_ORDER)
    ranks = _element_ranks(s_e)
    if all_critical:
        s_e_target   = compute_von_mises_range(delta_sigma, n_elem, COMPONENT_ORDER)
        elem_idx_map = {eid: i for i, eid in enumerate(ref_ids)}
        _log("[INFO] Critical element Von Mises range stress (target vs predicted):")
        _log(f"       {'elem_id':>12s}  {'target_VM':>12s}  {'pred_VM':>12s}  {'ratio':>8s}  {'pred_rank':>9s}")
        for eid in all_critical:
            if eid not in elem_idx_map:
                continue
            cidx    = elem_idx_map[eid]
            tgt_vm  = float(s_e_target[cidx])
            pred_vm = float(s_e[cidx])
            ratio   = pred_vm / (tgt_vm + 1e-12)
            _log(
                f"       {eid:>12d}  {tgt_vm:>12.3e}  {pred_vm:>12.3e}  "
                f"{ratio:>8.3f}  {int(ranks[cidx]):>9d}"
            )

    # ── Write outputs ─────────────────────────────────────────────────────
    ensure_dir(output_dir)
    write_csv_matrix(os.path.join(output_dir, f"stress_H{output_suffix}.csv"),                 H, subcase_ids)
    write_csv_vector(os.path.join(output_dir, f"stress_delta_target{output_suffix}.csv"),     delta_sigma)
    write_force_csv( os.path.join(output_dir, f"stress_f_range{output_suffix}.csv"),          subcase_ids, f_range)
    write_force_csv( os.path.join(output_dir, f"stress_f_mean{output_suffix}.csv"),           subcase_ids, f_mean)
    write_force_csv( os.path.join(output_dir, f"stress_f_max{output_suffix}.csv"),            subcase_ids, f_max)
    write_force_csv( os.path.join(output_dir, f"stress_f_min{output_suffix}.csv"),            subcase_ids, f_min)
    write_ranking_csv(
        os.path.join(output_dir, f"stress_ranking_check{output_suffix}.csv"),
        ref_ids, s_e, ranks, target_ranking,
    )

    metadata = {
        "ir_path": ir_path,
        "spc_path": spc_path,
        "ir_max_subcase": ir_max_subcase,
        "ir_min_subcase": ir_min_subcase,
        "subcase_ids": subcase_ids,
        "n_elem": n_elem,
        "n_groups": n_groups,
        "active_groups": active_groups,
        "max_active_groups_requested": max_active_groups,
        "auto_mandatory_top_k": auto_mandatory_top_k,
        "mandatory_groups_selected": selected_mandatory_groups,
        "alpha": alpha_used,
        "relative_error_range": err_range,
        "relative_error_max": err_max,
        "relative_error_min": err_min,
        "relative_error_mean": err_mean_ols,
        "critical_elem_ids": all_critical,
        "critical_weight": critical_weight,
        "target_ranking": target_ranking,
        "ranking_diagnostics": diag_ranking or None,
        "component_order": COMPONENT_ORDER,
        "scale": scale,
        "standardize": standardize,
        "gamma": gamma,
        "max_ranking_iter": max_ranking_iter,
    }

    return {"metadata": metadata}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_int_list(raw: str) -> List[int]:
    if not raw.strip():
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def _parse_float_list(raw: str) -> List[float]:
    return [float(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fatigue-aware joint group LASSO: match MAX+MIN stress tensors for fatigue.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input files
    parser.add_argument("--ir-strs",  required=True, help="Target stress file containing MAX and MIN subcases")
    parser.add_argument("--spc-strs", required=True, help="Unit load stress file (one subcase per force direction)")

    # Subcase IDs
    parser.add_argument("--ir-max-subcase", type=int, default=1000001, help="Subcase ID for σ_max (peak)")
    parser.add_argument("--ir-min-subcase", type=int, default=1000002, help="Subcase ID for σ_min (valley)")

    # Force model
    parser.add_argument("--scale-h", type=float, default=1000.0,
                        help="Scale factor applied to influence matrix columns (unit load magnitude)")
    parser.add_argument("--max-active-groups", type=int, default=None,
                        help="Max number of active force groups (FX/FY/FZ triplets); "
                             "triggers binary search for lambda. Omit for no limit.")

    # Critical element options
    parser.add_argument("--critical-elems", type=str, default="",
                        help="Comma-separated element IDs to up-weight (e.g. 10058616,10014072)")
    parser.add_argument("--critical-weight", type=float, default=10.0,
                        help="Weight multiplier applied to critical element residuals")
    parser.add_argument("--target-ranking", type=str, default="",
                        help="Ordered element IDs: position 0 → global rank 1, position 1 → rank 2, …  "
                             "Activates Phase-3 IRLS ranking refinement.")

    # Alpha / regularization
    parser.add_argument("--alpha-grid", type=str, default="0.1,1,10,100,1000",
                        help="Comma-separated alpha values for grid search (used when --max-active-groups is not set)")
    parser.add_argument("--alpha-lo",  type=float, default=0.01,    help="Binary search lower bound for alpha")
    parser.add_argument("--alpha-hi",  type=float, default=100000.0, help="Binary search upper bound for alpha")

    # Phase 0 — mandatory group pre-selection
    parser.add_argument("--auto-mandatory-top-k", type=int, default=2,
                        help="Top K force groups (by influence) per critical element to always activate. "
                             "Set 0 to disable Phase 0.")
    parser.add_argument("--mandatory-groups", type=str, default="",
                        help="Comma-separated 0-indexed group indices to always activate (Phase 0), "
                             "e.g. '0,3,7'")

    # IRLS settings
    parser.add_argument("--max-ranking-iter", type=int, default=10,
                        help="Max IRLS iterations for ranking refinement")
    parser.add_argument("--gamma", type=float, default=2.0,
                        help="IRLS weight amplification factor per rank-gap unit (γ^gap)")

    # Standardization
    parser.add_argument("--standardize",    dest="standardize", action="store_true")
    parser.add_argument("--no-standardize", dest="standardize", action="store_false")
    parser.set_defaults(standardize=True)

    # Output
    parser.add_argument("--output-dir",    type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamp",     dest="timestamp", action="store_true")
    parser.add_argument("--no-timestamp",  dest="timestamp", action="store_false")
    parser.set_defaults(timestamp=True)

    args = parser.parse_args()

    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S") if args.timestamp else ""
    output_suffix = f"_{timestamp}" if timestamp else ""

    critical_elem_ids = _parse_int_list(args.critical_elems)
    target_ranking    = _parse_int_list(args.target_ranking)
    alpha_grid        = _parse_float_list(args.alpha_grid)
    mandatory_groups  = _parse_int_list(args.mandatory_groups)

    result = run_fatigue_pipeline(
        ir_path=args.ir_strs,
        spc_path=args.spc_strs,
        output_dir=args.output_dir,
        ir_max_subcase=args.ir_max_subcase,
        ir_min_subcase=args.ir_min_subcase,
        scale=args.scale_h,
        max_active_groups=args.max_active_groups,
        critical_elem_ids=critical_elem_ids,
        critical_weight=args.critical_weight,
        target_ranking=target_ranking,
        alpha_grid=alpha_grid,
        alpha_lo=args.alpha_lo,
        alpha_hi=args.alpha_hi,
        standardize=args.standardize,
        output_suffix=output_suffix,
        max_ranking_iter=args.max_ranking_iter,
        gamma=args.gamma,
        auto_mandatory_top_k=args.auto_mandatory_top_k,
        mandatory_groups=mandatory_groups,
    )

    meta = result["metadata"]
    ensure_dir(args.output_dir)

    log_payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "files": {
            "ir_strs":  sha256_file(args.ir_strs),
            "spc_strs": sha256_file(args.spc_strs),
        },
        "metadata": meta,
    }
    write_log(    os.path.join(args.output_dir, f"run_log{output_suffix}.txt"),    log_payload)
    write_report( os.path.join(args.output_dir, f"report{output_suffix}.md"),      meta)
    _log(f"[DONE] Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
