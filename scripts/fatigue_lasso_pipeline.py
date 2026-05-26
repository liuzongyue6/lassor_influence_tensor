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
        f"- Critical elements: {meta['critical_elem_ids']}",
        f"- Target ranking: {meta['target_ranking']}",
        f"- Critical weight: {meta['critical_weight']}",
        "",
        "## Quality Metrics",
        f"- Alpha (regularization): {meta['alpha']:.6e}",
        f"- Relative error Δσ (range):  {meta['relative_error_range']:.4%}",
        f"- Relative error σ_max:        {meta['relative_error_max']:.4%}",
        f"- Relative error σ_min:        {meta['relative_error_min']:.4%}",
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
    gamma: float = 2.0,
    max_iter: int = 10,
) -> Tuple[np.ndarray, np.ndarray, float, dict]:
    """Run Phase 1 (weighted group LASSO on Δσ) with optional Phase 3 IRLS.

    If target_ranking is empty, runs Phase 1 once and returns immediately.

    Returns
    -------
    f_range       : force range vector (n,)
    active_mask   : boolean feature mask from best iteration
    alpha_used    : regularization alpha from best iteration
    diagnostics   : dict with ranking results (empty if no ranking requested)
    """
    weights = weights_init.copy()
    n_comp = len(COMPONENT_ORDER)
    n_elem = len(elem_ids)
    elem_idx = {eid: i for i, eid in enumerate(elem_ids)}
    K = len(target_ranking)

    # ── Phase 1 only (no ranking) ──────────────────────────────────────────
    if K == 0:
        if max_active_groups is not None:
            alpha, result = find_alpha_for_k_groups(
                H, delta_sigma, weights, group_size,
                max_active_groups, alpha_lo, alpha_hi, standardize,
            )
        else:
            alpha, result = _grid_search_alpha(H, delta_sigma, weights, group_size, alpha_grid, standardize)
        return result["coef"], result["active_mask"], alpha, {}

    # ── Phase 1 + Phase 3 (IRLS ranking loop) ─────────────────────────────
    # When max_active_groups is None, fix alpha after first grid search.
    # When max_active_groups is set, re-run binary search each iteration
    # with warm-started bounds for efficiency.
    alpha_fixed: Optional[float] = None
    if max_active_groups is None:
        print("[INFO] Pre-search: finding best alpha via grid search...")
        alpha_fixed, _ = _grid_search_alpha(H, delta_sigma, weights, group_size, alpha_grid, standardize)
        print(f"[INFO] Alpha fixed for IRLS: {alpha_fixed:.3e}")

    # Per-critical-element weight multipliers (updated independently per IRLS iter)
    per_elem_mult: Dict[int, float] = {
        eid: float(weights_init[elem_idx[eid] * n_comp])
        for eid in target_ranking
    }

    best_satisfied = -1
    best_state: Optional[dict] = None
    no_improve_count = 0
    history: List[dict] = []
    prev_alpha: Optional[float] = None

    for k in tqdm(range(max_iter), desc="IRLS ranking iterations"):
        # ── solve ────────────────────────────────────────────────────────
        if max_active_groups is not None:
            # Warm-start binary search bounds from previous alpha
            lo = max(alpha_lo, prev_alpha * 0.2) if prev_alpha else alpha_lo
            hi = min(alpha_hi, prev_alpha * 5.0) if prev_alpha else alpha_hi
            alpha, result = find_alpha_for_k_groups(
                H, delta_sigma, weights, group_size,
                max_active_groups, lo, hi, standardize,
            )
            # Safety: if warm bounds failed, widen back to full range
            if _count_active_groups(result["active_mask"], group_size) > max_active_groups:
                alpha, result = find_alpha_for_k_groups(
                    H, delta_sigma, weights, group_size,
                    max_active_groups, alpha_lo, alpha_hi, standardize,
                )
            prev_alpha = alpha
        else:
            alpha = alpha_fixed  # type: ignore[assignment]
            result = solve_group_lasso_weighted(H, delta_sigma, weights, group_size, alpha, standardize)

        f_range = result["coef"]

        # ── compute damage proxy and ranking ─────────────────────────────
        delta_pred = H @ f_range
        s_e = compute_von_mises_range(delta_pred, n_elem, COMPONENT_ORDER)
        ranks = _element_ranks(s_e)

        # ── check satisfaction ───────────────────────────────────────────
        n_satisfied = 0
        needs_boost: List[Tuple[int, int, int]] = []
        for p, c in enumerate(target_ranking, start=1):
            cidx = elem_idx[c]
            actual_rank = int(ranks[cidx])
            if actual_rank == p:
                n_satisfied += 1
            elif actual_rank > p:
                needs_boost.append((c, cidx, actual_rank - p))

        details = [
            {
                "elem": c,
                "desired": p,
                "actual": int(ranks[elem_idx[c]]),
                "proxy": float(s_e[elem_idx[c]]),
            }
            for p, c in enumerate(target_ranking, start=1)
        ]
        history.append({"iter": k, "n_satisfied": n_satisfied, "rel_err": result["relative_error"], "details": details})
        print(
            f"  [Iter {k:2d}] satisfied {n_satisfied}/{K} "
            f"| Δσ rel_err={result['relative_error']:.3%} "
            f"| active_groups={_count_active_groups(result['active_mask'], group_size)}"
        )

        # ── track best state ─────────────────────────────────────────────
        if n_satisfied > best_satisfied:
            best_satisfied = n_satisfied
            best_state = {
                "f_range": f_range.copy(),
                "active_mask": result["active_mask"].copy(),
                "alpha": alpha,
                "iter": k,
                "s_e": s_e.copy(),
                "ranks": ranks.copy(),
                "rel_err": result["relative_error"],
            }
            no_improve_count = 0
        else:
            no_improve_count += 1

        if n_satisfied == K:
            print(f"  All {K} ranking constraints satisfied at iteration {k}.")
            break
        if no_improve_count >= 3:
            print(f"  No ranking improvement for 3 consecutive iterations. Best: {best_satisfied}/{K}.")
            break

        # ── IRLS weight update ────────────────────────────────────────────
        # Each critical element's weight grows independently by γ^gap.
        # Elements already at their target rank are NOT touched.
        for c, cidx, gap in needs_boost:
            multiplier = gamma ** gap
            # 增加一个权重上限，防止无限增长
            per_elem_mult[c] = min(per_elem_mult[c] * multiplier, 1e8) 
            weights[cidx * n_comp : (cidx + 1) * n_comp] = per_elem_mult[c]

    assert best_state is not None
    final_f = best_state["f_range"]
    final_mask = best_state["active_mask"]
    final_alpha = best_state["alpha"]
    s_e_final = best_state["s_e"]
    ranks_final = best_state["ranks"]

    ranking_table = [
        {
            "elem_id": c,
            "desired_rank": p,
            "achieved_rank": int(ranks_final[elem_idx[c]]),
            "damage_proxy": float(s_e_final[elem_idx[c]]),
        }
        for p, c in enumerate(target_ranking, start=1)
    ]

    residual = H @ final_f - delta_sigma
    diag: dict = {
        "ranking_satisfied": best_satisfied == K,
        "satisfied_count": best_satisfied,
        "total_critical": K,
        "best_iteration": best_state["iter"],
        "iterations_run": len(history),
        "final_relative_error": float(np.linalg.norm(residual) / (np.linalg.norm(delta_sigma) + 1e-12)),
        "ranking_table": ranking_table,
        "history": history,
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
) -> dict:
    """Full fatigue LASSO pipeline.

    Phase 1: Weighted group LASSO on Δσ = σ_max − σ_min
    Phase 2: Restricted OLS on σ_mean on Phase-1 active groups
    Phase 3: IRLS ranking refinement (only if target_ranking is given)
    """
    print("\n[INFO] Parsing stress files (this may take a moment for large models)...")
    ir_lines  = read_text(ir_path)
    spc_lines = read_text(spc_path)

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
    print(f"[INFO] Elements: {n_elem}, SPC subcases: {len(spc_subcases)}")

    print("[INFO] Building stress vectors and influence matrix...")
    sigma_max   = build_vector_with_order(ir_subcases[ir_max_subcase], ref_ids)
    sigma_min   = build_vector_with_order(ir_subcases[ir_min_subcase], ref_ids)
    delta_sigma = sigma_max - sigma_min
    sigma_mean  = (sigma_max + sigma_min) / 2.0

    H, subcase_ids = build_matrix(spc_subcases, scale, ref_ids)
    n_groups = H.shape[1] // GROUP_SIZE
    print(f"[INFO] H shape: {H.shape}, force groups: {n_groups}")

    # Union of all critical + ranking elements for initial weight assignment
    all_critical = list(dict.fromkeys(critical_elem_ids + target_ranking))
    weights_init = build_element_weights(ref_ids, len(COMPONENT_ORDER), all_critical, critical_weight)

    # ── Phase 1 (+ Phase 3 if ranking requested) ──────────────────────────
    print("\n[INFO] Phase 1: Weighted Group LASSO on stress range Δσ...")
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
        gamma=gamma,
        max_iter=max_ranking_iter,
    )

    active_groups = _count_active_groups(active_mask, GROUP_SIZE)
    print(f"[INFO] Active groups: {active_groups} / {n_groups}, nonzero features: {int(active_mask.sum())}")

    # ── Phase 2 ───────────────────────────────────────────────────────────
    print("\n[INFO] Phase 2: Restricted OLS on mean stress σ_mean...")
    f_mean = solve_phase2_mean(H, sigma_mean, weights_init, active_mask)

    # ── Recover f_max, f_min ──────────────────────────────────────────────
    f_max = f_mean + f_range / 2.0
    f_min = f_mean - f_range / 2.0

    # ── Quality metrics ───────────────────────────────────────────────────
    delta_pred  = H @ f_range
    err_range   = float(np.linalg.norm(delta_pred - delta_sigma) / (np.linalg.norm(delta_sigma) + 1e-12))
    err_max     = float(np.linalg.norm(H @ f_max - sigma_max)    / (np.linalg.norm(sigma_max)   + 1e-12))
    err_min     = float(np.linalg.norm(H @ f_min - sigma_min)    / (np.linalg.norm(sigma_min)   + 1e-12))
    print(f"[SUCCESS] Δσ: {err_range:.3%} | σ_max: {err_max:.3%} | σ_min: {err_min:.3%}")

    # ── Damage proxy ranking ──────────────────────────────────────────────
    s_e   = compute_von_mises_range(delta_pred, n_elem, COMPONENT_ORDER)
    ranks = _element_ranks(s_e)

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
        "alpha": alpha_used,
        "relative_error_range": err_range,
        "relative_error_max": err_max,
        "relative_error_min": err_min,
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
    print(f"\n[DONE] Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
