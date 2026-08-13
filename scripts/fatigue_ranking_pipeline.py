"""
Critical-Ranked Local-Fit Pipeline
=====================================

Inverts the priority of fatigue_lasso_pipeline.py:

  PRIMARY   – critical elements (e.g. 10058616, 10014072) must rank above ALL others
              in Von Mises equivalent stress range
  SECONDARY – predicted stress range at critical element positions fits the target tensor
  GLOBAL    – global tensor matching is NOT required

Three-phase algorithm
---------------------
  Phase 0 – Influence-guided mandatory group pre-selection  (same as V2)
  Phase 1 – Group LASSO / BCD fitted on CRITICAL ELEMENT ROWS ONLY  ← key change
  Phase 3 – Ranking-constrained SLSQP with LOCAL fit objective       ← key change
  Phase 2 – Restricted OLS for mean stress → f_max, f_min

Why Phase 1 on critical rows?
  With all 109,524 rows the LASSO objective is dominated by the ~18,252 non-critical
  elements and may select groups with near-zero influence on the critical elements.
  Restricting to n_crit × 6 rows (e.g. 12 rows for 2 elements) forces the group
  selection to focus on coverage of the critical locations.

Why local fit objective in Phase 3?
  The SLSQP objective min ||H_crit @ f - Δσ_crit||² drives force magnitudes toward
  replicating the critical element stress while ranking constraints enforce
  VM_crit > VM_others.  No weight saturation, no global-fit collapse.

Usage (PowerShell)
------------------
  python scripts/fatigue_ranking_pipeline.py `
      --ir-strs  InfluenceMatrix/Cradle_HAZ_Element/Xpeng_Target_Stress.txt `
      --spc-strs InfluenceMatrix/Cradle_HAZ_Element/Xpeng_Unit_Load_Stress.txt `
      --ir-max-subcase 1000001 --ir-min-subcase 1000002 `
      --max-active-groups 4 --auto-mandatory-top-k 2 `
      --critical-elems 10058616,10014072 `
      --target-ranking 10058616,10014072 `
      --critical-weight 10.0 `
      --max-ranking-iter 20 `
      --output-dir outputs/ranking_v1
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

# Both scripts live in scripts/; Python adds that dir to sys.path when running
# directly, so the sibling import works.  The fallback handles edge cases.
try:
    from fatigue_lasso_pipeline import (
        read_text, sha256_file, parse_subcases, build_vector_with_order,
        build_matrix, validate_elem_ids, ensure_dir,
        COMPONENT_ORDER, GROUP_SIZE,
        _group_active_from_features, _count_active_groups,
        build_element_weights,
        compute_von_mises_range, _element_ranks, _vm_single,
        _weighted_ols_fixed_support,
        find_alpha_for_k_groups, solve_bcd,
        compute_group_influence, solve_phase2_mean,
        write_influence_csv, write_force_csv, write_ranking_csv,
        write_log, _log, reset_log_buffer, get_log_buffer, get_git_commit,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fatigue_lasso_pipeline import (
        read_text, sha256_file, parse_subcases, build_vector_with_order,
        build_matrix, validate_elem_ids, ensure_dir,
        COMPONENT_ORDER, GROUP_SIZE,
        _group_active_from_features, _count_active_groups,
        build_element_weights,
        compute_von_mises_range, _element_ranks, _vm_single,
        _weighted_ols_fixed_support,
        find_alpha_for_k_groups, solve_bcd,
        compute_group_influence, solve_phase2_mean,
        write_influence_csv, write_force_csv, write_ranking_csv,
        write_log, _log, reset_log_buffer, get_log_buffer, get_git_commit,
    )

DEFAULT_OUTPUT_DIR = "outputs"


# ---------------------------------------------------------------------------
# New helper: critical-element row mask
# ---------------------------------------------------------------------------

def _make_crit_row_mask(
    elem_ids: List[int],
    crit_ids: List[int],
    n_comp: int,
) -> np.ndarray:
    """Boolean mask over all m rows; True for rows belonging to critical elements."""
    mask = np.zeros(len(elem_ids) * n_comp, dtype=bool)
    crit_set = set(crit_ids)
    for idx, eid in enumerate(elem_ids):
        if eid in crit_set:
            mask[idx * n_comp : (idx + 1) * n_comp] = True
    return mask


# ---------------------------------------------------------------------------
# Phase 3 — Ranking-constrained SLSQP with LOCAL fit objective
# ---------------------------------------------------------------------------

def solve_local_ranking(
    H: np.ndarray,
    delta_sigma: np.ndarray,
    crit_row_mask: np.ndarray,
    active_mask: np.ndarray,
    target_ranking: List[int],
    elem_ids: List[int],
    n_comp: int,
    f_init: np.ndarray,
    margin: float = 0.0,
    max_iter: int = 20,
    max_blockers_per_iter: int = 200,
) -> Tuple[np.ndarray, dict]:
    """SLSQP: LOCAL fit objective at critical elements + ranking constraints.

    Objective  (local, not global):
        min  ||H_crit_active @ f - Δσ_crit||²
        where H_crit_active = H[crit_row_mask][:, active_mask]

    Constraints:
        VM_range(H_{crit_e,active} @ f) - VM_range(H_{blocker,active} @ f) >= margin
        for each blocker element currently ranked above the desired rank of crit_e

    Iteration loop:
        1. Evaluate current ranking from H_active @ f_current (uses FULL H, all elements)
        2. Collect blocker constraints for unsatisfied critical elements
        3. Solve SLSQP with local objective + blocker constraints
        4. Repeat until all satisfied, 3 no-improvement rounds, or max_iter
    """
    try:
        from scipy.optimize import minimize as _sp_minimize
    except ImportError as exc:
        raise SystemExit("scipy is required: pip install scipy") from exc

    H_active      = H[:, active_mask]                        # (m) × n_active
    H_crit_active = H[crit_row_mask][:, active_mask]         # (n_crit*6) × n_active
    delta_crit    = delta_sigma[crit_row_mask]                # (n_crit*6,)

    n_active = H_active.shape[1]
    n_elem   = len(elem_ids)
    elem_idx = {eid: i for i, eid in enumerate(elem_ids)}
    K        = len(target_ranking)

    # Precompute local normal equations — tiny (n_active × n_active)
    HtH_crit     = H_crit_active.T @ H_crit_active
    Htb_crit     = H_crit_active.T @ delta_crit
    b_norm_sq    = float(np.dot(delta_crit, delta_crit))

    def _obj(fa: np.ndarray) -> float:
        return float(fa @ HtH_crit @ fa) - 2.0 * float(Htb_crit @ fa) + b_norm_sq

    def _grad_obj(fa: np.ndarray) -> np.ndarray:
        return 2.0 * (HtH_crit @ fa - Htb_crit)

    f_current       = f_init[active_mask].copy()
    best_satisfied  = -1
    best_state: Optional[dict] = None
    no_improve_cnt  = 0
    history: List[dict] = []

    for k in range(max_iter):
        # ── Evaluate full ranking (uses all elements) ──────────────────────
        delta_pred_full = H_active @ f_current                # (m,)
        s_e   = compute_von_mises_range(delta_pred_full, n_elem, COMPONENT_ORDER)
        ranks = _element_ranks(s_e)

        # Local fit error (critical elements only)
        local_err = float(
            np.linalg.norm(H_crit_active @ f_current - delta_crit)
            / (np.linalg.norm(delta_crit) + 1e-12)
        )

        n_satisfied = 0
        details     = []
        constraints = []

        for p_idx, crit_id in enumerate(target_ranking):
            desired_rank = p_idx + 1
            if crit_id not in elem_idx:
                details.append({"elem": crit_id, "desired": desired_rank,
                                 "actual": -1, "proxy": 0.0})
                continue
            crit_i      = elem_idx[crit_id]
            actual_rank = int(ranks[crit_i])
            details.append({
                "elem": crit_id, "desired": desired_rank,
                "actual": actual_rank, "proxy": float(s_e[crit_i]),
            })

            if actual_rank == desired_rank:
                n_satisfied += 1
                continue

            H_crit_sub = H_active[crit_i * n_comp : (crit_i + 1) * n_comp, :]

            # Blockers = elements ranked at or above desired_rank (must be displaced)
            # plus elements between desired_rank and actual_rank
            blocker_set = set(
                j for j in range(n_elem)
                if ranks[j] <= desired_rank and j != crit_i
            )
            blocker_set.update(
                j for j in range(n_elem)
                if desired_rank < ranks[j] < actual_rank and j != crit_i
            )
            # Don't treat already-satisfied higher-priority critical elements as blockers
            for prev_idx in range(p_idx):
                prev_id = target_ranking[prev_idx]
                if prev_id in elem_idx and ranks[elem_idx[prev_id]] == prev_idx + 1:
                    blocker_set.discard(elem_idx[prev_id])

            # Sort by worst violation first (largest s_e[blocker] - s_e[crit])
            blockers_sorted = sorted(
                blocker_set,
                key=lambda j: float(s_e[j] - s_e[crit_i]),
                reverse=True,
            )[:max_blockers_per_iter]

            for bi in blockers_sorted:
                H_blocker_sub = H_active[bi * n_comp : (bi + 1) * n_comp, :]
                # Capture sub-matrices by value to avoid late-binding closure issue
                def _make_con(Hc=H_crit_sub.copy(), Hb=H_blocker_sub.copy()):
                    def _con(fa):
                        return _vm_single(Hc @ fa) - _vm_single(Hb @ fa) - margin
                    return _con
                constraints.append({"type": "ineq", "fun": _make_con()})

        history.append({
            "iter": k, "n_satisfied": n_satisfied,
            "local_err": local_err, "details": details,
        })
        _log(
            f"[Iter {k:2d}] satisfied {n_satisfied}/{K} "
            f"| local_fit_err={local_err:.3%} | constraints={len(constraints)}"
        )
        for d in details:
            status = "OK" if d["actual"] == d["desired"] else "--"
            print(
                f"         [{status}] elem={d['elem']:>10d}  "
                f"desired={d['desired']:>3d}  actual={d['actual']:>5d}  "
                f"proxy={d['proxy']:.3e}",
                flush=True,
            )

        # Track best state
        if n_satisfied > best_satisfied:
            best_satisfied = n_satisfied
            best_state = {
                "f_active":  f_current.copy(),
                "iter":      k,
                "s_e":       s_e.copy(),
                "ranks":     ranks.copy(),
                "local_err": local_err,
            }
            no_improve_cnt = 0
        else:
            no_improve_cnt += 1

        if n_satisfied == K:
            _log(f"[INFO] All {K} ranking constraints satisfied at iteration {k}.")
            break
        if no_improve_cnt >= 3:
            _log(f"[INFO] No improvement for 3 consecutive iterations. Best: {best_satisfied}/{K}.")
            break
        if not constraints:
            _log(f"[INFO] No active constraints at iteration {k} — done.")
            break

        # ── SLSQP solve ────────────────────────────────────────────────────
        result = _sp_minimize(
            _obj, f_current, jac=_grad_obj,
            method="SLSQP",
            constraints=constraints,
            options={"maxiter": 500, "ftol": 1e-12, "disp": False},
        )
        if not result.success and result.status not in (0, 9):
            _log(f"[WARNING] SLSQP iter {k}: {result.message} (continuing with best feasible point)")
        f_current = result.x

    assert best_state is not None

    final_f = np.zeros(H.shape[1])
    final_f[active_mask] = best_state["f_active"]

    ranking_table = [
        {
            "elem_id":       c,
            "desired_rank":  p,
            "achieved_rank": int(best_state["ranks"][elem_idx[c]]) if c in elem_idx else -1,
            "damage_proxy":  float(best_state["s_e"][elem_idx[c]])  if c in elem_idx else 0.0,
        }
        for p, c in enumerate(target_ranking, start=1)
    ]

    residual_local = H_crit_active @ best_state["f_active"] - delta_crit
    diag: dict = {
        "ranking_satisfied":        best_satisfied == K,
        "satisfied_count":          best_satisfied,
        "total_critical":           K,
        "best_iteration":           best_state["iter"],
        "iterations_run":           len(history),
        "final_local_fit_error":    float(np.linalg.norm(residual_local) / (np.linalg.norm(delta_crit) + 1e-12)),
        "ranking_table":            ranking_table,
        "history":                  history,
        "basis_insufficient_warning": (best_satisfied < K and best_state["iter"] == 0),
    }
    return final_f, diag


# ---------------------------------------------------------------------------
# Report writer
# ---------------------------------------------------------------------------

def write_ranking_report(path: str, meta: dict) -> None:
    rd = meta.get("ranking_diagnostics")
    lines = [
        "# Critical-Ranked Local-Fit Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Configuration",
        f"- MAX subcase: `{meta['ir_max_subcase']}`",
        f"- MIN subcase: `{meta['ir_min_subcase']}`",
        f"- Elements (total): {meta['n_elem']}",
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
        f"- Local fit error Δσ (critical elements only): {meta['local_fit_error_range']:.4%}",
        f"- Global Δσ error (informational, not optimised): {meta['global_error_range']:.4%}",
        f"- Mean stress error (σ_mean, global):            {meta.get('relative_error_mean', float('nan')):.4%}",
        "",
        "> Phase 1 optimises fit **at critical element positions only** (not globally).",
        "> The global error is informational; it is expected to be higher than the local error.",
        "",
    ]

    if rd:
        lines += [
            "## Ranking Results",
            f"- Ranking satisfied: **{rd['ranking_satisfied']}**",
            f"- Constraints satisfied: {rd['satisfied_count']} / {rd['total_critical']}",
            f"- Best iteration: {rd['best_iteration']}",
            f"- Iterations run: {rd['iterations_run']}",
            f"- Local fit error at best state: {rd['final_local_fit_error']:.4%}",
            "",
            "### Per-Element",
            "| Element | Desired rank | Achieved rank | Damage proxy |",
            "|---------|-------------|---------------|-------------|",
        ]
        for row in rd["ranking_table"]:
            ok = "✓" if row["achieved_rank"] == row["desired_rank"] else "✗"
            lines.append(
                f"| {row['elem_id']} | {row['desired_rank']} "
                f"| {row['achieved_rank']} {ok} "
                f"| {row['damage_proxy']:.3e} |"
            )
        lines.append("")

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Top-level pipeline
# ---------------------------------------------------------------------------

def run_ranking_pipeline(
    ir_path: str,
    spc_path: str,
    output_dir: str,
    ir_max_subcase: int,
    ir_min_subcase: int,
    scale: float,
    max_active_groups: int,
    critical_elem_ids: List[int],
    critical_weight: float,
    target_ranking: List[int],
    alpha_lo: float,
    alpha_hi: float,
    standardize: bool,
    output_suffix: str,
    max_ranking_iter: int = 20,
    auto_mandatory_top_k: int = 2,
    mandatory_groups: Optional[List[int]] = None,
    ranking_margin: float = 0.0,
    max_blockers_per_iter: int = 200,
    feasibility_check: bool = True,
) -> dict:
    """Critical-ranked local-fit pipeline.

    Phase 0: Influence analysis + mandatory group selection
    Phase 1: Group LASSO/BCD on CRITICAL ELEMENT ROWS ONLY → lock active_mask_fixed
    Phase 3: SLSQP ranking optimisation (local objective + ranking constraints)
    Phase 2: Restricted OLS for mean stress → f_max, f_min
    """
    if not critical_elem_ids and not target_ranking:
        raise ValueError("--critical-elems and/or --target-ranking must be specified.")
    if mandatory_groups is None:
        mandatory_groups = []

    _log("[INFO] Parsing stress files...")
    ir_lines  = read_text(ir_path)
    spc_lines = read_text(spc_path)
    _log(f"[INFO] Read {len(ir_lines)} IR lines + {len(spc_lines)} SPC lines")

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
    n_comp = len(COMPONENT_ORDER)
    _log(f"[INFO] Elements: {n_elem} | SPC subcases: {len(spc_subcases)}")

    _log("[INFO] Building stress vectors and influence matrix...")
    sigma_max   = build_vector_with_order(ir_subcases[ir_max_subcase], ref_ids)
    sigma_min   = build_vector_with_order(ir_subcases[ir_min_subcase], ref_ids)
    delta_sigma = sigma_max - sigma_min
    sigma_mean  = (sigma_max + sigma_min) / 2.0

    H, subcase_ids = build_matrix(spc_subcases, scale, ref_ids)
    n_groups = H.shape[1] // GROUP_SIZE
    _log(f"[INFO] H shape: {H.shape} ({H.nbytes / 1024**2:.1f} MB) | force groups: {n_groups}")

    # Unified critical element set (preserve order, deduplicate)
    all_critical = list(dict.fromkeys(critical_elem_ids + target_ranking))
    elem_idx_map = {eid: i for i, eid in enumerate(ref_ids)}

    weights_init = build_element_weights(ref_ids, n_comp, all_critical, critical_weight)

    # ── Phase 0: influence-guided mandatory group selection ────────────────
    mandatory_feature_mask   = np.zeros(H.shape[1], dtype=bool)
    influence_dict: Dict[int, np.ndarray] = {}
    selected_mandatory_groups: List[int] = []

    if all_critical and (auto_mandatory_top_k > 0 or mandatory_groups):
        _log("[INFO] Phase 0: Computing group influence on critical elements...")
        influence_dict = compute_group_influence(H, all_critical, ref_ids, GROUP_SIZE, n_comp)
        if auto_mandatory_top_k > 0:
            for eid, scores in sorted(influence_dict.items()):
                top_k_idx  = np.argsort(-scores)[:auto_mandatory_top_k]
                _log(
                    f"  Influence top-{auto_mandatory_top_k} for elem {eid}: "
                    f"groups {top_k_idx.tolist()} "
                    f"(Frob: {[f'{scores[i]:.3e}' for i in top_k_idx]})"
                )
        mandatory_group_set: set = set(mandatory_groups)
        if auto_mandatory_top_k > 0:
            for scores in influence_dict.values():
                mandatory_group_set.update(
                    np.argsort(-scores)[:auto_mandatory_top_k].tolist()
                )
        mandatory_group_set = {g for g in mandatory_group_set if 0 <= g < n_groups}
        selected_mandatory_groups = sorted(mandatory_group_set)
        for g in selected_mandatory_groups:
            s = g * GROUP_SIZE
            e = min((g + 1) * GROUP_SIZE, H.shape[1])
            mandatory_feature_mask[s:e] = True
        mand_sc_labels = [
            str(subcase_ids[g * 3]) if g * 3 < len(subcase_ids) else f"g{g}"
            for g in selected_mandatory_groups
        ]
        _log(
            f"[INFO] Mandatory groups ({len(selected_mandatory_groups)}): "
            f"{selected_mandatory_groups} (lead subcases: {mand_sc_labels})"
        )
        ensure_dir(output_dir)
        write_influence_csv(
            os.path.join(output_dir, f"stress_group_influence{output_suffix}.csv"),
            influence_dict, n_groups, subcase_ids,
        )

    # ── Phase 0 feasibility check (optional) ──────────────────────────────
    if feasibility_check and all_critical and target_ranking:
        _log(f"[INFO] Phase 0 feasibility: unconstrained OLS (all {n_groups} groups)...")
        f_full, _, _, _ = np.linalg.lstsq(H, delta_sigma, rcond=None)
        err_full = float(
            np.linalg.norm(H @ f_full - delta_sigma) / (np.linalg.norm(delta_sigma) + 1e-12)
        )
        s_e_full   = compute_von_mises_range(H @ f_full, n_elem, COMPONENT_ORDER)
        ranks_full = _element_ranks(s_e_full)
        _log(f"[INFO]   Unconstrained OLS error = {err_full:.3%}")
        _log(f"         {'elem_id':>12s}  {'unconstrained_rank':>18s}  {'desired_rank':>12s}")
        for eid in all_critical:
            if eid not in elem_idx_map:
                continue
            unc_rank    = int(ranks_full[elem_idx_map[eid]])
            desired     = target_ranking.index(eid) + 1 if eid in target_ranking else None
            desired_str = str(desired) if desired else "—"
            _log(f"         {eid:>12d}  {unc_rank:>18d}  {desired_str:>12s}")
            if desired is not None and unc_rank > desired:
                _log(
                    f"[WARNING] Phase 0 feasibility: elem {eid} ranks {unc_rank} with ALL "
                    f"{n_groups} groups (desired: {desired}). "
                    f"SPC basis cannot achieve this ranking under any sparse selection."
                )

    # ── Phase 1: group selection on CRITICAL ELEMENT ROWS ONLY ────────────
    _log("[INFO] Phase 1: Group selection fitted on critical element rows only...")
    crit_row_mask = _make_crit_row_mask(ref_ids, all_critical, n_comp)
    H_crit        = H[crit_row_mask, :]          # (n_crit*6) × n_features
    delta_crit    = delta_sigma[crit_row_mask]
    weights_crit  = weights_init[crit_row_mask]

    _log(
        f"[INFO] Phase 1 system: {H_crit.shape[0]} rows "
        f"({len(all_critical)} critical elements × {n_comp} components) "
        f"× {H_crit.shape[1]} cols"
    )

    if mandatory_feature_mask.any():
        n_mand_groups = _count_active_groups(mandatory_feature_mask, GROUP_SIZE)
        free_budget = max(0, max_active_groups - n_mand_groups)
        _log(f"[INFO] Phase 1: BCD — {n_mand_groups} mandatory + ≤{free_budget} free groups")
        p1_result = solve_bcd(
            H_crit, delta_crit, weights_crit, GROUP_SIZE,
            mandatory_feature_mask, free_budget,
            alpha_lo, alpha_hi, standardize,
        )
        alpha_used = float(p1_result.get("alpha") or alpha_lo)
    else:
        _log(f"[INFO] Phase 1: Binary-search LASSO on critical rows (max {max_active_groups} groups)")
        alpha_used, p1_result = find_alpha_for_k_groups(
            H_crit, delta_crit, weights_crit, GROUP_SIZE,
            max_active_groups, alpha_lo, alpha_hi, standardize,
        )

    # Lock the active group mask — never modified after Phase 1
    active_mask_fixed = _group_active_from_features(p1_result["active_mask"], GROUP_SIZE)
    n_active_fixed    = _count_active_groups(active_mask_fixed, GROUP_SIZE)
    p1_active_indices = [
        g for g in range(n_groups)
        if active_mask_fixed[g * GROUP_SIZE : (g + 1) * GROUP_SIZE].any()
    ]
    _log(
        f"[INFO] Phase 1 locked {n_active_fixed} groups (indices: {p1_active_indices}) "
        f"| local fit error: {p1_result['relative_error']:.3%} | alpha: {alpha_used:.4e}"
    )

    # ── Phase 3: ranking-constrained SLSQP, local fit objective ───────────
    if target_ranking:
        _log("[INFO] Phase 3: SLSQP ranking optimisation (local fit objective)...")
        f_range, diag_ranking = solve_local_ranking(
            H=H,
            delta_sigma=delta_sigma,
            crit_row_mask=crit_row_mask,
            active_mask=active_mask_fixed,
            target_ranking=target_ranking,
            elem_ids=ref_ids,
            n_comp=n_comp,
            f_init=p1_result["coef"],
            margin=ranking_margin,
            max_iter=max_ranking_iter,
            max_blockers_per_iter=max_blockers_per_iter,
        )
        if diag_ranking.get("basis_insufficient_warning"):
            _log("[WARNING] Phase 3: SLSQP never improved on Phase 1 result (best_iteration=0).")
            _log("[WARNING] The active groups may not span the required stress directions for ranking.")
            _log("[WARNING] Interventions:")
            _log("[WARNING]   1. --max-active-groups N  (try 8, 10, or 15 for full basis)")
            _log("[WARNING]   2. --mandatory-groups '0,3,7'  (force-include specific groups)")
            _log("[WARNING]   3. --no-feasibility-check to skip startup OLS if slow")
    else:
        f_range      = p1_result["coef"]
        diag_ranking = {}

    active_g_sc = [
        str(subcase_ids[g * 3]) if g * 3 < len(subcase_ids) else f"g{g}"
        for g in p1_active_indices
    ]
    _log(
        f"[INFO] Active groups ({n_active_fixed}): "
        f"indices={p1_active_indices}, lead_subcases={active_g_sc}"
    )

    # ── Phase 2: mean stress (restricted OLS on active groups) ────────────
    _log("[INFO] Phase 2: Restricted OLS on mean stress...")
    f_mean = solve_phase2_mean(H, sigma_mean, weights_init, active_mask_fixed)
    err_mean = float(
        np.linalg.norm(H @ f_mean - sigma_mean) / (np.linalg.norm(sigma_mean) + 1e-12)
    )
    _log(f"[INFO] Phase 2 mean stress global error: {err_mean:.3%}")

    # ── Recover f_max, f_min ──────────────────────────────────────────────
    f_max = f_mean + f_range / 2.0
    f_min = f_mean - f_range / 2.0

    # ── Quality metrics ───────────────────────────────────────────────────
    delta_pred = H @ f_range

    # Local fit at critical elements (primary metric)
    local_fit_err = float(
        np.linalg.norm(delta_pred[crit_row_mask] - delta_sigma[crit_row_mask])
        / (np.linalg.norm(delta_sigma[crit_row_mask]) + 1e-12)
    )
    # Global Δσ error (informational — not the optimisation target)
    global_err = float(
        np.linalg.norm(delta_pred - delta_sigma) / (np.linalg.norm(delta_sigma) + 1e-12)
    )
    _log(
        f"[SUCCESS] local_fit_err (crit elems): {local_fit_err:.3%} | "
        f"global_err (info only): {global_err:.3%}"
    )

    # Damage proxy ranking (Von Mises range over ALL elements)
    s_e   = compute_von_mises_range(delta_pred, n_elem, COMPONENT_ORDER)
    ranks = _element_ranks(s_e)

    s_e_target = compute_von_mises_range(delta_sigma, n_elem, COMPONENT_ORDER)
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
    write_force_csv(
        os.path.join(output_dir, f"stress_f_range{output_suffix}.csv"),  subcase_ids, f_range)
    write_force_csv(
        os.path.join(output_dir, f"stress_f_mean{output_suffix}.csv"),   subcase_ids, f_mean)
    write_force_csv(
        os.path.join(output_dir, f"stress_f_max{output_suffix}.csv"),    subcase_ids, f_max)
    write_force_csv(
        os.path.join(output_dir, f"stress_f_min{output_suffix}.csv"),    subcase_ids, f_min)
    write_ranking_csv(
        os.path.join(output_dir, f"stress_ranking_check{output_suffix}.csv"),
        ref_ids, s_e, ranks, target_ranking,
    )

    metadata = {
        "algorithm":                  "fatigue_ranking_pipeline_v1",
        "ir_path":                    ir_path,
        "spc_path":                   spc_path,
        "ir_max_subcase":             ir_max_subcase,
        "ir_min_subcase":             ir_min_subcase,
        "subcase_ids":                subcase_ids,
        "n_elem":                     n_elem,
        "n_groups":                   n_groups,
        "active_groups":              n_active_fixed,
        "max_active_groups_requested": max_active_groups,
        "auto_mandatory_top_k":       auto_mandatory_top_k,
        "mandatory_groups_selected":  selected_mandatory_groups,
        "alpha":                      alpha_used,
        "local_fit_error_range":      local_fit_err,
        "global_error_range":         global_err,
        "relative_error_mean":        err_mean,
        "critical_elem_ids":          all_critical,
        "critical_weight":            critical_weight,
        "target_ranking":             target_ranking,
        "ranking_diagnostics":        diag_ranking or None,
        "component_order":            COMPONENT_ORDER,
        "scale":                      scale,
        "standardize":                standardize,
        "max_ranking_iter":           max_ranking_iter,
        "ranking_margin":             ranking_margin,
        "max_blockers_per_iter":      max_blockers_per_iter,
        "feasibility_check":          feasibility_check,
    }
    return {"metadata": metadata}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_int_list(raw: str) -> List[int]:
    if not raw.strip():
        return []
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Critical-Ranked Local-Fit Pipeline: guarantees critical elements rank above "
            "all others in Von Mises stress range, fitting the target only at critical "
            "element positions (not globally)."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input files
    parser.add_argument("--ir-strs",        required=True,
                        help="Target stress file (MAX and MIN subcases)")
    parser.add_argument("--spc-strs",       required=True,
                        help="Unit load stress file (one subcase per force direction)")

    # Subcase IDs
    parser.add_argument("--ir-max-subcase", type=int, default=1000001)
    parser.add_argument("--ir-min-subcase", type=int, default=1000002)

    # Force model
    parser.add_argument("--scale-h", type=float, default=1000.0,
                        help="Unit load magnitude (H columns are divided by this value)")
    parser.add_argument("--max-active-groups", type=int, default=6,
                        help="Max number of active force groups (FX/FY/FZ triplets)")

    # Critical elements
    parser.add_argument("--critical-elems",  type=str, default="",
                        help="Comma-separated element IDs to track (e.g. 10058616,10014072)")
    parser.add_argument("--critical-weight", type=float, default=10.0,
                        help="Row weight for critical elements in Phase 1 BCD OLS step")
    parser.add_argument("--target-ranking",  type=str, default="",
                        help="Ordered element IDs: position 0 → rank 1, position 1 → rank 2, …")

    # Phase 0 mandatory groups
    parser.add_argument("--auto-mandatory-top-k", type=int, default=2,
                        help="Top-K groups by influence per critical element → always active. 0=off.")
    parser.add_argument("--mandatory-groups",     type=str, default="",
                        help="Additional 0-indexed group indices to always activate, e.g. '0,3,7'")

    # Alpha bounds
    parser.add_argument("--alpha-lo", type=float, default=0.01)
    parser.add_argument("--alpha-hi", type=float, default=100000.0)

    # Phase 3 SLSQP
    parser.add_argument("--max-ranking-iter",     type=int,   default=20,
                        help="Max SLSQP iterations for ranking refinement")
    parser.add_argument("--ranking-margin",        type=float, default=0.0,
                        help="Required VM margin (Pa): VM_crit - VM_blocker >= margin")
    parser.add_argument("--max-blockers-per-iter", type=int,   default=200,
                        help="Max blocker constraints per SLSQP call (worst violators first)")

    # Feasibility check
    parser.add_argument("--feasibility-check",    dest="feasibility_check",
                        action="store_true",  default=True,
                        help="Run unconstrained OLS before Phase 1 to report achievable ranks")
    parser.add_argument("--no-feasibility-check", dest="feasibility_check",
                        action="store_false",
                        help="Skip unconstrained OLS feasibility check (faster startup)")

    # Standardization
    parser.add_argument("--standardize",    dest="standardize", action="store_true")
    parser.add_argument("--no-standardize", dest="standardize", action="store_false")
    parser.set_defaults(standardize=True)

    # Output
    parser.add_argument("--output-dir",   type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamp",    dest="timestamp", action="store_true")
    parser.add_argument("--no-timestamp", dest="timestamp", action="store_false")
    parser.set_defaults(timestamp=True)

    args = parser.parse_args()

    timestamp     = datetime.now().strftime("%Y%m%d_%H%M%S") if args.timestamp else ""
    output_suffix = f"_{timestamp}" if timestamp else ""

    reset_log_buffer()
    run_started = time.time()

    result = run_ranking_pipeline(
        ir_path=args.ir_strs,
        spc_path=args.spc_strs,
        output_dir=args.output_dir,
        ir_max_subcase=args.ir_max_subcase,
        ir_min_subcase=args.ir_min_subcase,
        scale=args.scale_h,
        max_active_groups=args.max_active_groups,
        critical_elem_ids=_parse_int_list(args.critical_elems),
        critical_weight=args.critical_weight,
        target_ranking=_parse_int_list(args.target_ranking),
        alpha_lo=args.alpha_lo,
        alpha_hi=args.alpha_hi,
        standardize=args.standardize,
        output_suffix=output_suffix,
        max_ranking_iter=args.max_ranking_iter,
        auto_mandatory_top_k=args.auto_mandatory_top_k,
        mandatory_groups=_parse_int_list(args.mandatory_groups),
        ranking_margin=args.ranking_margin,
        max_blockers_per_iter=args.max_blockers_per_iter,
        feasibility_check=args.feasibility_check,
    )

    meta = result["metadata"]
    ensure_dir(args.output_dir)

    log_payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "elapsed_seconds": round(time.time() - run_started, 3),
        "command": " ".join(sys.argv),
        "cli_args": vars(args),
        "git_commit": get_git_commit(cwd=os.path.dirname(os.path.abspath(__file__))),
        "script": os.path.basename(__file__),
        "files": {
            "ir_strs":  sha256_file(args.ir_strs),
            "spc_strs": sha256_file(args.spc_strs),
        },
        "metadata": meta,
        "console_log": get_log_buffer(),
    }
    write_log(
        os.path.join(args.output_dir, f"run_log{output_suffix}.txt"),
        log_payload,
    )
    write_ranking_report(
        os.path.join(args.output_dir, f"report{output_suffix}.md"),
        meta,
    )
    _log(f"[DONE] Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
