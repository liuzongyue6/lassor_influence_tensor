#!/usr/bin/env python3
"""V4 — Unified fatigue channel-reduction pipeline (supersedes V2 and V3).

V2 (`fatigue_lasso_pipeline.py`) and V3 (`fatigue_ranking_pipeline.py`) are not two
algorithms.  They are the same four-phase method evaluated at two different values of
one parameter: the **fit row set** R ⊆ {1..m}, i.e. which rows of H enter the
least-squares objective.

    Phase 0  influence-guided mandatory group pre-selection      — identical, scope-free
    Phase 1  min_f ||W_R (H_R f − Δσ_R)||² + α Σ_g ||f_g||₂      — R depends on fit_scope
             (mandatory groups forced active, ≤ K groups total)
             → active_mask LOCKED here, never reopened (the V2 invariant)
    Phase 3  irls  : fixed-support weighted OLS + γ^gap weight boosting
             slsqp : min ||H_{R,A} f − Δσ_R||²  s.t.  VM(H_c f) − VM(H_b f) ≥ margin
    Phase 2  restricted OLS on σ_mean — always global, independent of fit_scope

This collapses to two orthogonal switches:

    fit_scope   ∈ {global, critical}    ×    phase3_mode ∈ {irls, slsqp, none}

      (global,   irls)   ≡ V2 default
      (global,   slsqp)  ≡ V2 --irls-mode ranking
      (critical, slsqp)  ≡ V3
      (critical, irls)   — newly reachable, no historical validation

Why the two scopes are the same knob
------------------------------------
`build_element_weights` already assigns weight w = --critical-weight to the critical
elements' rows.  `fit_scope=critical` is exactly the limit w → ∞ of that weighting.
V2's residual IRLS chases the same limit by repeatedly multiplying w by γ^gap — which
is precisely the path that triggers the documented weight-saturation failure (one step
to the 1e8 cap, global fit collapses, rank gets *worse*).  Restricting the rows reaches
the identical limit in a well-conditioned way.  See docs/fatigue_lasso_method.md §6.

All heavy lifting is imported from `fatigue_lasso_pipeline` (same STRS parser, same
Phase 0, same solvers) so there is exactly one implementation of each step.  V2 and V3
remain in the repo, unchanged, as frozen reference implementations.
"""

from __future__ import annotations

import argparse
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
        COMPONENT_ORDER, GROUP_SIZE, DEFAULT_OUTPUT_DIR,
        _group_active_from_features, _count_active_groups,
        build_element_weights, _make_crit_row_mask,
        compute_von_mises_range, _element_ranks,
        find_alpha_for_k_groups, solve_bcd, _grid_search_alpha,
        run_phase0, solve_residual_irls, solve_ranking_constrained,
        solve_phase2_mean,
        write_force_csv, write_ranking_csv, write_csv_matrix, write_csv_vector,
        write_log, _log, reset_log_buffer, get_log_buffer, get_git_commit,
    )
except ImportError:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from fatigue_lasso_pipeline import (
        read_text, sha256_file, parse_subcases, build_vector_with_order,
        build_matrix, validate_elem_ids, ensure_dir,
        COMPONENT_ORDER, GROUP_SIZE, DEFAULT_OUTPUT_DIR,
        _group_active_from_features, _count_active_groups,
        build_element_weights, _make_crit_row_mask,
        compute_von_mises_range, _element_ranks,
        find_alpha_for_k_groups, solve_bcd, _grid_search_alpha,
        run_phase0, solve_residual_irls, solve_ranking_constrained,
        solve_phase2_mean,
        write_force_csv, write_ranking_csv, write_csv_matrix, write_csv_vector,
        write_log, _log, reset_log_buffer, get_log_buffer, get_git_commit,
    )

ALGORITHM_ID = "fatigue_unified_pipeline_v4"

# Per-scope defaults.  These deliberately differ so that both V2 and V3 archived runs
# reproduce bit-for-bit under V4; do not "harmonise" them without re-validating
# outputs/cradle_v2_g6_mk6_* and outputs/cradle_v3_g4_mk2.
SCOPE_DEFAULTS: Dict[str, dict] = {
    "global": {
        "max_active_groups":     None,   # → --alpha-grid search
        "max_ranking_iter":      10,
        "max_blockers_per_iter": None,   # no cap, no sorting (V2)
        "slsqp_maxiter":         200,
        "slsqp_ftol":            1e-10,
    },
    "critical": {
        "max_active_groups":     6,
        "max_ranking_iter":      20,
        "max_blockers_per_iter": 200,    # worst violators first (V3)
        "slsqp_maxiter":         500,
        "slsqp_ftol":            1e-12,
    },
}


# ---------------------------------------------------------------------------
# Phase 1 — group selection on the selected fit rows
# ---------------------------------------------------------------------------

def run_phase1(
    H: np.ndarray,
    delta_sigma: np.ndarray,
    weights_init: np.ndarray,
    mandatory_feature_mask: Optional[np.ndarray],
    max_active_groups: Optional[int],
    alpha_grid: List[float],
    alpha_lo: float,
    alpha_hi: float,
    standardize: bool,
    fit_row_mask: Optional[np.ndarray] = None,
) -> Tuple[float, dict]:
    """Phase 1 — pick the active force groups, then lock them.

    Runs exactly once.  ``fit_row_mask=None`` fits all m rows (V2 global fit); a
    boolean mask restricts the fit to those rows (V3 critical-row fit).  Solver
    selection is unchanged from V2: BCD when mandatory groups exist, otherwise a
    binary search on α for a group budget, otherwise a grid search.

    Returns ``(alpha_used, phase1_result)``.
    """
    if fit_row_mask is None:
        H_fit, d_fit, w_fit = H, delta_sigma, weights_init
    else:
        H_fit = H[fit_row_mask, :]
        d_fit = delta_sigma[fit_row_mask]
        w_fit = weights_init[fit_row_mask]
        _log(
            f"[INFO] Phase 1 system: {H_fit.shape[0]} rows "
            f"(critical elements only) × {H_fit.shape[1]} cols"
        )

    if mandatory_feature_mask is not None and mandatory_feature_mask.any():
        n_mand = _count_active_groups(mandatory_feature_mask, GROUP_SIZE)
        free_budget = (
            max(0, max_active_groups - n_mand)
            if max_active_groups is not None
            else (H.shape[1] // GROUP_SIZE)
        )
        _log(f"[INFO] Phase 1: BCD — {n_mand} mandatory + ≤{free_budget} free groups")
        result = solve_bcd(
            H_fit, d_fit, w_fit, GROUP_SIZE,
            mandatory_feature_mask, free_budget,
            alpha_lo, alpha_hi, standardize,
        )
        return float(result.get("alpha") or alpha_lo), result

    if max_active_groups is not None:
        _log(f"[INFO] Phase 1: Binary-search LASSO (max {max_active_groups} groups)")
        return find_alpha_for_k_groups(
            H_fit, d_fit, w_fit, GROUP_SIZE,
            max_active_groups, alpha_lo, alpha_hi, standardize,
        )

    _log("[INFO] Phase 1: Grid search over --alpha-grid (no group budget)")
    return _grid_search_alpha(H_fit, d_fit, w_fit, GROUP_SIZE, alpha_grid, standardize)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def write_unified_report(path: str, meta: dict) -> None:
    """Markdown report covering both fit scopes.

    Always prints the local (critical-rows) and global Δσ errors side by side and
    marks which one the run actually optimised, so a V4 report is directly
    comparable against both a V2 and a V3 report.
    """
    rd = meta.get("ranking_diagnostics")
    scope = meta["fit_scope"]
    p3    = meta["phase3_mode"]
    local_is_target = scope == "critical"

    lines = [
        "# Unified Fatigue Pipeline Report (V4)",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
        "## Mode",
        f"- Fit scope: **{scope}** "
        + ("(Phase 1 and Phase 3 fit the critical elements' rows only)"
           if local_is_target else "(Phase 1 and Phase 3 fit all element rows)"),
        f"- Phase 3 mode: **{p3}**",
        f"- Equivalent legacy invocation: `{meta.get('legacy_equivalent', 'n/a')}`",
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
    ]

    local_err = meta.get("local_fit_error_range")
    local_txt = "n/a (no critical elements)" if local_err is None else f"{local_err:.4%}"
    mark_local  = " ← optimised" if local_is_target else ""
    mark_global = "" if local_is_target else " ← optimised"
    lines += [
        f"- Δσ error at critical elements (local): {local_txt}{mark_local}",
        f"- Δσ error over all elements (global):   {meta['relative_error_range']:.4%}{mark_global}",
        f"- Relative error σ_max:                  {meta['relative_error_max']:.4%}",
        f"- Relative error σ_min:                  {meta['relative_error_min']:.4%}",
        f"- Relative error σ_mean:                 {meta.get('relative_error_mean', float('nan')):.4%}",
        "",
    ]
    if local_is_target:
        lines += [
            "> Phase 1/3 optimise the fit **at the critical elements only**. The global",
            "> error is informational and is expected to exceed a comparable V2 run.",
            "",
        ]

    if rd:
        lines += [
            "## Ranking Results",
            f"- Ranking satisfied: **{rd['ranking_satisfied']}**",
            f"- Constraints satisfied: {rd['satisfied_count']} / {rd['total_critical']}",
            f"- Best iteration: {rd['best_iteration']}",
            f"- Iterations run: {rd['iterations_run']}",
            f"- Global Δσ error at best state: {rd['final_relative_error']:.4%}",
        ]
        if rd.get("final_local_fit_error") is not None:
            lines.append(
                f"- Local Δσ error at best state:  {rd['final_local_fit_error']:.4%}"
            )
        lines += [
            "",
            "### Per-Element",
            "| Element | Desired rank | Achieved rank | Damage proxy |",
            "|---------|-------------|---------------|-------------|",
        ]
        for row in rd["ranking_table"]:
            mark = "✓" if row["achieved_rank"] == row["desired_rank"] else "✗"
            lines.append(
                f"| {row['elem_id']} | {row['desired_rank']} "
                f"| {row['achieved_rank']} {mark} "
                f"| {row['damage_proxy']:.3e} |"
            )
        lines.append("")
        if rd.get("basis_insufficient_warning"):
            lines += [
                "> **Basis insufficiency**: Phase 3 never improved on Phase 1 "
                "(`best_iteration = 0`).",
                "> The active groups cannot span the stress directions the ranking needs. "
                "Try more",
                "> groups (`--max-active-groups`), explicit `--mandatory-groups`, or "
                "`--fit-scope critical`.",
                "",
            ]

    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------

def run_unified_pipeline(
    ir_path: str,
    spc_path: str,
    output_dir: str,
    ir_max_subcase: int,
    ir_min_subcase: int,
    scale: float,
    fit_scope: str,
    phase3_mode: str,
    max_active_groups: Optional[int],
    critical_elem_ids: List[int],
    critical_weight: float,
    target_ranking: List[int],
    alpha_grid: List[float],
    alpha_lo: float,
    alpha_hi: float,
    standardize: bool,
    output_suffix: str,
    max_ranking_iter: int,
    max_blockers_per_iter: Optional[int],
    slsqp_maxiter: int,
    slsqp_ftol: float,
    gamma: float = 2.0,
    auto_mandatory_top_k: int = 2,
    mandatory_groups: Optional[List[int]] = None,
    ranking_margin: float = 0.0,
    feasibility_check: bool = True,
    dump_matrix: bool = True,
) -> dict:
    """Phase 0 → 1 → 3 → 2 → outputs, dispatched on (fit_scope, phase3_mode)."""
    if mandatory_groups is None:
        mandatory_groups = []
    if fit_scope not in SCOPE_DEFAULTS:
        raise ValueError(f"fit_scope must be one of {sorted(SCOPE_DEFAULTS)}, got {fit_scope!r}")
    if phase3_mode not in ("irls", "slsqp", "none"):
        raise ValueError(f"phase3_mode must be irls/slsqp/none, got {phase3_mode!r}")

    all_critical = list(dict.fromkeys(critical_elem_ids + target_ranking))
    if fit_scope == "critical" and not all_critical:
        raise SystemExit(
            "--fit-scope critical requires --critical-elems and/or --target-ranking: "
            "there are no critical rows to fit."
        )

    # ── Parse inputs ──────────────────────────────────────────────────────
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
    n_comp = len(COMPONENT_ORDER)
    _log(f"[INFO] Elements: {n_elem} | SPC subcases: {len(spc_subcases)} | H scale: {scale}")

    # ── Build vectors and H ───────────────────────────────────────────────
    _log("[INFO] Building stress vectors and influence matrix...")
    sigma_max   = build_vector_with_order(ir_subcases[ir_max_subcase], ref_ids)
    sigma_min   = build_vector_with_order(ir_subcases[ir_min_subcase], ref_ids)
    delta_sigma = sigma_max - sigma_min
    sigma_mean  = (sigma_max + sigma_min) / 2.0

    H, subcase_ids = build_matrix(spc_subcases, scale, ref_ids)
    n_groups = H.shape[1] // GROUP_SIZE
    _log(
        f"[INFO] H shape: {H.shape} ({H.nbytes / (1024 ** 2):.1f} MB) "
        f"| force groups: {n_groups}"
    )
    _log(f"[INFO] Mode: fit_scope={fit_scope} | phase3_mode={phase3_mode}")

    weights_init = build_element_weights(ref_ids, n_comp, all_critical, critical_weight)
    elem_idx_map = {eid: i for i, eid in enumerate(ref_ids)}

    # Row masks.  crit_row_mask always exists when there are critical elements — in
    # global scope it is diagnostics-only, in critical scope it is the fit set too.
    crit_row_mask = (
        _make_crit_row_mask(ref_ids, all_critical, n_comp) if all_critical else None
    )
    fit_row_mask = crit_row_mask if fit_scope == "critical" else None

    # ── Phase 0 ───────────────────────────────────────────────────────────
    phase0 = run_phase0(
        H=H,
        delta_sigma=delta_sigma,
        elem_ids=ref_ids,
        all_critical=all_critical,
        target_ranking=target_ranking,
        subcase_ids=subcase_ids,
        n_groups=n_groups,
        auto_mandatory_top_k=auto_mandatory_top_k,
        mandatory_groups=mandatory_groups,
        output_dir=output_dir,
        output_suffix=output_suffix,
        feasibility_check=feasibility_check,
    )
    mandatory_feature_mask    = phase0["mandatory_feature_mask"]
    selected_mandatory_groups = phase0["mandatory_groups_selected"]

    # ── Phase 1 (group selection, then LOCK) ──────────────────────────────
    _log(f"[INFO] Phase 1: group selection (fit_scope={fit_scope}, runs once)...")
    alpha_used, p1_result = run_phase1(
        H=H,
        delta_sigma=delta_sigma,
        weights_init=weights_init,
        mandatory_feature_mask=(
            mandatory_feature_mask if mandatory_feature_mask.any() else None
        ),
        max_active_groups=max_active_groups,
        alpha_grid=alpha_grid,
        alpha_lo=alpha_lo,
        alpha_hi=alpha_hi,
        standardize=standardize,
        fit_row_mask=fit_row_mask,
    )

    active_mask_fixed = _group_active_from_features(p1_result["active_mask"], GROUP_SIZE)
    n_active_fixed    = _count_active_groups(active_mask_fixed, GROUP_SIZE)
    active_g_indices  = [
        g for g in range(n_groups)
        if active_mask_fixed[g * GROUP_SIZE : (g + 1) * GROUP_SIZE].any()
    ]
    err_label = "local fit error" if fit_scope == "critical" else "range error"
    _log(
        f"[INFO] Phase 1 locked {n_active_fixed} groups (indices: {active_g_indices}) "
        f"| initial {err_label}: {p1_result['relative_error']:.3%} | alpha: {alpha_used:.4e}"
    )

    # ── Phase 3 ───────────────────────────────────────────────────────────
    diag_ranking: dict = {}
    if not target_ranking or phase3_mode == "none":
        if target_ranking and phase3_mode == "none":
            _log("[INFO] Phase 3 skipped (--phase3-mode none): using the Phase 1 solution.")
        f_range = p1_result["coef"]
    elif phase3_mode == "slsqp":
        _log("[INFO] Phase 3 (slsqp): ranking-constrained optimisation on fixed support...")
        f_range, diag_ranking = solve_ranking_constrained(
            H, delta_sigma, active_mask_fixed, target_ranking,
            ref_ids, n_comp, f_init=p1_result["coef"],
            margin=ranking_margin, max_iter=max_ranking_iter,
            fit_row_mask=fit_row_mask,
            local_row_mask=crit_row_mask,
            max_blockers_per_iter=max_blockers_per_iter,
            slsqp_maxiter=slsqp_maxiter,
            slsqp_ftol=slsqp_ftol,
            prune_satisfied_criticals=(fit_scope == "critical"),
        )
    else:  # irls
        _log("[INFO] Phase 3 (irls): fixed-support weight boosting...")
        f_range, active_mask_fixed, alpha_used, diag_ranking = solve_residual_irls(
            H, delta_sigma, weights_init, active_mask_fixed, ref_ids, n_comp,
            target_ranking, alpha_used, gamma=gamma, max_iter=max_ranking_iter,
            fit_row_mask=fit_row_mask,
            local_row_mask=crit_row_mask,
        )

    active_groups = _count_active_groups(active_mask_fixed, GROUP_SIZE)
    active_g_sc   = [
        str(subcase_ids[g * 3]) if g * 3 < len(subcase_ids) else f"g{g}"
        for g in active_g_indices
    ]
    _log(
        f"[INFO] Active groups: {active_groups}/{n_groups} "
        f"(indices: {active_g_indices}, lead subcases: {active_g_sc}) "
        f"| nonzero features: {int(active_mask_fixed.sum())}"
    )

    # ── Phase 2 (always global) ───────────────────────────────────────────
    _log("[INFO] Phase 2: Restricted OLS on mean stress (sigma_mean)...")
    f_mean = solve_phase2_mean(H, sigma_mean, weights_init, active_mask_fixed)
    err_mean = float(
        np.linalg.norm(H @ f_mean - sigma_mean) / (np.linalg.norm(sigma_mean) + 1e-12)
    )
    _log(f"[INFO] Phase 2 mean stress error: {err_mean:.3%}")

    # ── Recover f_max, f_min ──────────────────────────────────────────────
    f_max = f_mean + f_range / 2.0
    f_min = f_mean - f_range / 2.0

    # ── Quality metrics (both scopes always reported) ─────────────────────
    delta_pred = H @ f_range
    err_range  = float(
        np.linalg.norm(delta_pred - delta_sigma) / (np.linalg.norm(delta_sigma) + 1e-12)
    )
    err_max = float(np.linalg.norm(H @ f_max - sigma_max) / (np.linalg.norm(sigma_max) + 1e-12))
    err_min = float(np.linalg.norm(H @ f_min - sigma_min) / (np.linalg.norm(sigma_min) + 1e-12))
    err_local: Optional[float] = None
    if crit_row_mask is not None:
        err_local = float(
            np.linalg.norm(delta_pred[crit_row_mask] - delta_sigma[crit_row_mask])
            / (np.linalg.norm(delta_sigma[crit_row_mask]) + 1e-12)
        )
    local_txt = "n/a" if err_local is None else f"{err_local:.3%}"
    _log(
        f"[SUCCESS] local_err: {local_txt} | range_err(global): {err_range:.3%} | "
        f"max_err: {err_max:.3%} | min_err: {err_min:.3%}"
    )

    # ── Damage proxy ranking ──────────────────────────────────────────────
    s_e   = compute_von_mises_range(delta_pred, n_elem, COMPONENT_ORDER)
    ranks = _element_ranks(s_e)
    if all_critical:
        s_e_target = compute_von_mises_range(delta_sigma, n_elem, COMPONENT_ORDER)
        _log("[INFO] Critical element Von Mises range stress (target vs predicted):")
        _log(
            f"       {'elem_id':>12s}  {'target_VM':>12s}  {'pred_VM':>12s}  "
            f"{'ratio':>8s}  {'pred_rank':>9s}"
        )
        for eid in all_critical:
            if eid not in elem_idx_map:
                continue
            cidx    = elem_idx_map[eid]
            tgt_vm  = float(s_e_target[cidx])
            pred_vm = float(s_e[cidx])
            _log(
                f"       {eid:>12d}  {tgt_vm:>12.3e}  {pred_vm:>12.3e}  "
                f"{pred_vm / (tgt_vm + 1e-12):>8.3f}  {int(ranks[cidx]):>9d}"
            )

    # ── Write outputs ─────────────────────────────────────────────────────
    ensure_dir(output_dir)
    if dump_matrix:
        write_csv_matrix(os.path.join(output_dir, f"stress_H{output_suffix}.csv"), H, subcase_ids)
        write_csv_vector(
            os.path.join(output_dir, f"stress_delta_target{output_suffix}.csv"), delta_sigma
        )
    write_force_csv(os.path.join(output_dir, f"stress_f_range{output_suffix}.csv"), subcase_ids, f_range)
    write_force_csv(os.path.join(output_dir, f"stress_f_mean{output_suffix}.csv"),  subcase_ids, f_mean)
    write_force_csv(os.path.join(output_dir, f"stress_f_max{output_suffix}.csv"),   subcase_ids, f_max)
    write_force_csv(os.path.join(output_dir, f"stress_f_min{output_suffix}.csv"),   subcase_ids, f_min)
    write_ranking_csv(
        os.path.join(output_dir, f"stress_ranking_check{output_suffix}.csv"),
        ref_ids, s_e, ranks, target_ranking,
    )

    # Metadata carries the UNION of the V2 and V3 key sets so downstream readers
    # written against either version keep working.
    metadata = {
        "algorithm": ALGORITHM_ID,
        "fit_scope": fit_scope,
        "phase3_mode": phase3_mode,
        "legacy_equivalent": _legacy_equivalent(fit_scope, phase3_mode),
        "resolved_defaults": {
            "max_active_groups":     max_active_groups,
            "max_ranking_iter":      max_ranking_iter,
            "max_blockers_per_iter": max_blockers_per_iter,
            "slsqp_maxiter":         slsqp_maxiter,
            "slsqp_ftol":            slsqp_ftol,
        },
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
        # V2 key names
        "relative_error_range": err_range,
        "relative_error_max": err_max,
        "relative_error_min": err_min,
        "relative_error_mean": err_mean,
        # V3 key names (same numbers, different names)
        "local_fit_error_range": err_local,
        "global_error_range": err_range,
        "critical_elem_ids": all_critical,
        "critical_weight": critical_weight,
        "target_ranking": target_ranking,
        "ranking_diagnostics": diag_ranking or None,
        "component_order": COMPONENT_ORDER,
        "scale": scale,
        "standardize": standardize,
        "gamma": gamma,
        "max_ranking_iter": max_ranking_iter,
        "max_blockers_per_iter": max_blockers_per_iter,
        "irls_mode": {"irls": "residual", "slsqp": "ranking", "none": "none"}[phase3_mode],
        "ranking_margin": ranking_margin,
        "feasibility_check": feasibility_check,
    }
    return {"metadata": metadata}


def _legacy_equivalent(fit_scope: str, phase3_mode: str) -> str:
    """Human-readable pointer to the frozen V2/V3 script this mode reproduces."""
    return {
        ("global",   "irls"):  "fatigue_lasso_pipeline.py --irls-mode residual (V2 default)",
        ("global",   "slsqp"): "fatigue_lasso_pipeline.py --irls-mode ranking (V2)",
        ("critical", "slsqp"): "fatigue_ranking_pipeline.py (V3)",
        ("critical", "irls"):  "none — new combination, no historical validation",
        ("global",   "none"):  "fatigue_lasso_pipeline.py without --target-ranking (V2)",
        ("critical", "none"):  "none — Phase 1 only, critical-row fit",
    }.get((fit_scope, phase3_mode), "n/a")


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
    p = argparse.ArgumentParser(
        description=(
            "V4 unified fatigue channel-reduction pipeline. "
            "--fit-scope selects which rows the least-squares objective uses; "
            "--phase3-mode selects the ranking refinement algorithm. "
            "Together they reproduce both V2 and V3."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=(
            "Mode map:  (global,irls)=V2 default  (global,slsqp)=V2 --irls-mode ranking  "
            "(critical,slsqp)=V3  (critical,irls)=new, unvalidated"
        ),
    )
    p.add_argument("--ir-strs", required=True,
                   help="Target stress file containing the MAX and MIN subcases")
    p.add_argument("--spc-strs", required=True,
                   help="SPC unit-load stress file (one subcase per force direction)")
    p.add_argument("--ir-max-subcase", type=int, default=1000001, help="Subcase ID for sigma_max")
    p.add_argument("--ir-min-subcase", type=int, default=1000002, help="Subcase ID for sigma_min")
    p.add_argument("--scale-h", type=float, default=1000.0,
                   help="Scale factor applied to influence matrix columns (unit load magnitude)")

    # ── The two mode switches ─────────────────────────────────────────────
    p.add_argument("--fit-scope", choices=["global", "critical"], default="global",
                   help="Rows entering the Phase 1/3 least-squares objective: all elements "
                        "(global) or the critical elements only (critical)")
    p.add_argument("--phase3-mode", choices=["irls", "slsqp", "none"], default=None,
                   help="Phase 3 algorithm: irls (weight boosting), slsqp "
                        "(ranking constraints), none (stop after Phase 1). "
                        "[default: irls]")
    p.add_argument("--irls-mode", choices=["residual", "ranking"], default=None,
                   help="DEPRECATED alias for --phase3-mode (residual->irls, ranking->slsqp)")

    p.add_argument("--critical-elems", type=str, default="",
                   help="Comma-separated element IDs to up-weight, e.g. 10058616,10014072")
    p.add_argument("--critical-weight", type=float, default=10.0,
                   help="Weight multiplier applied to critical element residuals")
    p.add_argument("--target-ranking", type=str, default="",
                   help="Ordered element IDs: position 0 -> rank 1, position 1 -> rank 2, ...")

    p.add_argument("--auto-mandatory-top-k", type=int, default=2,
                   help="Phase 0: top-K groups by influence per critical element to force "
                        "active. 0 disables Phase 0")
    p.add_argument("--mandatory-groups", type=str, default="",
                   help="Phase 0: additional 0-indexed group indices to always activate")

    p.add_argument("--alpha-grid", type=str, default="0.1,1,10,100,1000",
                   help="Alpha candidates for grid search (only when --max-active-groups unset)")
    p.add_argument("--alpha-lo", type=float, default=0.01, help="Binary-search lower bound for alpha")
    p.add_argument("--alpha-hi", type=float, default=100000.0, help="Binary-search upper bound for alpha")
    p.add_argument("--gamma", type=float, default=2.0,
                   help="(irls mode only) weight amplification factor per rank-gap unit")
    p.add_argument("--ranking-margin", type=float, default=0.0,
                   help="(slsqp mode only) required margin: VM_crit - VM_competitor >= margin")

    # ── Scope-dependent defaults (None sentinel, resolved after parsing) ───
    p.add_argument("--max-active-groups", type=int, default=None,
                   help="Max active force groups. [default: unset (grid search) for "
                        "--fit-scope global, 6 for critical]")
    p.add_argument("--max-ranking-iter", type=int, default=None,
                   help="Max Phase 3 iterations. [default: 10 for global, 20 for critical]")
    p.add_argument("--max-blockers-per-iter", type=int, default=None,
                   help="Cap on blocker constraints per SLSQP call, worst violators first. "
                        "[default: uncapped for global, 200 for critical]")
    p.add_argument("--slsqp-maxiter", type=int, default=None,
                   help="SLSQP maxiter. [default: 200 for global, 500 for critical]")
    p.add_argument("--slsqp-ftol", type=float, default=None,
                   help="SLSQP ftol. [default: 1e-10 for global, 1e-12 for critical]")

    p.add_argument("--feasibility-check", dest="feasibility_check", action="store_true",
                   default=True, help="Run unconstrained OLS before Phase 1 to log the "
                                      "achievable rank upper bound")
    p.add_argument("--no-feasibility-check", dest="feasibility_check", action="store_false",
                   help="Skip the Phase 0 feasibility OLS")
    p.add_argument("--standardize", dest="standardize", action="store_true", default=True,
                   help="Standardize columns before Group LASSO")
    p.add_argument("--no-standardize", dest="standardize", action="store_false",
                   help="Do not standardize columns")
    p.add_argument("--dump-matrix", dest="dump_matrix", action="store_true", default=True,
                   help="Write stress_H_*.csv and stress_delta_target_*.csv")
    p.add_argument("--no-dump-matrix", dest="dump_matrix", action="store_false",
                   help="Skip the large H / target dumps")
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR, help="Output directory")
    p.add_argument("--timestamp", dest="timestamp", action="store_true", default=True,
                   help="Suffix output filenames with a timestamp")
    p.add_argument("--no-timestamp", dest="timestamp", action="store_false",
                   help="Do not suffix output filenames")

    args = p.parse_args()

    # ── Resolve --phase3-mode / deprecated --irls-mode ────────────────────
    if args.irls_mode is not None:
        mapped = {"residual": "irls", "ranking": "slsqp"}[args.irls_mode]
        if args.phase3_mode is not None and args.phase3_mode != mapped:
            p.error(
                f"--irls-mode {args.irls_mode} maps to --phase3-mode {mapped}, "
                f"which conflicts with --phase3-mode {args.phase3_mode}"
            )
        args.phase3_mode = mapped
    if args.phase3_mode is None:
        args.phase3_mode = "irls"

    # ── Resolve scope-dependent defaults ──────────────────────────────────
    defaults = SCOPE_DEFAULTS[args.fit_scope]
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    critical_elems = _parse_int_list(args.critical_elems)
    target_ranking = _parse_int_list(args.target_ranking)

    if args.fit_scope == "critical" and not (critical_elems or target_ranking):
        p.error("--fit-scope critical requires --critical-elems and/or --target-ranking")

    output_suffix = f"_{datetime.now().strftime('%Y%m%d_%H%M%S')}" if args.timestamp else ""

    reset_log_buffer()
    run_started = time.time()

    if args.fit_scope == "critical" and args.phase3_mode == "irls":
        _log("[WARNING] fit_scope=critical + phase3_mode=irls is a new combination with no")
        _log("[WARNING] historical validation. IRLS weight boosting on top of an already")
        _log("[WARNING] critical-row-restricted fit is redundant and may saturate weights.")
        _log("[WARNING] Prefer --phase3-mode slsqp with --fit-scope critical.")

    result = run_unified_pipeline(
        ir_path=args.ir_strs,
        spc_path=args.spc_strs,
        output_dir=args.output_dir,
        ir_max_subcase=args.ir_max_subcase,
        ir_min_subcase=args.ir_min_subcase,
        scale=args.scale_h,
        fit_scope=args.fit_scope,
        phase3_mode=args.phase3_mode,
        max_active_groups=args.max_active_groups,
        critical_elem_ids=critical_elems,
        critical_weight=args.critical_weight,
        target_ranking=target_ranking,
        alpha_grid=_parse_float_list(args.alpha_grid),
        alpha_lo=args.alpha_lo,
        alpha_hi=args.alpha_hi,
        standardize=args.standardize,
        output_suffix=output_suffix,
        max_ranking_iter=args.max_ranking_iter,
        max_blockers_per_iter=args.max_blockers_per_iter,
        slsqp_maxiter=args.slsqp_maxiter,
        slsqp_ftol=args.slsqp_ftol,
        gamma=args.gamma,
        auto_mandatory_top_k=args.auto_mandatory_top_k,
        mandatory_groups=_parse_int_list(args.mandatory_groups),
        ranking_margin=args.ranking_margin,
        feasibility_check=args.feasibility_check,
        dump_matrix=args.dump_matrix,
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
    write_log(os.path.join(args.output_dir, f"run_log{output_suffix}.txt"), log_payload)
    write_unified_report(os.path.join(args.output_dir, f"report{output_suffix}.md"), meta)
    _log(f"[DONE] Outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()
