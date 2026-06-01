import argparse
import csv
import os
import re
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde, pearsonr

COMPONENT_ORDER = ["VON", "XX1", "YY1", "XY1", "XX2", "YY2", "XY2"]
DEFAULT_OUTPUT_DIR = "outputs"

PAPER_RC = {
    "font.family": "serif",
    "font.size": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 1.0,
    "savefig.dpi": 300,
}


@dataclass
class SubcaseData:
    subcase_id: int
    elem_ids: List[int]
    values: Dict[int, Dict[str, float]]


def read_text(path: str) -> List[str]:
    with open(path, "r", encoding="utf-8", errors="ignore") as handle:
        return handle.readlines()


def parse_subcases(lines: List[str]) -> Dict[int, SubcaseData]:
    subcase_pattern = re.compile(r"\$SUBCASE\s+(\d+)")
    data_row_pattern = re.compile(r"^\s*\d+\s+[-+0-9.Ee]+\s+[-+0-9.Ee]+")

    current_subcase = None
    in_block = False
    subcases: Dict[int, SubcaseData] = {}

    for line in lines:
        subcase_match = subcase_pattern.search(line)
        if subcase_match:
            subcase_id = int(subcase_match.group(1))
            current_subcase = subcase_id
            if subcase_id not in subcases:
                subcases[subcase_id] = SubcaseData(subcase_id, [], {})
            in_block = False
            continue

        if "$ELEMENT STRESS(PLATE)" in line:
            in_block = True
            continue

        if in_block and line.strip().startswith("--------"):
            continue

        if in_block and data_row_pattern.match(line):
            parts = line.split()
            elem_id = int(parts[0])
            values = {
                "VON": float(parts[1]),
                "XX1": float(parts[2]),
                "XX2": float(parts[3]),
                "YY1": float(parts[4]),
                "YY2": float(parts[5]),
                "XY1": float(parts[6]),
                "XY2": float(parts[7]),
            }
            if current_subcase is None:
                raise ValueError("Encountered data row before any $SUBCASE header.")
            if elem_id not in subcases[current_subcase].values:
                subcases[current_subcase].elem_ids.append(elem_id)
            subcases[current_subcase].values[elem_id] = values
            continue

        if in_block and line.strip() == "":
            in_block = False

    return subcases


def build_vector_with_order(subcase: SubcaseData, elem_ids: List[int]) -> np.ndarray:
    if not elem_ids:
        raise ValueError("Element list is empty.")
    vector = []
    for elem_id in elem_ids:
        if elem_id not in subcase.values:
            raise ValueError(f"Subcase {subcase.subcase_id} is missing element {elem_id}.")
        elem_values = subcase.values[elem_id]
        for comp in COMPONENT_ORDER:
            vector.append(elem_values[comp])
    return np.asarray(vector, dtype=float)


def validate_elem_ids(reference_ids: List[int], candidate_ids: List[int]) -> None:
    ref_set = set(reference_ids)
    cand_set = set(candidate_ids)
    if ref_set == cand_set:
        return
    missing = sorted(ref_set - cand_set)
    extra = sorted(cand_set - ref_set)
    message = [
        "Element mismatch between target and result.",
        f"Missing: {missing[:10]}" + (" ..." if len(missing) > 10 else ""),
        f"Extra: {extra[:10]}" + (" ..." if len(extra) > 10 else ""),
    ]
    raise ValueError(" ".join(message))


# ── Metrics ───────────────────────────────────────────────────────────────────

def _compute_metrics(target: np.ndarray, result: np.ndarray) -> dict:
    diff = result - target
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff ** 2)))
    max_ae = float(np.max(np.abs(diff)))
    ss_res = float(np.sum(diff ** 2))
    ss_tot = float(np.sum((target - np.mean(target)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    pr, _ = pearsonr(target, result)
    mask = np.abs(target) >= 1e-6
    mape = float(np.mean(np.abs(diff[mask] / target[mask])) * 100) if mask.any() else float("nan")
    return {"mae": mae, "rmse": rmse, "max_ae": max_ae, "r2": r2, "pearson": float(pr), "mape": mape}


def compute_metrics(target: np.ndarray, result: np.ndarray) -> tuple:
    n = len(COMPONENT_ORDER)
    per_comp = []
    for i, comp in enumerate(COMPONENT_ORDER):
        m = _compute_metrics(target[i::n], result[i::n])
        m["component"] = comp
        per_comp.append(m)
    global_m = _compute_metrics(target, result)
    return global_m, per_comp


# ── Text / CSV outputs ─────────────────────────────────────────────────────────

def write_report(
    path: str,
    target_path: str,
    result_path: str,
    target_subcase: int,
    result_subcase: int,
    n_elements: int,
    global_m: dict,
    per_comp: list,
) -> None:
    sep = "=" * 62
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"{sep}\n  STRESS TENSOR COMPARISON REPORT\n")
        f.write(f"  Generated : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n{sep}\n\n")
        f.write(f"TARGET  : {target_path}\n          subcase {target_subcase}\n")
        f.write(f"RESULT  : {result_path}\n          subcase {result_subcase}\n\n")
        f.write(f"Elements compared : {n_elements:,}\n")
        f.write(f"Components        : {', '.join(COMPONENT_ORDER)}\n\n")
        f.write("GLOBAL METRICS  (all components concatenated)\n")
        f.write("-" * 46 + "\n")
        f.write(f"  MAE       : {global_m['mae']:.6g} MPa\n")
        f.write(f"  RMSE      : {global_m['rmse']:.6g} MPa\n")
        f.write(f"  Max AE    : {global_m['max_ae']:.6g} MPa\n")
        f.write(f"  R²        : {global_m['r2']:.6f}\n")
        f.write(f"  Pearson r : {global_m['pearson']:.6f}\n")
        f.write(f"  MAPE      : {global_m['mape']:.4f} %\n\n")
        f.write("PER-COMPONENT METRICS\n")
        f.write("-" * 62 + "\n")
        hdr = f"{'Comp':>5}  {'MAE':>10}  {'RMSE':>10}  {'MaxAE':>10}  {'R²':>8}  {'Pearson':>8}  {'MAPE%':>8}\n"
        f.write(hdr)
        f.write("-" * 62 + "\n")
        for m in per_comp:
            f.write(
                f"{m['component']:>5}  "
                f"{m['mae']:>10.4f}  "
                f"{m['rmse']:>10.4f}  "
                f"{m['max_ae']:>10.4f}  "
                f"{m['r2']:>8.4f}  "
                f"{m['pearson']:>8.4f}  "
                f"{m['mape']:>8.3f}\n"
            )
        f.write(f"\n{sep}\n")


def write_metrics_csv(path: str, per_comp: list) -> None:
    fields = ["component", "mae", "rmse", "max_ae", "r2", "pearson", "mape"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for m in per_comp:
            writer.writerow({
                k: f"{m[k]:.6g}" if isinstance(m[k], float) else m[k]
                for k in fields
            })


# ── Figures ───────────────────────────────────────────────────────────────────

def _paper_style() -> None:
    plt.rcParams.update(PAPER_RC)


def _density_scatter(ax, x: np.ndarray, y: np.ndarray, cmap: str = "viridis", s: int = 3) -> None:
    """Scatter plot colored by local point density (KDE-based)."""
    xy = np.vstack([x, y])
    if len(x) > 4000:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(x), 4000, replace=False)
        kde = gaussian_kde(xy[:, idx])
    else:
        kde = gaussian_kde(xy)
    density = kde(xy)
    order = np.argsort(density)
    ax.scatter(x[order], y[order], c=density[order], cmap=cmap, s=s, alpha=0.7, linewidths=0)


# Fig 1a: Von Mises scatter (single panel)
def plot_von_scatter(
    path: str,
    target: np.ndarray,
    result: np.ndarray,
    per_comp: list,
    n_elements: int,
) -> None:
    _paper_style()
    n = len(COMPONENT_ORDER)
    fig, ax = plt.subplots(figsize=(5, 4))
    t, r = target[0::n], result[0::n]
    _density_scatter(ax, t, r)
    lo = min(t.min(), r.min())
    hi = max(t.max(), r.max())
    pad = (hi - lo) * 0.05 or 0.5
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, zorder=5)
    ax.set_xlim(lo - pad, hi + pad)
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Target (MPa)", fontsize=8)
    ax.set_ylabel("Result (MPa)", fontsize=8)
    m = per_comp[0]
    ax.text(
        0.04, 0.96,
        f"R² = {m['r2']:.4f}\nRMSE = {m['rmse']:.3f} MPa\nN = {n_elements:,}",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
    )
    fig.suptitle("Von Mises — Target vs Result", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# Fig 1b: tensor component scatter grid (2×3, top face / bottom face rows)
def plot_tensor_scatter_grid(
    path: str,
    target: np.ndarray,
    result: np.ndarray,
    per_comp: list,
    global_m: dict,
    n_elements: int,
) -> None:
    _paper_style()
    n = len(COMPONENT_ORDER)
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    flat = axes.flatten()

    for k, i in enumerate(range(1, 7)):
        ax = flat[k]
        comp = COMPONENT_ORDER[i]
        t, r = target[i::n], result[i::n]
        _density_scatter(ax, t, r)
        lo = min(t.min(), r.min())
        hi = max(t.max(), r.max())
        pad = (hi - lo) * 0.05 or 0.5
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], "k--", lw=0.8, zorder=5)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Target (MPa)", fontsize=8)
        ax.set_ylabel("Result (MPa)", fontsize=8)
        ax.set_title(comp, fontsize=9, fontweight="bold")
        m = per_comp[i]
        ax.text(
            0.04, 0.96,
            f"R² = {m['r2']:.4f}\nRMSE = {m['rmse']:.3f} MPa",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
        )

    fig.suptitle(
        "Tensor Components — Target vs Result\n"
        f"N={n_elements:,}  MAE={global_m['mae']:.3f} MPa  "
        f"RMSE={global_m['rmse']:.3f} MPa  R²={global_m['r2']:.5f}",
        fontsize=9, fontweight="bold",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# Fig 2a: Von Mises Bland-Altman (single panel)
def plot_von_bland_altman(path: str, target: np.ndarray, result: np.ndarray) -> None:
    _paper_style()
    n = len(COMPONENT_ORDER)
    fig, ax = plt.subplots(figsize=(5, 4))
    t, r = target[0::n], result[0::n]
    mean_val = (t + r) / 2.0
    diff = r - t
    bias = float(np.mean(diff))
    sigma = float(np.std(diff, ddof=1))
    loa_hi = bias + 1.96 * sigma
    loa_lo = bias - 1.96 * sigma
    ax.scatter(mean_val, diff, s=3, alpha=0.35, color="#2171b5", linewidths=0)
    ax.axhline(bias, color="black", lw=1.0)
    ax.axhline(loa_hi, color="#d62728", lw=0.8, linestyle="--")
    ax.axhline(loa_lo, color="#d62728", lw=0.8, linestyle="--")
    ax.set_xlabel("Mean (MPa)", fontsize=8)
    ax.set_ylabel("Difference (MPa)", fontsize=8)
    ax.text(
        0.04, 0.96,
        f"Bias={bias:+.3f}\n+1.96σ={loa_hi:.3f}\n−1.96σ={loa_lo:.3f}",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
    )
    fig.suptitle("Bland-Altman Agreement — Von Mises", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# Fig 2b: tensor component Bland-Altman grid (2×3)
def plot_tensor_bland_altman_grid(path: str, target: np.ndarray, result: np.ndarray) -> None:
    _paper_style()
    n = len(COMPONENT_ORDER)
    fig, axes = plt.subplots(2, 3, figsize=(9, 6))
    flat = axes.flatten()

    for k, i in enumerate(range(1, 7)):
        ax = flat[k]
        comp = COMPONENT_ORDER[i]
        t, r = target[i::n], result[i::n]
        mean_val = (t + r) / 2.0
        diff = r - t
        bias = float(np.mean(diff))
        sigma = float(np.std(diff, ddof=1))
        loa_hi = bias + 1.96 * sigma
        loa_lo = bias - 1.96 * sigma
        ax.scatter(mean_val, diff, s=3, alpha=0.35, color="#2171b5", linewidths=0)
        ax.axhline(bias, color="black", lw=1.0)
        ax.axhline(loa_hi, color="#d62728", lw=0.8, linestyle="--")
        ax.axhline(loa_lo, color="#d62728", lw=0.8, linestyle="--")
        ax.set_xlabel("Mean (MPa)", fontsize=8)
        ax.set_ylabel("Difference (MPa)", fontsize=8)
        ax.set_title(comp, fontsize=9, fontweight="bold")
        ax.text(
            0.04, 0.96,
            f"Bias={bias:+.3f}\n+1.96σ={loa_hi:.3f}\n−1.96σ={loa_lo:.3f}",
            transform=ax.transAxes, fontsize=7, va="top",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="none", alpha=0.85),
        )

    fig.suptitle("Bland-Altman Agreement — Tensor Components", fontsize=10, fontweight="bold")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# Fig 3: component MAE / RMSE bar chart
def plot_component_metrics(path: str, per_comp: list) -> None:
    _paper_style()
    comps = [m["component"] for m in per_comp]
    maes = [m["mae"] for m in per_comp]
    rmses = [m["rmse"] for m in per_comp]

    x = np.arange(len(comps))
    w = 0.35
    fig, ax = plt.subplots(figsize=(7, 4))
    b_mae = ax.bar(x - w / 2, maes, w, label="MAE", color="#4292c6", edgecolor="none")
    b_rms = ax.bar(x + w / 2, rmses, w, label="RMSE", color="#f16913", edgecolor="none")
    # highlight VON with a green outline
    for b in (b_mae[0], b_rms[0]):
        b.set_edgecolor("#00441b")
        b.set_linewidth(1.5)

    ax.set_xticks(x)
    ax.set_xticklabels(comps)
    ax.set_ylabel("Error (MPa)")
    ax.set_title("Per-Component Error Metrics", fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# Fig 4: ECDF of absolute relative error
def plot_ecdf(path: str, target: np.ndarray, result: np.ndarray) -> None:
    _paper_style()
    n = len(COMPONENT_ORDER)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig, ax = plt.subplots(figsize=(6, 4))
    for i, comp in enumerate(COMPONENT_ORDER):
        t, r = target[i::n], result[i::n]
        mask = np.abs(t) >= 1e-6
        if not mask.any():
            continue
        rel_err = np.abs((r[mask] - t[mask]) / t[mask]) * 100.0
        sorted_err = np.sort(rel_err)
        ecdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        frac_10 = float(np.mean(rel_err <= 10.0)) * 100
        ax.plot(sorted_err, ecdf, color=colors[i % len(colors)], lw=1.2,
                label=f"{comp}  (≤10 %: {frac_10:.1f} %)")

    for thr in (5.0, 10.0):
        ax.axvline(thr, color="gray", lw=0.8, linestyle="--", alpha=0.7)
        ax.text(thr * 1.05, 0.02, f"{thr:.0f} %", fontsize=7, color="gray", va="bottom")

    ax.set_xscale("log")
    ax.set_xlabel("Absolute Relative Error (%)")
    ax.set_ylabel("Cumulative Fraction of Elements")
    ax.set_title("ECDF of Absolute Relative Error by Component", fontweight="bold")
    ax.legend(fontsize=7, loc="lower right")
    ax.grid(True, which="both", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


# ── Entry point ───────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Compare target and result stress tensors.")
    parser.add_argument("--target", required=True, help="Target .strs / .txt file")
    parser.add_argument("--result", "--verified", dest="result", required=True,
                        help="Result .strs / .txt file  (--verified accepted for backwards compat)")
    parser.add_argument("--target-subcase", type=int, default=None,
                        help="Subcase ID to read from the target file")
    parser.add_argument("--result-subcase", type=int, default=None,
                        help="Subcase ID to read from the result file")
    parser.add_argument("--subcase", type=int, default=1,
                        help="Fallback subcase ID for both files when --target-subcase / --result-subcase are omitted")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--timestamp", dest="timestamp", action="store_true",
                        help="Append a timestamp suffix to all output files.")
    parser.add_argument("--no-timestamp", dest="timestamp", action="store_false")
    parser.set_defaults(timestamp=True)
    args = parser.parse_args()

    target_subcase = args.target_subcase if args.target_subcase is not None else args.subcase
    result_subcase = args.result_subcase if args.result_subcase is not None else args.subcase

    target_subcases = parse_subcases(read_text(args.target))
    result_subcases = parse_subcases(read_text(args.result))

    if target_subcase not in target_subcases:
        available = ", ".join(str(s) for s in sorted(target_subcases.keys()))
        raise ValueError(f"Target subcase {target_subcase} not found. Available: {available}")
    if result_subcase not in result_subcases:
        available = ", ".join(str(s) for s in sorted(result_subcases.keys()))
        raise ValueError(f"Result subcase {result_subcase} not found. Available: {available}")

    ref_ids = target_subcases[target_subcase].elem_ids
    validate_elem_ids(ref_ids, result_subcases[result_subcase].elem_ids)

    target_vec = build_vector_with_order(target_subcases[target_subcase], ref_ids)
    result_vec = build_vector_with_order(result_subcases[result_subcase], ref_ids)

    global_m, per_comp = compute_metrics(target_vec, result_vec)

    os.makedirs(args.output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S") if args.timestamp else ""
    suf = f"_{ts}" if ts else ""

    report_path       = os.path.join(args.output_dir, f"compare_report{suf}.txt")
    metrics_csv       = os.path.join(args.output_dir, f"compare_metrics{suf}.csv")
    von_scatter_png   = os.path.join(args.output_dir, f"compare_von_scatter{suf}.png")
    tensor_scatter_png = os.path.join(args.output_dir, f"compare_tensor_scatter_grid{suf}.png")
    von_ba_png        = os.path.join(args.output_dir, f"compare_von_bland_altman{suf}.png")
    tensor_ba_png     = os.path.join(args.output_dir, f"compare_tensor_bland_altman{suf}.png")
    bar_png           = os.path.join(args.output_dir, f"compare_component_metrics{suf}.png")
    ecdf_png          = os.path.join(args.output_dir, f"compare_ecdf{suf}.png")

    write_report(report_path, args.target, args.result,
                 target_subcase, result_subcase, len(ref_ids), global_m, per_comp)
    write_metrics_csv(metrics_csv, per_comp)
    plot_von_scatter(von_scatter_png, target_vec, result_vec, per_comp, len(ref_ids))
    plot_tensor_scatter_grid(tensor_scatter_png, target_vec, result_vec, per_comp, global_m, len(ref_ids))
    plot_von_bland_altman(von_ba_png, target_vec, result_vec)
    plot_tensor_bland_altman_grid(tensor_ba_png, target_vec, result_vec)
    plot_component_metrics(bar_png, per_comp)
    plot_ecdf(ecdf_png, target_vec, result_vec)

    print(f"Outputs written to: {args.output_dir}")
    for p in (report_path, metrics_csv, von_scatter_png, tensor_scatter_png,
              von_ba_png, tensor_ba_png, bar_png, ecdf_png):
        print(f"  {os.path.basename(p)}")


if __name__ == "__main__":
    main()
