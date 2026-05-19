import argparse
import csv
import os
import re
from datetime import datetime
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

COMPONENT_ORDER = ["XX1", "YY1", "XY1", "XX2", "YY2", "XY2"]
DEFAULT_OUTPUT_DIR = "outputs"


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


def build_vector(subcase: SubcaseData) -> np.ndarray:
    vector = []
    for elem_id in subcase.elem_ids:
        elem_values = subcase.values[elem_id]
        for comp in COMPONENT_ORDER:
            vector.append(elem_values[comp])
    return np.asarray(vector, dtype=float)


def build_vector_with_order(subcase: SubcaseData, elem_ids: List[int]) -> np.ndarray:
    if not elem_ids:
        raise ValueError("Element list is empty.")

    vector = []
    for elem_id in elem_ids:
        if elem_id not in subcase.values:
            raise ValueError(
                f"Subcase {subcase.subcase_id} is missing element {elem_id}."
            )
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
        "Element mismatch between target and verified.",
        f"Missing: {missing[:10]}" + (" ..." if len(missing) > 10 else ""),
        f"Extra: {extra[:10]}" + (" ..." if len(extra) > 10 else ""),
    ]
    raise ValueError(" ".join(message))


def write_stats_csv(path: str, target: np.ndarray, verified: np.ndarray) -> None:
    diff = verified - target
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "target", "verified", "diff", "abs_diff"])
        for idx, (t_val, v_val, d_val) in enumerate(zip(target, verified, diff), start=1):
            writer.writerow([idx, f"{t_val:.10e}", f"{v_val:.10e}", f"{d_val:.10e}", f"{abs(d_val):.10e}"])


def plot_scatter(path: str, target: np.ndarray, verified: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(target, verified, s=8, alpha=0.7)
    min_val = min(target.min(), verified.min())
    max_val = max(target.max(), verified.max())
    ax.plot([min_val, max_val], [min_val, max_val], "k--", linewidth=1)
    ax.set_xlabel("Target")
    ax.set_ylabel("Verified")
    ax.set_title("Target vs Verified Stress")
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_component_error(path: str, target: np.ndarray, verified: np.ndarray) -> None:
    diff = verified - target
    comps = COMPONENT_ORDER
    comp_errors = []
    for i in range(len(comps)):
        comp_errors.append(np.mean(np.abs(diff[i::len(comps)])))

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(comps, comp_errors)
    ax.set_ylabel("Mean Abs Error")
    ax.set_title("Component Mean Abs Error")
    ax.grid(True, axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare target and verified stress tensors.")
    parser.add_argument("--target", required=True, help="Target .strs file")
    parser.add_argument("--verified", required=True, help="Verified .strs file")
    parser.add_argument("--subcase", type=int, default=1, help="Subcase ID to compare")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--timestamp",
        dest="timestamp",
        action="store_true",
        help="Append a timestamp suffix to all output files.",
    )
    parser.add_argument(
        "--no-timestamp",
        dest="timestamp",
        action="store_false",
        help="Disable timestamp suffix for output files.",
    )
    parser.set_defaults(timestamp=True)
    args = parser.parse_args()

    target_lines = read_text(args.target)
    verified_lines = read_text(args.verified)

    target_subcases = parse_subcases(target_lines)
    verified_subcases = parse_subcases(verified_lines)

    if args.subcase not in target_subcases:
        available = ", ".join(str(s) for s in sorted(target_subcases.keys()))
        raise ValueError(f"Target subcase {args.subcase} not found. Available: {available}")
    if args.subcase not in verified_subcases:
        available = ", ".join(str(s) for s in sorted(verified_subcases.keys()))
        raise ValueError(f"Verified subcase {args.subcase} not found. Available: {available}")

    reference_elem_ids = target_subcases[args.subcase].elem_ids
    validate_elem_ids(reference_elem_ids, verified_subcases[args.subcase].elem_ids)

    target_vec = build_vector_with_order(target_subcases[args.subcase], reference_elem_ids)
    verified_vec = build_vector_with_order(verified_subcases[args.subcase], reference_elem_ids)

    if target_vec.shape != verified_vec.shape:
        raise ValueError("Target and verified vectors have different sizes.")

    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if args.timestamp else ""
    output_suffix = f"_{timestamp}" if timestamp else ""

    stats_path = os.path.join(args.output_dir, f"compare_stats{output_suffix}.csv")
    scatter_path = os.path.join(args.output_dir, f"compare_scatter{output_suffix}.png")
    comp_path = os.path.join(args.output_dir, f"compare_component_error{output_suffix}.png")

    write_stats_csv(stats_path, target_vec, verified_vec)
    plot_scatter(scatter_path, target_vec, verified_vec)
    plot_component_error(comp_path, target_vec, verified_vec)


if __name__ == "__main__":
    main()
