import argparse
import csv
import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple
from tqdm import tqdm
import numpy as np

try:
    from sklearn.linear_model import LassoCV, LinearRegression
except ImportError as exc:
    raise SystemExit(
        "scikit-learn is required. Install with: pip install scikit-learn"
    ) from exc

from group_lasso import GroupLasso

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


def sha256_file(path: str) -> str:
    sha = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            sha.update(chunk)
    return sha.hexdigest()


def parse_subcases(lines: List[str], expect_kind: str) -> Dict[int, SubcaseData]:
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

        if f"$ELEMENT {expect_kind}(PLATE)" in line:
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
    if not subcase.elem_ids:
        raise ValueError(f"Subcase {subcase.subcase_id} has no elements.")

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


def build_matrix(
    subcases: Dict[int, SubcaseData],
    scale: float,
    elem_ids: List[int],
) -> Tuple[np.ndarray, List[int]]:
    ordered_ids = sorted(subcases.keys())
    columns = []
    for subcase_id in ordered_ids:
        vector = build_vector_with_order(subcases[subcase_id], elem_ids)
        columns.append(vector / scale)
    return np.column_stack(columns), ordered_ids


def validate_elem_ids(
    reference_ids: List[int],
    candidate_ids: List[int],
    subcase_id: int,
) -> None:
    ref_set = set(reference_ids)
    cand_set = set(candidate_ids)
    if ref_set == cand_set:
        return

    missing = sorted(ref_set - cand_set)
    extra = sorted(cand_set - ref_set)
    message = [
        f"Element mismatch in subcase {subcase_id}.",
        f"Missing: {missing[:10]}" + (" ..." if len(missing) > 10 else ""),
        f"Extra: {extra[:10]}" + (" ..." if len(extra) > 10 else ""),
    ]
    raise ValueError(" ".join(message))


def write_csv_vector(path: str, vector: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["index", "value"])
        for idx, val in enumerate(vector, start=1):
            writer.writerow([idx, f"{val:.10e}"])


def write_csv_matrix(path: str, matrix: np.ndarray, subcase_ids: List[int]) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        header = ["index"] + [f"SUBCASE_{sid}" for sid in subcase_ids]
        writer.writerow(header)
        for i in range(matrix.shape[0]):
            row = [i + 1] + [f"{val:.10e}" for val in matrix[i, :]]
            writer.writerow(row)


def _safe_scale(values: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    scale = np.std(values, axis=0)
    scale = np.where(scale < eps, 1.0, scale)
    return scale


def solve_group_lasso(
    h_matrix: np.ndarray,
    target: np.ndarray,
    standardize: bool,
    alpha: float,  # 注意：这里需要提供正则化系数
) -> dict:
    
    # 1. 特征与目标的标准化
    if standardize:
        feature_scale = _safe_scale(h_matrix)
        target_scale = float(_safe_scale(target))
        h_scaled = h_matrix / feature_scale
        target_scaled = target / target_scale
    else:
        feature_scale = np.ones(h_matrix.shape[1], dtype=float)
        target_scale = 1.0
        h_scaled = h_matrix
        target_scaled = target

    # 2. 定义分组逻辑 
    # 假设你的 h_matrix 有 24 列，要每 3 个分为一组
    num_features = h_matrix.shape[1]
    group_size = 3
    
    # 生成类似 [1, 1, 1, 2, 2, 2, 3, 3, 3 ...] 的数组
    groups = np.repeat(np.arange(1, (num_features // group_size) + 1), group_size)
    
    # 如果列数不能被3整除，处理余数（单独分一组）
    remainder = num_features % group_size
    if remainder != 0:
        remainder_groups = np.full(remainder, groups[-1] + 1)
        groups = np.concatenate([groups, remainder_groups])

    # 3. 初始化并训练 Group LASSO
    # group-lasso 库需要手动设定具体的 reg (相当于 alpha)
    gl = GroupLasso(
        groups=groups,
        group_reg=alpha,     # 控制组级别的稀疏性
        l1_reg=0.0,          # 组内不进行 L1 稀疏化，保证同进同退
        fit_intercept=False,
        scale_reg="inverse_group_size", # 消除组大小不同的影响
        supress_warning=True,
        tol=1e-3
    )
    
    # group-lasso 要求的 target 形状必须是 (n_samples, 1)
    gl.fit(h_scaled, target_scaled.reshape(-1, 1))
    
    # 提取系数
    coef_scaled = gl.coef_.flatten()

    # 4. 再次执行 Relaxed OLS
    # Group LASSO 同样存在收缩偏差，选出 Active 组后，不再无惩罚回归
    active_features = np.abs(coef_scaled) > 1e-10
    coef_scaled_relaxed = np.zeros_like(coef_scaled)

    if np.any(active_features):
        from sklearn.linear_model import LinearRegression
        ols = LinearRegression(fit_intercept=False)
        ols.fit(h_scaled[:, active_features], target_scaled)
        coef_scaled_relaxed[active_features] = ols.coef_
    
    # 4. 还原尺度 (使用无偏差的 Relaxed 系数)
    coef = coef_scaled_relaxed * target_scale / feature_scale
    
    # 5. 计算最终残差
    prediction = h_matrix @ coef
    residual = prediction - target

    # 直接使用传入的 alpha

    result = {
        "coef": coef,
        "alpha": alpha, 
        "residual_norm": float(np.linalg.norm(residual)),
        "relative_error": float(np.linalg.norm(residual) / (np.linalg.norm(target) + 1e-12)),
        "nonzero": int(np.sum(active_features)),
        "standardize": standardize,
        "feature_scale": feature_scale.tolist(),
        "target_scale": target_scale,
    }

    return result


def write_result_csv(path: str, subcase_ids: List[int], coef: np.ndarray) -> None:
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["subcase_id", "force_value"])
        for sid, val in zip(subcase_ids, coef):
            writer.writerow([sid, f"{val:.10e}"])


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def run_pipeline(
    ir_path: str,
    spc_path: str,
    kind: str,
    output_dir: str,
    ir_subcase: int,
    scale: float,
    alpha_grid: List[float],
    max_iter: int,
    tol: float,
    standardize: bool,
    output_suffix: str,
) -> Dict[str, object]:
    ir_lines = read_text(ir_path)
    spc_lines = read_text(spc_path)

    ir_subcases = parse_subcases(ir_lines, kind)
    spc_subcases = parse_subcases(spc_lines, kind)

    if ir_subcase not in ir_subcases:
        available = ", ".join(str(s) for s in sorted(ir_subcases.keys()))
        raise ValueError(f"IR subcase {ir_subcase} not found. Available: {available}")

    reference_elem_ids = ir_subcases[ir_subcase].elem_ids
    target = build_vector_with_order(ir_subcases[ir_subcase], reference_elem_ids)

    for subcase_id, subcase in spc_subcases.items():
        validate_elem_ids(reference_elem_ids, subcase.elem_ids, subcase_id)

    h_matrix, subcase_ids = build_matrix(spc_subcases, scale, reference_elem_ids)

    if h_matrix.shape[0] != target.shape[0]:
        raise ValueError(
            f"Dimension mismatch: H has {h_matrix.shape[0]} rows, target has {target.shape[0]} entries."
        )
    print(f"\n[INFO] 开始对 {kind} 进行 Group LASSO 优化分析...")

# ================= Loading 进度显示 =================
    print(f"\n[INFO] 开始对 {kind} 进行 Group LASSO 优化分析...")
    
    best_result = None
    best_error = float('inf')
    
    # 使用 tqdm 包装 alpha_grid，生成可视化
    for current_alpha in tqdm(alpha_grid, desc=f"Testing Alphas for {kind}"):
        # 调用新的组惩罚函数
        result = solve_group_lasso(h_matrix, target, standardize, alpha=current_alpha)
        
        # 记录误差最小的那个结果
        if result["relative_error"] < best_error:
            best_error = result["relative_error"]
            best_result = result

    print(f"[SUCCESS] {kind} 优化完成! 最佳 Alpha: {best_result['alpha']:.2e}, 相对误差: {best_error:.4%}")
    # ===============================================

    ensure_dir(output_dir)
    write_csv_vector(
        os.path.join(output_dir, f"{kind.lower()}_E_target{output_suffix}.csv"),
        target,
    )
    write_csv_matrix(
        os.path.join(output_dir, f"{kind.lower()}_H{output_suffix}.csv"),
        h_matrix,
        subcase_ids,
    )
    write_result_csv(
        os.path.join(output_dir, f"{kind.lower()}_result{output_suffix}.csv"),
        subcase_ids,
        best_result["coef"],
    )

    metadata = {
        "kind": kind,
        "ir_path": ir_path,
        "spc_path": spc_path,
        "ir_subcase": ir_subcase,
        "subcase_ids": subcase_ids,
        "component_order": COMPONENT_ORDER,
        "scale": scale,
        "dimensions": {"H": list(h_matrix.shape), "target": int(target.shape[0])},
        "elem_ids": reference_elem_ids,
        "alpha": best_result["alpha"],
        "nonzero": best_result["nonzero"],
        "residual_norm": best_result["residual_norm"],
        "relative_error": best_result["relative_error"],
        "standardize": best_result["standardize"],
        "target_scale": best_result["target_scale"],
        "feature_scale": best_result["feature_scale"],
    }

    return {
        "metadata": metadata,
    }

def write_log(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2))


def write_report(path: str, entries: List[Dict[str, object]]) -> None:
    lines = [
        "# LASSO Report",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        "",
    ]

    for entry in entries:
        meta = entry["metadata"]
        lines.append(f"## {meta['kind']} results")
        lines.append("")
        lines.append(f"- IR subcase: {meta['ir_subcase']}")
        lines.append(f"- Subcases (H columns): {', '.join(str(s) for s in meta['subcase_ids'])}")
        lines.append(f"- H shape: {meta['dimensions']['H']}")
        lines.append(f"- Target length: {meta['dimensions']['target']}")
        lines.append(f"- Alpha: {meta['alpha']:.6e}")
        lines.append(f"- Nonzero forces: {meta['nonzero']}")
        lines.append(f"- Residual norm: {meta['residual_norm']:.6e}")
        lines.append(f"- Relative error: {meta['relative_error']:.6e}")
        lines.append(f"- Standardize: {meta['standardize']}")
        lines.append("")

    with open(path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(lines))


def parse_alpha_grid(raw: str) -> List[float]:
    return [float(val.strip()) for val in raw.split(",") if val.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Parse .strs/.strn and run LASSO.")
    parser.add_argument("--mode", choices=["stress", "strain", "both"], default="both")
    parser.add_argument("--ir-subcase", type=int, default=1)
    parser.add_argument("--scale-h", type=float, default=1000.0)
    parser.add_argument(
        "--alpha-grid",
        type=str,
        default="0.2,1,10,50,100,500,1000",
    )
    parser.add_argument("--max-iter", type=int, default=1000000)
    parser.add_argument("--tol", type=float, default=1e-3)
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
    parser.add_argument(
        "--standardize",
        dest="standardize",
        action="store_true",
        help="Scale features/target by standard deviation before LASSO.",
    )
    parser.add_argument(
        "--no-standardize",
        dest="standardize",
        action="store_false",
        help="Disable scaling; solve in original units.",
    )
    parser.set_defaults(standardize=True)
    parser.add_argument(
        "--ir-strs",
        type=str,
        default="InfluenceMatrix/le5quad4_IR/LE5Quad4_IR.strs",
    )
    parser.add_argument(
        "--ir-strn",
        type=str,
        default="InfluenceMatrix/le5quad4_IR/LE5Quad4_IR.strn",
    )
    parser.add_argument(
        "--spc-strs",
        type=str,
        default="InfluenceMatrix/le5quad4_SPC/LE5Quad4_SPC.strs",
    )
    parser.add_argument(
        "--spc-strn",
        type=str,
        default="InfluenceMatrix/le5quad4_SPC/LE5Quad4_SPC.strn",
    )

    args = parser.parse_args()
    alpha_grid = parse_alpha_grid(args.alpha_grid)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") if args.timestamp else ""
    output_suffix = f"_{timestamp}" if timestamp else ""

    entries = []
    log_payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "output_suffix": output_suffix,
        "alpha_grid": alpha_grid,
        "files": {},
    }

    if args.mode in ("stress", "both"):
        log_payload["files"]["ir_strs"] = sha256_file(args.ir_strs)
        log_payload["files"]["spc_strs"] = sha256_file(args.spc_strs)
        entries.append(
            run_pipeline(
                args.ir_strs,
                args.spc_strs,
                "STRESS",
                args.output_dir,
                args.ir_subcase,
                args.scale_h,
                alpha_grid,
                args.max_iter,
                args.tol,
                args.standardize,
                output_suffix,
            )
        )

    if args.mode in ("strain", "both"):
        log_payload["files"]["ir_strn"] = sha256_file(args.ir_strn)
        log_payload["files"]["spc_strn"] = sha256_file(args.spc_strn)
        entries.append(
            run_pipeline(
                args.ir_strn,
                args.spc_strn,
                "STRAIN",
                args.output_dir,
                args.ir_subcase,
                args.scale_h,
                alpha_grid,
                args.max_iter,
                args.tol,
                args.standardize,
                output_suffix,
            )
        )

    for entry in entries:
        kind = entry["metadata"]["kind"].lower()
        log_payload[f"{kind}_metadata"] = entry["metadata"]

    ensure_dir(args.output_dir)
    write_log(os.path.join(args.output_dir, f"run_log{output_suffix}.txt"), log_payload)
    write_report(os.path.join(args.output_dir, f"report{output_suffix}.md"), entries)


if __name__ == "__main__":
    main()
