"""
error_analysis_outlier_r.py
==============================
Standalone error-taxonomy script for the spatial outlier detection
task. Computes the SAME has_evidence_bool / ambiguous / correct logic
score_outlier_iterative_r.py uses (copied verbatim), classifies every
row into one of 5 error_type labels, and writes ONLY the error
analysis output. Never writes a scoring summary CSV, never touches
Scores/.

USAGE
-----
1. Uncomment exactly ONE of the 4 RESULTS_CSV / ERROR_OUT_DIR blocks.
2. Run: python error_analysis_outlier_r.py
3. Output goes to Errors/<configuration>/:
     error_analysis_outlier_r.csv
     error_analysis_outlier_r_counts.csv
"""

import os
import re
import pandas as pd

from statistic_plausibility import magnitude_plausible
from formula_invariance_check import check_shift_invariant

GROUND_TRUTH_METHODS_PATH = "ground truth/ground_truth_all_methods_outlierr.csv"
MANIFEST_PATH = "chunks_csv/outlier/routlier_iterative_r_manifest.csv"

# ==================== CONFIGURATION: uncomment ONE block ====================

# Multi agent
#RESULTS_CSV = "results/multi agent/outlier_iterative_r_results_multiagent_with_method.csv"
#ERROR_OUT_DIR = "Errors/multi agent"

# Single turn
#RESULTS_CSV = "outlier_iterative_r_results_with_method.csv"
#ERROR_OUT_DIR = "Errors/single turn"

# Multi turn
#RESULTS_CSV = "results/single agent/outlier_iterative_r_results_multi_with_method.csv"
#ERROR_OUT_DIR = "Errors/multi turn"

# ReAct
RESULTS_CSV = "ReAct/outlier_iterative_r_results_react.csv"
ERROR_OUT_DIR = "Errors/ReAct"

# ==============================================================================

CLASSES = ["outlier_high", "outlier_low", "not_outlier"]

METHOD_NAME_MAP = {
    "Iterative-R (Lu, Chen & Kou 2003)": "Iterative-R (Lu, Chen & Kou 2003)",
    "Single-pass OLS regression (no masking correction)": "Single-pass OLS regression (no masking correction)",
    "Robust regression (Huber/RANSAC/Theil-Sen)": "Robust regression (Huber/RANSAC/Theil-Sen)",
    "Neighbor mean/std z-score (non-regression)": "Neighbor mean/std z-score (non-regression)",
    "Local Moran's I / LISA": "Local Moran's I / LISA",
}

SELF_ADMITTED_NON_COMPUTATION_PHRASES = [
    "not computed", "could not be computed", "unable to compute", "impossible to compute",
    "no numerical results", "did not reach a valid", "no valid final answer",
    "not available", "cannot provide a computed number", "no computed",
    "did not execute", "not execute correctly", "syntaxerror", "not provided",
    "could not compute", "n/a", "[not computed]", "placeholder value",
    "not actual computation", "can't report a valid", "cannot report a valid",
    "failed to converge", "insufficient data", "insufficient information",
    "could not be completed", "cannot be determined", "don't include actual",
    "do not include actual", "numerical issue that prevented",
    "please replace the values", "replace the values with the actual",
    "i cannot provide the actual", "i recommend running the corrected code",
]


def _has_self_admitted_non_computation(text):
    if not isinstance(text, str):
        return False
    return any(phrase in text.lower() for phrase in SELF_ADMITTED_NON_COMPUTATION_PHRASES)


def _has_real_statistic(evidence_text, full_text=None):
    if not isinstance(evidence_text, str):
        return False
    if _has_self_admitted_non_computation(full_text if full_text else evidence_text):
        return False
    match = re.search(r"=\s*([^,)]+)[,)]", evidence_text)
    if not match:
        return False
    val = match.group(1).strip()
    try:
        f = float(val)
        return f == f and abs(f) != float("inf")
    except ValueError:
        return False


def _extract_raw_statistic(evidence_text):
    if not isinstance(evidence_text, str):
        return None
    match = re.search(r"=\s*([^,)]+)[,)]", evidence_text)
    return match.group(1).strip() if match else None


def _to_float(raw):
    if raw is None:
        return None
    try:
        f = float(str(raw).strip())
        return f if (f == f and abs(f) != float("inf")) else None
    except ValueError:
        return None


def verdict_z_consistent(predicted, evidence_text, threshold=1.96, alpha=0.05):
    import re as _re
    match = _re.search(r"([A-Za-z_\-\s]*)=\s*([^,)]+)[,)]", str(evidence_text))
    if not match:
        return True
    label = match.group(1).strip().lower()
    raw_value = match.group(2).strip()
    f = _to_float(raw_value)
    if f is None:
        return True
    Z_SCORE_LABELS = {"z", "z-score", "z_score", "standardized residual",
                       "standardized_residual", "std residual", "std_residual"}
    P_VALUE_LABELS = {"p", "p-value", "p_value", "significance", "significance measure"}
    is_confident_z = label in Z_SCORE_LABELS
    is_confident_p = label in P_VALUE_LABELS
    if not is_confident_z and not is_confident_p:
        return True
    if is_confident_p:
        is_significant = abs(f) < alpha
        direction_positive = None
    else:
        is_significant = abs(f) >= threshold
        direction_positive = f > 0
    if predicted == "not_outlier":
        return not is_significant
    if predicted == "outlier_high":
        if not is_significant:
            return False
        return direction_positive is None or direction_positive is True
    if predicted == "outlier_low":
        if not is_significant:
            return False
        return direction_positive is None or direction_positive is False
    return True


def outlier_magnitude_plausible(raw_value, threshold=12.0):
    f = _to_float(raw_value)
    if f is None:
        return True
    return abs(f) <= threshold


def load_ground_truth_methods():
    gt = pd.read_csv(GROUND_TRUTH_METHODS_PATH)
    table = {}
    for _, r in gt.iterrows():
        table[(r["chunk_id"], r["method"])] = {
            "gt_direction": r["gt_direction"],
            "gt_z": r["gt_z"],
        }
    return table


def resolve_ground_truth(chunk_id, statistical_method, manifest_direction, manifest_z, gt_methods):
    normalized_method = METHOD_NAME_MAP.get(statistical_method)
    if normalized_method is not None:
        key = (chunk_id, normalized_method)
        if key in gt_methods:
            gt = gt_methods[key]
            return gt["gt_direction"], gt["gt_z"], "method_specific", normalized_method
    return manifest_direction, manifest_z, "manifest_fallback", "Iterative-R (Lu, Chen & Kou 2003)"


def _load_manifest_index():
    try:
        return pd.read_csv(MANIFEST_PATH).set_index("chunk_id")
    except (FileNotFoundError, OSError):
        print(f"WARNING: manifest not found at {MANIFEST_PATH} -- "
              f"shift-invariance re-checks will be skipped (inconclusive) for every row.")
        return pd.DataFrame(columns=["csv_path", "adjacency_path"])


def run():
    os.makedirs(ERROR_OUT_DIR, exist_ok=True)

    df = pd.read_csv(RESULTS_CSV)

    if "statistical_method" not in df.columns:
        raise ValueError(
            f"'{RESULTS_CSV}' has no `statistical_method` column. "
            "Run attach_statistical_method_outlierr.py first and point "
            "RESULTS_CSV at its '_with_method' output."
        )

    gt_methods = load_ground_truth_methods()
    manifest = _load_manifest_index()

    resolved = df.apply(
        lambda r: resolve_ground_truth(
            r["chunk_id"], r.get("statistical_method"),
            r["ground_truth_direction"], r["ground_truth_z_score"], gt_methods),
        axis=1)
    df["gt_direction_used"] = resolved.apply(lambda t: t[0])

    df["has_real_statistic_bool"] = df.apply(
        lambda r: _has_real_statistic(r["evidence_text"], r.get("verdict_detail")), axis=1)

    total_checkable = int(df["has_real_statistic_bool"].sum())
    print(f"Running shift-invariance re-checks on {total_checkable} row(s)...", flush=True)
    _progress = {"n": 0}

    def _shift_check(row):
        if not row["has_real_statistic_bool"]:
            return None
        _progress["n"] += 1
        print(f"  [{_progress['n']}/{total_checkable}] {row['model']} | {row['chunk_id']}", flush=True)
        if row["chunk_id"] not in manifest.index:
            return None
        m = manifest.loc[row["chunk_id"]]
        raw_stat = _extract_raw_statistic(row["evidence_text"])
        return check_shift_invariant(
            row["run_dir"], m["csv_path"], m.get("adjacency_path"), raw_stat)

    df["shift_invariant"] = df.apply(_shift_check, axis=1)
    print("Shift-invariance re-checks complete.\n", flush=True)

    def _plausibility_check(row):
        if not row["has_real_statistic_bool"]:
            return True
        raw = _extract_raw_statistic(row["evidence_text"])
        if not magnitude_plausible(raw):
            return False
        if not outlier_magnitude_plausible(raw):
            return False
        if not verdict_z_consistent(row["predicted"], row["evidence_text"]):
            return False
        return True

    df["statistic_plausible"] = df.apply(_plausibility_check, axis=1)

    df["has_evidence_bool"] = (
        df["has_real_statistic_bool"]
        & (df["shift_invariant"] != False)  # noqa: E712
        & df["statistic_plausible"]
    )

    df["ambiguous"] = df["predicted"].isna() | (~df["predicted"].isin(CLASSES))
    df["correct"] = (~df["ambiguous"]) & (df["predicted"] == df["gt_direction_used"])

    WEIGHT_COLS = ["format_failure", "grounded_correctness", "grounding_failure",
                   "reasoning_failure", "ungrounded_correctness"]

    def _error_weights(row):
        w = {c: 0.0 for c in WEIGHT_COLS}
        if row["ambiguous"]:
            w["format_failure"] = 1.0
        elif row["has_evidence_bool"] and row["correct"]:
            w["grounded_correctness"] = 1.0
        elif (not row["has_evidence_bool"]) and row["correct"]:
            w["ungrounded_correctness"] = 1.0
        elif row["has_evidence_bool"] and (not row["correct"]):
            w["reasoning_failure"] = 1.0
        else:
            w["grounding_failure"] = 1.0
        return w

    weights_df = df.apply(_error_weights, axis=1, result_type="expand")
    df = pd.concat([df, weights_df], axis=1)

    out_path = os.path.join(ERROR_OUT_DIR, "error_analysis_outlier_r.csv")
    df[["chunk_id", "model", "mode"] + WEIGHT_COLS].assign(task="outlier_r").to_csv(out_path, index=False)
    print(f"Saved per-case error taxonomy: {out_path} ({len(df)} rows)")

    error_counts = (
        df.assign(task="outlier_r")
        .groupby(["task", "mode"])[WEIGHT_COLS].sum().reset_index()
    )
    counts_path = os.path.join(ERROR_OUT_DIR, "error_analysis_outlier_r_counts.csv")
    error_counts.to_csv(counts_path, index=False)
    print(f"Saved error-type counts: {counts_path}")


if __name__ == "__main__":
    run()
