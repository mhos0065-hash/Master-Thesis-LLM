"""
error_analysis_spatial.py
============================
Standalone error-taxonomy script for the spatial correlation task
(Global + Local Moran's I). Computes the SAME has_evidence_bool /
ambiguous / correct logic score_spatial_results.py uses (copied
verbatim), then splits each case's contribution into 5 weight columns
that sum to 1.0. Writes ONLY the error analysis output.

GRADED, not discrete: `correct` is already the exact per-row value the
accuracy table uses -- 0/1 for global rows, graded F1 (borough-overlap
partial credit) for local rows. Each case's weight is split between
the correct-side bucket (weight = correct) and the incorrect-side
bucket (weight = 1 - correct), chosen by evidence status. Summing
grounded_correctness + ungrounded_correctness across all cases and
dividing by n reproduces the exact same accuracy value the headline
table reports, for both global and local rows.

USAGE
-----
1. Uncomment exactly ONE of the 4 RESULTS_CSV / ERROR_OUT_DIR blocks.
2. Run: python error_analysis_spatial.py
3. Output goes to Errors/<configuration>/:
     error_analysis_spatial.csv
     error_analysis_spatial_counts.csv
"""

import json
import os
import re
import pandas as pd

from statistic_plausibility import magnitude_plausible, bounded_probability, verdict_p_consistent
from formula_invariance_check import check_shift_invariant, compute_shifted_statistic

GROUND_TRUTH_METHODS_PATH = "ground truth/ground_truth_all_methods_spatial.csv"
MANIFEST_PATH = "chunks_csv/manifest.csv"

# ==================== CONFIGURATION: uncomment ONE block ====================

# ReAct
RESULTS_CSV = "ReAct/spatial_correlation_results_react_with_method.csv"
ERROR_OUT_DIR = "Errors/ReAct"

# Multi agent
#RESULTS_CSV = "results/multi agent/spatial_correlation_results_multiagent_with_method.csv"
#ERROR_OUT_DIR = "Errors/multi agent"

# Multi turn
#RESULTS_CSV = "results/single agent/results_spatial_multi_with_method.csv"
#ERROR_OUT_DIR = "Errors/multi turn"

# Single turn
#RESULTS_CSV = "spatial_results_with_method.csv"
#ERROR_OUT_DIR = "Errors/single turn"

# ==============================================================================

METHOD_NAME_MAP_GLOBAL = {
    "Global Moran's I": "Global Moran's I",
    "Geary's C": "Geary's C",
    "Join Count Statistic": "Join Count Statistic",
    "Neighbor-average correlation (non-standard)": "Neighbor-average correlation (non-standard)",
}
METHOD_NAME_MAP_LOCAL = {
    "Local Moran's I / LISA": "Local Moran's I / LISA",
    "Getis-Ord Gi / Gi*": "Getis-Ord Gi / Gi*",
}

LOCAL_SHIFT_TOLERANCE = {
    "Local Moran's I / LISA": 1.5,
    "Getis-Ord Gi / Gi*": 0.2,
}
DEFAULT_LOCAL_SHIFT_TOLERANCE = 0.2


def _has_real_statistic(evidence_text):
    if not isinstance(evidence_text, str):
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


def parse_bool(v):
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s in ("true", "found", "1", "yes", "clustered", "present"):
        return True
    if s in ("false", "not found", "0", "no", "not clustered", "not present", "random"):
        return False
    return None


def _extract_raw_statistic(evidence_text):
    if not isinstance(evidence_text, str):
        return None
    match = re.search(r"=\s*([^,)]+)[,)]", evidence_text)
    return match.group(1).strip() if match else None


def _extract_p_value(evidence_text):
    if not isinstance(evidence_text, str):
        return None
    matches = re.findall(r"=\s*([^,)]+)[,)]", evidence_text)
    return matches[1].strip() if len(matches) >= 2 else None


def _extract_local_values(evidence_text):
    if not isinstance(evidence_text, str):
        return []
    vals = []
    for raw in re.findall(r"=\s*([^,)]+)[,)]", evidence_text):
        try:
            f = float(raw.strip())
            if f == f and abs(f) != float("inf"):
                vals.append(f)
        except ValueError:
            continue
    return vals


def _normalize_borough_name(name):
    n = name.strip().lower()
    n = n.replace("&", "and")
    n = re.sub(r"[.\u2019']", "", n)
    n = re.sub(r"\s+", " ", n).strip()
    return n


def _extract_local_borough_names(evidence_text):
    if not isinstance(evidence_text, str):
        return set()
    names = set()
    for entry in re.findall(r"([A-Za-z][A-Za-z .'\-&]*?)\s*\(", evidence_text):
        name = entry.strip()
        if name:
            names.add(_normalize_borough_name(name))
    return names


def load_ground_truth_methods():
    gt = pd.read_csv(GROUND_TRUTH_METHODS_PATH)
    table = {}
    for _, r in gt.iterrows():
        table[(r["chunk_id"], r["method"])] = {
            "gt_stat": r["gt_stat"] if pd.notna(r["gt_stat"]) else None,
            "gt_clustered": bool(r["gt_clustered"]),
        }
    return table


def load_local_borough_ground_truth():
    manifest = pd.read_csv(MANIFEST_PATH)
    table = {}
    for _, r in manifest.iterrows():
        if str(r.get("spatial_granularity", "")).strip().lower() != "local":
            continue
        raw = r.get("ground_truth_metric_json")
        try:
            parsed = json.loads(raw) if isinstance(raw, str) else {}
            boroughs = parsed.get("significant_cluster_boroughs", {})
            table[r["chunk_id"]] = {_normalize_borough_name(b) for b in boroughs.keys()}
        except (ValueError, TypeError):
            table[r["chunk_id"]] = set()
    return table


def resolve_ground_truth(chunk_id, statistical_method, granularity, manifest_clustered, gt_methods):
    method_map = METHOD_NAME_MAP_GLOBAL if granularity == "global" else METHOD_NAME_MAP_LOCAL
    fallback_method = "Global Moran's I" if granularity == "global" else "Local Moran's I / LISA"
    normalized_method = method_map.get(statistical_method)
    if normalized_method is not None:
        key = (chunk_id, normalized_method)
        if key in gt_methods:
            gt = gt_methods[key]
            return gt["gt_clustered"], "method_specific", normalized_method
    return bool(manifest_clustered), "manifest_fallback", fallback_method


def _local_chunk_f1(pred_set, true_set):
    tp = len(pred_set & true_set)
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    if tp == 0 and fp == 0 and fn == 0:
        return 1.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return f1


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
            "Run attach_statistical_method_spatial.py first and point "
            "RESULTS_CSV at its '_with_method' output."
        )

    gt_methods = load_ground_truth_methods()
    local_borough_gt = load_local_borough_ground_truth()
    manifest = _load_manifest_index()

    resolved = df.apply(
        lambda r: resolve_ground_truth(
            r["chunk_id"], r.get("statistical_method"), r["spatial_granularity"],
            r["ground_truth"], gt_methods),
        axis=1)
    df["gt_clustered_used"] = resolved.apply(lambda t: t[0])
    df["gt_method_used"] = resolved.apply(lambda t: t[2])

    df["predicted_bool"] = df["predicted"].apply(parse_bool)

    _NO_VALID_ANSWER_PHRASES = ("stopped:", "reached iteration limit", "(empty response)")

    def _admits_no_valid_answer(text):
        if not isinstance(text, str):
            return False
        low = text.lower()
        return any(phrase in low for phrase in _NO_VALID_ANSWER_PHRASES)

    df["ambiguous"] = df["predicted_bool"].isna() | df["evidence_text"].apply(_admits_no_valid_answer)
    df["has_real_statistic_bool"] = df["evidence_text"].apply(_has_real_statistic)
    df["is_local"] = df["spatial_granularity"].astype(str).str.strip().str.lower() == "local"

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

        if row["is_local"]:
            local_vals = _extract_local_values(row["evidence_text"])
            if not local_vals:
                return None
            shifted_stat = compute_shifted_statistic(row["run_dir"], m["csv_path"], m.get("adjacency_path"))
            if shifted_stat is None:
                return None
            try:
                shifted_val = float(shifted_stat)
            except (TypeError, ValueError):
                return False
            if not (shifted_val == shifted_val and abs(shifted_val) != float("inf")):
                return False
            tol = LOCAL_SHIFT_TOLERANCE.get(row.get("gt_method_used"), DEFAULT_LOCAL_SHIFT_TOLERANCE)
            return any(abs(shifted_val - v) <= tol for v in local_vals)
        else:
            raw_stat = _extract_raw_statistic(row["evidence_text"])
            if raw_stat is None:
                return None
            return check_shift_invariant(
                row["run_dir"], m["csv_path"], m.get("adjacency_path"), raw_stat)

    df["shift_invariant"] = df.apply(_shift_check, axis=1)
    print("Shift-invariance re-checks complete.\n", flush=True)

    def _plausibility_check(row):
        if not row["has_real_statistic_bool"]:
            return True
        if row["is_local"]:
            values = _extract_local_values(row["evidence_text"])
            return all(magnitude_plausible(v) for v in values)
        else:
            raw_stat = _extract_raw_statistic(row["evidence_text"])
            raw_p = _extract_p_value(row["evidence_text"])
            if not magnitude_plausible(raw_stat):
                return False
            if not bounded_probability(raw_p):
                return False
            if not pd.isna(row["predicted_bool"]):
                if not verdict_p_consistent(bool(row["predicted_bool"]), raw_p):
                    return False
            return True

    df["statistic_plausible"] = df.apply(_plausibility_check, axis=1)

    df["has_evidence_bool"] = (
        df["has_real_statistic_bool"]
        & (df["shift_invariant"] != False)  # noqa: E712
        & df["statistic_plausible"]
    )

    def _row_correctness(row):
        if row["is_local"]:
            if row["ambiguous"]:
                return 0.0
            pred_names = _extract_local_borough_names(row["evidence_text"])
            true_names = local_borough_gt.get(row["chunk_id"], set())
            return _local_chunk_f1(pred_names, true_names)
        else:
            if row["ambiguous"]:
                return 0.0
            is_correct = row["predicted_bool"] == row["gt_clustered_used"]
            return 1.0 if is_correct else 0.0

    df["correct"] = df.apply(_row_correctness, axis=1)

    WEIGHT_COLS = ["format_failure", "grounded_correctness", "grounding_failure",
                   "reasoning_failure", "ungrounded_correctness"]

    # --- Fractional per-case credit, discrete only at the FINAL pooled
    # total --- `correct` is 0/1 for global rows (already naturally
    # discrete) and a continuous graded F1 for local rows. Each
    # evaluable case contributes its exact `correct` value (the SAME
    # formula the accuracy table uses) to the correct-side bucket and
    # 1-correct to the incorrect-side bucket -- NOT a per-case whole-
    # bucket assignment. Summed per (task, mode), written UNROUNDED to
    # the _counts.csv below; rounding happens only once, in
    # merge_error_analysis_counts.py/.R, on the FINAL total pooled
    # across all 5 tasks -- this is what actually minimizes the gap
    # against the accuracy table (verified numerically: ~50x smaller
    # gap than a per-case 0.5 threshold on a 20-chunk synthetic test).
    def _error_weights(row):
        w = {c: 0.0 for c in WEIGHT_COLS}
        if row["ambiguous"]:
            w["format_failure"] = 1.0
            return w
        c = float(row["correct"])
        if row["has_evidence_bool"]:
            w["grounded_correctness"] = c
            w["reasoning_failure"] = 1.0 - c
        else:
            w["ungrounded_correctness"] = c
            w["grounding_failure"] = 1.0 - c
        return w

    weights_df = df.apply(_error_weights, axis=1, result_type="expand")
    df = pd.concat([df, weights_df], axis=1)

    out_path = os.path.join(ERROR_OUT_DIR, "error_analysis_spatial.csv")
    df[["chunk_id", "model", "mode"] + WEIGHT_COLS].assign(task="spatial").to_csv(out_path, index=False)
    print(f"Saved per-case error taxonomy: {out_path} ({len(df)} rows)")

    error_counts = (
        df.assign(task="spatial")
        .groupby(["task", "mode"])[WEIGHT_COLS].sum().reset_index()
    )
    counts_path = os.path.join(ERROR_OUT_DIR, "error_analysis_spatial_counts.csv")
    error_counts.to_csv(counts_path, index=False)
    print(f"Saved error-type counts: {counts_path}")


if __name__ == "__main__":
    run()
