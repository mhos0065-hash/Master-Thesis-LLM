"""
error_analysis_changespillover.py
====================================
Standalone error-taxonomy script for the spatiotemporal change
clustering task. Computes the SAME has_evidence_bool / ambiguous /
correct logic score_changespillover.py uses (copied verbatim),
classifies every row into one of 5 error_type labels, and writes ONLY
the error analysis output. Never writes a scoring summary CSV, never
touches Scores/.

USAGE
-----
1. Uncomment exactly ONE of the 4 RESULTS_CSV / ERROR_OUT_DIR blocks.
2. Run: python error_analysis_changespillover.py
3. Output goes to Errors/<configuration>/:
     error_analysis_changespillover.csv
     error_analysis_changespillover_counts.csv
"""

import os
import re
import pandas as pd

from statistic_plausibility import magnitude_plausible, bounded_probability, verdict_p_consistent
from formula_invariance_check import check_shift_invariant

GROUND_TRUTH_METHODS_PATH = "ground truth/ground_truth_all_methods_changespillover.csv"
MANIFEST_PATH = "chunks_csv/manifest.csv"

# ==================== CONFIGURATION: uncomment ONE block ====================

#Multi agent
RESULTS_CSV = "results/multi agent/changespillover_results_multiagent_with_method.csv"
ERROR_OUT_DIR = "Errors/multi agent"

#Multi turn
#RESULTS_CSV = "results/single agent/changespillover_results_multi_with_method.csv"
#ERROR_OUT_DIR = "Errors/multi turn"

#Single turn
#RESULTS_CSV = "changespillover_results_with_method.csv"
#ERROR_OUT_DIR = "Errors/single turn"

#ReAct
#RESULTS_CSV = "ReAct/changespillover_results_react_with_method.csv"
#ERROR_OUT_DIR = "Errors/ReAct"

# ==============================================================================

METHOD_NAME_MAP = {
    "Global Moran's I": "Global Moran's I",
    "Geary's C": "Geary's C",
    "Join Count Statistic": "Join Count Statistic",
    "Neighbor-average correlation (non-standard)": "Neighbor-average correlation (non-standard)",
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


def _extract_p_value(evidence_text):
    if not isinstance(evidence_text, str):
        return None
    matches = re.findall(r"=\s*([^,)]+)[,)]", evidence_text)
    return matches[1].strip() if len(matches) >= 2 else None


def load_ground_truth_methods():
    gt = pd.read_csv(GROUND_TRUTH_METHODS_PATH)
    table = {}
    for _, r in gt.iterrows():
        table[(r["chunk_id"], r["method"])] = {
            "gt_stat": r["gt_stat"],
            "gt_clustered": bool(r["gt_clustered"]),
        }
    return table


def resolve_ground_truth(chunk_id, statistical_method, manifest_clustered, gt_methods):
    normalized_method = METHOD_NAME_MAP.get(statistical_method)
    if normalized_method is not None:
        key = (chunk_id, normalized_method)
        if key in gt_methods:
            gt = gt_methods[key]
            return gt["gt_clustered"], "method_specific", normalized_method
    return bool(manifest_clustered), "manifest_fallback", "Global Moran's I"


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
            "Run attach_statistical_method_changespillover.py first and "
            "point RESULTS_CSV at its '_with_method' output."
        )

    gt_methods = load_ground_truth_methods()
    manifest = _load_manifest_index()

    resolved = df.apply(
        lambda r: resolve_ground_truth(
            r["chunk_id"], r.get("statistical_method"), r["ground_truth_present"], gt_methods),
        axis=1)
    df["gt_clustered_used"] = resolved.apply(lambda t: t[0])

    df["predicted_bool"] = df["predicted"].map({"True": True, "False": False, True: True, False: False})
    df["ambiguous"] = df["predicted_bool"].isna()
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
        window_end = m.get("window_end")
        shift_col = str(int(window_end)) if pd.notna(window_end) else None
        return check_shift_invariant(
            row["run_dir"], m["csv_path"], m.get("adjacency_path"), raw_stat,
            shift_only_column=shift_col)

    df["shift_invariant"] = df.apply(_shift_check, axis=1)
    print("Shift-invariance re-checks complete.\n", flush=True)

    def _plausibility_check(row):
        if not row["has_real_statistic_bool"]:
            return True
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

    df["correct"] = (~df["ambiguous"]) & (df["predicted_bool"] == df["gt_clustered_used"])

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

    out_path = os.path.join(ERROR_OUT_DIR, "error_analysis_changespillover.csv")
    df[["chunk_id", "model", "mode"] + WEIGHT_COLS].assign(task="changespillover").to_csv(out_path, index=False)
    print(f"Saved per-case error taxonomy: {out_path} ({len(df)} rows)")

    error_counts = (
        df.assign(task="changespillover")
        .groupby(["task", "mode"])[WEIGHT_COLS].sum().reset_index()
    )
    counts_path = os.path.join(ERROR_OUT_DIR, "error_analysis_changespillover_counts.csv")
    error_counts.to_csv(counts_path, index=False)
    print(f"Saved error-type counts: {counts_path}")


if __name__ == "__main__":
    run()
