"""
error_analysis_bivariate.py
==============================
Standalone error-taxonomy script for the bivariate spatial correlation
task. Computes the SAME has_evidence_bool / ambiguous / correct logic
score_bivariate.py uses (copied verbatim, not reinterpreted, so the
counts here always agree with that script's accuracy/F1), then splits
each case's contribution into 5 weight columns that sum to 1.0:
    format_failure          -- no parseable verdict at all
    grounding_failure       -- verdict given, wrong, and unevidenced
    ungrounded_correctness  -- verdict given, right, but unevidenced
    reasoning_failure       -- verdict given, wrong, but evidenced
    grounded_correctness    -- verdict given, right, and evidenced
(the first 4 are the paper's "4 failure types"; the 5th is success)

Correctness here is binary (0 or 1), so each case's weight is either
0.0 or 1.0 in exactly one column -- mathematically identical to a
discrete label, kept as weight columns anyway so the output format
matches getisord's and spatial's (where correctness is graded), and
the merge script can sum all 5 files' counts with no task-specific
handling. Writes ONLY the error
analysis output. It never writes a scoring summary CSV and never
touches the Scores/ directory.

error_type is one of:
    format_failure          -- no parseable verdict at all
    grounding_failure       -- verdict given, wrong, and unevidenced
    ungrounded_correctness  -- verdict given, right, but unevidenced
    reasoning_failure       -- verdict given, wrong, but evidenced
    grounded_correctness    -- verdict given, right, and evidenced
(the first 4 are the paper's "4 failure types"; the 5th is success)

USAGE
-----
1. Uncomment exactly ONE of the 4 RESULTS_CSV / ERROR_OUT_DIR blocks
   below, matching whichever configuration you want to analyze.
2. Run: python error_analysis_bivariate.py
3. Output goes to Errors/<configuration>/ -- two files:
     error_analysis_bivariate.csv         (one row per test case)
     error_analysis_bivariate_counts.csv  (counts per mode x error_type)
"""

import os
import re
import pandas as pd

from bivariate_verdict import _has_self_admitted_non_computation
from formula_invariance_check import check_shift_invariant
from statistic_plausibility import magnitude_plausible, bounded_probability, verdict_p_consistent

MANIFEST_PATH = "chunks_csv/bivariate/bivariate_manifest.csv"
GROUND_TRUTH_METHODS_PATH = "ground truth/ground_truth_all_methods_bivariate.csv"

# ==================== CONFIGURATION: uncomment ONE block ====================

# For single turn
#RESULTS_CSV = "bivariate_results_with_method.csv"
#ERROR_OUT_DIR = "Errors/single turn"

# For multi turn
#RESULTS_CSV = "results/single agent/bivariate_results_multi_with_method.csv"
#ERROR_OUT_DIR = "Errors/multi turn"

# For multi agent
#RESULTS_CSV = "results/multi agent/bivariate_results_multiagent_with_method.csv"
#ERROR_OUT_DIR = "Errors/multi agent"

# For ReAct agent
RESULTS_CSV = "ReAct/bivariate_results_react.csv"
ERROR_OUT_DIR = "Errors/ReAct"

# ==============================================================================

CLASSES = ["positive", "negative", "not_significant"]

METHOD_NAME_MAP = {
    "Lee's L": "Lee's L",
    "Bivariate / Cross Moran's I": "Bivariate / Cross Moran's I",
    "Moran's I": "Moran's I (X only, ref.)",
    "Pearson correlation": "Pearson correlation",
    "Generic Pearson (np.corrcoef)": "Pearson correlation",
}


def _has_real_test_statistic(evidence_text, full_text=None):
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
            "gt_p": r["gt_p"],
            "gt_significant": bool(r["gt_significant"]),
        }
    return table


def _direction_from_stat(stat, significant):
    try:
        s = float(stat)
    except (TypeError, ValueError):
        return "not_significant"
    if not significant or s != s:
        return "not_significant"
    return "positive" if s > 0 else "negative"


def _resolve_ground_truth(row, gt_methods, manifest):
    normalized_method = METHOD_NAME_MAP.get(row.get("statistical_method"))
    if normalized_method is not None:
        gt = gt_methods.get((row["chunk_id"], normalized_method))
        if gt is not None:
            return {
                "direction": _direction_from_stat(gt["gt_stat"], gt["gt_significant"]),
                "method": normalized_method,
            }
    if row["chunk_id"] in manifest.index:
        m = manifest.loc[row["chunk_id"]]
        return {
            "direction": m["ground_truth_direction"],
            "method": "Lee's L",
        }
    return None


def run():
    os.makedirs(ERROR_OUT_DIR, exist_ok=True)

    df = pd.read_csv(RESULTS_CSV)
    manifest = pd.read_csv(MANIFEST_PATH).set_index("chunk_id")
    gt_methods = load_ground_truth_methods()

    if "statistical_method" not in df.columns:
        raise ValueError(
            f"'{RESULTS_CSV}' has no `statistical_method` column. "
            "Run attach_statistical_method.py first and point RESULTS_CSV "
            "at its '_with_method' output."
        )

    df["has_real_statistic_bool"] = df.apply(
        lambda r: _has_real_test_statistic(r["evidence_text"], r.get("verdict_detail")), axis=1)

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
        raw_stat = _extract_raw_statistic(row["evidence_text"])
        raw_p = _extract_p_value(row["evidence_text"])
        if not magnitude_plausible(raw_stat):
            return False
        if not bounded_probability(raw_p):
            return False
        claims_significance = row["predicted"] in ("positive", "negative")
        if row["predicted"] in ("positive", "negative", "not_significant"):
            if not verdict_p_consistent(claims_significance, raw_p):
                return False
        return True

    df["statistic_plausible"] = df.apply(_plausibility_check, axis=1)

    df["has_evidence_bool"] = (
        df["has_real_statistic_bool"]
        & (df["shift_invariant"] != False)
        & df["statistic_plausible"]
    )

    resolved = df.apply(lambda r: _resolve_ground_truth(r, gt_methods, manifest), axis=1)
    df["gt_direction_used"] = resolved.apply(lambda d: d["direction"] if d else None)

    df["ambiguous"] = df["predicted"].isna() | (~df["predicted"].isin(CLASSES))
    df["correct"] = (~df["ambiguous"]) & (df["predicted"] == df["gt_direction_used"])

    WEIGHT_COLS = ["format_failure", "grounded_correctness", "grounding_failure",
                   "reasoning_failure", "ungrounded_correctness"]

    # Binary correctness here (unlike getisord/spatial-local), so this
    # is mathematically identical to a discrete label -- kept as
    # weight columns anyway for a uniform output format across all 5
    # tasks, so the merge script needs no task-specific handling.
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

    out_path = os.path.join(ERROR_OUT_DIR, "error_analysis_bivariate.csv")
    df[["chunk_id", "model", "mode"] + WEIGHT_COLS].assign(task="bivariate").to_csv(out_path, index=False)
    print(f"Saved per-case error taxonomy: {out_path} ({len(df)} rows)")

    error_counts = (
        df.assign(task="bivariate")
        .groupby(["task", "mode"])[WEIGHT_COLS].sum().reset_index()
    )
    counts_path = os.path.join(ERROR_OUT_DIR, "error_analysis_bivariate_counts.csv")
    error_counts.to_csv(counts_path, index=False)
    print(f"Saved error-type counts: {counts_path}")


if __name__ == "__main__":
    run()
