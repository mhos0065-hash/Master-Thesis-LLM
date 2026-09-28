"""
score_changespillover.py
===========================
Scores the spatial change-spillover task's binary classification
(CLUSTERED / NOT CLUSTERED, via predicted vs ground truth), with the
SAME column structure used for bivariate, getisord, and outlier-r, so
all tables are directly comparable.

Binary, not 3-way -- predicted and the resolved ground truth are both
simple booleans.

Evidence check (UNCHANGED apart from the shift-invariance addition
below): a chunk only counts as evidenced if at least one REAL, finite,
non-placeholder number is cited (not nan, not a method name, not a
self-admitted fabrication), the statistic is magnitude-plausible, the
p-value is a genuine probability, and the claimed verdict is consistent
with its own reported p-value.

VERDICT correctness is resolved PER ROW against the ground truth for
the SPECIFIC method that row used (from
ground_truth_all_methods_changespillover.csv: Geary's C, Join Count
Statistic, or Neighbor-average correlation), falling back to the
manifest's official Global Moran's I-derived ground truth (already
present via `ground_truth_present`) only when the row's method has no
computed ground truth. `n_method_specific_gt` records how many chunks
per model used a method-matched comparison vs. the fallback.

A per-chunk VALUE-verification layer: for chunks that are verdict-
correct AND evidenced, the reported test statistic is compared against
the RESOLVED ground truth's statistic using a flat, ABSOLUTE 0.05
tolerance (not relative -- relative error is meaningless near zero, and
these statistics are all roughly bounded in a similar small range).
0.05 is reused directly from the bivariate task's Cross-Moran's I /
Lee's L derivation rather than re-derived from scratch, since Global
Moran's I is the same statistic family. This is applied uniformly
across all methods here as a deliberate simplification, per explicit
instruction -- Geary's C's different natural scale (centered at 1, not
0) and the correlation-based method's tighter true ambiguity are both
real caveats worth flagging in the thesis write-up rather than treated
as settled. `accuracy_fully_verified` is the fraction of chunks passing
ALL of: correct verdict, real evidence, AND a statistic value that
actually matches its claimed method.

NEW -- SHIFT-INVARIANCE RE-EXECUTION (the same mechanism used in
score_bivariate.py, added here so `accuracy_evidence_filtered` and
`f1_evidence_filtered` are process-based metrics, not just a stricter
outcome check): Global Moran's I, Geary's C, the Join Count statistic's
underlying values, and the neighbor-average correlation are all built
from deviations around a mean (or a binary split derived from the raw
values), so all are invariant to a constant additive shift applied to
every borough's change value. For each chunk with a real cited number,
the model's OWN saved code is re-executed on a SHIFTED copy of the
input data, and the recomputed statistic is compared against the
original -- interrogating the code itself, not just the number it
happened to output once. `has_evidence_bool` now additionally requires
this re-check to NOT have returned a confirmed failure (True or
inconclusive both pass; only a confirmed False disqualifies), same
tri-state handling as bivariate.

Note on p-values: the task prompt itself warns that permutation-based
p-values need >=500 shuffles to be stable, i.e. p-values carry real,
expected Monte Carlo noise even from a correct implementation. This
version does not add a separate p-value tolerance check -- only the
test STATISTIC is value-verified above -- so a correct statistic with
a p-value that landed just the wrong side of 0.05 purely from
permutation noise is still handled by the existing `verdict_p_consistent`
plausibility check, not by the new tolerance layer.

Usage:
    python score_changespillover.py
Reads:  <mode>_results_with_method.csv (must include `statistical_method`
        -- see attach_statistical_method_changespillover.py),
        ground_truth_all_methods_changespillover.csv, changespillover_manifest.csv
Writes: changespillover_scoring_summary.csv
"""

import re
import pandas as pd

from statistic_plausibility import magnitude_plausible, bounded_probability, verdict_p_consistent
from formula_invariance_check import check_shift_invariant

GROUND_TRUTH_METHODS_PATH = "ground truth/ground_truth_all_methods_changespillover.csv"
MANIFEST_PATH = "chunks_csv/manifest.csv"  # must have csv_path / adjacency_path per chunk_id


#Multi agent
RESULTS_CSV = "results/multi agent/changespillover_results_multiagent_with_method.csv"
OUT_CSV = "Scores/multi agent/changespillover_scoring_summary_multiagent.csv"

#Multi turn
#RESULTS_CSV = "results/single agent/changespillover_results_multi_with_method.csv"
#OUT_CSV = "Scores/multi turn/changespillover_scoring_summary_multi.csv"

#Single turn
#RESULTS_CSV = "changespillover_results_with_method.csv"
#OUT_CSV = "Scores/single turn/changespillover_scoring_summary.csv"

#ReAct multi turn
#RESULTS_CSV = "ReAct/changespillover_results_react.csv"
#OUT_CSV = "Scores/ReAct/changespillover_scoring_summary_ReAct.csv"

# Maps attach_statistical_method_changespillover.py's free-text labels
# onto the (fewer) method names that actually have ground truth
# computed. "Local Moran's I / LISA" and the generic "Permutation /
# Monte Carlo (unspecified statistic)" label both fall back to the
# manifest's Global Moran's I answer -- see the R script's docstring
# for why Local Moran's I specifically isn't computed for this task.
METHOD_NAME_MAP = {
    "Global Moran's I": "Global Moran's I",
    "Geary's C": "Geary's C",
    "Join Count Statistic": "Join Count Statistic",
    "Neighbor-average correlation (non-standard)": "Neighbor-average correlation (non-standard)",
}

# Flat absolute tolerance, per explicit instruction -- see module
# docstring for the reuse rationale and caveats.
VALUE_TOLERANCE = 0.05

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
    """Checks that the first '=...,' or '=...)' value in the text is a
    genuine, finite number -- method-agnostic, and rejects a number that
    LOOKS real if the model's own words admit it isn't."""
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
    """Second '=...' value -- the p-value, which always follows the
    primary statistic in this task's required format."""
    if not isinstance(evidence_text, str):
        return None
    matches = re.findall(r"=\s*([^,)]+)[,)]", evidence_text)
    return matches[1].strip() if len(matches) >= 2 else None


def _to_float(raw):
    if raw is None:
        return None
    try:
        f = float(str(raw).strip())
        return f if (f == f and abs(f) != float("inf")) else None
    except ValueError:
        return None


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
    """Returns (clustered_bool, gt_stat, source, method_used_for_tolerance).
    Falls back to the manifest's Global Moran's I answer -- note the
    manifest doesn't carry a single gt_stat column directly on the
    results rows (only ground_truth_present), so the fallback's gt_stat
    is None -- value-verification is simply skipped (not penalized) for
    fallback rows, same principle as "unverifiable, not wrong"
    elsewhere in this pipeline."""
    normalized_method = METHOD_NAME_MAP.get(statistical_method)
    if normalized_method is not None:
        key = (chunk_id, normalized_method)
        if key in gt_methods:
            gt = gt_methods[key]
            return gt["gt_clustered"], gt["gt_stat"], "method_specific", normalized_method
    return bool(manifest_clustered), None, "manifest_fallback", "Global Moran's I"


def value_within_tolerance(model_val, gt_val):
    m = _to_float(model_val)
    g = _to_float(gt_val)
    if m is None or g is None:
        return None
    return abs(m - g) <= VALUE_TOLERANCE


def _load_manifest_index():
    """Loads the manifest indexed by chunk_id, for the shift-invariance
    re-check. Degrades to an always-missing stand-in (shift check
    becomes inconclusive/None for every row, never blocking scoring) if
    the file isn't found."""
    try:
        return pd.read_csv(MANIFEST_PATH).set_index("chunk_id")
    except (FileNotFoundError, OSError):
        print(f"WARNING: manifest not found at {MANIFEST_PATH} -- "
              f"shift-invariance re-checks will be skipped (inconclusive) for every row.")
        return pd.DataFrame(columns=["csv_path", "adjacency_path"])


def load():
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
    df["gt_stat_used"] = resolved.apply(lambda t: t[1])
    df["ground_truth_source"] = resolved.apply(lambda t: t[2])
    df["gt_method_used"] = resolved.apply(lambda t: t[3])

    df["predicted_bool"] = df["predicted"].map({"True": True, "False": False, True: True, False: False})
    df["ambiguous"] = df["predicted_bool"].isna()
    df["has_real_statistic_bool"] = df.apply(
        lambda r: _has_real_statistic(r["evidence_text"], r.get("verdict_detail")), axis=1)

    total_checkable = int(df["has_real_statistic_bool"].sum())
    print(f"Running shift-invariance re-checks on {total_checkable} row(s) with a real statistic "
          f"(this re-executes each row's saved code, so it's the slow part)...", flush=True)
    _progress = {"n": 0}

    def _shift_check(row):
        # --- Shift-invariance re-execution (process-based check) ---
        # Global Moran's I / Geary's C / Join Count / neighbor-average
        # correlation are all invariant to a constant additive shift of
        # the CHANGE variable -- see module docstring. Shifting BOTH
        # year columns (the old default) cancels exactly in the
        # difference (year2-year1), making the check trivially pass
        # regardless of correctness -- so only the LATER year column is
        # shifted here, identified via the manifest's own window_end
        # field (the same field ground_truth_all_methods_changespillover.R
        # already uses to identify that column).
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
            return True  # already disqualified elsewhere -- avoid double-penalizing
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

    # Evidence requires ALL of: a real, non-fabricated number; the
    # formula behaving correctly under a constant shift (where the
    # re-check was conclusive); AND the statistic/p-value being
    # plausible and internally consistent with the stated verdict --
    # this is what makes accuracy_evidence_filtered/f1_evidence_filtered
    # process-based, not just a stricter outcome check.
    df["has_evidence_bool"] = (
        df["has_real_statistic_bool"]
        & (df["shift_invariant"] != False)  # noqa: E712
        & df["statistic_plausible"]
    )

    df["correct"] = (~df["ambiguous"]) & (df["predicted_bool"] == df["gt_clustered_used"])

    def _correct_ef_value(row):
        # Match compute_metrics()'s OWN lenient rule for
        # f1_evidence_filtered/precision_ef/recall_ef -- a NOT_CLUSTERED
        # (False) claim doesn't need real evidence to count, since
        # there's nothing to fabricate when claiming "nothing was
        # found"; only a CLUSTERED (True) claim does. Confirmed
        # necessary (found via the identical bug in score_spatial_results.py):
        # without this, correct_evidence_filtered used a STRICTER bar
        # than f1_evidence_filtered, which already exempts False
        # predictions via compute_metrics' keep_as_is logic -- causing
        # accuracy_evidence_filtered to read systematically below
        # f1_evidence_filtered for every model, not a real finding.
        if row["ambiguous"]:
            return False
        if row["predicted_bool"] == False and row["correct"]:  # noqa: E712
            return True
        return bool(row["correct"] and row["has_evidence_bool"])

    df["correct_evidence_filtered"] = df.apply(_correct_ef_value, axis=1)

    def _value_check(row):
        if not (row["correct"] and row["has_evidence_bool"]):
            return None
        if row["gt_stat_used"] is None:
            return None  # manifest fallback with no comparable stat -- unverifiable, not wrong
        raw_stat = _extract_raw_statistic(row["evidence_text"])
        return value_within_tolerance(raw_stat, row["gt_stat_used"])

    df["value_verified"] = df.apply(_value_check, axis=1)
    df["fully_verified"] = df["correct"] & df["has_evidence_bool"] & (df["value_verified"] == True)  # noqa: E712

    return df


def compute_metrics(g, use_evidence_filter):
    """precision/recall/F1 are 2-CLASS MACRO-averaged (CLUSTERED and
    NOT_CLUSTERED each treated as their own class, then averaged) --
    UNCHANGED from the previous version of this script. tp/fp returned
    are still the raw CLUSTERED-class counts (descriptive only);
    precision/recall/F1 no longer derive from them directly."""
    y_true = g["gt_clustered_used"].astype(bool)
    has_evidence = g["has_evidence_bool"].astype(bool)

    if use_evidence_filter:
        keep_as_is = g["predicted_bool"].isna() | (g["predicted_bool"] == False) | has_evidence  # noqa: E712
        y_pred = g["predicted_bool"].where(keep_as_is, None)
    else:
        y_pred = g["predicted_bool"]

    tp = int(((y_pred == True) & (y_true == True)).sum())  # noqa: E712
    fp = int(((y_pred == True) & (y_true == False)).sum())  # noqa: E712

    f1s, precisions, recalls = [], [], []
    for target_class in (True, False):
        n_real = int((y_true == target_class).sum())
        if n_real == 0:
            continue
        tp_c = int(((y_pred == target_class) & (y_true == target_class)).sum())
        fp_c = int(((y_pred == target_class) & (y_true != target_class)).sum())
        fn_c = int(((y_pred != target_class) & (y_true == target_class)).sum())
        p = tp_c / (tp_c + fp_c) if (tp_c + fp_c) > 0 else 0.0
        r = tp_c / (tp_c + fn_c) if (tp_c + fn_c) > 0 else 0.0
        f1_c = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        precisions.append(p)
        recalls.append(r)
        f1s.append(f1_c)

    precision = sum(precisions) / len(precisions) if precisions else 0.0
    recall = sum(recalls) / len(recalls) if recalls else 0.0
    f1 = sum(f1s) / len(f1s) if f1s else 0.0

    n_total = len(g)
    correct_col = "correct_evidence_filtered" if use_evidence_filter else "correct"
    accuracy = g[correct_col].sum() / n_total if n_total > 0 else 0.0
    return accuracy, precision, recall, f1, tp, fp


def run():
    df = load()
    rows = []

    for (mode, model), g in df.groupby(["mode", "model"]):
        n = len(g)
        n_ambiguous = int(g["ambiguous"].sum())
        non_ambiguous = n - n_ambiguous
        correct_number = int(g["correct"].sum())
        correct_number_ef = int(g["correct_evidence_filtered"].sum())
        pct_with_evidence = round(g["has_evidence_bool"].mean(), 3) if n else 0.0

        if n == 0:
            continue

        accuracy, precision, recall, f1, tp, fp = compute_metrics(g, use_evidence_filter=False)
        accuracy_ef, precision_ef, recall_ef, f1_ef, tp_ef, fp_ef = compute_metrics(g, use_evidence_filter=True)

        n_method_specific_gt = int((g["ground_truth_source"] == "method_specific").sum())
        n_fully_verified = int(g["fully_verified"].sum())
        accuracy_fully_verified = round(n_fully_verified / n, 3)

        rows.append({
            "mode": mode, "model": model, "n": n, "n_ambiguous": n_ambiguous,
            "non_ambiguous": non_ambiguous, "correct_number": correct_number,
            "pct_with_evidence": pct_with_evidence,
            "tp": tp, "fp": fp,
            "accuracy": f"{accuracy * 100:05.2f}", "precision": f"{precision * 100:05.2f}",
            "recall": f"{recall * 100:05.2f}", "f1": f"{f1 * 100:05.2f}",
            "correct_number_evidence_filtered": correct_number_ef,
            "tp_evidence_filtered": tp_ef, "fp_evidence_filtered": fp_ef,
            "accuracy_evidence_filtered": f"{accuracy_ef * 100:05.2f}",
            "precision_evidence_filtered": f"{precision_ef * 100:05.2f}",
            "recall_evidence_filtered": f"{recall_ef * 100:05.2f}",
            "f1_evidence_filtered": f"{f1_ef * 100:05.2f}",
            # the 2 columns beyond the original schema -- see docstring
            "n_method_specific_gt": n_method_specific_gt,
            "accuracy_fully_verified": accuracy_fully_verified,
        })

    out = pd.DataFrame(rows).sort_values(["mode", "model"]).reset_index(drop=True)
    out = out[["mode", "model", "n", "n_ambiguous", "non_ambiguous", "correct_number",
               "pct_with_evidence", "tp", "fp", "accuracy", "precision", "recall", "f1",
               "correct_number_evidence_filtered", "tp_evidence_filtered", "fp_evidence_filtered",
               "accuracy_evidence_filtered", "precision_evidence_filtered",
               "recall_evidence_filtered", "f1_evidence_filtered",
               "n_method_specific_gt", "accuracy_fully_verified"]]
    out.to_csv(OUT_CSV, index=False)

    print(out.to_string(index=False))
    print(f"\nSaved to {OUT_CSV}")


if __name__ == "__main__":
    run()
