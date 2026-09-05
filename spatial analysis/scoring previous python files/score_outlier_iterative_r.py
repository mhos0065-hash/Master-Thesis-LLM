"""
score_outlier_iterative_r.py
===============================
Scores the iterative-r spatial outlier task's 3-way classification
(outlier_high / outlier_low / not_outlier) with the SAME column
structure used for bivariate, distance-decay, and the original Moran's I
task, so all tables are directly comparable.

Evidence check (UNCHANGED from the previous version of this script --
these are real fixes to real bugs found by inspecting actual logs, not
new logic):
  - `verdict_z_consistent`: a claimed verdict must be justified by the
    model's OWN reported value, but a value is only treated as a
    z-score or p-value when its OWN label says so (not guessed from
    magnitude). This fixed a real false positive where "z = 0.031" was
    misread as a significant p-value purely because 0.031 < 0.05.
  - `outlier_magnitude_plausible`: a task-specific magnitude ceiling
    (12.0), set at ~2x the real maximum |ground_truth_z_score| observed
    across the 50-chunk manifest (5.5441). This caught a real case
    (gpt-4o-mini, outlierr_2021_energy_city_of_london) where genuine,
    non-fabricated code omitted the masking-bias correction the prompt
    asks for, producing z=30.41 against a true value of 5.54 -- a
    directionally-correct but numerically nonsensical answer that the
    prior (pre-this-script) scorer had been crediting as fully correct.

NEW in this version, layered on top of the above (not replacing it):

1. VERDICT correctness is now resolved PER ROW against the ground
   truth for the SPECIFIC method that row used (from
   ground_truth_all_methods_outlierr.csv: Single-pass OLS regression,
   Robust regression, Neighbor mean/std z-score, or Local Moran's
   I/LISA), falling back to the manifest's official Iterative-R-
   derived ground truth (already present in the `ground_truth_direction`
   column) only when the row's method has no computed ground truth.
   `n_method_specific_gt` records how many chunks per model used a
   method-matched comparison vs. the fallback.

2. A per-chunk VALUE-verification layer: for chunks that are verdict-
   correct AND evidenced, the reported z-value is compared against the
   RESOLVED ground truth's value using a METHOD-SPECIFIC ABSOLUTE
   tolerance (not relative -- same principle used for the getisord
   task, where relative tolerance was shown to blow up meaninglessly
   for values near zero):
       Iterative-R, Single-pass regression, Robust regression:  0.1
       Neighbor mean/std z-score:                                0.05
       Local Moran's I / LISA:                                   1.5
   The regression-based tolerance (0.1) is empirically derived: a
   simulation compared standardized vs. leverage-adjusted (studentized)
   residual z-scores on the SAME synthetic regression fits (33-node
   network, 3000 trials) -- the two legitimate ways to turn a residual
   into a z-score differ by a median of 0.017, p95 of 0.073; 0.1 covers
   that with a small safety margin. Neighbor mean/std gets a tighter
   0.05 since that formula (a plain mean/sd comparison) has no such
   fitting-choice ambiguity. Local Moran's I reuses the tolerance
   derived for the getisord task (same statistic, same analytic-vs-
   permutation variance-estimation ambiguity). `accuracy_fully_verified`
   is the fraction of chunks passing ALL THREE bars: correct verdict,
   real evidence, AND a number that actually matches its claimed
   method -- the strictest, hardest-to-game metric in the table.

Usage:
    python score_outlier_iterative_r.py
Reads:  <mode>_results_with_method.csv (must include `statistical_method`
        -- see attach_statistical_method_outlierr.py),
        ground_truth_all_methods_outlierr.csv
Writes: outlier_iterative_r_scoring_summary.csv
"""

import re
import pandas as pd

from statistic_plausibility import magnitude_plausible

GROUND_TRUTH_METHODS_PATH = "ground truth/ground_truth_all_methods_outlierr.csv"

# Multi agent
#RESULTS_CSV = "results/multi agent/outlier_iterative_r_results_multiagent_with_method.csv"
#OUT_CSV = "Scores/multi agent/outlier_iterative_r_scoring_summary_multiagent.csv"

# Single turn
#RESULTS_CSV = "outlier_iterative_r_results_with_method.csv"
#OUT_CSV = "Scores/single turn/outlier_iterative_r_scoring_summary.csv"

# Multi turn
#RESULTS_CSV = "results/single agent/outlier_iterative_r_results_multi_with_method.csv"
#OUT_CSV = "Scores/multi turn/outlier_iterative_r_scoring_summary_multi.csv"

#ReAct multi agent
RESULTS_CSV = "ReAct/outlier_iterative_r_results_react.csv"
OUT_CSV = "Scores/ReAct/outlier_iterative_r_scoring_summary_ReAct.csv"


CLASSES = ["outlier_high", "outlier_low", "not_outlier"]

# Maps attach_statistical_method_outlierr.py's free-text labels onto the
# (fewer) method names that actually have ground truth computed. Anything
# not listed here (Getis-Ord, Mahalanobis, IQR, MAD, "none", or a bare
# generic z-score) falls back to the manifest's Iterative-R answer.
METHOD_NAME_MAP = {
    "Iterative-R (Lu, Chen & Kou 2003)": "Iterative-R (Lu, Chen & Kou 2003)",
    "Single-pass OLS regression (no masking correction)": "Single-pass OLS regression (no masking correction)",
    "Robust regression (Huber/RANSAC/Theil-Sen)": "Robust regression (Huber/RANSAC/Theil-Sen)",
    "Neighbor mean/std z-score (non-regression)": "Neighbor mean/std z-score (non-regression)",
    "Local Moran's I / LISA": "Local Moran's I / LISA",
}

# ABSOLUTE, method-specific tolerances -- see module docstring for
# the empirical derivation of these figures.
METHOD_TOLERANCE = {
    "Iterative-R (Lu, Chen & Kou 2003)": 0.1,
    "Single-pass OLS regression (no masking correction)": 0.1,
    "Robust regression (Huber/RANSAC/Theil-Sen)": 0.1,
    "Neighbor mean/std z-score (non-regression)": 0.05,
    "Local Moran's I / LISA": 1.5,
}
DEFAULT_TOLERANCE = 0.1  # used for the manifest_fallback source

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


def _to_float(raw):
    if raw is None:
        return None
    try:
        f = float(str(raw).strip())
        return f if (f == f and abs(f) != float("inf")) else None
    except ValueError:
        return None


def verdict_z_consistent(predicted, evidence_text, threshold=1.96, alpha=0.05):
    """A verdict must be justified by the model's OWN reported value --
    but ONLY when we can be confident what kind of value it is, based
    on its own label, not just its magnitude.

    Two real issues found and fixed here (both discovered via direct
    inspection of real evidence_text this session):

    1. A value in [0,1] is now treated as a p-value ONLY if its own
       label says so (e.g. "p =", "p_value =", "significance ="). A
       genuinely small, correctly labeled z-score near zero (e.g.
       "z = 0.031", outlierr_2023_crime_hillingdon, seen identically
       from both gpt-4o-mini and deepseek-r1-0528) was previously
       misread as a "significant p-value" purely because 0.031 < 0.05
       -- a false positive with nothing to do with the model's actual
       reasoning.

    2. The 1.96 threshold is now applied ONLY when the label confidently
       claims to be a standard z-score equivalent (z, z-score,
       standardized/std residual). The task's own prompt explicitly
       permits other methods ("z is just an example label -- if your
       method produces a different statistic, use its name instead"),
       and labels actually observed this session include t, R, M, r,
       robust_z, iterative_r -- a t-statistic has a different critical
       value depending on degrees of freedom, robust_z is often
       MAD-scaled rather than standard-normal, and R/M/r are ambiguous.
       Applying 1.96 uniformly to all of these would risk false
       positives the same way issue #1 did. For any label outside the
       confident z-score set, this check is skipped (returns True) --
       trading recall for precision, since a false accusation is worse
       than a missed one when this feeds a supervisor-facing analysis.

    Also checks sign consistency: outlier_high requires a positive
    value, outlier_low requires a negative one -- inherent to what
    those verdicts mean, not an invented rule.

    If no real number is present at all, this check is skipped
    (already disqualified elsewhere by the has_real_statistic check)."""
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
        return True  # unrecognized/ambiguous statistic (t, R, M, r, robust_z, etc.) -- skip, don't guess

    if is_confident_p:
        is_significant = abs(f) < alpha
        direction_positive = None  # a p-value alone carries no direction
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
    return True  # ambiguous/unrecognized verdict -- handled elsewhere


def outlier_magnitude_plausible(raw_value, threshold=12.0):
    """Task-specific magnitude check for the Iterative-R outlier task,
    tighter than the generic magnitude_plausible() (which uses a
    method-agnostic 1000 cutoff, far too loose to be useful here).

    Threshold is grounded directly in this task's own real ground truth,
    not chosen to produce any particular scoring outcome: the maximum
    |ground_truth_z_score| across all 50 chunks in
    outlier_iterative_r_manifest.csv is 5.5441 (outlierr_2023_gas_city_of_london).
    12.0 is a generous ~2x margin above that observed maximum -- same
    "generous 2x margin above real max" principle already used for
    Getis-Ord's IMPLAUSIBLE_THRESHOLD=10 in score_getisord.py, applied
    consistently here rather than invented fresh.

    NOTE: this stays a single, generic threshold regardless of which
    method a row used -- it's a coarse sanity filter for catching
    grossly broken computations (like the masking-bias-omission bug
    below), not the fine-grained per-method comparison that's a
    separate, stricter check (see accuracy_fully_verified).

    Motivation: direct inspection of real conversation logs this session
    (openai/gpt-4o-mini, single-turn, outlierr_2021_energy_city_of_london)
    showed a genuine, real pandas computation -- no fabrication -- that
    omitted the masking-bias correction the prompt explicitly asks for,
    producing z=30.41 against a ground truth of 5.5435. The verdict
    (OUTLIER-HIGH) was directionally correct and so counted as fully
    correct and evidenced under the prior scorer, despite the magnitude
    being off by more than 5x. This check does not penalize direction
    (already scored by `correct`); it only withholds evidence credit
    when the model's own cited number is implausibly large for this
    specific task, regardless of which mode or model produced it."""
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
    """Returns (direction, z, source, method_used_for_tolerance): the
    method-specific ground truth if computed, else the manifest's
    Iterative-R answer (already present on every row)."""
    normalized_method = METHOD_NAME_MAP.get(statistical_method)
    if normalized_method is not None:
        key = (chunk_id, normalized_method)
        if key in gt_methods:
            gt = gt_methods[key]
            return gt["gt_direction"], gt["gt_z"], "method_specific", normalized_method
    return manifest_direction, manifest_z, "manifest_fallback", "Iterative-R (Lu, Chen & Kou 2003)"


def value_within_tolerance(model_val, gt_val, method_for_tolerance):
    """ABSOLUTE tolerance check (not relative -- see module docstring)."""
    m = _to_float(model_val)
    g = _to_float(gt_val)
    if m is None or g is None:
        return None
    tol = METHOD_TOLERANCE.get(method_for_tolerance, DEFAULT_TOLERANCE)
    return abs(m - g) <= tol


def load():
    df = pd.read_csv(RESULTS_CSV)

    if "statistical_method" not in df.columns:
        raise ValueError(
            f"'{RESULTS_CSV}' has no `statistical_method` column. "
            "Run attach_statistical_method_outlierr.py first and point "
            "RESULTS_CSV at its '_with_method' output."
        )

    gt_methods = load_ground_truth_methods()
    resolved = df.apply(
        lambda r: resolve_ground_truth(
            r["chunk_id"], r.get("statistical_method"),
            r["ground_truth_direction"], r["ground_truth_z_score"], gt_methods),
        axis=1)
    df["gt_direction_used"] = resolved.apply(lambda t: t[0])
    df["gt_z_used"] = resolved.apply(lambda t: t[1])
    df["ground_truth_source"] = resolved.apply(lambda t: t[2])
    df["gt_method_used"] = resolved.apply(lambda t: t[3])

    df["has_real_statistic_bool"] = df.apply(
        lambda r: _has_real_statistic(r["evidence_text"], r.get("verdict_detail")), axis=1)

    def _plausibility_check(row):
        if not row["has_real_statistic_bool"]:
            return True  # already disqualified elsewhere -- avoid double-penalizing
        raw = _extract_raw_statistic(row["evidence_text"])
        if not magnitude_plausible(raw):
            return False
        if not outlier_magnitude_plausible(raw):
            return False
        if not verdict_z_consistent(row["predicted"], row["evidence_text"]):
            return False
        return True

    df["statistic_plausible"] = df.apply(_plausibility_check, axis=1)
    df["has_evidence_bool"] = df["has_real_statistic_bool"] & df["statistic_plausible"]
    df["ambiguous"] = df["predicted"].isna() | (~df["predicted"].isin(CLASSES))
    df["correct"] = (~df["ambiguous"]) & (df["predicted"] == df["gt_direction_used"])
    df["correct_evidence_filtered"] = df["correct"] & df["has_evidence_bool"]

    def _value_check(row):
        if not (row["correct"] and row["has_evidence_bool"]):
            return None
        raw = _extract_raw_statistic(row["evidence_text"])
        return value_within_tolerance(raw, row["gt_z_used"], row["gt_method_used"])

    df["value_verified"] = df.apply(_value_check, axis=1)
    df["fully_verified"] = df["correct"] & df["has_evidence_bool"] & (df["value_verified"] == True)  # noqa: E712

    return df


def per_class_metrics(g, use_evidence_filter):
    y_true = g["gt_direction_used"]

    def effective_pred(r):
        if r["ambiguous"]:
            return "__no_answer__"
        if use_evidence_filter and not r["has_evidence_bool"]:
            return "__unevidenced__"
        return r["predicted"]

    y_pred = g.apply(effective_pred, axis=1)

    tp_total = fp_total = 0
    precisions, recalls, f1_scores = [], [], []
    for cls in CLASSES:
        n_real = int((y_true == cls).sum())
        if n_real == 0:
            # class never occurs in this group's real ground truth -- excluded
            # from the macro average entirely (e.g. outlier_low occurred only
            # 2/50 times in the real data -- rare occurrences like this also
            # deserve a documented caveat about metric stability, but a
            # ZERO-occurrence class specifically should never silently
            # contribute a 0.0 that penalizes an otherwise-correct model)
            continue

        tp = int(((y_pred == cls) & (y_true == cls)).sum())
        fp = int(((y_pred == cls) & (y_true != cls)).sum())
        fn = int(((y_pred != cls) & (y_true == cls)).sum())
        tp_total += tp
        fp_total += fp

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)

    macro_precision = round(sum(precisions) / len(precisions), 3) if precisions else 0.0
    macro_recall = round(sum(recalls) / len(recalls), 3) if recalls else 0.0
    macro_f1 = round(sum(f1_scores) / len(f1_scores), 3) if f1_scores else 0.0
    return tp_total, fp_total, macro_precision, macro_recall, macro_f1


def run():
    df = load()
    rows = []

    for (mode, model), g in df.groupby(["mode", "model"]):
        n = len(g)
        n_ambiguous = int(g["ambiguous"].sum())
        non_ambiguous = n - n_ambiguous
        correct_number = int(g["correct"].sum())
        correct_number_ef = int(g["correct_evidence_filtered"].sum())
        pct_with_evidence = round(g["has_evidence_bool"].mean(), 3)

        if n == 0:
            continue

        accuracy = round(correct_number / n, 3)
        accuracy_ef = round(correct_number_ef / n, 3)

        tp, fp, precision, recall, f1 = per_class_metrics(g, use_evidence_filter=False)
        tp_ef, fp_ef, precision_ef, recall_ef, f1_ef = per_class_metrics(g, use_evidence_filter=True)

        n_method_specific_gt = int((g["ground_truth_source"] == "method_specific").sum())
        n_fully_verified = int(g["fully_verified"].sum())
        accuracy_fully_verified = round(n_fully_verified / n, 3)

        rows.append({
            "mode": mode, "model": model, "n": n, "n_ambiguous": n_ambiguous,
            "non_ambiguous": non_ambiguous, "correct_number": correct_number,
            "pct_with_evidence": pct_with_evidence,
            "tp": tp, "fp": fp,
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "correct_number_evidence_filtered": correct_number_ef,
            "tp_evidence_filtered": tp_ef, "fp_evidence_filtered": fp_ef,
            "accuracy_evidence_filtered": accuracy_ef,
            "precision_evidence_filtered": precision_ef,
            "recall_evidence_filtered": recall_ef,
            "f1_evidence_filtered": f1_ef,
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
