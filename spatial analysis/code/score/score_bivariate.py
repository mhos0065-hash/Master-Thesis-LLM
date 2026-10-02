"""
score_bivariate.py
=====================
Scores the bivariate spatial correlation task's 3-way classification
(positive / negative / not_significant) -- NOT a simple binary accuracy
check, since there are 3 classes and "wrong" can mean different things
(predicting positive when it's actually negative is a different mistake
than predicting not_significant).

VERDICT correctness (positive/negative/not_significant) is now judged
PER ROW against the ground truth for the SPECIFIC statistical method
that row actually used (from ground_truth_all_methods.csv), not always
against Lee's L. If a row used Pearson correlation, it's scored against
what Pearson correlation actually says about that chunk -- not against
Lee's L, which is a different (if related) statistic. This separates
"did the model execute/interpret its own chosen method correctly" from
"did the model choose the right method for a spatial question", which
were previously conflated by always scoring against Lee's L.

If a row's method has no computed ground truth (e.g. Geary's C, join
count, LISA, Mantel, Spearman/Kendall, chi-square, t-test, regression,
z-test, or a bare "permutation test" label -- see METHOD_NAME_MAP),
it FALLS BACK to the manifest's Lee's L-derived ground truth, since
that's the only ground truth guaranteed to exist for every chunk. Which
source was used for each row is recorded in `ground_truth_source`
("method_specific" vs "manifest_fallback") so this is fully auditable,
not silent.

A secondary, INFORMATIONAL-ONLY column `correct_vs_manifest` is also
kept: this always scores against Lee's L regardless of method, i.e. "did
the model answer the officially-posed question correctly". This is a
different, also-legitimate question from the primary `correct` column
and both are reported separately rather than merged.

NEW in this version: a value-verification layer. For rows that are
(a) verdict-correct (against whichever ground truth source was used)
AND (b) have real evidence (a genuine, non-fabricated statistic), we
additionally check whether the model's reported NUMBER is close enough
to the ground-truth value it was scored against. This catches the case
where a model gets the right VERDICT (possibly by luck, or from a
qualitatively-right-but-numerically-broken calculation) while reporting
a statistic value that doesn't actually match what its own chosen
method should produce.

TOLERANCE is now METHOD-SPECIFIC and ABSOLUTE, not a flat relative
percentage (see METHOD_TOLERANCE below). This was empirically derived,
not guessed: a simulation (spatial data on a 33-node Delaunay-
triangulated network, structurally comparable to the real London
borough adjacency graph, 3000 trials varying true correlation and
spatial-autocorrelation strength) measured how much Lee's L and
Cross-Moran's I actually differ between two legitimate weight-matrix
conventions (row-standardized vs. binary weights) on the SAME data.

Key finding: RELATIVE tolerance is the wrong metric for these
statistics -- because they're often close to zero, relative error
blows up meaninglessly (e.g. 0.01 vs 0.02 is "100% relative
difference" but a trivial absolute gap). Absolute differences were far
more stable:
    Lee's L:          median abs diff = 0.012, p90 = 0.033, p95 = 0.044
    Cross-Moran's I:  median abs diff = 0.012, p90 = 0.031, p95 = 0.037
An absolute tolerance of 0.05 covers roughly the 95th percentile of
legitimate convention-driven variation observed in simulation, while
remaining far tighter than the multi-hundred to multi-billion-magnitude
gaps actually seen in broken runs. This simulation only varied
row-standardized vs. binary weights -- real implementations could also
differ in neighbor definition (queen vs. rook, distance-based), so 0.05
is a reasonable, evidence-based default rather than a proven upper
bound; re-running the simulation with the actual borough adjacency
structure (rather than a synthetic stand-in) would tighten this further
if you want a fully data-specific number.

Pearson correlation keeps a much tighter tolerance (0.01) since it has
no such convention-driven ambiguity -- cor(x,y) is cor(x,y); the small
window is a floating-point safety margin, not an expected range.

Produces THREE outputs:
  1. bivariate_scoring_summary.csv  -- per-model accuracy + per-class
     precision/recall/F1 (one-vs-rest) + macro-F1, raw AND
     evidence-filtered, PLUS the new value-verification metrics.
  2. bivariate_confusion_matrix.csv -- per-model 3x3 confusion matrix
     (rows = real ground truth, columns = what the model predicted).
  3. Printed summary to console.

Usage:
    python score_bivariate.py
Reads:  <mode>_results_with_method.csv (must include a
        `statistical_method` column -- see attach_statistical_method.py),
        bivariate_manifest.csv, ground_truth_all_methods.csv
Writes: bivariate_scoring_summary.csv, bivariate_confusion_matrix.csv
"""

import re
import pandas as pd

from bivariate_verdict import _has_self_admitted_non_computation
from formula_invariance_check import check_shift_invariant
from statistic_plausibility import magnitude_plausible, bounded_probability, verdict_p_consistent

MANIFEST_PATH = "chunks_csv/bivariate/bivariate_manifest.csv"
GROUND_TRUTH_METHODS_PATH = "ground truth/ground_truth_all_methods_bivariate.csv"

# For single turn
#RESULTS_CSV = "bivariate_results_with_method.csv"
#SUMMARY_OUT = "Scores/single turn/bivariate_scoring_summary.csv"
#CONFUSION_OUT = "Scores/single turn/bivariate_confusion_matrix.csv"

# For multi turn
#RESULTS_CSV = "results/single agent/bivariate_results_multi_with_method.csv"
#SUMMARY_OUT = "Scores/multi turn/bivariate_scoring_summary_multi.csv"
#CONFUSION_OUT = "Scores/multi turn/bivariate_confusion_matrix_multi.csv"

# For multi agent
#RESULTS_CSV = "results/multi agent/bivariate_results_multiagent_with_method.csv"
#SUMMARY_OUT = "Scores/multi agent/bivariate_scoring_summary_multiagent.csv"
#CONFUSION_OUT = "Scores/multi agent/bivariate_confusion_matrix_multiagent.csv"

# For ReAct agent
RESULTS_CSV = "ReAct/bivariate_results_react.csv"
SUMMARY_OUT = "Scores/ReAct/bivariate_scoring_summary_ReAct.csv"
CONFUSION_OUT = "Scores/ReAct/bivariate_confusion_matrix_ReAct.csv"


CLASSES = ["positive", "negative", "not_significant"]

# Maps the free-text labels produced by attach_statistical_method.py's
# keyword scan onto the (fewer) method names that actually have a
# computed ground truth in ground_truth_all_methods.csv. Anything not
# listed here (Geary's C, join count, LISA, Getis-Ord, Mantel, Spearman,
# Kendall, chi-square, t-test, regression, permutation/Monte Carlo as a
# bare label, z-test, "none") has NO ground truth to check the value
# against -- those rows fall through to "unverifiable", not "wrong".
METHOD_NAME_MAP = {
    "Lee's L": "Lee's L",
    "Bivariate / Cross Moran's I": "Bivariate / Cross Moran's I",
    "Moran's I": "Moran's I (X only, ref.)",
    "Pearson correlation": "Pearson correlation",
    "Generic Pearson (np.corrcoef)": "Pearson correlation",
}

# Method-specific, ABSOLUTE tolerances. See the module docstring's
# TOLERANCE section for the simulation these are derived from.
METHOD_TOLERANCE = {
    "Lee's L": 0.05,
    "Bivariate / Cross Moran's I": 0.05,
    "Moran's I (X only, ref.)": 0.05,
    "Pearson correlation": 0.01,
}
DEFAULT_TOLERANCE = 0.05  # fallback for any method not listed above


def _has_real_test_statistic(evidence_text, full_text=None):
    """Checks that the TEST STATISTIC specifically (the first '=...,' or
    '=...)' in the answer line -- not the p-value, which comes second) is
    a genuine, finite number -- never nan, never inf, and never a method
    name like 'Lee's L' left in place of a real value (the exact bug
    found earlier this session). Method-agnostic: doesn't care what label
    precedes the '=', or which statistical method was used, only that a
    real number was actually computed and cited for it.

    ALSO rejects a number that LOOKS real if the model's own words admit
    it isn't -- e.g. "test statistic = 10.5 ... please replace the values
    with the actual observed statistic" is a real observed case: the
    number-parsing check alone would incorrectly accept 10.5 as valid
    evidence, missing the model's own admission that it's a placeholder."""
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
        return f == f and abs(f) != float("inf")  # excludes nan and inf
    except ValueError:
        return False  # e.g. a method name like "Lee's L" left in place of a number


def _extract_raw_statistic(evidence_text):
    """Same extraction logic as _has_real_test_statistic, isolated so the
    raw value can also be passed into the shift-invariance re-check and
    the method-specific ground-truth value comparison."""
    if not isinstance(evidence_text, str):
        return None
    match = re.search(r"=\s*([^,)]+)[,)]", evidence_text)
    return match.group(1).strip() if match else None


def _extract_p_value(evidence_text):
    """Extracts the SECOND '=...' value in the answer line -- the p-value
    / significance measure, which always comes after the primary test
    statistic in this task's required format. Returns None if there is
    no second value (some correct answers omit a secondary measure)."""
    if not isinstance(evidence_text, str):
        return None
    matches = re.findall(r"=\s*([^,)]+)[,)]", evidence_text)
    return matches[1].strip() if len(matches) >= 2 else None


def value_within_tolerance(model_val, gt_val, normalized_method):
    """Compares a model's reported statistic against the ground-truth
    value for the SAME method, using a method-specific ABSOLUTE
    tolerance (see METHOD_TOLERANCE / module docstring for how these
    were derived). Returns True/False, or None if either value isn't a
    usable number (can't be checked, not the same as "wrong")."""
    try:
        m = float(model_val)
        g = float(gt_val)
    except (TypeError, ValueError):
        return None
    if m != m or g != g:  # NaN
        return None
    if abs(m) == float("inf") or abs(g) == float("inf"):
        return None
    tol = METHOD_TOLERANCE.get(normalized_method, DEFAULT_TOLERANCE)
    return abs(m - g) <= tol


def load_ground_truth_methods():
    """Loads ground_truth_all_methods.csv into a {(chunk_id, method): row}
    lookup dict for fast per-row access."""
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
    """Same rule the manifest itself uses: direction is the sign of the
    statistic, but ONLY when the result is significant; otherwise the
    verdict is not_significant regardless of sign. Applied uniformly to
    every method so the derivation is consistent across the table."""
    try:
        s = float(stat)
    except (TypeError, ValueError):
        return "not_significant"
    if not significant or s != s:  # not significant, or stat is NaN
        return "not_significant"
    return "positive" if s > 0 else "negative"


def _resolve_ground_truth(row, gt_methods, manifest):
    """Returns the ground-truth entry to score THIS row against: the
    entry for the specific method it used, if we have one computed;
    otherwise falls back to the manifest's Lee's L-derived ground truth
    (the only ground truth guaranteed to exist for every chunk). Returns
    None only if the chunk_id itself isn't found anywhere."""
    normalized_method = METHOD_NAME_MAP.get(row.get("statistical_method"))
    if normalized_method is not None:
        gt = gt_methods.get((row["chunk_id"], normalized_method))
        if gt is not None:
            return {
                "direction": _direction_from_stat(gt["gt_stat"], gt["gt_significant"]),
                "stat": gt["gt_stat"],
                "p": gt["gt_p"],
                "significant": gt["gt_significant"],
                "source": "method_specific",
                "method": normalized_method,
            }

    if row["chunk_id"] in manifest.index:
        m = manifest.loc[row["chunk_id"]]
        return {
            "direction": m["ground_truth_direction"],
            "stat": m["ground_truth_lee_l"],
            "p": m["ground_truth_p_value"],
            "significant": str(m["ground_truth_is_significant"]).strip().upper() == "TRUE",
            "source": "manifest_fallback",
            "method": "Lee's L",
        }

    return None


def load():
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
    print(f"Running shift-invariance re-checks on {total_checkable} row(s) with a real statistic "
          f"(this re-executes each row's saved code, so it's the slow part)...", flush=True)
    _progress = {"n": 0}

    def _shift_check(row):
        if not row["has_real_statistic_bool"]:
            return None  # already disqualified -- no need to also re-run the code
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
            return True  # already disqualified elsewhere -- avoid double-penalizing
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

    # Evidence requires ALL of: a real, non-fabricated number; the
    # formula behaving correctly under a constant shift (where the
    # re-check was conclusive); AND the statistic/p-value being
    # plausible and internally consistent with the stated verdict.
    df["has_evidence_bool"] = (
        df["has_real_statistic_bool"]
        & (df["shift_invariant"] != False)
        & df["statistic_plausible"]
    )

    # --- Resolve which ground truth each row is scored against --------
    resolved = df.apply(lambda r: _resolve_ground_truth(r, gt_methods, manifest), axis=1)
    df["gt_direction_used"] = resolved.apply(lambda d: d["direction"] if d else None)
    df["gt_stat_used"] = resolved.apply(lambda d: d["stat"] if d else None)
    df["gt_method_used"] = resolved.apply(lambda d: d["method"] if d else None)
    df["ground_truth_source"] = resolved.apply(lambda d: d["source"] if d else "missing")

    df["ambiguous"] = df["predicted"].isna() | (~df["predicted"].isin(CLASSES))

    # PRIMARY: scored against the method the row actually used (with the
    # manifest_fallback above for methods we have no ground truth for).
    df["correct"] = (~df["ambiguous"]) & (df["predicted"] == df["gt_direction_used"])
    df["correct_evidence_filtered"] = df["correct"] & df["has_evidence_bool"]

    # SECONDARY / informational only: "did it answer the officially-posed
    # question correctly", always against Lee's L regardless of method.
    # Deliberately NOT used for accuracy/F1/confusion-matrix below -- a
    # separate, clearly-labeled column so the two questions aren't merged.
    df["correct_vs_manifest"] = (~df["ambiguous"]) & (df["predicted"] == df["ground_truth_direction"])

    # --- Value verification -----------------------------------------
    # Only checked for rows that are already verdict-correct (against
    # whichever ground truth source was used above) AND evidenced.
    def _value_check(row):
        if not (row["correct"] and row["has_evidence_bool"]):
            return None
        if row["gt_stat_used"] is None:
            return None
        raw_stat = _extract_raw_statistic(row["evidence_text"])
        return value_within_tolerance(raw_stat, row["gt_stat_used"], row["gt_method_used"])

    df["value_verified"] = df.apply(_value_check, axis=1)
    # value_verified is True / False / None(=unverifiable: no ground
    # truth exists for that method, so we can't confirm OR deny it)
    df["value_verifiable"] = df["value_verified"].notna()
    df["value_matched"] = df["value_verified"] == True  # noqa: E712

    # Strict: an unverifiable value does NOT count as fully verified.
    df["fully_verified_strict"] = df["correct"] & df["has_evidence_bool"] & df["value_matched"]
    # Lenient: an unverifiable value doesn't count AGAINST the row either
    # -- only an actual value MISMATCH downgrades it.
    df["fully_verified_lenient"] = (
        df["correct"] & df["has_evidence_bool"] & (df["value_verified"] != False)  # noqa: E712
    )

    return df


def per_class_metrics(g, use_evidence_filter):
    """Returns (per_class dict, macro_precision, macro_recall, macro_f1).
    Macro = unweighted average across the 3 classes -- deliberately NOT
    weighted by class frequency, since not_significant dominates this
    dataset (39/50 chunks) and a weighted average would let a model hide
    poor positive/negative detection behind good majority-class guessing.
    g is the FULL group (including ambiguous rows) -- an ambiguous row
    never matches any real class (so it can't be a TP or FP), but it DOES
    count as a false negative for whatever the true class was, the same
    treatment as evidence-filtering's '__unevidenced__' substitution.

    Uses `gt_direction_used` (per-row, method-matched ground truth, with
    manifest fallback) to stay consistent with the primary `correct`
    column -- NOT the always-Lee's-L `ground_truth_direction` column."""
    y_true = g["gt_direction_used"]

    def effective_pred(r):
        if r["ambiguous"]:
            return "__no_answer__"
        if use_evidence_filter and not r["has_evidence_bool"]:
            return "__unevidenced__"
        return r["predicted"]

    y_pred = g.apply(effective_pred, axis=1)

    per_class = {}
    precisions, recalls, f1_scores = [], [], []
    for cls in CLASSES:
        tp = int(((y_pred == cls) & (y_true == cls)).sum())
        fp = int(((y_pred == cls) & (y_true != cls)).sum())
        fn = int(((y_pred != cls) & (y_true == cls)).sum())

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        n_real = int((y_true == cls).sum())
        if n_real > 0:
            # only classes with at least one real ground-truth occurrence
            # count toward the macro average -- excluding a structurally
            # absent class prevents it from dragging F1 down for a model
            # that's genuinely correct on every class that actually exists
            # (confirmed concrete case: gpt-4o-mini scored 100% accuracy on
            # distance-decay but F1=0.667, purely from an empty third class)
            precisions.append(precision)
            recalls.append(recall)
            f1_scores.append(f1)

        per_class[cls] = {"precision": round(precision, 3), "recall": round(recall, 3),
                           "f1": round(f1, 3), "tp": tp, "fp": fp, "fn": fn}

    macro_precision = sum(precisions) / len(precisions) if precisions else 0.0
    macro_recall = sum(recalls) / len(recalls) if recalls else 0.0
    macro_f1 = sum(f1_scores) / len(f1_scores) if f1_scores else 0.0
    return per_class, macro_precision, macro_recall, macro_f1


def confusion_matrix(g_clear, model, mode):
    # gt_direction_used, not ground_truth_direction -- see per_class_metrics
    y_true = g_clear["gt_direction_used"]
    y_pred = g_clear["predicted"]
    rows = []
    for true_cls in CLASSES:
        row = {"model": model, "mode": mode, "true_class": true_cls}
        subset = y_pred[y_true == true_cls]
        for pred_cls in CLASSES:
            row[f"predicted_{pred_cls}"] = int((subset == pred_cls).sum())
        row["n_total"] = len(subset)
        rows.append(row)
    return rows


def run():
    df = load()
    summary_rows = []
    confusion_rows = []

    for (mode, model), g in df.groupby(["mode", "model"]):
        g_clear = g[~g["ambiguous"]]  # confusion matrix still only meaningful for real predictions
        n = len(g)
        n_ambiguous = int(g["ambiguous"].sum())
        non_ambiguous = n - n_ambiguous
        correct_number = int(g["correct"].sum())
        correct_number_ef = int(g["correct_evidence_filtered"].sum())
        pct_with_evidence = round(g["has_evidence_bool"].mean(), 3)

        if n == 0:
            continue

        # accuracy's denominator is the FULL count -- an ambiguous chunk is
        # never "correct" just because it was excluded from consideration
        accuracy = correct_number / n
        accuracy_ef = correct_number_ef / n

        per_class, precision, recall, f1 = per_class_metrics(g, use_evidence_filter=False)
        per_class_ef, precision_ef, recall_ef, f1_ef = per_class_metrics(g, use_evidence_filter=True)

        # micro-aggregated TP/FP: sum the per-class confusion counts across
        # all 3 classes -- TP_total ends up equal to correct_number (every
        # correct classification is a TP for exactly one class), FP_total
        # is every wrong classification (each miss is a false positive for
        # whichever class it was wrongly assigned to)
        tp = sum(per_class[c]["tp"] for c in CLASSES)
        fp = sum(per_class[c]["fp"] for c in CLASSES)
        tp_ef = sum(per_class_ef[c]["tp"] for c in CLASSES)
        fp_ef = sum(per_class_ef[c]["fp"] for c in CLASSES)

        # --- NEW: value-verification stats -------------------------
        # denominator = rows eligible for value-checking at all
        # (verdict-correct AND evidenced)
        n_value_checkable = correct_number_ef
        n_value_verifiable = int(g["value_verifiable"].sum())
        n_value_matched = int(g["value_matched"].sum())
        n_value_no_ground_truth = n_value_checkable - n_value_verifiable

        pct_value_matched_of_verifiable = (
            round(n_value_matched / n_value_verifiable, 3) if n_value_verifiable > 0 else None
        )

        accuracy_fully_verified_strict = round(int(g["fully_verified_strict"].sum()) / n, 3)
        accuracy_fully_verified_lenient = round(int(g["fully_verified_lenient"].sum()) / n, 3)

        # transparency: how much of this model's score rests on
        # method-specific ground truth vs. the Lee's L fallback
        n_method_specific_gt = int((g["ground_truth_source"] == "method_specific").sum())
        n_manifest_fallback_gt = int((g["ground_truth_source"] == "manifest_fallback").sum())

        # informational: accuracy against the officially-posed question
        # (Lee's L) regardless of which method the row used -- separate
        # from the primary, method-matched `accuracy` above
        accuracy_vs_manifest = round(int(g["correct_vs_manifest"].sum()) / n, 3)

        row = {
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
            # value-verification layer (only meaningful on top of
            # verdict-correct + evidenced rows)
            "n_value_checkable": n_value_checkable,
            "n_value_has_ground_truth": n_value_verifiable,
            "n_value_no_ground_truth_for_method": n_value_no_ground_truth,
            "n_value_matched": n_value_matched,
            "pct_value_matched_of_verifiable": pct_value_matched_of_verifiable,
            "accuracy_fully_verified_strict": accuracy_fully_verified_strict,
            "accuracy_fully_verified_lenient": accuracy_fully_verified_lenient,
            # transparency / secondary metrics
            "n_method_specific_gt": n_method_specific_gt,
            "n_manifest_fallback_gt": n_manifest_fallback_gt,
            "accuracy_vs_manifest_lee_l": accuracy_vs_manifest,
        }

        summary_rows.append(row)
        confusion_rows.extend(confusion_matrix(g_clear, model, mode))

    summary = pd.DataFrame(summary_rows).sort_values(["mode", "model"])
    summary.to_csv(SUMMARY_OUT, index=False)

    confusion = pd.DataFrame(confusion_rows)
    confusion.to_csv(CONFUSION_OUT, index=False)

    print(f"Saved: {SUMMARY_OUT} ({len(summary)} rows)")
    print(f"Saved: {CONFUSION_OUT} ({len(confusion)} rows)\n")

    print("=== Summary (raw vs evidence-filtered vs value-verified) ===")
    print(summary[["model", "n_ambiguous", "accuracy", "f1",
                    "accuracy_evidence_filtered", "f1_evidence_filtered",
                    "n_value_checkable", "n_value_has_ground_truth", "n_value_matched",
                    "pct_value_matched_of_verifiable",
                    "accuracy_fully_verified_strict", "accuracy_fully_verified_lenient"]]
          .to_string(index=False))


if __name__ == "__main__":
    run()
