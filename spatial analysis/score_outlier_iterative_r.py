"""
score_outlier_iterative_r.py
===============================
Scores the iterative-r spatial outlier task's 3-way classification
(outlier_high / outlier_low / not_outlier) with the SAME column
structure used for bivariate, distance-decay, and the original Moran's I
task, so all tables are directly comparable.

Evidence check: same principle as score_bivariate.py / score_distance_decay.py
-- a chunk only counts as evidenced if at least one REAL, finite,
non-placeholder number is cited (not nan, not a method name, not a
self-admitted "please replace this" fabrication).

Usage:
    python score_outlier_iterative_r.py
Reads:  outlier_iterative_r_results.csv
Writes: outlier_iterative_r_scoring_summary.csv
"""

import re
import pandas as pd

RESULTS_CSV = "outlier_iterative_r_results.csv"
OUT_CSV = "outlier_iterative_r_scoring_summary.csv"

CLASSES = ["outlier_high", "outlier_low", "not_outlier"]

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


def load():
    df = pd.read_csv(RESULTS_CSV)
    df["has_evidence_bool"] = df.apply(
        lambda r: _has_real_statistic(r["evidence_text"], r.get("verdict_detail")), axis=1)
    df["ambiguous"] = df["predicted"].isna() | (~df["predicted"].isin(CLASSES))
    df["correct"] = (~df["ambiguous"]) & (df["predicted"] == df["ground_truth_direction"])
    df["correct_evidence_filtered"] = df["correct"] & df["has_evidence_bool"]
    return df


def per_class_metrics(g, use_evidence_filter):
    y_true = g["ground_truth_direction"]

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
        })

    out = pd.DataFrame(rows).sort_values(["mode", "model"]).reset_index(drop=True)
    out = out[["mode", "model", "n", "n_ambiguous", "non_ambiguous", "correct_number",
               "pct_with_evidence", "tp", "fp", "accuracy", "precision", "recall", "f1",
               "correct_number_evidence_filtered", "tp_evidence_filtered", "fp_evidence_filtered",
               "accuracy_evidence_filtered", "precision_evidence_filtered",
               "recall_evidence_filtered", "f1_evidence_filtered"]]
    out.to_csv(OUT_CSV, index=False)

    print(out.to_string(index=False))
    print(f"\nSaved to {OUT_CSV}")


if __name__ == "__main__":
    run()
