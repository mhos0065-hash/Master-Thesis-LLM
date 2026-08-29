"""
score_bivariate.py
=====================
Scores the bivariate spatial correlation task's 3-way classification
(positive / negative / not_significant) -- NOT a simple binary accuracy
check, since there are 3 classes and "wrong" can mean different things
(predicting positive when it's actually negative is a different mistake
than predicting not_significant).

Produces THREE outputs:
  1. bivariate_scoring_summary.csv  -- per-model accuracy + per-class
     precision/recall/F1 (one-vs-rest) + macro-F1, BOTH raw and
     evidence-filtered (same "was it actually grounded in a real
     citation" logic as the original task's evidence-filtered scoring).
  2. bivariate_confusion_matrix.csv -- per-model 3x3 confusion matrix
     (rows = real ground truth, columns = what the model predicted),
     so you can see exactly WHICH mistakes each model tends to make.
  3. Printed summary to console.

Usage:
    python score_bivariate.py
Reads:  bivariate_results.csv
Writes: bivariate_scoring_summary.csv, bivariate_confusion_matrix.csv
"""

import re
import pandas as pd

from bivariate_verdict import _has_self_admitted_non_computation

# For single turn
#RESULTS_CSV = "bivariate_results.csv"
#SUMMARY_OUT = "Scores/single turn/bivariate_scoring_summary.csv"
#CONFUSION_OUT = "Scores/single turn/bivariate_confusion_matrix.csv"


# For multi turn
RESULTS_CSV = "results/bivariate_results_multi.csv"
SUMMARY_OUT = "Scores/multi turn/bivariate_scoring_summary_multi.csv"
CONFUSION_OUT = "Scores/multi turn/bivariate_confusion_matrix_multi.csv"


# For multi agent
RESULTS_CSV = "results/multi agent/bivariate_results_multiagent.csv"
SUMMARY_OUT = "Scores/multi agent/bivariate_scoring_summary_multi_agent.csv"
CONFUSION_OUT = "Scores/multi agent/bivariate_confusion_matrix_multi_agent.csv"

CLASSES = ["positive", "negative", "not_significant"]


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


def load():
    df = pd.read_csv(RESULTS_CSV)
    df["has_evidence_bool"] = df.apply(
        lambda r: _has_real_test_statistic(r["evidence_text"], r.get("verdict_detail")), axis=1)
    df["ambiguous"] = df["predicted"].isna() | (~df["predicted"].isin(CLASSES))
    df["correct"] = (~df["ambiguous"]) & (df["predicted"] == df["ground_truth_direction"])
    df["correct_evidence_filtered"] = df["correct"] & df["has_evidence_bool"]
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
    treatment as evidence-filtering's '__unevidenced__' substitution."""
    y_true = g["ground_truth_direction"]

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

    macro_precision = round(sum(precisions) / len(precisions), 3) if precisions else 0.0
    macro_recall = round(sum(recalls) / len(recalls), 3) if recalls else 0.0
    macro_f1 = round(sum(f1_scores) / len(f1_scores), 3) if f1_scores else 0.0
    return per_class, macro_precision, macro_recall, macro_f1


def confusion_matrix(g_clear, model, mode):
    y_true = g_clear["ground_truth_direction"]
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
        accuracy = round(correct_number / n, 3)
        accuracy_ef = round(correct_number_ef / n, 3)

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

        row = {
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
        }

        summary_rows.append(row)
        confusion_rows.extend(confusion_matrix(g_clear, model, mode))

    summary = pd.DataFrame(summary_rows).sort_values(["mode", "model"])
    summary.to_csv(SUMMARY_OUT, index=False)

    confusion = pd.DataFrame(confusion_rows)
    confusion.to_csv(CONFUSION_OUT, index=False)

    print(f"Saved: {SUMMARY_OUT} ({len(summary)} rows)")
    print(f"Saved: {CONFUSION_OUT} ({len(confusion)} rows)\n")

    print("=== Summary (raw vs evidence-filtered macro-F1) ===")
    print(summary[["model", "n_ambiguous", "tp", "fp", "accuracy", "precision", "recall", "f1",
                    "tp_evidence_filtered", "fp_evidence_filtered",
                    "accuracy_evidence_filtered", "f1_evidence_filtered"]]
          .to_string(index=False))


if __name__ == "__main__":
    run()
