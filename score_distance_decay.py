"""
score_distance_decay.py
==========================
Scores distance-decay's 3-way classification (negative_significant /
positive_significant / not_significant), matching the exact table
structure used for your other tasks: N_Query, Reply_ambiguous,
n_implausible, non_ambiguous, correct_predicted, pct_with_evidence, TP,
FP, accuracy, precision, recall, F1 -- raw and evidence-filtered, one
CSV, two blocks.

Column mapping from check_results.py's flagged CSV:
  Reply_ambiguous = issue_total_failure  (no verdict given at all)
  n_implausible   = issue_broken_template OR issue_fabricated_evidence
                    (gave SOME answer, but it's untrustworthy -- an
                    unfilled template or a cited number that doesn't
                    match the real computed value, e.g. qwen3-32b's
                    repeated -1.2345 placeholder)

Both ambiguous and implausible rows are treated as a genuine miss
(contributing to FN for their true class, never a TP/FP), matching every
other scorer this session -- never silently excluded from the denominator.

Evidence-filtering (second table) is the SAME has_evidence-based logic
used in bivariate/outlier-r: on top of the above, a correct answer only
counts if has_evidence=True, otherwise it's downgraded to a miss too.

Usage:
    python score_distance_decay.py
Reads:  distance_decay_results_flagged.csv (from check_results.py --save-flagged)
Writes: distance_decay_scoring_summary.csv
"""

import pandas as pd

RESULTS_CSV = "distance_decay_results_flagged.csv"
OUT_CSV = "distance_decay_scoring_summary.csv"

CLASSES = ["negative_significant", "positive_significant", "not_significant"]


def per_class_metrics(g, use_evidence_filter):
    y_true = g["ground_truth_direction"]

    def effective_pred(r):
        if bool(r.get("is_ambiguous", False)) or bool(r.get("is_implausible", False)):
            return "__no_answer__"
        if use_evidence_filter and not bool(r.get("has_evidence", False)):
            return "__unevidenced__"
        return r["predicted"]

    y_pred = g.apply(effective_pred, axis=1)

    tp_total = fp_total = 0
    precisions, recalls, f1_scores = [], [], []
    for cls in CLASSES:
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

    macro_precision = round(sum(precisions) / len(precisions), 3)
    macro_recall = round(sum(recalls) / len(recalls), 3)
    macro_f1 = round(sum(f1_scores) / len(f1_scores), 3)
    return tp_total, fp_total, macro_precision, macro_recall, macro_f1


def run():
    df = pd.read_csv(RESULTS_CSV)
    df["is_ambiguous"] = df.get("issue_total_failure", False).fillna(False).astype(bool)
    df["is_implausible"] = (
        df.get("issue_broken_template", False).fillna(False).astype(bool)
        | df.get("issue_fabricated_evidence", False).fillna(False).astype(bool)
    )
    df["has_evidence"] = df["has_evidence"].fillna(False).astype(bool)

    rows = []
    for (mode, model), g in df.groupby(["mode", "model"]):
        n = len(g)
        n_ambiguous = int(g["is_ambiguous"].sum())
        n_implausible = int((g["is_implausible"] & ~g["is_ambiguous"]).sum())
        non_ambiguous = n - n_ambiguous - n_implausible

        clear = g[~g["is_ambiguous"] & ~g["is_implausible"]]
        correct = (clear["predicted"] == clear["ground_truth_direction"])
        correct_predicted = int(correct.sum())
        correct_ef = int((correct & clear["has_evidence"]).sum())
        pct_with_evidence = round(clear["has_evidence"].mean(), 3) if len(clear) else 0.0

        accuracy = round(correct_predicted / n, 3) if n else 0.0
        accuracy_ef = round(correct_ef / n, 3) if n else 0.0

        tp, fp, precision, recall, f1 = per_class_metrics(g, use_evidence_filter=False)
        tp_ef, fp_ef, precision_ef, recall_ef, f1_ef = per_class_metrics(g, use_evidence_filter=True)

        rows.append({
            "model": model, "mode": mode,
            "N_Query": n, "Reply_ambiguous": n_ambiguous, "n_implausible": n_implausible,
            "non_ambiguous": non_ambiguous, "correct_predicted": correct_predicted,
            "pct_with_evidence": pct_with_evidence,
            "TP": tp, "FP": fp, "accuracy": accuracy,
            "precision": precision, "recall": recall, "F1": f1,
            "correct_predicted_Evidence": correct_ef,
            "TP_Evidence": tp_ef, "FP_Evidence": fp_ef, "accuracy_Evidence": accuracy_ef,
            "precision_Evidence": precision_ef, "recall_Evidence": recall_ef, "F1_Evidence": f1_ef,
        })

    summary = pd.DataFrame(rows).sort_values(["mode", "model"])
    summary.to_csv(OUT_CSV, index=False)

    print("=== Without filtering based on correct evidence ===")
    raw_cols = ["model", "N_Query", "Reply_ambiguous", "n_implausible", "non_ambiguous",
                "correct_predicted", "pct_with_evidence", "TP", "FP", "accuracy",
                "precision", "recall", "F1"]
    print(summary[raw_cols].to_string(index=False))
    print()
    print("=== After filtering based on correct evidence ===")
    ef_cols = ["model", "N_Query", "Reply_ambiguous", "n_implausible", "non_ambiguous",
               "correct_predicted_Evidence", "pct_with_evidence", "TP_Evidence", "FP_Evidence",
               "accuracy_Evidence", "precision_Evidence", "recall_Evidence", "F1_Evidence"]
    print(summary[ef_cols].to_string(index=False))
    print()
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    run()
