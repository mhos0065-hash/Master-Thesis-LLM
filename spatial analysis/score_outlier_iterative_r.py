"""
score_outlier_iterative_r.py
===============================
Scores the iterative-r spatial outlier task's 3-way classification
(outlier_high / outlier_low / not_outlier) -- one classification per
chunk, per target borough (NOT a variable-length list like Getis-Ord,
so this uses the same 3-way macro-averaged approach as score_bivariate.py,
not the pooled borough-level approach used for Getis-Ord).

TP/FP here are MICRO-aggregated across the 3 classes (summed per-class
confusion counts) -- TP_total = correct_number, FP_total = every wrong
classification, regardless of which wrong class it landed in.

"Implausible" for this task means the model's own computed z-score is
wildly outside any plausible range (|z| > 20 -- your real ground truth
never exceeds ~5), suggesting a computation bug rather than a genuine
disagreement -- these get excluded from pooling and counted separately,
same principle as Getis-Ord's implausibility filter.

Produces ONE CSV with both raw and evidence-filtered columns side by
side, one row per model, matching your requested column layout.

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
IMPLAUSIBLE_Z_THRESHOLD = 20  # real ground truth never exceeds ~5


def extract_z_from_evidence(evidence_text):
    if not isinstance(evidence_text, str):
        return None
    match = re.search(r"z\s*=\s*([-+]?\d*\.?\d+)", evidence_text, re.IGNORECASE)
    return float(match.group(1)) if match else None


def per_class_metrics(g, use_evidence_filter):
    """g is the FULL group (including ambiguous/implausible rows). Ambiguous
    and implausible rows are treated as a genuine miss -- they never match any
    real class (so they can't count as a TP or FP), but they DO count as an
    FN for whatever the true class was, since the model failed to identify it.
    This makes accuracy AND precision/recall/F1 all consistently penalize
    non-answers, rather than quietly excluding them from the denominator."""
    y_true = g["ground_truth_direction"]

    def effective_pred(r):
        if r["is_ambiguous"] or r["is_implausible"]:
            return "__no_answer__"
        if use_evidence_filter and not r["has_evidence_bool"]:
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
    df["has_evidence_bool"] = df["has_evidence"].fillna(False).astype(bool)
    df["z_value"] = df["evidence_text"].apply(extract_z_from_evidence)
    df["is_implausible"] = df["z_value"].abs() > IMPLAUSIBLE_Z_THRESHOLD
    df["is_ambiguous"] = df["predicted"].isna() | (~df["predicted"].isin(CLASSES))

    rows = []
    for (mode, model), g in df.groupby(["mode", "model"]):
        n = len(g)
        n_ambiguous = int((g["is_ambiguous"] & ~g["is_implausible"]).sum())
        n_implausible = int(g["is_implausible"].sum())
        g_clear = g[~g["is_ambiguous"] & ~g["is_implausible"]]
        non_ambiguous = len(g_clear)

        if n == 0:
            continue

        correct = (g_clear["predicted"] == g_clear["ground_truth_direction"])
        correct_predicted = int(correct.sum())
        correct_ef = int((correct & g_clear["has_evidence_bool"]).sum())
        pct_with_evidence = round(g_clear["has_evidence_bool"].mean(), 3) if non_ambiguous else 0.0

        # ambiguous/implausible chunks count as a genuine miss throughout --
        # both in accuracy's denominator AND in precision/recall/F1, via
        # per_class_metrics treating them as an automatic FN for the true class
        tp, fp, precision, recall, f1 = per_class_metrics(g, use_evidence_filter=False)
        tp_ef, fp_ef, precision_ef, recall_ef, f1_ef = per_class_metrics(g, use_evidence_filter=True)

        accuracy = round(correct_predicted / n, 3)
        accuracy_ef = round(correct_ef / n, 3)

        rows.append({
            "model": model, "mode": mode,
            "N_Query": n, "Reply_ambiguous": n_ambiguous, "n_implausible": n_implausible,
            "non_ambiguous": non_ambiguous, "correct_predicted": correct_predicted,
            "pct_with_evidence": pct_with_evidence,
            "TP_Borough_level": tp, "FP_Borough_level": fp,
            "accuracy": accuracy,
            "precision": precision, "recall": recall, "F1": f1,
            "correct_predicted_Evidence": correct_ef,
            "TP_Evidence_Borough_level": tp_ef, "FP_Evidence_Borough_level": fp_ef,
            "accuracy_Evidence": accuracy_ef,
            "precision_Evidence": precision_ef,
            "recall_Evidence": recall_ef, "F1_Evidence": f1_ef,
        })

    summary = pd.DataFrame(rows).sort_values(["mode", "model"])
    summary.to_csv(OUT_CSV, index=False)

    print("=== Without filtering based on correct evidence ===")
    raw_cols = ["model", "N_Query", "Reply_ambiguous", "n_implausible", "non_ambiguous",
                "correct_predicted", "pct_with_evidence", "TP_Borough_level", "FP_Borough_level",
                "accuracy", "precision", "recall", "F1"]
    print(summary[raw_cols].to_string(index=False))
    print()
    print("=== After filtering based on correct evidence ===")
    ef_cols = ["model", "N_Query", "Reply_ambiguous", "n_implausible", "non_ambiguous",
               "correct_predicted_Evidence", "pct_with_evidence", "TP_Evidence_Borough_level",
               "FP_Evidence_Borough_level", "accuracy_Evidence",
               "precision_Evidence", "recall_Evidence", "F1_Evidence"]
    print(summary[ef_cols].to_string(index=False))
    print()
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    run()
