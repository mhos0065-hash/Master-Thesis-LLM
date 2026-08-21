"""
score_changespillover.py
===========================
Scores the spatial change-spillover task's binary classification
(CLUSTERED / NOT CLUSTERED, via predicted vs ground_truth_present), with
the SAME column structure used for bivariate, distance-decay, and
outlier-r, so all tables are directly comparable.

Binary, not 3-way -- this task reuses casual_verdict.py's exact
CLUSTERED/NOT CLUSTERED format from the original Moran's I task, so
predicted and ground_truth_present are both simple booleans.

Evidence check: same principle as the other scorers -- a chunk only
counts as evidenced if at least one REAL, finite, non-placeholder number
is cited (not nan, not a method name, not a self-admitted fabrication).

Usage:
    python score_changespillover.py
Reads:  changespillover_results.csv
Writes: changespillover_scoring_summary.csv
"""

import re
import pandas as pd

#Multi
RESULTS_CSV = "results/changespillover_results_multi.csv"
OUT_CSV = "Scores/multi turn/changespillover_scoring_summary.csv"

#Single
#RESULTS_CSV = "changespillover_results.csv"
#OUT_CSV = "Scores/single turn/changespillover_scoring_summary.csv"

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
    df["predicted_bool"] = df["predicted"].map({"True": True, "False": False, True: True, False: False})
    df["ground_truth_bool"] = df["ground_truth_present"].astype(bool)
    df["ambiguous"] = df["predicted_bool"].isna()
    df["has_evidence_bool"] = df.apply(
        lambda r: _has_real_statistic(r["evidence_text"], r.get("verdict_detail")), axis=1)
    df["correct"] = (~df["ambiguous"]) & (df["predicted_bool"] == df["ground_truth_bool"])
    df["correct_evidence_filtered"] = df["correct"] & df["has_evidence_bool"]
    return df


def compute_metrics(g, use_evidence_filter):
    """precision/recall/F1 are 2-CLASS MACRO-averaged (CLUSTERED and
    NOT_CLUSTERED each treated as their own class, then averaged), not a
    single-class version that only measures CLUSTERED detection. The
    single-class version had a blind spot: an ambiguous row could only
    ever lower F1 when the real answer was CLUSTERED -- an ambiguous row
    whose real answer was NOT_CLUSTERED contributed nothing to F1 at all,
    even though it correctly lowered accuracy. Treating NOT_CLUSTERED as
    its own class closes this gap. tp/fp returned are still the raw
    CLUSTERED-class counts (descriptive only); precision/recall/F1 no
    longer derive from them directly."""
    y_true = g["ground_truth_bool"].astype(bool)
    has_evidence = g["has_evidence_bool"].astype(bool)

    if use_evidence_filter:
        keep_as_is = g["predicted_bool"].isna() | (g["predicted_bool"] == False) | has_evidence
        y_pred = g["predicted_bool"].where(keep_as_is, None)
    else:
        y_pred = g["predicted_bool"]

    tp = int(((y_pred == True) & (y_true == True)).sum())
    fp = int(((y_pred == True) & (y_true == False)).sum())

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

    precision = round(sum(precisions) / len(precisions), 3) if precisions else 0.0
    recall = round(sum(recalls) / len(recalls), 3) if recalls else 0.0
    f1 = round(sum(f1s) / len(f1s), 3) if f1s else 0.0

    n_total = len(g)
    correct_col = "correct_evidence_filtered" if use_evidence_filter else "correct"
    accuracy = round(g[correct_col].sum() / n_total, 3) if n_total > 0 else 0.0
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
