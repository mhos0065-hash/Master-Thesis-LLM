"""
score_results.py
==================
Scores results.csv (single/multi-agent, Global + Local Moran's I) with:
  - accuracy over the FULL count (an ambiguous/unparseable row is never
    "correct" just because it was excluded -- same principle used across
    every other scorer this session)
  - a STRICTER, method-agnostic evidence check: a citation only counts as
    real evidence if the reported statistic itself is a genuine, finite
    number -- never nan, never inf, and never a leftover method name
    (e.g. "Lee's L") in place of a value. Checking only "is any number
    present anywhere in the text" is too lenient -- a nan statistic sitting
    next to a perfectly real p-value would otherwise still count as
    "has_evidence=True", which is exactly the bug found and fixed in
    score_bivariate.py earlier this session (confirmed: 20/21 nan rows in
    this exact dataset were mislabeled this way before this fix).

Usage:
    python score_results.py
Reads:  results.csv
Writes: scoring_summary.csv
"""

import re
import pandas as pd

# Multi turn
RESULTS_CSV = "results/results_spatial_multi.csv"
OUT_CSV = "Scores/multi turn/scoring_spatial_summary.csv"

# Single turn
#RESULTS_CSV = "results.csv"
#OUT_CSV = "Scores/single turn/scoring_spatial_summary.csv"


def _has_real_statistic(evidence_text):
    """Checks that the FIRST '=...,' or '=...)' value in the text is a
    genuine, finite number -- method-agnostic, doesn't care what label
    precedes the '=' or which statistical method was used, only that a
    real number was actually computed and cited for it."""
    if not isinstance(evidence_text, str):
        return False
    match = re.search(r"=\s*([^,)]+)[,)]", evidence_text)
    if not match:
        return False
    val = match.group(1).strip()
    try:
        f = float(val)
        return f == f and abs(f) != float("inf")  # excludes nan and inf
    except ValueError:
        return False  # e.g. a method name left in place of a number


def parse_bool(v):
    if pd.isna(v):
        return None
    s = str(v).strip().lower()
    if s in ("true", "found", "1", "yes", "clustered", "present"):
        return True
    if s in ("false", "not found", "0", "no", "not clustered", "not present", "random"):
        return False
    return None


def load():
    df = pd.read_csv(RESULTS_CSV)
    df["predicted_bool"] = df["predicted"].apply(parse_bool)
    df["ground_truth_bool"] = df["ground_truth"].astype(bool)
    df["ambiguous"] = df["predicted_bool"].isna()
    df["has_evidence_bool"] = df["evidence_text"].apply(_has_real_statistic)
    df["correct"] = (~df["ambiguous"]) & (df["predicted_bool"] == df["ground_truth_bool"])
    df["correct_evidence_filtered"] = df["correct"] & df["has_evidence_bool"]
    return df


def compute_metrics(g, use_evidence_filter):
    """g is the FULL group (including ambiguous rows).

    precision/recall/F1 are 2-CLASS MACRO-averaged (CLUSTERED and
    NOT_CLUSTERED each treated as their own class, then averaged) --
    NOT the older single-class version that only measured detection of
    CLUSTERED. That version had a real blind spot: an ambiguous row
    (predicted_bool left unfilled) can NEVER become a TP or FP under
    either scheme, but under single-class scoring it could ONLY ever
    lower F1 when the real answer was CLUSTERED -- an ambiguous row
    whose real answer was NOT_CLUSTERED contributed nothing at all to
    F1, even though it correctly lowered accuracy. Treating
    NOT_CLUSTERED as its own class with its own recall closes this gap:
    an ambiguous row now counts as a missed detection of whichever class
    was actually true, mirroring exactly how accuracy already treats it.
    tp/fp returned are still the raw CLUSTERED-class counts (kept as
    descriptive figures), but precision/recall/F1 no longer derive from
    them directly."""
    y_true = g["ground_truth_bool"].astype(bool)
    has_evidence = g["has_evidence_bool"].astype(bool)

    if use_evidence_filter:
        # an unevidenced CLUSTERED claim is downgraded to "no real answer"
        # for macro-averaging purposes, same principle as every other
        # evidence-filtered scorer this session
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
            continue  # class never occurs -- excluded from macro average
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
    return round(accuracy, 3), round(precision, 3), round(recall, 3), round(f1, 3), tp, fp


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

        accuracy, precision, recall, f1, tp, fp = compute_metrics(g, use_evidence_filter=False)
        acc_ef, prec_ef, rec_ef, f1_ef, tp_ef, fp_ef = compute_metrics(g, use_evidence_filter=True)

        rows.append({
            "mode": mode, "model": model, "n": n, "n_ambiguous": n_ambiguous,
            "non_ambiguous": non_ambiguous, "correct_number": correct_number,
            "pct_with_evidence": pct_with_evidence,
            "tp": tp, "fp": fp,
            "accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1,
            "correct_number_evidence_filtered": correct_number_ef,
            "tp_evidence_filtered": tp_ef, "fp_evidence_filtered": fp_ef,
            "accuracy_evidence_filtered": acc_ef,
            "precision_evidence_filtered": prec_ef,
            "recall_evidence_filtered": rec_ef,
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
