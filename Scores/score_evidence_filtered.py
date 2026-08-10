"""
score_evidence_filtered.py
=============================
Extends score_results.py's scoring with a supervisor-presentable table
that shows BOTH the raw accuracy/precision/recall/F1 (predicted vs
ground_truth only, same as scoring_summary.csv) AND a stricter,
evidence-filtered version that additionally requires has_evidence=True --
i.e. a prediction only counts as correct if it was ALSO backed by a real,
non-fabricated cited number (Section 4.3's requirement).

This directly answers: "how much does the headline accuracy drop once
lucky-but-ungrounded guesses are excluded?" -- both numbers sit side by
side in one table so the gap itself is visible, not hidden.

Usage:
    python score_evidence_filtered.py
Reads:  results.csv
Writes: scoring_summary_evidence_filtered.csv
"""

import pandas as pd

RESULTS_CSV = "results.csv"
OUT_CSV = "scoring_summary_evidence_filtered.csv"


def load():
    df = pd.read_csv(RESULTS_CSV)

    def parse_bool(v):
        if pd.isna(v):
            return None
        s = str(v).strip().lower()
        if s in ("true", "found", "1", "yes", "clustered", "present"):
            return True
        if s in ("false", "not found", "0", "no", "not clustered", "not present", "random"):
            return False
        return None

    df["predicted_bool"] = df["predicted"].apply(parse_bool)
    df["ground_truth_bool"] = df["ground_truth"].astype(bool)
    df["ambiguous"] = df["predicted_bool"].isna()
    df["has_evidence_bool"] = df["has_evidence"].fillna(False).astype(bool) if "has_evidence" in df else False
    df["correct"] = (~df["ambiguous"]) & (df["predicted_bool"] == df["ground_truth_bool"])
    # evidence-filtered correctness: correct AND actually backed by a real citation
    df["correct_evidence_filtered"] = df["correct"] & df["has_evidence_bool"]
    return df


def compute_metrics(g_clear, use_evidence_filter):
    """g_clear already has ambiguous rows dropped.

    RAW (use_evidence_filter=False): standard predicted vs ground_truth.

    EVIDENCE-FILTERED (use_evidence_filter=True): a correct answer only
    counts as correct if it ALSO has real evidence -- applying to BOTH
    verdict directions (an ungrounded "NOT CLUSTERED" guess that happens
    to be right is penalized exactly the same as an ungrounded "CLUSTERED"
    guess), matching Section 4.3's requirement that correctness requires
    citing computed evidence regardless of which way the verdict points.
    A wrong answer (FP or the non-evidence-required half of FN) is not
    affected by evidence -- it was already wrong either way.
    """
    y_true = g_clear["ground_truth_bool"].astype(bool)
    y_pred_raw = g_clear["predicted_bool"].astype(bool)
    has_evidence = g_clear["has_evidence_bool"].astype(bool)

    if use_evidence_filter:
        # TP only counts if the correct positive call was also evidenced;
        # an unevidenced correct positive becomes a miss (FN), not a hit
        tp = int((y_pred_raw & y_true & has_evidence).sum())
        # FP is unaffected -- a wrong "True" claim is wrong regardless of
        # whether it was evidenced or fabricated
        fp = int((y_pred_raw & ~y_true).sum())
        # TN only counts if the correct negative call was also evidenced
        tn = int((~y_pred_raw & ~y_true & has_evidence).sum())
        # FN = originally-FN rows, PLUS originally-TP rows that lacked evidence
        fn = int((~y_pred_raw & y_true).sum()) + int((y_pred_raw & y_true & ~has_evidence).sum())
    else:
        tp = int((y_pred_raw & y_true).sum())
        fp = int((y_pred_raw & ~y_true).sum())
        tn = int((~y_pred_raw & ~y_true).sum())
        fn = int((~y_pred_raw & y_true).sum())

    n_total = len(g_clear)
    accuracy = (tp + tn) / n_total if n_total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return round(accuracy, 3), round(precision, 3), round(recall, 3), round(f1, 3), tp, fp


def run():
    df = load()
    rows = []

    for (mode, model), g in df.groupby(["mode", "model"]):
        g_clear = g.dropna(subset=["predicted_bool"])
        n = len(g)
        n_ambiguous = int(g["ambiguous"].sum())
        non_ambiguous = n - n_ambiguous
        correct_number = int(g["correct"].sum())
        correct_number_evidence_filtered = int(g["correct_evidence_filtered"].sum())
        pct_with_evidence = round(g["has_evidence_bool"].mean(), 3)

        if len(g_clear) == 0:
            continue

        accuracy, precision, recall, f1, tp, fp = compute_metrics(g_clear, use_evidence_filter=False)
        acc_ef, prec_ef, rec_ef, f1_ef, tp_ef, fp_ef = compute_metrics(g_clear, use_evidence_filter=True)

        rows.append({
            "mode": mode,
            "model": model,
            "n": n,
            "n_ambiguous": n_ambiguous,
            "non_ambiguous": non_ambiguous,
            "correct_number": correct_number,
            "pct_with_evidence": pct_with_evidence,
            "tp": tp,
            "fp": fp,
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "correct_number_evidence_filtered": correct_number_evidence_filtered,
            "tp_evidence_filtered": tp_ef,
            "fp_evidence_filtered": fp_ef,
            "accuracy_evidence_filtered": acc_ef,
            "precision_evidence_filtered": prec_ef,
            "recall_evidence_filtered": rec_ef,
            "f1_evidence_filtered": f1_ef,
        })

    out = pd.DataFrame(rows).sort_values(["mode", "model"]).reset_index(drop=True)
    out.to_csv(OUT_CSV, index=False)

    print(out.to_string(index=False))
    print(f"\nSaved to {OUT_CSV}")
    print("\nHow to read this: 'accuracy'/'precision'/'recall'/'f1' are the raw numbers")
    print("(same as scoring_summary.csv). The '_evidence_filtered' columns additionally")
    print("require a real, non-fabricated citation for a prediction to count as correct --")
    print("the gap between the two shows how much of each model's raw accuracy was")
    print("actually grounded in real computation vs. a lucky/ungrounded guess.")


if __name__ == "__main__":
    run()
