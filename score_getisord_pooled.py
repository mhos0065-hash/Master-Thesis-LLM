"""
score_getisord_pooled.py
===========================
Alternative to score_getisord_boroughs.py's per-chunk breakdown --
this pools EVERY (chunk, borough) pair into one set of predictions per
model, matching the simple per-model summary table format used for your
other tasks (N, ambiguous, non_ambiguous, correct_predicted,
pct_with_evidence, TP, FP, accuracy, precision, recall, F1).

Why this is more comparable: Getis-Ord chunks ask for a variable-length
LIST of boroughs, unlike your other tasks' single per-chunk decision.
Treating each (chunk, borough) combination as one classification decision
("is THIS borough a hot/cold spot in THIS chunk's real answer, and did
the model say so") makes the units directly comparable to your other
tasks' per-chunk decisions, and gives ONE clean row per model instead of
21 rows needing separate averaging.

"Positive" here = hot OR cold (i.e. "this borough IS significant") --
collapsed to binary since TP/FP need one positive class, same reasoning
as your original CLUSTERED/NOT_CLUSTERED task. A separate direction-
accuracy figure (hot vs cold, among correctly-flagged boroughs) is also
reported, since getting the row right but the type wrong is a distinct,
worth-tracking kind of error.

Chunks where predicted_boroughs is genuinely unparseable (not a real
empty list, not real names -- i.e. has_evidence is irrelevant here, this
is about whether ANY answer was given at all) are excluded from pooling
for that chunk, matching how "ambiguous" works in your other scorers.

Usage:
    python score_getisord_pooled.py
Reads:  getisord_results.csv
Writes: getisord_scoring_summary.csv
"""

import pandas as pd

RESULTS_CSV = "getisord_results.csv"
OUT_CSV = "getisord_scoring_summary.csv"

ALL_BOROUGHS = [
    "Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley", "Camden",
    "City of London", "Croydon", "Ealing", "Enfield", "Greenwich", "Hackney",
    "Hammersmith and Fulham", "Haringey", "Harrow", "Havering", "Hillingdon",
    "Hounslow", "Islington", "Kensington and Chelsea", "Kingston upon Thames",
    "Lambeth", "Lewisham", "Merton", "Newham", "Redbridge",
    "Richmond upon Thames", "Southwark", "Sutton", "Tower Hamlets",
    "Waltham Forest", "Wandsworth", "Westminster",
]


def parse_borough_string(s):
    if not isinstance(s, str) or s.strip() in ("(none)", "none", ""):
        return {}
    result = {}
    for entry in s.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) >= 2:
            result[parts[0].strip()] = parts[1].strip()
    return result


def row_is_ambiguous(predicted_str):
    """A chunk is ambiguous only if the model gave no parseable answer at
    all -- NOT if it validly answered 'none'. Matches predicted_boroughs
    being NaN/empty in a way that ISN'T a legitimate '(none)' response."""
    if not isinstance(predicted_str, str):
        return True
    return False  # predicted_boroughs is always either real entries or empty for a valid "none"


def run():
    df = pd.read_csv(RESULTS_CSV)
    rows = []
    IMPLAUSIBLE_THRESHOLD = 10  # real Gi* rarely flags more than a handful of boroughs

    for (mode, model), g in df.groupby(["mode", "model"]):
        n = len(g)
        n_ambiguous = 0
        n_implausible = 0
        correct_predicted_chunks = 0
        ef_correct = 0
        evidence_weighted_sum = 0

        tp = fp = fn = tn = 0
        tp_ef = fp_ef = fn_ef = tn_ef = 0

        for _, r in g.iterrows():
            pred_str = r.get("predicted_boroughs")
            truth_str = r.get("ground_truth_boroughs")
            evidence_text = str(r.get("evidence_text", "")).strip().lower()
            has_ev = bool(r.get("has_evidence", False))

            pred = parse_borough_string(pred_str)
            truth = parse_borough_string(truth_str)

            # a chunk is ambiguous only if it's empty AND wasn't a genuine "none" answer
            is_valid_none = evidence_text == "none"
            is_empty = not isinstance(pred_str, str) or pred_str.strip() in ("", "(none)", "none") \
                or pd.isna(pred_str)
            is_ambiguous_chunk = is_empty and not is_valid_none

            raw_entries = [e.strip() for e in str(pred_str).split(";") if e.strip()] if not is_empty else []
            has_duplicates = len(raw_entries) > len(pred)
            is_implausible_chunk = len(pred) > IMPLAUSIBLE_THRESHOLD or has_duplicates

            if is_ambiguous_chunk:
                n_ambiguous += 1
            if is_implausible_chunk:
                n_implausible += 1

            # an ambiguous or implausible chunk gives no TRUSTWORTHY prediction --
            # treated as if it predicted NOTHING for scoring purposes, so every real
            # significant borough in the true answer becomes a missed detection (FN),
            # but it never gets to inflate FP with an unreliable/broken claim, and it
            # never counts toward correct_predicted even if this happens to coincide
            # with an empty real answer
            if is_ambiguous_chunk or is_implausible_chunk:
                pred_effective = {}
            else:
                pred_effective = pred
                evidence_weighted_sum += has_ev
                if pred == truth:
                    correct_predicted_chunks += 1
                    if len(pred) == 0 or has_ev:
                        ef_correct += 1

            for borough in ALL_BOROUGHS:
                pred_flagged = borough in pred_effective
                truth_flagged = borough in truth

                if pred_flagged and truth_flagged:
                    tp += 1
                    if has_ev:
                        tp_ef += 1
                    else:
                        fn_ef += 1  # lost its evidence-backed credit, counted as missed
                elif pred_flagged and not truth_flagged:
                    fp += 1
                    fp_ef += 1  # wrong claims stay wrong regardless of evidence
                elif not pred_flagged and truth_flagged:
                    fn += 1
                    fn_ef += 1
                else:
                    tn += 1
                    tn_ef += 1

        non_ambiguous = n - n_ambiguous - n_implausible
        pct_with_evidence = round(evidence_weighted_sum / n, 3) if n else 0.0

        def compute_metrics(tp_, fp_, fn_, tn_):
            n_inst = tp_ + fp_ + fn_ + tn_
            accuracy = round((tp_ + tn_) / n_inst, 3) if n_inst else 0.0
            precision = round(tp_ / (tp_ + fp_), 3) if (tp_ + fp_) > 0 else 0.0
            recall = round(tp_ / (tp_ + fn_), 3) if (tp_ + fn_) > 0 else 0.0
            f1 = round(2 * precision * recall / (precision + recall), 3) if (precision + recall) > 0 else 0.0
            return accuracy, precision, recall, f1

        accuracy, precision, recall, f1 = compute_metrics(tp, fp, fn, tn)
        accuracy_ev, precision_ev, recall_ev, f1_ev = compute_metrics(tp_ef, fp_ef, fn_ef, tn_ef)

        rows.append({
            "model": model, "mode": mode,
            "N_Query": n, "Reply_ambiguous": n_ambiguous, "n_implausible": n_implausible,
            "non_ambiguous": non_ambiguous,
            "correct_predicted": correct_predicted_chunks, "pct_with_evidence": pct_with_evidence,
            "TP_borough": tp, "FP_borough": fp, "accuracy": accuracy, "precision": precision,
            "recall": recall, "F1": f1,
            "correct_predicted_Evidence": ef_correct,
            "TP_borough_Evidence": tp_ef, "FP_borough_Evidence": fp_ef, "accuracy_Evidence": accuracy_ev,
            "precision_Evidence": precision_ev, "recall_Evidence": recall_ev, "F1_Evidence": f1_ev,
        })

    summary = pd.DataFrame(rows).sort_values(["mode", "model"])
    summary.to_csv(OUT_CSV, index=False)

    print("=== Without filtering based on correct evidence ===")
    raw_cols = ["model", "N_Query", "Reply_ambiguous", "n_implausible", "non_ambiguous", "correct_predicted",
                "pct_with_evidence", "TP_borough", "FP_borough", "accuracy", "precision", "recall", "F1"]
    print(summary[raw_cols].to_string(index=False))
    print()
    print("=== After filtering based on correct evidence ===")
    ef_cols = ["model", "N_Query", "Reply_ambiguous", "n_implausible", "non_ambiguous", "correct_predicted_Evidence",
               "pct_with_evidence", "TP_borough_Evidence", "FP_borough_Evidence", "accuracy_Evidence",
               "precision_Evidence", "recall_Evidence", "F1_Evidence"]
    print(summary[ef_cols].to_string(index=False))
    print()
    print(f"Saved: {OUT_CSV}")


if __name__ == "__main__":
    run()
