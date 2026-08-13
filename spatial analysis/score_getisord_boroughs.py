"""
score_getisord_boroughs.py
=============================
Scores Getis-Ord results at the borough level, the same way
score_moran_boroughs.py scores Local Moran's I -- NOT a raw string
comparison (which would almost never match even when the model is
genuinely correct, due to ordering/formatting differences and real
z-scores never matching exactly).

For each chunk, parses both predicted_boroughs and ground_truth_boroughs
(format: "BoroughName:direction:zvalue; BoroughName2:direction:zvalue")
into sets, then computes:
  - name_only precision/recall/f1: did it name the right boroughs at all
    (ignoring hot/cold direction)?
  - with_type precision/recall/f1: did it ALSO get hot vs cold right?

Also flags rows with implausible results (more than ~10 boroughs flagged,
or the same borough appearing more than once) as likely computation bugs
rather than genuine disagreements -- these should be investigated, not
silently scored as "wrong" alongside genuine reasoning misses.

Usage:
    python score_getisord_boroughs.py
Reads:  getisord_results.csv
Writes: getisord_borough_scoring.csv
"""

import pandas as pd

RESULTS_CSV = "getisord_results.csv"
OUT_CSV = "getisord_borough_scoring.csv"
IMPLAUSIBLE_COUNT_THRESHOLD = 10  # real Gi* rarely flags more than a handful of boroughs


def parse_borough_string(s):
    """Returns dict {borough_name: direction}. Handles '(none)' and NaN."""
    if not isinstance(s, str) or s.strip() in ("(none)", "none", ""):
        return {}
    result = {}
    for entry in s.split(";"):
        entry = entry.strip()
        if not entry:
            continue
        parts = entry.split(":")
        if len(parts) >= 2:
            borough, direction = parts[0].strip(), parts[1].strip()
            result[borough] = direction  # last occurrence wins if duplicated
    return result


def score_row(predicted_str, ground_truth_str):
    pred = parse_borough_string(predicted_str)
    truth = parse_borough_string(ground_truth_str)

    pred_names = set(pred.keys())
    truth_names = set(truth.keys())

    # implausibility flag -- not a score, a data-quality warning
    raw_predicted_count = len([e for e in str(predicted_str).split(";") if e.strip()]) \
        if isinstance(predicted_str, str) and predicted_str.strip() not in ("(none)", "none", "") else 0
    duplicate_names = raw_predicted_count > len(pred_names)
    implausible = duplicate_names or len(pred_names) > IMPLAUSIBLE_COUNT_THRESHOLD

    if len(pred_names) == 0 and len(truth_names) == 0:
        return dict(name_precision=1.0, name_recall=1.0, name_f1=1.0,
                    type_precision=1.0, type_recall=1.0, type_f1=1.0,
                    implausible=implausible, n_predicted=0, n_ground_truth=0)

    name_correct = len(pred_names & truth_names)
    name_precision = name_correct / len(pred_names) if pred_names else 0.0
    name_recall = name_correct / len(truth_names) if truth_names else 0.0
    name_f1 = (2 * name_precision * name_recall / (name_precision + name_recall)
               if (name_precision + name_recall) > 0 else 0.0)

    type_correct = sum(1 for b in (pred_names & truth_names) if pred[b] == truth[b])
    type_precision = type_correct / len(pred_names) if pred_names else 0.0
    type_recall = type_correct / len(truth_names) if truth_names else 0.0
    type_f1 = (2 * type_precision * type_recall / (type_precision + type_recall)
               if (type_precision + type_recall) > 0 else 0.0)

    return dict(name_precision=round(name_precision, 3), name_recall=round(name_recall, 3),
                name_f1=round(name_f1, 3), type_precision=round(type_precision, 3),
                type_recall=round(type_recall, 3), type_f1=round(type_f1, 3),
                implausible=implausible, n_predicted=len(pred_names), n_ground_truth=len(truth_names))


def run():
    df = pd.read_csv(RESULTS_CSV)
    rows = []
    for _, r in df.iterrows():
        scores = score_row(r.get("predicted_boroughs"), r.get("ground_truth_boroughs"))
        rows.append({
            "chunk_id": r["chunk_id"], "model": r["model"], "mode": r["mode"],
            **scores,
        })

    out = pd.DataFrame(rows)
    out.to_csv(OUT_CSV, index=False)

    print(f"Saved: {OUT_CSV} ({len(out)} rows)\n")

    implausible = out[out["implausible"]]
    if len(implausible) > 0:
        print(f"WARNING: {len(implausible)} row(s) flagged as IMPLAUSIBLE "
              f"(duplicate boroughs, or >{IMPLAUSIBLE_COUNT_THRESHOLD} boroughs flagged) "
              f"-- likely computation bugs, not genuine disagreements. Investigate before "
              f"trusting these rows' scores:")
        print(implausible[["chunk_id", "model", "n_predicted"]].to_string(index=False))
        print()

    print("Per-model summary (excluding implausible rows):")
    clean = out[~out["implausible"]]
    summary = clean.groupby("model")[["name_precision", "name_recall", "name_f1",
                                        "type_precision", "type_recall", "type_f1"]].mean().round(3)
    print(summary)


if __name__ == "__main__":
    run()
