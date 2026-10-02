"""
score_moran_boroughs.py
=========================
Set-based precision/recall/F1 for LOCAL Moran's I (LISA) chunks -- checks
WHICH boroughs the model named as significant HH/LL clusters, not just
whether it said "yes, some cluster exists" (that binary check is already
in score_results.py; this is a stricter, complementary metric).

Why set-based P/R/F1 instead of an ad hoc "% correct" score: this is the
standard evaluation approach for any task where the answer is a SET of
named items compared against a gold set (the same logic used to evaluate
extracted entities, retrieved documents, etc.). It reuses the exact
precision/recall/F1 framework your proposal already commits to in Section
4.7, rather than introducing an unprincipled new scoring rule.

Usage:
    python score_moran_boroughs.py
Reads results.csv + chunks_csv/manifest.csv, prints a report, writes
moran_borough_scoring.csv.
"""

import pandas as pd
import json
import re

RESULTS_CSV = "results.csv"
MANIFEST_CSV = "chunks_csv/manifest.csv"

# All 33 London borough names, longest-first so "Kingston upon Thames" matches
# before a shorter false-positive substring could
LONDON_BOROUGHS = [
    "Barking and Dagenham", "Barnet", "Bexley", "Brent", "Bromley", "Camden",
    "City of London", "Croydon", "Ealing", "Enfield", "Greenwich", "Hackney",
    "Hammersmith and Fulham", "Haringey", "Harrow", "Havering", "Hillingdon",
    "Hounslow", "Islington", "Kensington and Chelsea", "Kingston upon Thames",
    "Lambeth", "Lewisham", "Merton", "Newham", "Redbridge",
    "Richmond upon Thames", "Southwark", "Sutton", "Tower Hamlets",
    "Waltham Forest", "Wandsworth", "Westminster",
]
LONDON_BOROUGHS_BY_LENGTH = sorted(LONDON_BOROUGHS, key=len, reverse=True)


CLUSTER_TYPES = ["HH", "LL", "HL", "LH"]


def extract_boroughs_from_text(text):
    """Find every London borough name mentioned in a piece of text (the
    model's verdict_detail + evidence descriptions/values, concatenated).
    Case-insensitive, whole-word-ish matching."""
    if not isinstance(text, str):
        return set()
    found = set()
    for borough in LONDON_BOROUGHS_BY_LENGTH:
        if re.search(re.escape(borough), text, re.IGNORECASE):
            found.add(borough)
    return found


def extract_borough_type_pairs(text):
    """Stricter version: for each borough mentioned, also try to find an
    HH/LL/HL/LH label near it -- specifically the NEAREST such label within
    a search window, not just any label in the window (a naive scan can
    otherwise grab a neighboring borough's label instead of the right
    one). Returns {borough: cluster_type or None if a borough was named
    but no type could be confidently associated with it -- None counts as
    WRONG when scored against ground truth, since the task requires
    stating the type, not just the borough name."""
    if not isinstance(text, str):
        return {}
    pairs = {}
    for borough in LONDON_BOROUGHS_BY_LENGTH:
        m = re.search(re.escape(borough), text, re.IGNORECASE)
        if not m:
            continue
        borough_center = (m.start() + m.end()) / 2
        window_start, window_end = max(0, m.start() - 40), m.end() + 40
        window_text = text[window_start:window_end]

        best_type, best_distance = None, None
        for ctype in CLUSTER_TYPES:
            for tm in re.finditer(rf"\b{ctype}\b", window_text):
                abs_pos = window_start + (tm.start() + tm.end()) / 2
                distance = abs(abs_pos - borough_center)
                if best_distance is None or distance < best_distance:
                    best_type, best_distance = ctype, distance
        pairs[borough] = best_type
    return pairs


def score_local_moran_chunk(predicted_text, true_boroughs):
    """Returns (precision, recall, f1, predicted_set, true_set).
    NAME-ONLY: did the model name the right boroughs, regardless of
    whether it also got the HH vs LL label right for them."""
    predicted_set = extract_boroughs_from_text(predicted_text)
    true_set = set(true_boroughs)

    if len(predicted_set) == 0 and len(true_set) == 0:
        return 1.0, 1.0, 1.0, predicted_set, true_set  # correctly identified "no clusters"
    if len(predicted_set) == 0:
        return 0.0, 0.0, 0.0, predicted_set, true_set  # said nothing, but should have named boroughs
    if len(true_set) == 0:
        return 0.0, 0.0, 0.0, predicted_set, true_set  # named boroughs, but truly none are significant

    tp = len(predicted_set & true_set)
    precision = tp / len(predicted_set)
    recall = tp / len(true_set)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, predicted_set, true_set


def score_local_moran_chunk_with_type(predicted_text, true_borough_types):
    """STRICTER: did the model name the right borough AND assign it the
    correct HH/LL/HL/LH label? A borough only counts as a true positive if
    both match. true_borough_types: {borough: "HH"/"LL"/etc.}"""
    predicted_pairs = extract_borough_type_pairs(predicted_text)
    true_pairs = dict(true_borough_types)

    if len(predicted_pairs) == 0 and len(true_pairs) == 0:
        return 1.0, 1.0, 1.0, predicted_pairs, true_pairs
    if len(predicted_pairs) == 0 or len(true_pairs) == 0:
        return 0.0, 0.0, 0.0, predicted_pairs, true_pairs

    correct = sum(1 for b, t in predicted_pairs.items()
                  if b in true_pairs and t is not None and t == true_pairs[b])
    precision = correct / len(predicted_pairs)
    recall = correct / len(true_pairs)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1, predicted_pairs, true_pairs


def run():
    results = pd.read_csv(RESULTS_CSV)
    manifest = pd.read_csv(MANIFEST_CSV)

    local_results = results[results["chunk_id"].str.startswith("moran_local_")]
    if len(local_results) == 0:
        print("No moran_local_* chunks found in results.csv yet -- run single_agent.py/multi_agent.py "
              "on at least one Local Moran's I chunk first.")
        return

    rows = []
    for _, r in local_results.iterrows():
        manifest_row = manifest[manifest["chunk_id"] == r["chunk_id"]]
        if len(manifest_row) == 0:
            continue
        metric = json.loads(manifest_row.iloc[0]["ground_truth_metric_json"])
        true_types = metric.get("significant_cluster_boroughs", {})
        combined_text = f"{r['verdict_detail']} {r.get('evidence_text', '')}"

        p1, r1, f1_1, pred_set, true_set = score_local_moran_chunk(combined_text, list(true_types.keys()))
        p2, r2, f1_2, pred_pairs, true_pairs = score_local_moran_chunk_with_type(combined_text, true_types)

        rows.append({
            "chunk_id": r["chunk_id"], "model": r["model"], "mode": r["mode"],
            "true_boroughs": ", ".join(f"{b}({t})" for b, t in true_types.items()) or "(none)",
            "predicted_boroughs": ", ".join(sorted(pred_set)) or "(none)",
            "name_only_precision": round(p1, 3), "name_only_recall": round(r1, 3), "name_only_f1": round(f1_1, 3),
            "with_type_precision": round(p2, 3), "with_type_recall": round(r2, 3), "with_type_f1": round(f1_2, 3),
        })

    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    print(f"\nMean name-only F1 (right borough, type may be wrong): {out['name_only_f1'].mean():.3f}")
    print(f"Mean with-type F1 (right borough AND right HH/LL/HL/LH label): {out['with_type_f1'].mean():.3f}")
    out.to_csv("moran_borough_scoring.csv", index=False)
    print("Saved to moran_borough_scoring.csv")


if __name__ == "__main__":
    run()
