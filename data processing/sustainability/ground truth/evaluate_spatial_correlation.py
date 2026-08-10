"""
evaluate_spatial_correlation.py
==================================
Standalone evaluator using the single combined ground truth file
(spatial_correlation_ground_truth.csv) -- works for both Global and Local
Moran's I answers. No dependency on the rest of the pipeline.

Usage:
    python evaluate_spatial_correlation.py
"""

import re
import pandas as pd

GROUND_TRUTH_CSV = "spatial_correlation_ground_truth.csv"

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


def get_local_ground_truth(year, indicator, csv_path=GROUND_TRUTH_CSV):
    """{borough: cluster_label} for TRUE significant local clusters (HH/LL only)."""
    df = pd.read_csv(csv_path)
    sub = df[(df["granularity"] == "local") & (df["year"] == year)
             & (df["indicator"] == indicator) & (df["is_significant"])]
    return dict(zip(sub["borough"], sub["cluster_label"]))


def get_global_ground_truth(year, indicator, csv_path=GROUND_TRUTH_CSV):
    """Single bool + detail for the Global Moran's I ground truth."""
    df = pd.read_csv(csv_path)
    row = df[(df["granularity"] == "global") & (df["year"] == year) & (df["indicator"] == indicator)]
    if len(row) == 0:
        return None
    r = row.iloc[0]
    return {"is_significant": bool(r["is_significant"]), "morans_i": r["morans_i"],
            "p_value": r["p_value"], "interpretation": r["interpretation"]}


def extract_borough_type_pairs(text):
    match = re.search(r"final answer:", text, re.IGNORECASE)
    search_text = text[match.start():] if match else text
    pairs = {}
    for borough in LONDON_BOROUGHS_BY_LENGTH:
        m = re.search(re.escape(borough), search_text, re.IGNORECASE)
        if not m:
            continue
        center = (m.start() + m.end()) / 2
        window_start = max(0, m.start() - 40)
        window = search_text[window_start: m.end() + 40]
        best_type, best_dist = None, None
        for ctype in CLUSTER_TYPES:
            for tm in re.finditer(rf"\b{ctype}\b", window):
                pos = window_start + (tm.start() + tm.end()) / 2
                dist = abs(pos - center)
                if best_dist is None or dist < best_dist:
                    best_type, best_dist = ctype, dist
        pairs[borough] = best_type
    return pairs


def evaluate_local(year, indicator, model_response_text, csv_path=GROUND_TRUTH_CSV):
    true_types = get_local_ground_truth(year, indicator, csv_path)
    pred_types = extract_borough_type_pairs(model_response_text)

    if not true_types and not pred_types:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "n_correct": 0, "true": true_types, "predicted": pred_types}
    if not true_types or not pred_types:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "n_correct": 0, "true": true_types, "predicted": pred_types}

    correct = sum(1 for b, t in pred_types.items() if b in true_types and t == true_types[b])
    precision = correct / len(pred_types)
    recall = correct / len(true_types)
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    return {"precision": round(precision, 3), "recall": round(recall, 3), "f1": round(f1, 3),
            "n_correct": correct, "true": true_types, "predicted": pred_types}


if __name__ == "__main__":
    RESPONSES = {
        "Gemini": "FINAL ANSWER: Camden (HH), Kensington and Chelsea (HH), Kingston upon Thames (LL), Sutton (LL), Westminster (HH)",
        "ChatGPT": "FINAL ANSWER: Tower Hamlets (HH), Newham (HH), Hackney (HH), Lambeth (HH), Southwark (HH), Richmond upon Thames (LL), Kingston upon Thames (LL), Sutton (LL), Bromley (LL), Harrow (LL)",
        "Copilot": "FINAL ANSWER: Islington (HH), Hackney (HH), Camden (HH), Westminster (HH), Tower Hamlets (HH), Kingston upon Thames (LL), Sutton (LL), Merton (LL)",
        "Deepseek": "FINAL ANSWER: Westminster (HH), Camden (HH), Islington (HH), Hackney (HH), Kensington and Chelsea (HH), Lambeth (HH), Haringey (HH), Harrow (LL), Richmond upon Thames (LL), Sutton (LL), Merton (LL), Bexley (LL), Bromley (LL), Kingston upon Thames (LL), Barnet (LL), Hillingdon (LL), Redbridge (LL), Waltham Forest (LL), Havering (LL), Enfield (LL), Barking and Dagenham (LL), Greenwich (LL), Lewisham (LL), Croydon (LL), Hounslow (LL), Ealing (LL), Newham (LL)",
        "Claude": "FINAL ANSWER: Westminster (HH), Camden (HH), Kensington and Chelsea (HH), Kingston upon Thames (LL), Sutton (LL)",
    }

    true_types = get_local_ground_truth(2023, "Crime")
    print(f"Ground truth (granularity=local, 2023, Crime): {true_types}\n")

    print(f"{'Model':<10}{'#named':<8}{'#correct':<10}{'precision':<11}{'recall':<9}{'F1':<6}")
    for model, text in RESPONSES.items():
        r = evaluate_local(2023, "Crime", text)
        print(f"{model:<10}{len(r['predicted']):<8}{r['n_correct']:<10}"
              f"{r['precision']:<11}{r['recall']:<9}{r['f1']:<6}")
