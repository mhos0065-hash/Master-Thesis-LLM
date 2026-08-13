"""
check_citation_accuracy.py
=============================
Checks whether the numbers a model CITED as evidence are actually close to
the REAL ground truth statistic -- not just whether its final verdict
(CLUSTERED/NOT CLUSTERED) happened to be correct. This is exactly the gap
the hybrid prompt format was built to expose: a model can get the right
verdict while citing a fabricated or wildly-off number (see the mixed
results across your 8-model test, e.g. cited Moran's I for the same real
Crime 2023 value of 0.167 ranged from 0.165 to 0.452 across models).

For each row in results.csv with a spatial_correlation chunk and a cited
number, extracts the FIRST cited number from evidence_text (by convention,
the primary statistic -- Moran's I -- is always cited first in the hybrid
prompts) and compares it against the real value in
chunks_csv/manifest.csv's ground_truth_metric_json.

Usage:
    python check_citation_accuracy.py
Reads:  results.csv, chunks_csv/manifest.csv
Writes: citation_accuracy.csv
"""

import pandas as pd
import json
import re

RESULTS_CSV = "results.csv"
MANIFEST_CSV = "chunks_csv/manifest.csv"
OUT_CSV = "citation_accuracy.csv"

NUMBER_PATTERN = r"[-+]?\d*\.?\d+"


def extract_moran_i_citation(evidence_text):
    """Specifically finds the number cited as Moran's I -- looking for
    'I =', 'I=', "I's", etc. immediately preceding a number -- rather than
    naively grabbing the first number in the text. A model that only cites
    a p-value (e.g. "CLUSTERED (p = 0.001)", no I value at all) should
    return None here, not have its p-value mistaken for the I citation."""
    if not isinstance(evidence_text, str):
        return None
    # look for "I" followed by optional apostrophe-s, optional whitespace,
    # an equals sign, optional whitespace, then a number
    match = re.search(r"I(?:'s)?\s*=\s*(" + NUMBER_PATTERN + r")", evidence_text)
    if match:
        return float(match.group(1))
    return None


def run():
    results = pd.read_csv(RESULTS_CSV)
    manifest = pd.read_csv(MANIFEST_CSV)

    spatial = results[results["insight_type"] == "spatial_correlation"].copy()
    if len(spatial) == 0:
        print("No spatial_correlation rows found in results.csv yet.")
        return

    rows = []
    for _, r in spatial.iterrows():
        manifest_row = manifest[manifest["chunk_id"] == r["chunk_id"]]
        if len(manifest_row) == 0:
            continue
        try:
            gt_metric = json.loads(manifest_row.iloc[0]["ground_truth_metric_json"])
        except (json.JSONDecodeError, TypeError):
            continue

        real_moran_i = gt_metric.get("moran_i")
        cited_number = extract_moran_i_citation(r.get("evidence_text"))

        if real_moran_i is None or cited_number is None:
            continue

        abs_error = abs(cited_number - real_moran_i)
        # "close" defined loosely -- same sign AND within 0.15 of the real value,
        # since even a genuinely-computed I can vary somewhat by implementation
        # detail (e.g. exact permutation count). This is a coarse sanity check,
        # not a precise equivalence test.
        same_sign = (cited_number >= 0) == (real_moran_i >= 0)
        plausibly_real = same_sign and abs_error < 0.15

        rows.append({
            "chunk_id": r["chunk_id"], "model": r["model"], "mode": r["mode"],
            "predicted": r["predicted"], "ground_truth": r["ground_truth"],
            "verdict_correct": (str(r["predicted"]) == str(r["ground_truth"])),
            "real_moran_i": real_moran_i, "cited_number": cited_number,
            "abs_error": round(abs_error, 4), "plausibly_computed": plausibly_real,
        })

    out = pd.DataFrame(rows)
    if len(out) == 0:
        print("No rows had both a real moran_i and a parseable cited number.")
        return

    print(out.to_string(index=False))
    print()
    print("=== Summary: verdict-correct vs citation-plausible (these should usually agree) ===")
    print(out.groupby(["verdict_correct", "plausibly_computed"]).size().rename("count").reset_index().to_string(index=False))
    print()
    n_suspicious = ((out["verdict_correct"]) & (~out["plausibly_computed"])).sum()
    print(f"Rows with a CORRECT verdict but an implausible cited number: {n_suspicious}")
    print("These are the 'right answer, questionable reasoning' cases worth manually reviewing --")
    print("a correct CLUSTERED/NOT CLUSTERED call that isn't backed by a number anywhere near the")
    print("real statistic may reflect luck or pattern-matching rather than genuine computation.")

    out.to_csv(OUT_CSV, index=False)
    print(f"\nSaved to {OUT_CSV}")


if __name__ == "__main__":
    run()
