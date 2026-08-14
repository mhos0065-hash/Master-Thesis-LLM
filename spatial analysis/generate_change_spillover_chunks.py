"""
generate_change_spillover_chunks.py
======================================
Builds chunk-level data files and the manifest for the new spatial
change-spillover task (Task 5), from the real R output
(spatial_change_spillover_all_years.csv).

For each (year_pair x indicator) row in the R output:
  - Writes a real per-borough delta CSV: chunks_csv/changespillover/
    changespillover_<yearpair>_<indicator>.csv (columns: borough, delta_value)
  - Adds one manifest row, reusing the EXACT SAME schema as your original
    Global Moran's I manifest.csv, and the EXACT SAME CLUSTERED/NOT
    CLUSTERED answer format (reuses casual_verdict.py, answer_format=
    "casual_with_citation") -- no new parser needed.

Usage:
    python generate_change_spillover_chunks.py
Reads:  spatial_change_spillover_all_years.csv (real R output)
        chunks_csv/borough_adjacency.csv (existing, real adjacency file)
Writes: chunks_csv/changespillover/changespillover_<yearpair>_<indicator>.csv (21 files)
        chunks_csv/changespillover/changespillover_manifest.csv
"""

import json
import os
import pandas as pd

R_OUTPUT_CSV = "chunks_csv/changespillover/spatial_change_spillover_all_years.csv"
ADJACENCY_PATH = "chunks_csv/borough_adjacency.csv"
CHUNK_DIR = "chunks_csv/changespillover"
MANIFEST_OUT = os.path.join(CHUNK_DIR, "changespillover_manifest.csv")

os.makedirs(CHUNK_DIR, exist_ok=True)

YEAR_PAIR_LABELS = {
    "2021_2022": ("2021", "2022"),
    "2022_2023": ("2022", "2023"),
    "2021_2023": ("2021", "2023"),
}


def parse_borough_deltas(s):
    """Turns 'Borough1:1.23; Borough2:-4.56; ...' into a DataFrame with
    columns borough, delta_value."""
    rows = []
    for entry in s.split("; "):
        entry = entry.strip()
        if not entry:
            continue
        borough, val = entry.rsplit(":", 1)
        rows.append({"borough": borough.strip(), "delta_value": float(val)})
    return pd.DataFrame(rows)


def build_prompt(indicator, year_start, year_end):
    return (
        f"I've attached the CHANGE in {indicator} for 33 London boroughs between "
        f"{year_start} and {year_end} (delta_value = {year_end} value minus {year_start} "
        f"value), and a list showing which boroughs are geographic neighbors of each other.\n\n"
        f"Question: Do boroughs that saw a large CHANGE in {indicator} between {year_start} "
        f"and {year_end} tend to be located near other boroughs that ALSO saw a similarly-sized "
        f"change in the SAME direction (both increasing, or both decreasing), or does a "
        f"borough's change show no relationship to its neighbors' changes? This tests whether "
        f"the CHANGE itself clusters spatially, not just the raw level in any single year.\n\n"
        f"Explain your reasoning based on the data, including the actual computed statistic "
        f"and its significance. You can use a statistical method such as Global Moran's I, or "
        f"another appropriate method of your choosing, to support your conclusion.\n\n"
        f"Note: dedicated spatial-statistics packages (pysal, esda, libpysal, geopandas, or "
        f"similar) are NOT installed in this environment -- don't spend turns trying different "
        f"import paths for them. Implement your chosen method directly using only pandas and "
        f"numpy. If you test statistical significance via a permutation/randomization test, "
        f"use at least 500 random shuffles -- fewer permutations can produce an unstable "
        f"p-value that lands on the wrong side of the 0.05 threshold just by chance, "
        f"especially when the true effect is only moderate.\n\n"
        f"End your response with exactly one line, including the computed number(s) you used:\n"
        f"FINAL ANSWER: CLUSTERED (statistical method = <value>, p = <value>)\n"
        f"or\n"
        f"FINAL ANSWER: NOT CLUSTERED (statistical method = <value>, p = <value>)"
    )


def run():
    df = pd.read_csv(R_OUTPUT_CSV)
    manifest_rows = []

    for i, r in df.iterrows():
        year_pair = r["year_pair"]
        indicator = r["Indicator"]
        year_start, year_end = YEAR_PAIR_LABELS[year_pair]

        chunk_id = f"changespillover_{year_pair}_{indicator.lower()}"
        csv_filename = f"{chunk_id}.csv"
        csv_path = f"{CHUNK_DIR}/{csv_filename}"

        deltas_df = parse_borough_deltas(r["borough_deltas"])
        deltas_df.to_csv(csv_path, index=False)

        ground_truth_present = bool(r["clustered"])
        ground_truth_detail = (
            f"{'CLUSTERED' if ground_truth_present else 'NOT CLUSTERED'}: real Global Moran's I "
            f"on the {year_start}->{year_end} change in {indicator} = {r['moran_i']}, "
            f"p = {r['p_value']}"
        )
        ground_truth_rule = "p_value < 0.05 => CLUSTERED, else NOT CLUSTERED"
        ground_truth_metric_json = json.dumps({
            "moran_i": r["moran_i"], "p_value": r["p_value"],
            "n_missing_imputed": int(r["n_missing"]), "year_start": year_start, "year_end": year_end,
        })

        manifest_rows.append({
            "chunk_id": chunk_id,
            "dataset": f"Spatial Change Spillover ({indicator}, {year_start}-{year_end})",
            "insight_type": "spatial_change_spillover",
            "modality": "spatial",
            "csv_path": csv_path,
            "n_rows": len(deltas_df),
            "insight_def": build_prompt(indicator, year_start, year_end),
            "ground_truth_present": ground_truth_present,
            "ground_truth_detail": ground_truth_detail,
            "ground_truth_rule": ground_truth_rule,
            "ground_truth_metric_json": ground_truth_metric_json,
            "window_start": year_start,
            "window_end": year_end,
            "window_index": i,
            "adjacency_path": ADJACENCY_PATH,
            "spatial_granularity": "global",
            "answer_format": "casual_with_citation",
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(MANIFEST_OUT, index=False)
    print(f"Wrote {len(manifest)} chunk data files to {CHUNK_DIR}/")
    print(f"Wrote manifest: {MANIFEST_OUT} ({len(manifest)} rows)")
    print()
    print("ground_truth_present distribution:")
    print(manifest["ground_truth_present"].value_counts())


if __name__ == "__main__":
    run()
