"""
generate_change_spillover_chunks.py
======================================
Builds chunk-level data files and the manifest for the spatial
change-spillover task, from the RAW YEARLY PANEL DATA (2021.csv,
2022.csv, 2023.csv) -- NOT from pre-computed deltas.

CHANGE FROM PREVIOUS VERSION: each chunk now contains the indicator's
RAW value for ALL THREE years (2021, 2022, 2023) for all 33 boroughs,
plus the adjacency list. The model must compute the change (delta)
between the two relevant years ITSELF -- it is told which two years
matter for this specific question, but is given all three (Option A),
so it must correctly select and use only the right two.

Ground truth (moran_i, p_value, clustered) is UNCHANGED -- still sourced
from the existing R output (spatial_change_spillover_all_years.csv),
since the real ground-truth statistical test (Moran's I on the delta)
has not changed, only what data the model is shown.

Usage:
    python generate_change_spillover_chunks.py
Reads:  2021.csv, 2022.csv, 2023.csv (raw yearly panel data)
        chunks_csv/changespillover/spatial_change_spillover_all_years.csv (R output, ground truth)
        chunks_csv/borough_adjacency.csv (existing, real adjacency file)
Writes: chunks_csv/changespillover/changespillover_<yearpair>_<indicator>.csv (21 files)
        chunks_csv/changespillover/changespillover_manifest.csv
"""

import json
import os
import pandas as pd

YEAR_FILES = {"2021": "data processing/sustainability/dataset/2021.csv", "2022": "data processing/sustainability/dataset/2022.csv", "2023": "data processing/sustainability/dataset/2023.csv"}
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

# Maps the R output's indicator label (title case) to the raw panel
# file's actual column name (confirmed directly against the real
# uploaded 2021/2022/2023.csv files).
INDICATOR_COLUMN_MAP = {
    "NO2": "no2_annual_mean",
    "Gas": "gas_kwh_per_consumer",
    "Energy": "energy_kwh_per_capita",
    "GHG": "ghg_tco2e_per_capita",
    "Transport": "transport_kwh_per_capita",
    "Crime": "crime_per_1k",
    "Waste": "waste_tonnes_per_capita",
}


def load_year_frames():
    """Loads all three yearly panel files, keeping only borough + the
    columns we actually need (indicator columns), indexed by borough."""
    frames = {}
    for year, path in YEAR_FILES.items():
        df = pd.read_csv(path)
        frames[year] = df.set_index("borough")
    return frames


def build_prompt(indicator, year_start, year_end):
    return (
        f"I've attached {indicator} values for 33 London boroughs for the years 2021, "
        f"2022, and 2023, and a list showing which boroughs are geographic neighbors of "
        f"each other.\n\n"
        f"Question: Do boroughs that saw a large CHANGE in {indicator} between {year_start} "
        f"and {year_end} tend to be located near other boroughs that ALSO saw a similarly-sized "
        f"change in the SAME direction (both increasing, or both decreasing), or does a "
        f"borough's change show no relationship to its neighbors' changes? This tests whether "
        f"the CHANGE itself clusters spatially, not just the raw level in any single year.\n\n"
        f"(Note: the data includes a third year not needed for this question.)\n\n"
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
        f"FINAL ANSWER: CLUSTERED (test statistic = <value>, p = <value>)\n"
        f"or\n"
        f"FINAL ANSWER: NOT CLUSTERED (test statistic = <value>, p = <value>)\n"
        f"(test statistic is just an example label -- if your method produces a "
        f"different statistic, use its name instead, but the value after '=' must be a "
        f"real number)"
    )


def run():
    year_frames = load_year_frames()
    gt_df = pd.read_csv(R_OUTPUT_CSV)
    manifest_rows = []

    for i, r in gt_df.iterrows():
        year_pair = r["year_pair"]
        indicator = r["Indicator"]
        year_start, year_end = YEAR_PAIR_LABELS[year_pair]
        col = INDICATOR_COLUMN_MAP[indicator]

        chunk_id = f"changespillover_{year_pair}_{indicator.lower()}"
        csv_filename = f"{chunk_id}.csv"
        csv_path = f"{CHUNK_DIR}/{csv_filename}"

        # Option A: build a wide chunk with ALL THREE years' raw values
        # for this indicator, borough as the join key. Missing values
        # (NaN) are kept as-is, not dropped or imputed here -- the model
        # is expected to handle them, same as every other task.
        wide = pd.DataFrame(index=year_frames["2021"].index)
        for year in ["2021", "2022", "2023"]:
            wide[f"{indicator}_{year}"] = year_frames[year][col]
        wide = wide.reset_index().rename(columns={"index": "borough"})
        wide.to_csv(csv_path, index=False)

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
            "n_rows": len(wide),
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
