"""
generate_bivariate_chunks.py
===============================
Builds the 50 runnable bivariate spatial correlation chunks from
bivariate_ground_truth.csv + the real 2021/2022/2023 panel CSVs.

For each of the 50 (year, indicator_x, indicator_y) rows:
  1. Writes a small, focused CSV (borough, indicator_x_col, indicator_y_col)
     -- same "one small CSV per chunk" convention as your existing
     moran_global_2023_crime.csv-style files, not one giant shared file.
  2. Writes the casual-with-citation-style prompt text.
  3. Adds a row to bivariate_manifest.csv with everything single_agent_bivariate.py
     needs to run it.

SAFE FOR YOUR EXISTING PIPELINE: writes only to NEW paths
(chunks_csv/bivariate/*.csv, bivariate_manifest.csv) -- never touches
chunks_csv/manifest.csv, borough_adjacency.csv, or any existing chunk file.

Usage:
    python generate_bivariate_chunks.py
Reads:  bivariate_ground_truth.csv, 2021.csv, 2022.csv, 2023.csv
Writes: chunks_csv/bivariate/<chunk_id>.csv  (50 files)
        bivariate_manifest.csv
"""

import pandas as pd
import json

GROUND_TRUTH_PATH = "ground truth/bivariate_ground_truth.csv"
PANEL_PATHS = {2021: "spatial analysis/sustainability/dataset/2021.csv", 2022: "spatial analysis/sustainability/dataset/2022.csv", 2023: "spatial analysis/sustainability/dataset/2023.csv"}
ADJACENCY_PATH = "chunks_csv/borough_adjacency.csv"
CHUNKS_DIR = "chunks_csv/bivariate"
MANIFEST_OUT = "chunks_csv/bivariate/bivariate_manifest.csv"

IND_TO_COL = {
    "NO2": "no2_annual_mean",
    "Gas": "gas_kwh_per_consumer",
    "Energy": "energy_kwh_per_capita",
    "GHG": "ghg_tco2e_per_capita",
    "Transport": "transport_kwh_per_capita",
    "Crime": "crime_per_1k",
    "Waste": "waste_tonnes_per_capita",
}

PERMUTATION_INSTRUCTION = (
    "If you test statistical significance via a permutation/randomization test, use at least "
    "500 random shuffles -- fewer permutations can produce an unstable p-value that lands on "
    "the wrong side of the 0.05 threshold just by chance, especially when the true effect is "
    "only moderate."
)

NO_GEOSPATIAL_LIBS_NOTICE = (
    "Note: dedicated spatial-statistics packages (pysal, esda, libpysal, geopandas, or similar) "
    "are NOT installed in this environment -- don't spend turns trying different import paths "
    "for them. Implement this directly using only pandas and numpy."
)


def build_prompt(label_x, label_y):
    return f"""I've attached {label_x} and {label_y} data for London boroughs, and a list showing which boroughs are geographic neighbors of each other.

Question: Do boroughs with high {label_x} tend to be located near boroughs with high {label_y} (a POSITIVE spatial association), near boroughs with low {label_y} (a NEGATIVE spatial association), or is there no significant relationship between where high-{label_x} and high/low-{label_y} boroughs are located (NOT SIGNIFICANT)?

This is testing BIVARIATE spatial correlation (Lee's L statistic) -- a genuinely different question from asking whether {label_x} alone is spatially clustered. You are testing whether {label_x} in one place relates to {label_y} in NEIGHBORING places.

To compute this: standardize both variables (z-scores), compute the spatial lag of {label_y} (the weighted average of {label_y} among each borough's neighbors), then compute the correlation between {label_x}'s own z-score and {label_y}'s spatially-lagged z-score. {NO_GEOSPATIAL_LIBS_NOTICE} {PERMUTATION_INSTRUCTION}

End your response with exactly one line, including the computed number(s) you used:
FINAL ANSWER: POSITIVE (Lee's L = <value>, p = <value>)
or
FINAL ANSWER: NEGATIVE (Lee's L = <value>, p = <value>)
or
FINAL ANSWER: NOT SIGNIFICANT (Lee's L = <value>, p = <value>)"""


def run():
    gt = pd.read_csv(GROUND_TRUTH_PATH)
    assert len(gt) == 50, f"expected 50 ground truth rows, got {len(gt)}"

    panels = {yr: pd.read_csv(path) for yr, path in PANEL_PATHS.items()}

    manifest_rows = []
    for _, r in gt.iterrows():
        yr = int(r["year"])
        ind_x, ind_y = r["indicator_x"], r["indicator_y"]
        col_x, col_y = IND_TO_COL[ind_x], IND_TO_COL[ind_y]
        chunk_id = r["chunk_id"]

        panel = panels[yr]
        chunk_df = panel[["borough", col_x, col_y]].rename(
            columns={col_x: f"{ind_x}_value", col_y: f"{ind_y}_value"})

        csv_path = f"{CHUNKS_DIR}/{chunk_id}.csv"
        chunk_df.to_csv(csv_path, index=False)

        manifest_rows.append({
            "chunk_id": chunk_id,
            "dataset": f"Bivariate Spatial Correlation ({ind_x} vs {ind_y}, {yr})",
            "insight_type": "bivariate_spatial_correlation",
            "modality": "spatial",
            "csv_path": csv_path,
            "adjacency_path": ADJACENCY_PATH,
            "insight_def": build_prompt(ind_x, ind_y),
            "ground_truth_direction": r["direction"],
            "ground_truth_is_significant": r["is_significant"],
            "ground_truth_lee_l": r["lee_l"],
            "ground_truth_p_value": r["p_value_mc"],
            "ground_truth_rule": r["acceptance_rule"],
            "answer_format": "bivariate_with_citation",
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(MANIFEST_OUT, index=False)

    print(f"Wrote {len(manifest)} chunk CSVs to {CHUNKS_DIR}/")
    print(f"Wrote manifest: {MANIFEST_OUT} ({len(manifest)} rows)")
    print()
    print("Sample chunk data file (first one):")
    print(pd.read_csv(f"{CHUNKS_DIR}/{gt.iloc[0]['chunk_id']}.csv").to_string(index=False))
    print()
    print("Sample prompt (first chunk):")
    print(manifest.iloc[0]["insight_def"])


if __name__ == "__main__":
    run()
