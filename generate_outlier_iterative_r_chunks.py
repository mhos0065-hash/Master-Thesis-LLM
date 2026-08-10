"""
generate_outlier_iterative_r_chunks.py
=========================================
Builds the 50 per-borough iterative-r spatial outlier chunks from
outlier_iterative_r_ground_truth.csv + the real panel CSVs. Each chunk
asks about ONE specific borough, but the data file always contains ALL
33 boroughs (the computation genuinely needs the whole city -- you
can't compute a regression or a neighbor average from one row alone).
Data files are shared across boroughs that need the same (year,
indicator) combination, to avoid duplicating identical data 50 times.

SAFE FOR YOUR EXISTING PIPELINE: writes only to NEW paths
(chunks_csv/outlierr/*.csv, outlier_iterative_r_manifest.csv) -- never
touches chunks_csv/manifest.csv or any other existing chunk file.

Usage:
    python generate_outlier_iterative_r_chunks.py
Reads:  outlier_iterative_r_ground_truth.csv, 2021.csv, 2022.csv, 2023.csv
Writes: chunks_csv/outlierr/<year>_<indicator>.csv  (one per unique year+indicator needed)
        outlier_iterative_r_manifest.csv  (50 rows, one per chunk)
"""

import pandas as pd

GROUND_TRUTH_PATH = "ground truth/outlier_iterative_r_ground_truth.csv"
PANEL_PATHS = {2021: "spatial analysis/sustainability/dataset/2021.csv", 2022: "spatial analysis/sustainability/dataset/2022.csv", 2023: "spatial analysis/sustainability/dataset/2023.csv"}
ADJACENCY_PATH = "chunks_csv/borough_adjacency.csv"
CHUNKS_DIR = "chunks_csv/outlierr"
MANIFEST_OUT = "chunks_csv/outlierr/outlier_iterative_r_manifest.csv"

IND_TO_COL = {
    "NO2": "no2_annual_mean", "Gas": "gas_kwh_per_consumer", "Energy": "energy_kwh_per_capita",
    "GHG": "ghg_tco2e_per_capita", "Transport": "transport_kwh_per_capita",
    "Crime": "crime_per_1k", "Waste": "waste_tonnes_per_capita",
}
 
METHOD_NOTE = (
    "Use the ITERATIVE-R method (Lu, Chen & Kou, 2003) -- this is more rigorous than a simple "
    "neighbor-difference z-test, and requires ALL of these exact steps, in order:\n"
    "1. For every borough, compute its own value and its neighbors' average value (the spatial lag).\n"
    "2. Fit an OLS regression: own_value ~ neighbor_avg, using ALL 33 boroughs (pass 1).\n"
    "3. Compute residuals, standardize into z-scores.\n"
    "4. Flag any borough with |z| > 1.96 in pass 1 as a CANDIDATE to exclude.\n"
    "5. Refit the SAME regression, this time EXCLUDING those candidates -- this corrects for a known "
    "bias called 'masking', where an extreme borough can distort the very line used to judge it.\n"
    "6. Apply this refined line to ALL 33 boroughs (including the excluded candidates) to get final "
    "predicted values and residuals.\n"
    "7. Standardize the final residuals into z-scores. Do NOT skip step 5 -- fitting only once, "
    "on all boroughs including outliers, is a materially different (and weaker) method."
)
NO_GEOSPATIAL_LIBS_NOTICE = (
    "Note: dedicated spatial-statistics packages (pysal, esda, libpysal, geopandas, or similar) "
    "are NOT installed in this environment -- implement this directly using only pandas and numpy."
)
 
 
def build_prompt(label, borough):
    return f"""I've attached {label} data for 33 London boroughs and a list showing which boroughs are geographic neighbors of each other.
 
Question: Is {borough} a spatial outlier for {label} -- meaning its value is unusually different from what would be expected given its neighbors' values, once you account for the citywide relationship between a borough's value and its neighborhood's average?
 
{METHOD_NOTE}
 
{NO_GEOSPATIAL_LIBS_NOTICE}
 
Focus your final answer specifically on {borough} -- report its OWN final z-score (after the refit in step 6-7), not any other borough's.
 
End your response with exactly one line, including the computed number you used:
FINAL ANSWER: OUTLIER - HIGH (z = <value>)
or
FINAL ANSWER: OUTLIER - LOW (z = <value>)
or
FINAL ANSWER: NOT AN OUTLIER (z = <value>)"""
 
 
def run():
    gt = pd.read_csv(GROUND_TRUTH_PATH)
    assert len(gt) == 50, f"expected 50 ground truth rows, got {len(gt)}"
 
    panels = {yr: pd.read_csv(path) for yr, path in PANEL_PATHS.items()}
 
    # write one shared data file per unique (year, indicator) combination
    written_data_files = {}
    manifest_rows = []
 
    for _, r in gt.iterrows():
        yr = int(r['year'])
        ind = r['indicator']
        borough = r['borough']
        chunk_id = r['chunk_id']
 
        data_key = (yr, ind)
        if data_key not in written_data_files:
            col = IND_TO_COL[ind]
            panel = panels[yr]
            data_df = panel[["borough", col]].rename(columns={col: f"{ind}_value"})
            value_col = f"{ind}_value"
            if data_df[value_col].isna().any():
                data_df[value_col] = data_df[value_col].fillna(data_df[value_col].mean())
            csv_path = f"{CHUNKS_DIR}/{yr}_{ind.lower()}.csv"
            data_df.to_csv(csv_path, index=False)
            written_data_files[data_key] = csv_path
        csv_path = written_data_files[data_key]
 
        manifest_rows.append({
            "chunk_id": chunk_id,
            "dataset": f"Iterative-r Spatial Outlier ({ind}, {yr}, {borough})",
            "insight_type": "iterative_r_outlier",
            "modality": "spatial",
            "csv_path": csv_path,
            "adjacency_path": ADJACENCY_PATH,
            "insight_def": build_prompt(ind, borough),
            "target_borough": borough,
            "ground_truth_direction": r["direction"],
            "ground_truth_z_score": r["z_score_final"],
            "ground_truth_outlier_type": r["outlier_type"],
            "answer_format": "outlierr_with_citation",
        })
 
    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(MANIFEST_OUT, index=False)
 
    print(f"Wrote {len(written_data_files)} shared data CSVs to {CHUNKS_DIR}/")
    print(f"Wrote manifest: {MANIFEST_OUT} ({len(manifest)} rows)")
    print()
    sample = manifest.iloc[0]
    print("Sample chunk:", sample['chunk_id'])
    print("Ground truth:", sample['ground_truth_outlier_type'], "z =", sample['ground_truth_z_score'])
    print()
    print(sample['insight_def'])
 
 
if __name__ == "__main__":
    run()