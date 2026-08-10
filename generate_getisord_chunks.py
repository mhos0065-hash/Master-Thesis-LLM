"""
generate_getisord_chunks.py
==============================
Builds the 21 Getis-Ord Gi* chunks (7 indicators x 3 years) from
getisord_ground_truth.csv + the real panel CSVs. Deliberately mirrors
your existing Local Moran's I chunk structure -- one chunk per
(indicator, year), asking the model to find ALL significant boroughs at
once, listed with hot/cold labels and a computed value each -- so
Getis-Ord results are directly comparable to your existing Local Moran's
I results using the same prompt shape and scoring approach.

SAFE FOR YOUR EXISTING PIPELINE: writes only to NEW paths
(chunks_csv/getisord/*.csv, getisord_manifest.csv) -- never touches
chunks_csv/manifest.csv or any existing chunk file.

Usage:
    python generate_getisord_chunks.py
Reads:  getisord_ground_truth.csv, 2021.csv, 2022.csv, 2023.csv
Writes: chunks_csv/getisord/<chunk_id>.csv  (21 files)
        getisord_manifest.csv
"""

import pandas as pd
import json

GROUND_TRUTH_PATH = "ground truth/getisord_ground_truth.csv"
PANEL_PATHS = {2021: "spatial analysis/sustainability/dataset/2021.csv", 2022: "spatial analysis/sustainability/dataset/2022.csv", 2023: "spatial analysis/sustainability/dataset/2023.csv"}
ADJACENCY_PATH = "chunks_csv/borough_adjacency.csv"
CHUNKS_DIR = "chunks_csv/getisord"
MANIFEST_OUT = "chunks_csv/getisord/getisord_manifest.csv"

IND_TO_COL = {
    "NO2": "no2_annual_mean", "Gas": "gas_kwh_per_consumer", "Energy": "energy_kwh_per_capita",
    "GHG": "ghg_tco2e_per_capita", "Transport": "transport_kwh_per_capita",
    "Crime": "crime_per_1k", "Waste": "waste_tonnes_per_capita",
}

PERMUTATION_NOTE = (
    "Getis-Ord Gi* does not use a permutation test -- it uses a z-score computed directly "
    "from the data (comparing the sum of each borough's value AND its neighbors' values "
    "against what would be expected under complete spatial randomness). Use the standard "
    "critical values: |z| > 1.96 for 95% confidence, |z| > 2.576 for 99% confidence."
)
NO_GEOSPATIAL_LIBS_NOTICE = (
    "Note: dedicated spatial-statistics packages (pysal, esda, libpysal, geopandas, or similar) "
    "are NOT installed in this environment -- don't spend turns trying different import paths "
    "for them. Implement this directly using only pandas and numpy."
)


MISSING_VALUE_NOTICE = (
    "Note: some boroughs may have a missing (NaN) value for this indicator. You MUST explicitly "
    "replace it before any computation, using code like: "
    "df['col'] = df['col'].fillna(df['col'].mean()). "
    "Do NOT rely on pandas functions like .sum() or .mean() to 'skip' NaN values automatically -- "
    "they do this silently and INCONSISTENTLY (e.g. a borough's neighbor-count may still include a "
    "missing neighbor even though its value gets silently dropped from a sum, corrupting the result "
    "for every borough near it). After the explicit fillna() step, there should be ZERO remaining "
    "NaN values anywhere in your data before you compute anything else. "
    "However, a borough whose OWN value was originally missing should NOT itself be reported as "
    "a hot or cold spot in your final answer (even though its imputed value should still be used "
    "when computing OTHER boroughs' neighborhood sums, since it's still a real neighbor)."
)


def build_prompt(label):
    return f"""I've attached {label} data for 33 London boroughs and a list showing which boroughs are geographic neighbors of each other.

Question: Using the Getis-Ord Gi* statistic, identify boroughs where there is an unusually intense CONCENTRATION of high {label} (a HOT spot) or low {label} (a COLD spot), considering each borough TOGETHER WITH its neighbors as one combined local area.

This is different from asking whether a borough simply resembles its immediate neighbors (that would be Local Moran's I) -- Gi* tests whether the SUM of values in a borough and its neighbors combined is unusually large or small compared to the citywide average, regardless of how similar the borough is to any one neighbor individually.

To compute this: for each borough, sum its own value and its neighbors' values (a borough always counts itself as part of its own neighborhood for this specific statistic -- this is what distinguishes Gi* from other local statistics), then standardize this sum into a z-score using the mean and variance expected under complete spatial randomness. {MISSING_VALUE_NOTICE} {NO_GEOSPATIAL_LIBS_NOTICE} {PERMUTATION_NOTE}

End your response with exactly one line listing every borough you identified, using HOT for hot spot and COLD for cold spot, and including the computed Gi* z-score for each:
FINAL ANSWER: BoroughName1 (HOT, Gi*=<value>), BoroughName2 (COLD, Gi*=<value>), ...

If you find none, write: FINAL ANSWER: none"""


def run():
    gt = pd.read_csv(GROUND_TRUTH_PATH)
    assert len(gt) == 693, f"expected 693 ground truth rows, got {len(gt)}"

    panels = {yr: pd.read_csv(path) for yr, path in PANEL_PATHS.items()}

    manifest_rows = []
    for (yr, ind), group in gt.groupby(['year', 'indicator']):
        col = IND_TO_COL[ind]
        chunk_id = f"getisord_{yr}_{ind.lower()}"

        panel = panels[yr]
        chunk_df = panel[["borough", col]].rename(columns={col: f"{ind}_value"})
        csv_path = f"{CHUNKS_DIR}/{chunk_id}.csv"
        chunk_df.to_csv(csv_path, index=False)

        # ground truth for scoring: boroughs with real hot/cold spots, borough:direction:z
        sig = group[group['is_significant']]
        gt_boroughs = "; ".join(f"{r.borough}:{r.direction}:{r.gi_star_z}" for _, r in sig.iterrows())

        manifest_rows.append({
            "chunk_id": chunk_id,
            "dataset": f"Getis-Ord Gi* ({ind}, {yr})",
            "insight_type": "getisord_hotspot",
            "modality": "spatial",
            "csv_path": csv_path,
            "adjacency_path": ADJACENCY_PATH,
            "insight_def": build_prompt(ind),
            "ground_truth_boroughs": gt_boroughs if gt_boroughs else "(none)",
            "answer_format": "getisord_with_citation",
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(MANIFEST_OUT, index=False)

    print(f"Wrote {len(manifest)} chunk CSVs to {CHUNKS_DIR}/")
    print(f"Wrote manifest: {MANIFEST_OUT} ({len(manifest)} rows)")
    print()
    print("Sample chunk (Crime, 2021):")
    sample = manifest[manifest['chunk_id'] == 'getisord_2021_crime'].iloc[0]
    print("Ground truth boroughs:", sample['ground_truth_boroughs'])
    print()
    print(sample['insight_def'])


if __name__ == "__main__":
    run()
