"""
generate_distance_decay_chunks.py
=====================================
Builds the 50 distance-decay (gravity model) chunks from
distance_decay_ground_truth.csv + the real borough-pair trip/distance
data. Each chunk gives the model a REAL subset of one origin borough's
actual outgoing trips (5-7 real destination pairs, genuinely sampled --
never fabricated), and asks it to test whether trip volume declines with
distance via a log-log regression, matching the exact method your R
script used for the citywide aggregate.

SAFE FOR YOUR EXISTING PIPELINE: writes only to NEW paths
(chunks_csv/distdecay/*.csv, chunks_csv/distdecay/distance_decay_manifest.csv)
-- never touches any other task's files.

Usage:
    python generate_distance_decay_chunks.py
Reads:  distance_decay_ground_truth.csv, distance_decay_cyclehire.csv
Writes: chunks_csv/distdecay/<chunk_id>.csv  (50 files)
        chunks_csv/distdecay/distance_decay_manifest.csv
"""

import pandas as pd

GROUND_TRUTH_PATH = "ground truth/distance_decay_ground_truth.csv"
RAW_DATA_PATH = "chunks_csv/distdecay/distance_decay_cyclehire.csv"
CHUNKS_DIR = "chunks_csv/distdecay"
MANIFEST_OUT = "chunks_csv/distdecay/distance_decay_manifest.csv"

METHOD_NOTE = (
    "This is a GRAVITY MODEL / spatial interaction analysis (Wilson, 1971) -- it tests whether "
    "trip volume between places falls off as the distance between them grows, the foundational "
    "relationship in transportation geography.\n"
    "Method, in order:\n"
    "1. Take the log of n_trips and the log of distance_km for every row.\n"
    "2. Fit an OLS regression: log(n_trips) ~ log(distance_km).\n"
    "3. Check the slope's p-value for significance (p < 0.05) and its sign.\n"
    "4. Classify: DISTANCE_DECAY_PRESENT if significant AND the slope is NEGATIVE (trips fall "
    "off with distance, the expected gravity-model pattern); POSITIVE_DISTANCE_RELATIONSHIP if "
    "significant AND the slope is POSITIVE (trips rise with distance -- unusual); "
    "NO_DISTANCE_DECAY if not statistically significant either way."
)
NO_GEOSPATIAL_LIBS_NOTICE = (
    "Note: dedicated spatial-statistics or geocoding packages (pysal, esda, geopandas, or "
    "similar) are NOT installed in this environment -- implement this directly using only "
    "pandas, numpy, and scipy.stats (scipy IS available for the regression's p-value)."
)


def build_prompt(borough, n_pairs):
    return f"""I've attached real bike-share trip data for {n_pairs} destination boroughs reachable from {borough}, including the number of real trips taken and the real distance between the two boroughs' centroids.

Question: Does the number of trips from {borough} to each destination decline as the distance to that destination increases -- the "distance-decay" or gravity-model relationship?

{METHOD_NOTE}

{NO_GEOSPATIAL_LIBS_NOTICE}

End your response with exactly one line, including the computed slope and p-value:
FINAL ANSWER: DISTANCE_DECAY_PRESENT (slope = <value>, p = <value>)
or
FINAL ANSWER: POSITIVE_DISTANCE_RELATIONSHIP (slope = <value>, p = <value>)
or
FINAL ANSWER: NO_DISTANCE_DECAY (slope = <value>, p = <value>)"""


def run():
    gt = pd.read_csv(GROUND_TRUTH_PATH)
    assert len(gt) == 50, f"expected 50 ground truth rows, got {len(gt)}"

    raw = pd.read_csv(RAW_DATA_PATH)

    manifest_rows = []
    for _, r in gt.iterrows():
        start_borough = r["start_borough"]
        end_boroughs = r["end_boroughs"].split(";")
        chunk_id = r["chunk_id"]

        chunk_data = raw[(raw["StartBorough"] == start_borough) & (raw["EndBorough"].isin(end_boroughs))]
        chunk_data = chunk_data[["StartBorough", "EndBorough", "n_trips", "distance_km"]].copy()
        assert len(chunk_data) == len(end_boroughs), \
            f"{chunk_id}: expected {len(end_boroughs)} rows, got {len(chunk_data)}"

        csv_path = f"{CHUNKS_DIR}/{chunk_id}.csv"
        chunk_data.to_csv(csv_path, index=False)

        manifest_rows.append({
            "chunk_id": chunk_id,
            "dataset": f"Distance-Decay Gravity Model ({start_borough})",
            "insight_type": "distance_decay",
            "modality": "spatial",
            "csv_path": csv_path,
            "insight_def": build_prompt(start_borough, len(end_boroughs)),
            "ground_truth_direction": r["direction"],
            "ground_truth_interpretation": r["interpretation"],
            "ground_truth_slope": r["slope_beta"],
            "ground_truth_p_value": r["p_value"],
            "ground_truth_r_squared": r["r_squared"],
            "answer_format": "distdecay_with_citation",
        })

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(MANIFEST_OUT, index=False)

    print(f"Wrote {len(manifest)} chunk CSVs to {CHUNKS_DIR}/")
    print(f"Wrote manifest: {MANIFEST_OUT} ({len(manifest)} rows)")
    print()
    sample = manifest.iloc[0]
    print("Sample chunk:", sample["chunk_id"])
    print("Ground truth:", sample["ground_truth_interpretation"],
          "slope=", sample["ground_truth_slope"], "p=", sample["ground_truth_p_value"])
    print()
    print(sample["insight_def"])


if __name__ == "__main__":
    run()
