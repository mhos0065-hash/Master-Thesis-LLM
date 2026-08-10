"""
generate_cyclehire_chunks.py
===============================
Builds real, runnable chunks for the Cycle Hire self-containment analysis
(previously only tested manually) and merges them into the existing
chunks_csv/manifest.csv alongside the 14 sustainability Moran's I chunks --
so single_agent.py/multi_agent.py/run_experiment.py can run against BOTH
ground truth sets in one sweep.

For each of the 8 windows x 2 metrics (SelfContainmentRatio,
SelfContainmentIndex) x 2 granularities (global, local) = 32 new chunks.
Ground truth comes directly from cyclehire_self_containment_ground_truth.csv
(itself built from your real R output) -- nothing is recomputed here.

Each chunk points the Coder agent at the RAW per-window trip file (not a
pre-aggregated table) plus the shared borough_adjacency.csv, exactly
matching the manual prompts already validated earlier -- the LLM must
compute the self-containment metric itself from raw StartBorough/EndBorough
columns, then reason about spatial clustering using the adjacency list.

Usage:
    python generate_cyclehire_chunks.py
"""

import pandas as pd
import json
import os
import shutil
from manifest_io import load_manifest, save_manifest

GROUND_TRUTH_CSV = "/mnt/user-data/uploads/cyclehire_self_containment_ground_truth.csv"
WINDOWS_DIR = "cyclehire_windows"
CSV_DIR = "chunks_csv"
MANIFEST_PATH = os.path.join(CSV_DIR, "manifest.csv")
ADJACENCY_MANIFEST_PATH = "chunks_csv/borough_adjacency.csv"  # already present

GLOBAL_SELFCONTAINMENT_INSIGHT_DEF = """Does this Self-Containment metric show significant GLOBAL spatial autocorrelation (clustering) across the Cycle Hire service area boroughs? You are given RAW trip-level data (DATASET_PATH, columns include StartBorough, EndBorough) AND a borough adjacency list (ADJACENCY_PATH, columns: borough, neighbor).

First compute the metric per borough using EXACTLY this method:
1. Departures = number of trips that started in that borough
2. Self-contained trips = number of those departures that also ended in the same borough
3. SelfContainmentRatio = Self-contained trips / Departures
4. Arrival share = (trips ending in that borough) / (total trips ending anywhere)
5. Expected self-contained trips (by chance) = Departures x Arrival share
6. SelfContainmentIndex = Self-contained trips / Expected self-contained trips

Then compute Global Moran's I on the requested metric: I = (n/S0) * [sum_i sum_j w_ij*(x_i-xbar)*(x_j-xbar)] / [sum_i (x_i-xbar)^2], using row-standardized weights w_ij = 1/(number of neighbors of borough i) from the adjacency list. Estimate significance via permutation: shuffle the metric values across boroughs at least 200 times, recompute I each time, and estimate a p-value as the fraction of shuffled I values >= your observed I. State FOUND (significant positive clustering, e.g. permutation p < 0.05 AND I > 0) or NOT FOUND, and cite your computed I, p-value, and which metric you used as evidence."""

LOCAL_SELFCONTAINMENT_INSIGHT_DEF = """Does at least one borough form a significant LOCAL spatial cluster (LISA) for this Self-Containment metric? You are given RAW trip-level data (DATASET_PATH, columns include StartBorough, EndBorough) AND a borough adjacency list (ADJACENCY_PATH, columns: borough, neighbor).

First compute the metric per borough using EXACTLY this method:
1. Departures = number of trips that started in that borough
2. Self-contained trips = number of those departures that also ended in the same borough
3. SelfContainmentRatio = Self-contained trips / Departures
4. Arrival share = (trips ending in that borough) / (total trips ending anywhere)
5. Expected self-contained trips (by chance) = Departures x Arrival share
6. SelfContainmentIndex = Self-contained trips / Expected self-contained trips

Then standardize the requested metric to z-scores, and for each borough compute its spatial lag = the average z-score of its neighbors (row-standardized weights from the adjacency list). Classify each borough: HH if z_i>0 and lag>0 (high value surrounded by high values); LL if z_i<0 and lag<0 (low surrounded by low); HL/LH if they disagree (spatial outliers, not clusters). A borough counts as a significant cluster if it is HH or LL AND its local statistic is unusually extreme relative to a permutation-based null (shuffle values across boroughs many times, recompute, treat the observed value as significant if it falls in the extreme ~10% tail). State FOUND if at least one borough qualifies as a significant HH or LL cluster, name the borough(s), state which metric you used, and cite your computed z-score/lag/classification as evidence."""


def run():
    gt = pd.read_csv(GROUND_TRUTH_CSV)
    manifest = load_manifest(MANIFEST_PATH)
    windows_index = pd.read_csv(os.path.join(WINDOWS_DIR, "windows_index.csv"))

    # Copy each window's raw trip file into chunks_csv/ so it's part of the
    # standard chunk data location (matching every other chunk's convention)
    window_csv_paths = {}
    for _, w in windows_index.iterrows():
        src = os.path.join(WINDOWS_DIR, w["filepath"])
        dst_name = f"cyclehire_{w['window_id']}.csv"
        dst = os.path.join(CSV_DIR, dst_name)
        shutil.copy(src, dst)
        window_csv_paths[w["window_id"]] = f"chunks_csv/{dst_name}"

    n_added = 0
    for (window_id, indicator, granularity), group in gt.groupby(["window_id", "indicator", "granularity"]):
        chunk_id = f"cyclehire_{granularity}_{indicator.lower()}_{window_id}"

        if granularity == "global":
            row = group.iloc[0]
            ground_truth = {
                "insight_present": bool(row["is_significant"]),
                "verdict_detail": f"{indicator} Global Moran's I = {row['morans_i']:.4f} "
                                   f"(p={row['p_value']}) -> {row['interpretation']}",
                "metric": {"moran_i": float(row["morans_i"]), "p_value": float(row["p_value"])},
                "acceptance_rule": row["acceptance_rule"],
            }
            insight_def = GLOBAL_SELFCONTAINMENT_INSIGHT_DEF.replace(
                "this Self-Containment metric", f"the {indicator} metric"
            ).replace("the requested metric", indicator)
        else:
            sig = group[group["is_significant"]]
            cluster_boroughs = dict(zip(sig["borough"], sig["cluster_label"]))
            # Preserve the FULL per-borough breakdown -- every borough, not just
            # the significant ones -- matching the exact structure/columns of
            # cyclehire_self_containment_ground_truth.csv / spatial_correlation_ground_truth.csv
            # (z_score, lag_z, morans_i, p_value, cluster_label). Collapsing to
            # only significant boroughs was a real fidelity loss vs. the source.
            per_borough = [
                {
                    "borough": r["borough"],
                    "z_score": float(r["z_score"]),
                    "lag_z": float(r["lag_z"]),
                    "morans_i": float(r["morans_i"]),
                    "p_value": float(r["p_value"]),
                    "cluster_label": r["cluster_label"],
                }
                for _, r in group.iterrows()
            ]
            ground_truth = {
                "insight_present": len(cluster_boroughs) > 0,
                "verdict_detail": (f"{len(cluster_boroughs)} borough(s) form significant local clusters: "
                                    f"{cluster_boroughs}" if cluster_boroughs else
                                    "No borough forms a significant HH or LL local cluster"),
                "metric": {
                    "significant_cluster_boroughs": cluster_boroughs,  # kept for score_moran_boroughs.py
                    "per_borough": per_borough,                         # full detail, all boroughs
                },
                "acceptance_rule": group.iloc[0]["acceptance_rule"],
            }
            insight_def = LOCAL_SELFCONTAINMENT_INSIGHT_DEF.replace(
                "this Self-Containment metric", f"the {indicator} metric"
            ).replace("the requested metric", indicator)

        window_meta = windows_index[windows_index["window_id"] == window_id].iloc[0]
        manifest[chunk_id] = {
            "dataset": f"Cycle Hire Self-Containment ({indicator}, {window_id})",
            "insight_type": "spatial_correlation",
            "modality": "spatial",
            "csv_path": window_csv_paths[window_id],
            "n_rows": -1,  # trip-level file, row count not meaningful here
            "insight_def": insight_def,
            "ground_truth": ground_truth,
            "window_start": window_meta["window_start"],
            "window_end": window_meta["window_end"],
            "window_index": window_id,
            "adjacency_path": ADJACENCY_MANIFEST_PATH,
            "spatial_granularity": granularity,
        }
        n_added += 1

    save_manifest(manifest, MANIFEST_PATH)
    print(f"Added {n_added} Cycle Hire chunks. Total chunks in manifest: {len(manifest)}")


if __name__ == "__main__":
    run()
