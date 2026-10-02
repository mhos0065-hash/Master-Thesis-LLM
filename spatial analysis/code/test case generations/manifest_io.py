"""
manifest_io.py
================
Load/save the chunk manifest as a flat CSV instead of manifest.json.

CSV columns:
    chunk_id, dataset, insight_type, modality, csv_path, n_rows, insight_def,
    ground_truth_present, ground_truth_detail, ground_truth_rule,
    ground_truth_metric_json, window_start, window_end, window_index,
    adjacency_path, spatial_granularity, answer_format

`modality` is "spatial" or "non_spatial" -- Section 4.7 requires results
disaggregated by this dimension alongside pipeline config/model/insight type.

`window_start`/`window_end`/`window_index` are populated only for chunks
produced by the sliding-window streaming generator (Section 4.5) -- empty
string for ordinary static chunks. `window_index` lets you group a
dataset's chunks back into their original window sequence to measure
classification consistency across consecutive/overlapping windows.

`adjacency_path` is populated only for spatial_correlation chunks that need
a second input file (a borough adjacency/weights list) alongside the main
csv_path -- empty string otherwise. Threaded through to agent_loop()'s
extra_files={"ADJACENCY_PATH": ...} so the Coder agent can load both files
and compute Moran's I itself.

`spatial_granularity` is "global", "local", or "" (for non-spatial-
correlation chunks). Global and Local Moran's I chunks BOTH use
insight_type="spatial_correlation" (they're the same insight category at
different granularities per your literature review), but they need
DIFFERENT evaluation: Global is a single binary FOUND/NOT FOUND per chunk
(aggregate across many chunks for P/R/F1 -- see score_results.py), while
Local is a set of named boroughs per chunk (set-based P/R/F1 WITHIN one
chunk -- see score_moran_boroughs.py). Without this field, grouping
results by insight_type alone would silently average these two different
tasks together into one misleading number.

`answer_format` is "casual" or "json" (default "json" for backward
compatibility with other insight types, when reintroduced). "casual" means
the insight_def prompt ends with a plain-text FINAL ANSWER line
(CLUSTERED/NOT CLUSTERED, or a BoroughName (HH)/(LL) list) rather than a
JSON schema -- single_agent.py/multi_agent.py check this field to decide
whether to call agent_loop() with require_structured_verdict=True (JSON,
parsed via schemas.py) or False (plain text, parsed via
casual_verdict.py's parse_casual_final_answer()).

`ground_truth_metric_json` holds the insight-specific numbers (z-scores,
Gini, Pearson r, share-by-category, etc.) as a JSON string in one cell,
since that part's shape genuinely varies by insight type and doesn't
flatten cleanly into fixed columns. Everything else is a normal column.

load_manifest() returns the exact same nested dict shape the rest of the
pipeline (single_agent.py, multi_agent.py, run_experiment.py) already
expects, so only the two lines that do the loading/saving need to change --
agent_loop, scoring, etc. are untouched.
"""

import json
import pandas as pd

MANIFEST_CSV = "chunks_csv/manifest.csv"


def save_manifest(records: dict, path: str = MANIFEST_CSV):
    """records: {chunk_id: {dataset, insight_type, modality, csv_path, n_rows,
    insight_def, ground_truth: {insight_present, verdict_detail,
    acceptance_rule, metric}, window_start?, window_end?, window_index?}}"""
    rows = []
    for chunk_id, meta in records.items():
        gt = meta["ground_truth"]
        rows.append({
            "chunk_id": chunk_id,
            "dataset": meta["dataset"],
            "insight_type": meta["insight_type"],
            "modality": meta.get("modality", "non_spatial"),
            "csv_path": meta["csv_path"],
            "n_rows": meta["n_rows"],
            "insight_def": meta["insight_def"],
            "ground_truth_present": gt["insight_present"],
            "ground_truth_detail": gt["verdict_detail"],
            "ground_truth_rule": gt["acceptance_rule"],
            "ground_truth_metric_json": json.dumps(gt.get("metric", {})),
            "window_start": meta.get("window_start", ""),
            "window_end": meta.get("window_end", ""),
            "window_index": meta.get("window_index", ""),
            "adjacency_path": meta.get("adjacency_path", ""),
            "spatial_granularity": meta.get("spatial_granularity", ""),
            "answer_format": meta.get("answer_format", "json"),
        })
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return path


def load_manifest(path: str = MANIFEST_CSV) -> dict:
    df = pd.read_csv(path)
    # pandas reads empty CSV cells as NaN (a float), and NaN is truthy in
    # Python -- without this fillna, `if meta.get("adjacency_path")` would
    # incorrectly evaluate True for every chunk, not just ones that actually
    # have an adjacency file. Force these optional columns to real empty
    # strings instead.
    for optional_col in ("window_start", "window_end", "window_index", "adjacency_path",
                         "spatial_granularity", "answer_format"):
        if optional_col in df.columns:
            df[optional_col] = df[optional_col].fillna("")

    manifest = {}
    for _, row in df.iterrows():
        manifest[row["chunk_id"]] = {
            "dataset": row["dataset"],
            "insight_type": row["insight_type"],
            "modality": row.get("modality", "non_spatial"),
            "csv_path": row["csv_path"],
            "n_rows": int(row["n_rows"]),
            "insight_def": row["insight_def"],
            "ground_truth": {
                "insight_present": bool(row["ground_truth_present"]),
                "verdict_detail": row["ground_truth_detail"],
                "acceptance_rule": row["ground_truth_rule"],
                "metric": json.loads(row["ground_truth_metric_json"]),
            },
            "window_start": row.get("window_start", ""),
            "window_end": row.get("window_end", ""),
            "window_index": row.get("window_index", ""),
            "adjacency_path": row.get("adjacency_path", ""),
            "spatial_granularity": row.get("spatial_granularity", ""),
            "answer_format": row.get("answer_format", "json") or "json",
        }
    return manifest
