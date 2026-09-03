"""
multi_agent_spatial_correlation.py
======================================
Runner for the multi-agent (Profiler -> Planner -> Coder -> Validator)
experiment on the spatial_correlation task -- Global + Local Moran's I --
using the SAME 5-check deterministic validator + retry architecture
(multi_agent_utils.py) as bivariate, Getis-Ord, change-spillover, and
outlier, for methodological consistency across every multi-agent task
this session. This deliberately does NOT reuse the older multi_agent.py
(Profiler->Planner->Coder->Reporter, no validator/retry loop) that
already existed for this task -- that architecture isn't comparable to
the other five tasks' results.

Uses the DEFAULT answer_format_instructions (no customization needed):
the manifest's own insight_def already fully specifies the exact
required FINAL ANSWER format for BOTH granularities:
  - global: "CLUSTERED (statistical method = <value>, p = <value>)"
  - local:  "BoroughName (HH, I=<value>), ..." or "none"
-- same reasoning as bivariate/changespillover never needing a custom
format either.

Ground truth is accessed via manifest_io.load_manifest()'s nested
{"ground_truth": {"insight_present": bool, ...}} shape (this task's own
convention, unlike the other four tasks' flat manifest columns) --
reused here rather than rebuilt, since it's already tested
infrastructure shared with single_agent.py.

Usage:
    python multi_agent_spatial_correlation.py --model openai/o4-mini
    python multi_agent_spatial_correlation.py --model openai/o4-mini --chunk moran_global_2021_crime
"""

import argparse
import csv
import json
import os
import traceback
from datetime import datetime, timezone

from multi_agent_utils import multi_agent_loop
from manifest_io import load_manifest
from casual_verdict import parse_casual_with_citation

MANIFEST_PATH = "chunks_csv/manifest.csv"
RESULTS_CSV = "results/multi agent/spatial_correlation_results_multiagent.csv"
FAILED_LOG = "logs/multi agent/spatial_correlation_multiagent_failed_runs.log"


def append_result(chunk_id, meta, model, predicted, verdict_detail, evidence_text,
                   has_evidence, n_evidence_items, run_dir, ma_result):
    header = [
        "chunk_id", "dataset", "insight_type", "modality", "spatial_granularity", "model", "mode",
        "predicted", "ground_truth", "has_evidence", "n_evidence_items",
        "run_dir", "verdict_detail", "evidence_text",
        "validation_score", "validation_total", "plan_attempts_used", "status",
    ]
    exists = os.path.exists(RESULTS_CSV) and os.path.getsize(RESULTS_CSV) > 0
    encoding = "utf-8" if exists else "utf-8-sig"
    with open(RESULTS_CSV, "a", newline="", encoding=encoding) as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow([
            chunk_id, meta["dataset"], meta["insight_type"], meta.get("modality", ""),
            meta.get("spatial_granularity", ""),
            model, "multiagent",
            predicted, meta["ground_truth"]["insight_present"],
            has_evidence, n_evidence_items,
            run_dir,
            str(verdict_detail).replace("\n", " ")[:500],
            str(evidence_text).replace("\n", " ")[:1000],
            ma_result["validation_score"], ma_result["validation_total"],
            ma_result["plan_attempts_used"], ma_result["status"],
        ])


def run_one(chunk_id, meta, model):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join("runs", f"{ts}_spatialcorr_multiagent_{model.replace('/', '-')}_{chunk_id}")
    os.makedirs(run_dir, exist_ok=True)

    extra_files = {"ADJACENCY_PATH": meta["adjacency_path"]} if meta.get("adjacency_path") else None

    ma_result = multi_agent_loop(
        question=meta["insight_def"], dataset_path=meta["csv_path"], run_dir=run_dir,
        model=model, extra_files=extra_files, max_plan_count=3, max_coder_retries=2,
    )

    with open(os.path.join(run_dir, "attempts_log.json"), "w", encoding="utf-8") as f:
        json.dump(ma_result, f, indent=2, default=str)

    final_text = ma_result["final_answer"] or ""
    predicted, verdict_detail, evidence_text, has_evidence, cited_numbers = \
        parse_casual_with_citation(final_text, meta.get("spatial_granularity"))

    append_result(chunk_id, meta, model, predicted, verdict_detail, evidence_text,
                  has_evidence, len(cited_numbers), run_dir, ma_result)

    print(f"[multiagent | {model} | {chunk_id}] granularity={meta.get('spatial_granularity')} "
          f"predicted={predicted} ground_truth={meta['ground_truth']['insight_present']} "
          f"validation={ma_result['validation_score']}/{ma_result['validation_total']} "
          f"status={ma_result['status']}",
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", default=None, help="run only this chunk_id (default: all)")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    manifest = load_manifest(MANIFEST_PATH)
    chunk_ids = [args.chunk] if args.chunk else list(manifest.keys())

    failed = 0
    for cid in chunk_ids:
        try:
            run_one(cid, manifest[cid], args.model)
        except Exception as e:
            failed += 1
            ts = datetime.now(timezone.utc).isoformat()
            with open(FAILED_LOG, "a", encoding="utf-8") as f:
                f.write(f"\n[{ts}] FAILED: mode=multiagent model={args.model} chunk={cid}\n")
                f.write(traceback.format_exc())
                f.write("\n" + "-" * 70 + "\n")
            print(f"[multiagent | {args.model} | {cid}] CRASHED -- logged to {FAILED_LOG}, continuing...",
                  flush=True)

    print(f"\nDone. {failed} chunk(s) failed (see {FAILED_LOG} if >0).")
