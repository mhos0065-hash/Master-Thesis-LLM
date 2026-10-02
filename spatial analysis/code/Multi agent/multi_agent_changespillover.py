"""
multi_agent_changespillover.py
==================================
Runner for the multi-agent (Profiler -> Planner -> Coder -> Validator)
experiment on the spatial change-spillover task (V2: raw yearly values,
model computes the change itself -- NOT pre-computed deltas).

Uses the DEFAULT answer_format_instructions (no customization needed):
the manifest's own question text already spells out the exact required
format ("FINAL ANSWER: CLUSTERED (test statistic = <value>, p = <value>)"
or "NOT CLUSTERED"), and multi_agent_utils.py's default Coder prompt
already uses a generic <verdict> placeholder, not hardcoded to specific
words -- the same reason multi_agent_bivariate.py never needed a custom
format either.

check_shift_invariance is not referenced here since it has been removed
from multi_agent_utils.py entirely.

Note for reference: Moran's I computed on a DELTA (year2 - year1) is
still shift-invariant to a constant added to BOTH years, since the
shift cancels in the subtraction before Moran's I (itself shift-
invariant) is even applied -- confirmed algebraically earlier this
project, though moot now that the check has been removed.

Usage:
    python multi_agent_changespillover.py --model openai/o4-mini
    python multi_agent_changespillover.py --model openai/o4-mini --chunk changespillover_2021_2022_crime
"""

import argparse
import csv
import json
import os
import traceback
from datetime import datetime, timezone

import pandas as pd

from multi_agent_utils import multi_agent_loop
from casual_verdict import parse_casual_with_citation

#MANIFEST_PATH = "chunks_csv/changespillover/changespillover_manifest.csv"
MANIFEST_PATH = "chunks_csv/changespillover/changespillover_manifest2.csv"  # For additional test cases


#RESULTS_CSV = "results/multi agent/changespillover_results_multiagent.csv"
RESULTS_CSV = "results/multi agent/changespillover_results_multiagent_additional.csv"  # For additional test cases


FAILED_LOG = "logs/multi agent/changespillover_multiagent_failed_runs.log"


def load_manifest():
    df = pd.read_csv(MANIFEST_PATH)
    return {row["chunk_id"]: row.to_dict() for _, row in df.iterrows()}


def append_result(chunk_id, meta, model, predicted, verdict_detail, evidence_text,
                   has_evidence, n_evidence_items, run_dir, ma_result):
    header = [
        "chunk_id", "dataset", "insight_type", "modality", "model", "mode",
        "predicted", "ground_truth_present", "has_evidence", "n_evidence_items",
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
            model, "multiagent",
            predicted, meta["ground_truth_present"],
            has_evidence, n_evidence_items,
            run_dir,
            str(verdict_detail).replace("\n", " ")[:500],
            str(evidence_text).replace("\n", " ")[:1000],
            ma_result["validation_score"], ma_result["validation_total"],
            ma_result["plan_attempts_used"], ma_result["status"],
        ])


def run_one(chunk_id, meta, model):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join("runs", f"{ts}_changespillover_multiagent_{model.replace('/', '-')}_{chunk_id}")
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

    print(f"[multiagent | {model} | {chunk_id}] predicted={predicted} "
          f"ground_truth_present={meta['ground_truth_present']} "
          f"validation={ma_result['validation_score']}/{ma_result['validation_total']} "
          f"status={ma_result['status']}",
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", default=None, help="run only this chunk_id (default: all 21)")
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    manifest = load_manifest()
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
