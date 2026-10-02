"""
single_agent_changespillover.py
==================================
Dedicated single-agent runner for the spatial change-spillover task
(Task 5). Completely separate from all other runners -- reuses
agent_utils.agent_loop() and casual_verdict.py's parse_casual_with_citation()
(the SAME parser used for the original Global Moran's I task, since the
answer format is identical: CLUSTERED / NOT CLUSTERED).

Usage:
    python single_agent_changespillover.py --model openai/gpt-4o-mini
    python single_agent_changespillover.py --model openai/gpt-4o-mini --chunk changespillover_2021_2022_crime
"""

import argparse
import csv
import json
import os
import traceback
from datetime import datetime, timezone

import pandas as pd

#from agent_utils import agent_loop, LLM_MODEL # Single turn

from agent_utils_multi_turn import agent_loop, LLM_MODEL # Multi turn

from casual_verdict import parse_casual_with_citation

#MANIFEST_PATH = "chunks_csv/changespillover/changespillover_manifest.csv"
MANIFEST_PATH = "chunks_csv/changespillover/changespillover_manifest2.csv" # For additional test cases

#RESULTS_CSV = "results/single agent/changespillover_results_multi.csv" # Multi turn

RESULTS_CSV = "results/single agent/changespillover_results_multi_additional.csv" # Multi turn additional test cases .


#RESULTS_CSV = "changespillover_results.csv" # Single turn

#RESULTS_CSV = "changespillover_results_additional.csv" # Single-turn For additional test cases  


FAILED_LOG = "changespillover_failed_runs.log"

SYSTEM_PROMPT = (
    "You are a single AI agent analyzing spatial data. You have access to a Python "
    "sandbox where you can run code to load, inspect, and analyze data files."
)


def load_manifest():
    df = pd.read_csv(MANIFEST_PATH)
    return {row["chunk_id"]: row.to_dict() for _, row in df.iterrows()}


def append_result(chunk_id, meta, model, predicted, verdict_detail, evidence_text,
                   has_evidence, n_evidence_items, run_dir):
    header = [
        "chunk_id", "dataset", "insight_type", "modality", "model", "mode",
        "window_start", "window_end", "predicted", "ground_truth_present",
        "ground_truth_detail", "has_evidence", "n_evidence_items", "run_dir",
        "verdict_detail", "evidence_text",
    ]
    exists = os.path.exists(RESULTS_CSV) and os.path.getsize(RESULTS_CSV) > 0
    encoding = "utf-8" if exists else "utf-8-sig"
    with open(RESULTS_CSV, "a", newline="", encoding=encoding) as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow([
            chunk_id, meta["dataset"], meta["insight_type"], meta.get("modality", ""),
            model, "single",
            meta["window_start"], meta["window_end"],
            predicted, meta["ground_truth_present"], meta["ground_truth_detail"],
            has_evidence, n_evidence_items,
            run_dir,
            str(verdict_detail).replace("\n", " ")[:500],
            str(evidence_text).replace("\n", " ")[:1000],
        ])


def run_one(chunk_id, meta, model):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join("runs", f"{ts}_changespillover_{model.replace('/', '-')}_{chunk_id}")
    os.makedirs(run_dir, exist_ok=True)

    extra_files = {"ADJACENCY_PATH": meta["adjacency_path"]} if meta.get("adjacency_path") else None


    # multi turn
    verdict, final_text, log = agent_loop(
        SYSTEM_PROMPT, meta["insight_def"], meta["csv_path"], run_dir,
        model=model, require_structured_verdict=False,
        extra_files=extra_files, max_iterations=6, answer_from_code_output=True)

    # Single turn
    #verdict, final_text, log = agent_loop(
        #SYSTEM_PROMPT, meta["insight_def"], meta["csv_path"], run_dir,
        #model=model, require_structured_verdict=False,
        #extra_files=extra_files, max_iterations=1, answer_from_code_output=True)

    with open(os.path.join(run_dir, "conversation_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

    predicted, verdict_detail, evidence_text, has_evidence, cited_numbers = \
        parse_casual_with_citation(final_text, meta.get("spatial_granularity", "global"))

    append_result(chunk_id, meta, model, predicted, verdict_detail, evidence_text,
                  has_evidence, len(cited_numbers), run_dir)

    print(f"[single | {model} | {chunk_id}] predicted={predicted} "
          f"ground_truth={meta['ground_truth_present']} has_evidence={has_evidence}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", default=None, help="run only this chunk_id (default: all 21)")
    parser.add_argument("--model", default=LLM_MODEL)
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
                f.write(f"\n[{ts}] FAILED: mode=single model={args.model} chunk={cid}\n")
                f.write(traceback.format_exc())
                f.write("\n" + "-" * 70 + "\n")
            print(f"[single | {args.model} | {cid}] CRASHED -- logged to {FAILED_LOG}, continuing...", flush=True)

    print(f"\nDone. {failed} chunk(s) failed (see {FAILED_LOG} if >0).")
