"""
multi_agent_bivariate_6.py
=============================
Identical to multi_agent_bivariate.py, EXCEPT it imports from
multi_agent_utils_6.py (the 6-check engine, including shift_invariance)
instead of multi_agent_utils.py (the current, 5-check version).

Use this specifically to run the remaining bivariate models
(deepseek-r1, qwen3-32b, kimi-k2.6) so they land on the SAME 6-check
baseline as the models already completed (o3, o4-mini, gpt-4o-mini),
keeping the full bivariate comparison consistent across all 7 models.

Writes to the SAME bivariate_results_multiagent.csv as the original
runner, so all models' rows end up in one consistent file.

Runner for the multi-agent (Profiler -> Planner -> Coder -> Validator)
experiment on the bivariate spatial correlation task.

Completely separate from single_agent_bivariate.py -- does not import
from, call, or modify it. Writes to its own output file so it never
collides with your single-agent or hybrid multi-turn results.

Usage:
    python multi_agent_bivariate.py --model openai/o4-mini
    python multi_agent_bivariate.py --model openai/o4-mini --chunk bivariate_2021_gas_waste
"""

import argparse
import csv
import json
import os
import traceback
from datetime import datetime, timezone

import pandas as pd

from multi_agent_utils_6 import multi_agent_loop
from bivariate_verdict import parse_bivariate_with_citation

MANIFEST_PATH = "chunks_csv/bivariate/bivariate_manifest.csv"
RESULTS_CSV = "results/multi agent/bivariate_results_multiagent.csv"
FAILED_LOG = "logs/multi agent/bivariate_multiagent_failed_runs.log"


def load_manifest():
    df = pd.read_csv(MANIFEST_PATH)
    return {row["chunk_id"]: row.to_dict() for _, row in df.iterrows()}


def append_result(chunk_id, meta, model, predicted, verdict_detail, evidence_text,
                   has_evidence, n_evidence_items, run_dir, ma_result):
    header = [
        "chunk_id", "dataset", "insight_type", "modality", "model", "mode",
        "predicted", "ground_truth_direction", "ground_truth_is_significant",
        "has_evidence", "n_evidence_items", "run_dir", "verdict_detail", "evidence_text",
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
            predicted, meta["ground_truth_direction"], meta["ground_truth_is_significant"],
            has_evidence, n_evidence_items,
            run_dir,
            str(verdict_detail).replace("\n", " ")[:500],
            str(evidence_text).replace("\n", " ")[:1000],
            ma_result["validation_score"], ma_result["validation_total"],
            ma_result["plan_attempts_used"], ma_result["status"],
        ])


def run_one(chunk_id, meta, model):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join("runs", f"{ts}_bivariate_multiagent_{model.replace('/', '-')}_{chunk_id}")
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
        parse_bivariate_with_citation(final_text)

    append_result(chunk_id, meta, model, predicted, verdict_detail, evidence_text,
                  has_evidence, len(cited_numbers), run_dir, ma_result)

    print(f"[multiagent | {model} | {chunk_id}] predicted={predicted} "
          f"ground_truth_direction={meta['ground_truth_direction']} "
          f"validation={ma_result['validation_score']}/{ma_result['validation_total']} "
          f"status={ma_result['status']}",
          flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", default=None, help="run only this chunk_id (default: all 50)")
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
