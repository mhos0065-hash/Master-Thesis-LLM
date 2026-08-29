"""
multi_agent_getisord.py  (compatible with the OLDER multi_agent_utils.py
-- the one already used for your bivariate runs, WITHOUT modification)
=============================================================================
Same task as before (Getis-Ord Gi* hotspot detection), but adapted to
work with a multi_agent_utils.py that predates answer_format_instructions
and the '*' character / list-building fixes. Nothing in
multi_agent_utils.py needs to change for this to work.

How the workarounds function, each confirmed by direct test:
1. No answer_format_instructions parameter exists in the old file, so
   the custom format instructions are embedded directly into the
   `question` text instead (question already flows through to every
   agent's prompt).
2. The old file's value-parsing regex can't handle the '*' in "Gi*=" --
   so the Coder is instructed to ALSO print a redundant, asterisk-free
   "gi_value=<value>" right next to each required "Gi*=<value>". This
   is read by the old parser for internal validation (finite/shift/
   determinism checks); the required "Gi*=" is still present for
   getisord_verdict.py, the real downstream scorer, which needs it
   and is unaffected by the extra label.
3. The old file's real_variable_reference check only recognizes an
   f-string with a placeholder ON THE SAME LINE as "FINAL ANSWER" --
   so the Coder is instructed to build the joined string INSIDE the
   f-string's {...} (e.g. f"FINAL ANSWER: {', '.join(results)}"),
   not via string concatenation outside an f-string.

Usage:
    python multi_agent_getisord.py --model openai/o4-mini
    python multi_agent_getisord.py --model openai/o4-mini --chunk getisord_2021_crime
"""

import argparse
import csv
import json
import os
import traceback
from datetime import datetime, timezone

import pandas as pd

from multi_agent_utils import multi_agent_loop
from getisord_verdict import parse_getisord_with_citation

MANIFEST_PATH = "chunks_csv/getisord/getisord_manifest.csv"
RESULTS_CSV = "results/multi agent/getisord_results_multiagent.csv"
FAILED_LOG = "logs/multi agent/getisord_multiagent_failed_runs.log"

FORMAT_INSTRUCTIONS_BLOCK = (
    "\n\n---\nREQUIRED OUTPUT FORMAT (read carefully, this overrides any "
    "other format example you may have seen):\n"
    "Your code must ALSO print `data_points_used = <integer>` -- the "
    "number of boroughs actually used in your analysis (after "
    "excluding any missing values), same as any other task.\n"
    "Then print one FINAL ANSWER line in this form:\n"
    "FINAL ANSWER: Borough1 (HOT, Gi*=<value>), Borough2 (COLD, Gi*=<value>), ...\n"
    "-- list EVERY borough that is a statistically significant hot or "
    "cold spot. 'Gi*' is just an example label -- if your chosen method "
    "produces a different statistic (e.g. a z-score from a different "
    "formulation,  etc.), use its own name instead, "
    "but the value after '=' must always be a real, computed number "
    "for that borough. If no borough is significant, print exactly:\n"
    "FINAL ANSWER: none\n"
    "Build the line using an f-string where the borough list is joined "
    "INSIDE the curly braces, e.g.:\n"
    "    print(f\"FINAL ANSWER: {', '.join(results)}\")\n"
    "-- not string concatenation like print(f\"FINAL ANSWER: \" + \", \".join(results)), "
    "which is NOT acceptable. Use real, computed values from your own "
    "code -- never hand-typed numbers.\n---\n"
)


def load_manifest():
    df = pd.read_csv(MANIFEST_PATH)
    return {row["chunk_id"]: row.to_dict() for _, row in df.iterrows()}


def append_result(chunk_id, meta, model, predicted, verdict_detail, evidence_text,
                   has_evidence, n_evidence_items, run_dir, ma_result):
    header = [
        "chunk_id", "dataset", "insight_type", "modality", "model", "mode",
        "predicted_boroughs", "n_predicted_boroughs", "ground_truth_boroughs",
        "has_evidence", "n_evidence_items", "run_dir", "verdict_detail", "evidence_text",
        "validation_score", "validation_total", "plan_attempts_used", "status",
    ]
    exists = os.path.exists(RESULTS_CSV) and os.path.getsize(RESULTS_CSV) > 0
    encoding = "utf-8" if exists else "utf-8-sig"

    if predicted is None:
        predicted_str, n_predicted = "", 0
    else:
        predicted_str = "; ".join(f"{b}:{d}:{z}" for b, (d, z) in predicted.items())
        n_predicted = len(predicted)

    with open(RESULTS_CSV, "a", newline="", encoding=encoding) as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow([
            chunk_id, meta["dataset"], meta["insight_type"], meta.get("modality", ""),
            model, "multiagent",
            predicted_str, n_predicted, meta["ground_truth_boroughs"],
            has_evidence, n_evidence_items,
            run_dir,
            str(verdict_detail).replace("\n", " ")[:500],
            str(evidence_text).replace("\n", " ")[:1000],
            ma_result["validation_score"], ma_result["validation_total"],
            ma_result["plan_attempts_used"], ma_result["status"],
        ])


def run_one(chunk_id, meta, model):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join("runs", f"{ts}_getisord_multiagent_{model.replace('/', '-')}_{chunk_id}")
    os.makedirs(run_dir, exist_ok=True)

    extra_files = {"ADJACENCY_PATH": meta["adjacency_path"]} if meta.get("adjacency_path") else None
    question_with_format = meta["insight_def"] + FORMAT_INSTRUCTIONS_BLOCK

    ma_result = multi_agent_loop(
        question=question_with_format, dataset_path=meta["csv_path"], run_dir=run_dir,
        model=model, extra_files=extra_files, max_plan_count=3, max_coder_retries=2,
    )

    with open(os.path.join(run_dir, "attempts_log.json"), "w", encoding="utf-8") as f:
        json.dump(ma_result, f, indent=2, default=str)

    final_text = ma_result["final_answer"] or ""
    predicted, verdict_detail, evidence_text, has_evidence, cited_numbers = \
        parse_getisord_with_citation(final_text)

    append_result(chunk_id, meta, model, predicted, verdict_detail, evidence_text,
                  has_evidence, len(cited_numbers), run_dir, ma_result)

    n_pred = len(predicted) if predicted is not None else "unparseable"
    print(f"[multiagent | {model} | {chunk_id}] predicted_boroughs={n_pred} "
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
