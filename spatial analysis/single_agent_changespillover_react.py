"""
single_agent_changespillover_react.py
========================================
ReAct-condition runner for the spatiotemporal change-spillover task.
Mirrors single_agent_changespillover.py's structure closely -- SAME
manifest, SAME parser (casual_verdict.parse_casual_with_citation, same
one used for the original Global/Local Moran's I task since the answer
format is identical), SAME CSV column layout -- so results are directly
comparable to your other 3 conditions on this task.

CHANGES FROM single_agent_changespillover.py, clearly marked below:
  1. Imports agent_loop_react from agent_utils_react.py (the SAME file
     already built, tested, and used for the Getis-Ord and Outlier
     ReAct conditions -- no new loop logic here, just reused).
     agent_utils.py, agent_utils_multi_turn.py, and
     single_agent_changespillover.py are all left completely untouched.
  2. Output goes to the SAME "ReAct" folder as the other two ReAct
     result files, but a separate file -- never overwrites or appends
     to changespillover_results_multi.csv or any other existing file.
  3. One extra column at the END of the CSV: statistical_method --
     extracted from the model's own evidence_text (e.g. "Moran's I",
     "z", "I", "statistical method" -- the same prompt-template
     placeholder bug seen elsewhere in this project), same extraction
     approach as the other two ReAct runners, adapted to strip a
     leading "CLUSTERED ("/"NOT CLUSTERED (" prefix (this task's
     global-granularity format) instead of the outlier task's
     "OUTLIER - HIGH ("/"NOT AN OUTLIER (" or Getis-Ord's "(HOT/COLD, ".
     Returns "none" (not blank) if no label-value pattern is found.
  4. Also writes a react_fidelity.json alongside the run's conversation
     log -- same as the other two ReAct runners.
  5. spatial_granularity is passed through to the parser exactly as in
     the original file, read from the manifest per chunk (this task has
     both global and local variants, same as the Moran's I task).

Usage:
    python single_agent_changespillover_react.py --model openai/gpt-4o-mini
    python single_agent_changespillover_react.py --model openai/gpt-4o-mini --chunk changespillover_2021_2022_crime
"""

import argparse
import csv
import json
import os
import re
import traceback
from datetime import datetime, timezone

import pandas as pd

# CHANGE 1: new loop, new file -- agent_utils.py / agent_utils_multi_turn.py untouched.
from agent_utils import LLM_MODEL
from agent_utils_react import agent_loop_react

from casual_verdict import parse_casual_with_citation

#MANIFEST_PATH = "chunks_csv/changespillover/changespillover_manifest.csv"

MANIFEST_PATH = "chunks_csv/changespillover/changespillover_manifest2.csv" # For additional test cases

# CHANGE 2: separate output location -- never touches any existing results file.
RESULTS_DIR = "ReAct"
#RESULTS_CSV = os.path.join(RESULTS_DIR, "changespillover_results_react.csv")

RESULTS_CSV = os.path.join(RESULTS_DIR, "changespillover_results_react_additional.csv") # For additional test cases
FAILED_LOG = os.path.join(RESULTS_DIR, "changespillover_react_failed_runs.log")

SYSTEM_PROMPT = (
    "You are a single AI agent analyzing spatial data, working in the ReAct "
    "(Reasoning + Acting) style. You have access to a Python sandbox where "
    "you can run code to load, inspect, and analyze data files."
)


def load_manifest():
    df = pd.read_csv(MANIFEST_PATH)
    return {row["chunk_id"]: row.to_dict() for _, row in df.iterrows()}


GENERIC_NON_METHOD_LABELS = {"statistical method", "test statistic", "significance measure"}


def _extract_statistical_method(evidence_text):
    """CHANGE 3: pulls the label the model actually used before its
    FIRST '=' in evidence_text. Strips a leading "CLUSTERED ("/
    "NOT CLUSTERED (" prefix (global format) or a
    "BoroughName (HH/LL/HOT/COLD, " prefix (local format) if present,
    keeping just the statistic label itself. Some labels a model uses
    aren't real method names at all -- "statistical method", "test
    statistic", "significance measure" -- these are generic descriptive
    phrases, not a named statistic, so they're treated as 'none' rather
    than reported as if they identified the method used. Returns 'none'
    (not blank) if no label-value pattern is found."""
    if not isinstance(evidence_text, str):
        return "none"
    match = re.search(r"([A-Za-z_\-\s\*']*)=", evidence_text)
    if match:
        label = match.group(1).strip()
        label = re.sub(r"^.*?(?:NOT\s+CLUSTERED|CLUSTERED)\s*\(\s*", "", label).strip()
        label = re.sub(r"^.*?\((?:HH|LL|HOT|COLD)[^,]*,\s*", "", label).strip()
        if not label or label.lower() in GENERIC_NON_METHOD_LABELS:
            return "none"
        return label
    return "none"


def append_result(chunk_id, meta, model, predicted, verdict_detail, evidence_text,
                   has_evidence, n_evidence_items, run_dir, react_fidelity):
    header = [
        "chunk_id", "dataset", "insight_type", "modality", "model", "mode",
        "window_start", "window_end", "predicted", "ground_truth_present",
        "ground_truth_detail", "has_evidence", "n_evidence_items", "run_dir",
        "verdict_detail", "evidence_text",
        "react_fidelity",       # fraction of turns that followed Thought-before-Action
        "statistical_method",   # CHANGE 3: new column, always last
    ]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    exists = os.path.exists(RESULTS_CSV) and os.path.getsize(RESULTS_CSV) > 0
    encoding = "utf-8" if exists else "utf-8-sig"

    statistical_method = _extract_statistical_method(evidence_text)
    fidelity_pct = round(sum(react_fidelity) / len(react_fidelity), 3) if react_fidelity else None

    with open(RESULTS_CSV, "a", newline="", encoding=encoding) as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow([
            chunk_id, meta["dataset"], meta["insight_type"], meta.get("modality", ""),
            model, "react",
            meta["window_start"], meta["window_end"],
            predicted, meta["ground_truth_present"], meta["ground_truth_detail"],
            has_evidence, n_evidence_items,
            run_dir,
            str(verdict_detail).replace("\n", " ")[:500],
            str(evidence_text).replace("\n", " ")[:1000],
            fidelity_pct,
            statistical_method,
        ])


def run_one(chunk_id, meta, model):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(RESULTS_DIR, "runs", f"{ts}_changespillover_react_{model.replace('/', '-')}_{chunk_id}")
    os.makedirs(run_dir, exist_ok=True)

    extra_files = {"ADJACENCY_PATH": meta["adjacency_path"]} if meta.get("adjacency_path") else None

    verdict, final_text, log, react_fidelity = agent_loop_react(
        SYSTEM_PROMPT, meta["insight_def"], meta["csv_path"], run_dir,
        model=model, require_structured_verdict=False,
        extra_files=extra_files, max_iterations=6)

    with open(os.path.join(run_dir, "conversation_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    # CHANGE 4: fidelity record saved alongside the log, for later audit.
    with open(os.path.join(run_dir, "react_fidelity.json"), "w", encoding="utf-8") as f:
        json.dump({"per_turn": react_fidelity,
                   "fraction_compliant": round(sum(react_fidelity) / len(react_fidelity), 3) if react_fidelity else None},
                  f, indent=2)

    # CHANGE 5: spatial_granularity read from the manifest per chunk, same as original.
    predicted, verdict_detail, evidence_text, has_evidence, cited_numbers = \
        parse_casual_with_citation(final_text, meta.get("spatial_granularity", "global"))

    append_result(chunk_id, meta, model, predicted, verdict_detail, evidence_text,
                  has_evidence, len(cited_numbers), run_dir, react_fidelity)

    fidelity_str = f"{round(sum(react_fidelity)/len(react_fidelity)*100)}%" if react_fidelity else "n/a"
    print(f"[react | {model} | {chunk_id}] predicted={predicted} "
          f"ground_truth={meta['ground_truth_present']} has_evidence={has_evidence} "
          f"react_fidelity={fidelity_str}", flush=True)


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
            os.makedirs(RESULTS_DIR, exist_ok=True)
            with open(FAILED_LOG, "a", encoding="utf-8") as f:
                f.write(f"\n[{ts}] FAILED: mode=react model={args.model} chunk={cid}\n")
                f.write(traceback.format_exc())
                f.write("\n" + "-" * 70 + "\n")
            print(f"[react | {args.model} | {cid}] CRASHED -- logged to {FAILED_LOG}, continuing...", flush=True)

    print(f"\nDone. {failed} chunk(s) failed (see {FAILED_LOG} if >0).")
