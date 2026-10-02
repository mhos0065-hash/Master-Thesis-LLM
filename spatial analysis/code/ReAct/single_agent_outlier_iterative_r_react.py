"""
single_agent_outlier_iterative_r_react.py
============================================
ReAct-condition runner for the Iterative-R spatial outlier task. Mirrors
single_agent_outlier_iterative_r.py's structure closely -- SAME manifest,
SAME parser (outlier_iterative_r_verdict.parse_outlierr_with_citation),
SAME CSV column layout -- so results are directly comparable to your
other 3 conditions on this task.

CHANGES FROM single_agent_outlier_iterative_r.py, clearly marked below:
  1. Imports agent_loop_react from agent_utils_react.py (the SAME file
     already built and tested for the Getis-Ord ReAct condition -- no
     new loop logic here, just reused). agent_utils.py and
     single_agent_outlier_iterative_r.py are both left completely
     untouched.
  2. Output goes to the SAME "ReAct" folder as the Getis-Ord ReAct
     results, but a separate file -- never overwrites or appends to
     outlier_iterative_r_results.csv, and never touches
     getisord_results_react.csv either.
  3. One extra column at the END of the CSV: statistical_method --
     extracted from the model's own evidence_text (e.g. "z", "t", "M",
     "R", "std_residual" -- this task's models have used all of these
     at different points this session), same extraction approach as
     the Getis-Ord ReAct runner, adapted for this task's single-value
     "NOT AN OUTLIER (label = value)" format instead of a per-borough
     list. Returns "none" (not blank) if no label-value pattern is
     found.
  4. Also writes a react_fidelity.json alongside the run's conversation
     log -- same as the Getis-Ord ReAct runner -- so ReAct-format
     adherence can be reported honestly rather than assumed.
  5. No answer_from_code_output parameter is passed (agent_loop_react
     doesn't have one) -- it doesn't need it: the loop already accepts
     a FINAL ANSWER from either the model's own prose OR real printed
     code output, whichever the model actually uses, which covers the
     same ground answer_from_code_output=True was doing in the original
     multi-turn block for this task.

Usage:
    python single_agent_outlier_iterative_r_react.py --model openai/gpt-4o-mini
    python single_agent_outlier_iterative_r_react.py --model openai/gpt-4o-mini --chunk outlierr_2021_crime_tower_hamlets
"""

import argparse
import csv
import json
import os
import re
import traceback
from datetime import datetime, timezone

import pandas as pd

# CHANGE 1: new loop, new file -- agent_utils.py itself is untouched.
from agent_utils import LLM_MODEL
from agent_utils_react import agent_loop_react

from outlier_iterative_r_verdict import parse_outlierr_with_citation

MANIFEST_PATH = "chunks_csv/outlierr/outlier_iterative_r_manifest.csv"

# CHANGE 2: separate output location -- never touches outlier_iterative_r_results.csv.
RESULTS_DIR = "ReAct"
RESULTS_CSV = os.path.join(RESULTS_DIR, "outlier_iterative_r_results_react.csv")
FAILED_LOG = os.path.join(RESULTS_DIR, "outlier_iterative_r_react_failed_runs.log")

SYSTEM_PROMPT = (
    "You are a single AI agent analyzing spatial data, working in the ReAct "
    "(Reasoning + Acting) style. You have access to a Python sandbox where "
    "you can run code to load, inspect, and analyze data files."
)


def load_manifest():
    df = pd.read_csv(MANIFEST_PATH)
    return {row["chunk_id"]: row.to_dict() for _, row in df.iterrows()}


def _extract_statistical_method(evidence_text):
    """CHANGE 3: pulls the label the model actually used before its
    FIRST '=' in evidence_text (e.g. "z", "t", "M", "R", "std_residual"
    -- all seen from real runs of this task this session). Strips a
    leading "OUTLIER - HIGH (" / "OUTLIER - LOW (" / "NOT AN OUTLIER ("
    prefix if present, keeping just the statistic label itself. Returns
    'none' (not blank) if no label-value pattern is found."""
    if not isinstance(evidence_text, str):
        return "none"
    match = re.search(r"([A-Za-z_\-\s\*]*)=", evidence_text)
    if match:
        label = match.group(1).strip()
        label = re.sub(r"^.*?(?:OUTLIER\s*-\s*(?:HIGH|LOW)|NOT AN OUTLIER)\s*\(\s*", "", label).strip()
        return label if label else "none"
    return "none"


def append_result(chunk_id, meta, model, predicted, verdict_detail, evidence_text,
                   has_evidence, n_evidence_items, run_dir, react_fidelity):
    header = [
        "chunk_id", "dataset", "insight_type", "modality", "model", "mode", "target_borough",
        "predicted", "ground_truth_direction", "ground_truth_z_score", "ground_truth_outlier_type",
        "has_evidence", "n_evidence_items", "run_dir", "verdict_detail", "evidence_text",
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
            model, "react", meta["target_borough"],
            predicted, meta["ground_truth_direction"], meta["ground_truth_z_score"],
            meta["ground_truth_outlier_type"],
            has_evidence, n_evidence_items,
            run_dir,
            str(verdict_detail).replace("\n", " ")[:500],
            str(evidence_text).replace("\n", " ")[:1000],
            fidelity_pct,
            statistical_method,
        ])


def run_one(chunk_id, meta, model):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(RESULTS_DIR, "runs", f"{ts}_outlierr_react_{model.replace('/', '-')}_{chunk_id}")
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

    predicted, verdict_detail, evidence_text, has_evidence, cited_numbers = \
        parse_outlierr_with_citation(final_text)

    append_result(chunk_id, meta, model, predicted, verdict_detail, evidence_text,
                  has_evidence, len(cited_numbers), run_dir, react_fidelity)

    fidelity_str = f"{round(sum(react_fidelity)/len(react_fidelity)*100)}%" if react_fidelity else "n/a"
    print(f"[react | {model} | {chunk_id}] predicted={predicted} "
          f"ground_truth={meta['ground_truth_direction']} has_evidence={has_evidence} "
          f"react_fidelity={fidelity_str}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", default=None, help="run only this chunk_id (default: all 50)")
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
