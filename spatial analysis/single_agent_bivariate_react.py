"""
single_agent_bivariate_react.py
==================================
ReAct-condition runner for the bivariate spatial correlation task
(Cross-variable Spatial Association Detection). Mirrors
single_agent_bivariate.py's structure closely -- SAME manifest, SAME
parser (bivariate_verdict.parse_bivariate_with_citation), SAME CSV
column layout -- so results are directly comparable to your other 3
conditions on this task.

NOTE ON THE UPLOADED FILE: single_agent_bivariate.py is currently
configured for MULTI-turn (imports from agent_utils_multi_turn,
max_iterations=6, RESULTS_CSV points at the "_multi" file) -- the
single-turn block is present but commented out. This doesn't affect
what's built here; only the parser and CSV schema matter for this file.

CHANGES FROM single_agent_bivariate.py, clearly marked below:
  1. Imports agent_loop_react from agent_utils_react.py (the SAME file
     already built, tested, and used for Getis-Ord, Outlier, and
     Change-Spillover). No new loop logic. agent_utils.py,
     agent_utils_multi_turn.py, and single_agent_bivariate.py are all
     left completely untouched.
  2. Output goes to the SAME "ReAct" folder as the other three ReAct
     result files, but a separate file -- never touches
     bivariate_results_multi.csv or any other existing file.
  3. One extra column at the END of the CSV: statistical_method --
     extracted from the model's own evidence_text. This task's exact
     verdict wording wasn't known in advance (e.g. it may be a Lee's L
     statistic, or something else), so rather than guess and hardcode
     task-specific prefix words to strip, this reuses ONLY the core
     extraction regex + the generic-label blocklist -- confirmed by
     direct testing to already correctly handle every verdict format
     seen across this project's other three tasks (Getis-Ord's
     "(HOT/COLD, ", the outlier task's "OUTLIER - HIGH (", change-
     spillover's "CLUSTERED (") without needing any task-specific
     stripping rule at all, since the character-class-restricted regex
     naturally stops at any '(' or ',' regardless of what precedes it.
     Same blocklist as the other three runners: "statistical method",
     "test statistic", "significance measure" map to "none". Returns
     "none" (not blank) if no label-value pattern is found.
  4. Also writes a react_fidelity.json alongside the run's conversation
     log -- same as the other three ReAct runners.

Usage:
    python single_agent_bivariate_react.py --model openai/gpt-4o-mini
    python single_agent_bivariate_react.py --model openai/gpt-4o-mini --chunk bivariate_2021_gas_waste
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

from bivariate_verdict import parse_bivariate_with_citation

MANIFEST_PATH = "chunks_csv/bivariate/bivariate_manifest.csv"

# CHANGE 2: separate output location -- never touches any existing results file.
RESULTS_DIR = "ReAct"
RESULTS_CSV = os.path.join(RESULTS_DIR, "bivariate_results_react.csv")
FAILED_LOG = os.path.join(RESULTS_DIR, "bivariate_react_failed_runs.log")

SYSTEM_PROMPT = (
    "You are a single AI agent analyzing spatial data, working in the ReAct "
    "(Reasoning + Acting) style. You have access to a Python sandbox where "
    "you can run code to load, inspect, and analyze data files."
)

GENERIC_NON_METHOD_LABELS = {"statistical method", "test statistic", "significance measure"}


def load_manifest():
    df = pd.read_csv(MANIFEST_PATH)
    return {row["chunk_id"]: row.to_dict() for _, row in df.iterrows()}


def _extract_statistical_method(evidence_text):
    """CHANGE 3: pulls the label the model actually used before its
    FIRST '=' in evidence_text. Deliberately does NOT strip any
    task-specific prefix -- confirmed unnecessary by direct testing,
    since the character-class-restricted regex below only ever matches
    contiguous letters/spaces/apostrophe/underscore/hyphen/asterisk
    immediately before '=', which already naturally excludes anything
    before a '(' or ',' regardless of what verdict words precede it.
    Maps known generic non-method labels to 'none'. Returns 'none' (not
    blank) if no label-value pattern is found."""
    if not isinstance(evidence_text, str):
        return "none"
    match = re.search(r"([A-Za-z_\-\s\*']*)=", evidence_text)
    if match:
        label = match.group(1).strip()
        if not label or label.lower() in GENERIC_NON_METHOD_LABELS:
            return "none"
        return label
    return "none"


def append_result(chunk_id, meta, model, predicted, verdict_detail, evidence_text,
                   has_evidence, n_evidence_items, run_dir, react_fidelity):
    header = [
        "chunk_id", "dataset", "insight_type", "modality", "model", "mode",
        "predicted", "ground_truth_direction", "ground_truth_is_significant",
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
            model, "react",
            predicted, meta["ground_truth_direction"], meta["ground_truth_is_significant"],
            has_evidence, n_evidence_items,
            run_dir,
            str(verdict_detail).replace("\n", " ")[:500],
            str(evidence_text).replace("\n", " ")[:1000],
            fidelity_pct,
            statistical_method,
        ])


def run_one(chunk_id, meta, model):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(RESULTS_DIR, "runs", f"{ts}_bivariate_react_{model.replace('/', '-')}_{chunk_id}")
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
        parse_bivariate_with_citation(final_text)

    append_result(chunk_id, meta, model, predicted, verdict_detail, evidence_text,
                  has_evidence, len(cited_numbers), run_dir, react_fidelity)

    fidelity_str = f"{round(sum(react_fidelity)/len(react_fidelity)*100)}%" if react_fidelity else "n/a"
    print(f"[react | {model} | {chunk_id}] predicted={predicted} "
          f"ground_truth_direction={meta['ground_truth_direction']} has_evidence={has_evidence} "
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
            print(f"[react | {args.model} | {cid}] CRASHED -- logged to {FAILED_LOG}, continuing...",
                  flush=True)

    print(f"\nDone. {failed} chunk(s) failed (see {FAILED_LOG} if >0).")
