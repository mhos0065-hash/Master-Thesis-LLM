"""
single_agent_getisord_react.py
=================================
ReAct-condition runner for the Getis-Ord Gi* task. Mirrors
single_agent_getisord.py's structure closely -- SAME manifest, SAME
parser (getisord_verdict.parse_getisord_with_citation), SAME CSV column
layout -- so results are directly comparable to your other 3 conditions.

CHANGES FROM single_agent_getisord.py, clearly marked below:
  1. Imports agent_loop_react from the NEW agent_utils_react.py, not
     agent_loop from agent_utils.py. agent_utils.py and
     single_agent_getisord.py are both left completely untouched.
  2. Output goes to a NEW "ReAct" folder, not getisord_results.csv --
     never overwrites or appends to any existing results file.
  3. One extra column at the END of the CSV: statistical_method --
     extracted from the model's own evidence_text (the label it used
     before "=", e.g. "Gi*", "z-score", "statistical method"), reusing
     the same extraction approach as score_getisord.py's
     parse_borough_labels(). This is what the MODEL actually reported
     using for that chunk, not an assumed constant -- consistent with
     this project's existing finding that models don't always use the
     same label even when performing the same underlying computation.
  4. Also writes a react_fidelity.json alongside the run's conversation
     log: what fraction of this chunk's turns actually followed the
     Thought-before-Action pattern, from agent_utils_react's
     check_react_fidelity() -- so fidelity can be reported honestly
     rather than assumed from the prompt.

Usage:
    python single_agent_getisord_react.py --model openai/gpt-4o-mini
    python single_agent_getisord_react.py --model openai/gpt-4o-mini --chunk getisord_2021_crime
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

from getisord_verdict import parse_getisord_with_citation

MANIFEST_PATH = "chunks_csv/getisord/getisord_manifest.csv"

# CHANGE 2: separate output location -- never touches getisord_results.csv.
RESULTS_DIR = "ReAct"
RESULTS_CSV = os.path.join(RESULTS_DIR, "getisord_results_react.csv")
FAILED_LOG = os.path.join(RESULTS_DIR, "getisord_react_failed_runs.log")

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
    FIRST '=' in evidence_text (e.g. "Gi*", "z-score", "statistical
    method") -- same approach as score_getisord.py's
    parse_borough_labels(), applied here to a single chunk's overall
    evidence_text rather than per-borough. Returns '' if no label-value
    pattern is found (e.g. evidence_text is 'none' or empty)."""
    if not isinstance(evidence_text, str):
        return ""
    match = re.search(r"([A-Za-z_\-\s\*]*)=", evidence_text)
    if match:
        label = match.group(1).strip()
        # strip a leading borough name + "(HOT/COLD, " prefix if present,
        # keeping just the statistic label itself
        label = re.sub(r"^.*?\((?:HOT|COLD)[^,]*,\s*", "", label).strip()
        return label
    return ""


def append_result(chunk_id, meta, model, predicted, verdict_detail, evidence_text,
                   has_evidence, n_evidence_items, run_dir, react_fidelity):
    header = [
        "chunk_id", "dataset", "insight_type", "modality", "model", "mode",
        "predicted_boroughs", "n_predicted_boroughs", "ground_truth_boroughs",
        "has_evidence", "n_evidence_items", "run_dir", "verdict_detail", "evidence_text",
        "react_fidelity",       # fraction of turns that followed Thought-before-Action
        "statistical_method",   # CHANGE 3: new column, always last
    ]
    os.makedirs(RESULTS_DIR, exist_ok=True)
    exists = os.path.exists(RESULTS_CSV) and os.path.getsize(RESULTS_CSV) > 0
    encoding = "utf-8" if exists else "utf-8-sig"

    if predicted is None:
        predicted_str, n_predicted = "", 0
    else:
        predicted_str = "; ".join(f"{b}:{d}:{z}" for b, (d, z) in predicted.items())
        n_predicted = len(predicted)

    statistical_method = _extract_statistical_method(evidence_text)
    fidelity_pct = round(sum(react_fidelity) / len(react_fidelity), 3) if react_fidelity else None

    with open(RESULTS_CSV, "a", newline="", encoding=encoding) as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow([
            chunk_id, meta["dataset"], meta["insight_type"], meta.get("modality", ""),
            model, "react",
            predicted_str, n_predicted, meta["ground_truth_boroughs"],
            has_evidence, n_evidence_items,
            run_dir,
            str(verdict_detail).replace("\n", " ")[:500],
            str(evidence_text).replace("\n", " ")[:1000],
            fidelity_pct,
            statistical_method,
        ])


def run_one(chunk_id, meta, model):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(RESULTS_DIR, "runs", f"{ts}_getisord_react_{model.replace('/', '-')}_{chunk_id}")
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
        parse_getisord_with_citation(final_text)

    append_result(chunk_id, meta, model, predicted, verdict_detail, evidence_text,
                  has_evidence, len(cited_numbers), run_dir, react_fidelity)

    n_pred = len(predicted) if predicted is not None else "unparseable"
    fidelity_str = f"{round(sum(react_fidelity)/len(react_fidelity)*100)}%" if react_fidelity else "n/a"
    print(f"[react | {model} | {chunk_id}] predicted_boroughs={n_pred} has_evidence={has_evidence} "
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
            print(f"[react | {args.model} | {cid}] CRASHED -- logged to {FAILED_LOG}, continuing...",
                  flush=True)

    print(f"\nDone. {failed} chunk(s) failed (see {FAILED_LOG} if >0).")
