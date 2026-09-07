"""
single_agent_spatial_correlation_react.py
============================================
ReAct-condition runner for the spatial correlation task (Global/Local
Moran's I). Mirrors single_agent.py's structure closely -- SAME
manifest loader (manifest_io.load_manifest), SAME dual verdict-format
branching (json / casual / casual_with_citation, chosen per-chunk via
the manifest's answer_format field), SAME CSV column layout -- so
results are directly comparable to your other 3 conditions on this
task.

NOTE ON THE UPLOADED FILE: single_agent.py is currently configured for
MULTI-turn (imports from agent_utils_multi_turn, max_iterations=6,
RESULTS_CSV points at the "_spatial_multi" file) -- the single-turn
block is present but commented out. Doesn't affect what's built here;
only the parser branching and CSV schema matter for this file.

CHANGES FROM single_agent.py, clearly marked below:
  1. Imports agent_loop_react from agent_utils_react.py (the SAME file
     already built, tested, and used for Getis-Ord, Outlier, Change-
     Spillover, and Bivariate). No new loop logic. agent_utils.py,
     agent_utils_multi_turn.py, and single_agent.py are all left
     completely untouched.
  2. Output goes to the SAME "ReAct" folder as the other four ReAct
     result files, but a separate file -- never touches results.csv,
     results_spatial_multi.csv, or any other existing file.
  3. One extra column at the END of the CSV: statistical_method --
     extracted from evidence_text using the same generic label=value
     extraction + blocklist as the other ReAct runners. This only
     produces a real label for the "casual"/"casual_with_citation"
     format (the format the file's own docstring says current
     spatial_correlation chunks actually use); for the "json" format
     evidence_text is built as "description: value" pairs, which the
     '=' based regex correctly does not match -- returns "none" for
     those rows rather than a wrong/partial extraction, which is the
     safe, correct behaviour for a format this extraction was never
     meant to parse.
  4. Also writes a react_fidelity.json alongside the run's conversation
     log -- same as the other four ReAct runners.
  5. require_structured_verdict is passed through per-chunk exactly as
     the original does (True for "json" format, False otherwise) --
     agent_loop_react already supports this parameter unchanged.

Usage:
    python single_agent_spatial_correlation_react.py --model openai/gpt-4o-mini
    python single_agent_spatial_correlation_react.py --model openai/gpt-4o-mini --chunk moran_global_2021_crime
"""

import argparse
import csv
import json
import os
import re
import traceback
from datetime import datetime, timezone

# CHANGE 1: new loop, new file -- agent_utils.py / agent_utils_multi_turn.py untouched.
from agent_utils import LLM_MODEL
from agent_utils_react import agent_loop_react

from manifest_io import load_manifest
from casual_verdict import parse_casual_final_answer, parse_casual_with_citation

MANIFEST_PATH = "chunks_csv/manifest.csv"

# CHANGE 2: separate output location -- never touches any existing results file.
RESULTS_DIR = "ReAct"
RESULTS_CSV = os.path.join(RESULTS_DIR, "spatial_correlation_results_react.csv")
FAILED_LOG = os.path.join(RESULTS_DIR, "spatial_correlation_react_failed_runs.log")

SYSTEM_PROMPT = """You are a single AI agent responsible for the entire analysis pipeline,
working in the ReAct (Reasoning + Acting) style: profiling the dataset, planning the
analysis, writing and running code, and reporting results. You must ground every claim in
code output -- never guess numbers. Always print() the numbers you compute so they appear
in the tool output."""

GENERIC_NON_METHOD_LABELS = {"statistical method", "test statistic", "significance measure"}


def _extract_statistical_method(evidence_text):
    """CHANGE 3: same generic extraction as the other four ReAct
    runners, extended to also accept a colon separator -- confirmed
    necessary from real data: this task's models sometimes print
    natural f-strings like print(f"Global Moran's I: {value}") rather
    than the '=' format seen consistently in the other four tasks. Only
    ever produces a real label for the casual / casual_with_citation
    text format -- the json format's "description: value" evidence_text
    would otherwise ALSO match this colon pattern, so callers relying on
    'none' for json-format rows should be aware this is now permissive
    to both separators; in practice current spatial_correlation chunks
    use casual formats only, per this task's own docstring.

    Fix (found from real data): the pipeline's own iteration-limit
    sentinel, "(Stopped: reached iteration limit...)", was being matched
    by the colon separator and returned as if "Stopped" were a method
    name. This sentinel only ever comes from agent_utils_react.py's own
    fallback message, never from genuine model output, so excluding it
    here can never discard real data."""
    if not isinstance(evidence_text, str):
        return "none"
    if evidence_text.strip().startswith("(Stopped:"):
        return "none"
    match = re.search(r"([A-Za-z_\-\s\*']*)[=:]", evidence_text)
    if match:
        label = match.group(1).strip()
        if not label or label.lower() in GENERIC_NON_METHOD_LABELS:
            return "none"
        return label
    return "none"


def build_user_prompt(insight_type, insight_def):
    return f"""Analyse the dataset for the following insight:

INSIGHT TYPE: {insight_type}
{insight_def}
"""


def run_one(chunk_id, meta, model):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(RESULTS_DIR, "runs", f"{ts}_spatialcorr_react_{model.replace('/', '-')}_{chunk_id}")
    os.makedirs(run_dir, exist_ok=True)  # must exist BEFORE agent_loop_react -- sandbox writes here

    answer_format = meta.get("answer_format", "json")
    require_structured = (answer_format == "json")

    prompt = build_user_prompt(meta["insight_type"], meta["insight_def"])
    extra_files = {"ADJACENCY_PATH": meta["adjacency_path"]} if meta.get("adjacency_path") else None

    verdict, final_text, log, react_fidelity = agent_loop_react(
        SYSTEM_PROMPT, meta["insight_def"], meta["csv_path"], run_dir,
        model=model, require_structured_verdict=require_structured,
        extra_files=extra_files, max_iterations=6)

    with open(os.path.join(run_dir, "conversation_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    # CHANGE 4: fidelity record saved alongside the log, for later audit.
    with open(os.path.join(run_dir, "react_fidelity.json"), "w", encoding="utf-8") as f:
        json.dump({"per_turn": react_fidelity,
                   "fraction_compliant": round(sum(react_fidelity) / len(react_fidelity), 3) if react_fidelity else None},
                  f, indent=2)

    if answer_format == "json":
        predicted = verdict.insight_present if verdict else None
        verdict_detail = verdict.verdict_detail if verdict else final_text
        evidence_text = "; ".join(f"{ev.description}: {ev.value}" for ev in verdict.evidence) if verdict else ""
        has_evidence = bool(verdict and len(verdict.evidence) > 0)
        n_evidence_items = len(verdict.evidence) if verdict else 0
    elif answer_format == "casual_with_citation":
        predicted, verdict_detail, evidence_text, has_citation, cited_numbers = parse_casual_with_citation(
            final_text, meta.get("spatial_granularity"))
        has_evidence = has_citation
        n_evidence_items = len(cited_numbers)
    else:  # "casual"
        predicted, verdict_detail, evidence_text = parse_casual_final_answer(
            final_text, meta.get("spatial_granularity"))
        has_evidence = bool(evidence_text)
        n_evidence_items = 1 if evidence_text else 0

    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(f"# ReAct single-agent report: {chunk_id}\n\n")
        f.write(f"**Predicted:** {predicted}\n\n")
        f.write(f"**Detail:** {verdict_detail}\n\n")
        if evidence_text:
            f.write(f"**Answer line:** {evidence_text}\n")

    append_result(chunk_id, meta, model, "react", predicted, verdict_detail,
                  evidence_text, has_evidence, n_evidence_items, run_dir, react_fidelity, answer_format)

    fidelity_str = f"{round(sum(react_fidelity)/len(react_fidelity)*100)}%" if react_fidelity else "n/a"
    print(f"[react | {model} | {chunk_id}] predicted={predicted} "
          f"ground_truth={meta['ground_truth']['insight_present']} "
          f"react_fidelity={fidelity_str}"
          + ("" if predicted is not None else "  <-- NO VALID VERDICT (counts as incorrect)"), flush=True)


def append_result(chunk_id, meta, model, mode, predicted, verdict_detail,
                   evidence_text, has_evidence, n_evidence_items, run_dir, react_fidelity, answer_format):
    header = ["chunk_id", "dataset", "insight_type", "modality", "spatial_granularity", "model", "mode",
              "predicted", "ground_truth", "has_evidence", "n_evidence_items",
              "run_dir", "verdict_detail", "evidence_text",
              "react_fidelity",       # fraction of turns that followed Thought-before-Action
              "statistical_method"]   # CHANGE 3: new column, always last
    os.makedirs(RESULTS_DIR, exist_ok=True)
    exists = os.path.exists(RESULTS_CSV) and os.path.getsize(RESULTS_CSV) > 0
    encoding = "utf-8" if exists else "utf-8-sig"

    # json-format rows use a "description: value" evidence_text -- the
    # colon-aware extraction below would misread that as a method label,
    # so json-format rows are explicitly excluded rather than relying on
    # the current manifest never using this path for this task.
    statistical_method = "none" if answer_format == "json" else _extract_statistical_method(evidence_text)
    fidelity_pct = round(sum(react_fidelity) / len(react_fidelity), 3) if react_fidelity else None

    with open(RESULTS_CSV, "a", newline="", encoding=encoding) as f:
        w = csv.writer(f)
        if not exists:
            w.writerow(header)
        w.writerow([
            chunk_id, meta["dataset"], meta["insight_type"], meta.get("modality", ""),
            meta.get("spatial_granularity", ""),
            model, mode, predicted, meta["ground_truth"]["insight_present"],
            has_evidence, n_evidence_items,
            run_dir,
            str(verdict_detail).replace("\n", " ")[:500],
            str(evidence_text).replace("\n", " ")[:1000],
            fidelity_pct,
            statistical_method,
        ])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunk", default=None, help="run only this chunk_id (default: all)")
    parser.add_argument("--model", default=LLM_MODEL)
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
            os.makedirs(RESULTS_DIR, exist_ok=True)
            with open(FAILED_LOG, "a", encoding="utf-8") as f:
                f.write(f"\n[{ts}] FAILED: mode=react model={args.model} chunk={cid}\n")
                f.write(traceback.format_exc())
                f.write("\n" + "-" * 70 + "\n")
            print(f"[react | {args.model} | {cid}] CRASHED -- logged to {FAILED_LOG}, continuing...", flush=True)

    print(f"\nDone. {failed} chunk(s) failed (see {FAILED_LOG} if >0).")
