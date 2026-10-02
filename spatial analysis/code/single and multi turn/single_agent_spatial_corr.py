"""
single_agent.py
=================
SINGLE-AGENT baseline. One agent does profiling, planning, coding, and
reporting all in one loop, for every chunk in chunks_csv/manifest.csv.
Every code block runs in the real sandbox (sandbox.py: subprocess isolation,
timeout, network blocked).

Two verdict formats, chosen per-chunk via the manifest's answer_format field:
  - "json"   (default, for non-spatial_correlation insight types when
              reintroduced): FINAL ANSWER must be valid JSON matching
              schemas.AgentVerdict, citing computed evidence.
  - "casual" (used by the current spatial_correlation chunks): FINAL ANSWER
              is plain text -- "CLUSTERED"/"NOT CLUSTERED" for Global chunks,
              or a "BoroughName (HH), BoroughName (LL), ..."/"none" list for
              Local chunks -- parsed by casual_verdict.parse_casual_final_answer().

Usage:
    python single_agent.py                     # run all chunks, default model
    python single_agent.py --chunk cyclehire_outlier_sample
    python single_agent.py --model qwen/qwen3-32b

Writes one folder per run to:
    runs/<timestamp>_single_<model>_<chunk_id>/
        conversation_log.json
        report.md
        artifacts/           (created by the sandbox if the model saves files)
    and appends one row to results.csv (used by score_results.py)
"""

import argparse
import csv
import json
import os
import traceback
from datetime import datetime, timezone

#from agent_utils import agent_loop, LLM_MODEL # Single turn

from agent_utils_multi_turn import agent_loop, LLM_MODEL # Multi turn

from manifest_io import load_manifest
from casual_verdict import parse_casual_final_answer, parse_casual_with_citation

#MANIFEST_PATH = "chunks_csv/manifest.csv"

MANIFEST_PATH = "chunks_csv/manifest2.csv"  # For additional test cases
RUNS_DIR = "runs"

#RESULTS_CSV = "results/single agent/results_spatial_multi.csv" # multi turn

RESULTS_CSV = "results/single agent/results_spatial_multi_additional.csv" # For additional test cases multi turn

#RESULTS_CSV = "spatial_results.csv" # single turn

#RESULTS_CSV = "spatial_results_additional.csv" # For additional test cases


SYSTEM_PROMPT = """You are a single AI agent responsible for the entire analysis pipeline:
profiling the dataset, planning the analysis, writing and running code, and reporting results.
You must ground every claim in code output -- never guess numbers. Always print() the numbers
you compute so they appear in the tool output."""


def build_user_prompt(insight_type, insight_def):
    return f"""Analyse the dataset for the following insight:

INSIGHT TYPE: {insight_type}
{insight_def}
"""


def run_one(chunk_id, meta, model):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(RUNS_DIR, f"{ts}_single_{model.replace('/', '-')}_{chunk_id}")
    os.makedirs(run_dir, exist_ok=True)  # must exist BEFORE agent_loop -- sandbox writes here

    answer_format = meta.get("answer_format", "json")
    require_structured = (answer_format == "json")

    prompt = build_user_prompt(meta["insight_type"], meta["insight_def"])
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
                #max_iterations=1,
                #answer_from_code_output=True,    
                #extra_files=extra_files)

    with open(os.path.join(run_dir, "conversation_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)

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
        f.write(f"# Single-agent report: {chunk_id}\n\n")
        f.write(f"**Predicted:** {predicted}\n\n")
        f.write(f"**Detail:** {verdict_detail}\n\n")
        if evidence_text:
            f.write(f"**Answer line:** {evidence_text}\n")

    append_result(chunk_id, meta, model, "single", predicted, verdict_detail,
                  evidence_text, has_evidence, n_evidence_items, run_dir)
    print(f"[single | {model} | {chunk_id}] predicted={predicted} "
          f"ground_truth={meta['ground_truth']['insight_present']}"
          + ("" if predicted is not None else "  <-- NO VALID VERDICT (counts as incorrect)"))


def append_result(chunk_id, meta, model, mode, predicted, verdict_detail,
                   evidence_text, has_evidence, n_evidence_items, run_dir):
    header = ["chunk_id", "dataset", "insight_type", "modality", "spatial_granularity", "model", "mode",
              "predicted", "ground_truth", "has_evidence", "n_evidence_items",
              "run_dir", "verdict_detail", "evidence_text"]
    # Check the file actually HAS CONTENT, not just that it exists -- an
    # empty (0-byte) file (e.g. from an accidental empty save, or a stray
    # file-creation command) would otherwise silently skip the header
    # forever, causing every real result row to be misread as column names.
    exists = os.path.exists(RESULTS_CSV) and os.path.getsize(RESULTS_CSV) > 0
    # utf-8-sig ONLY on first creation (writes a BOM so Excel/Windows tools
    # correctly detect UTF-8 instead of defaulting to cp1252 and mangling
    # any non-ASCII characters a model's response might contain, e.g. smart
    # quotes). Subsequent appends use plain utf-8 -- writing utf-8-sig on
    # every append would insert a stray BOM mid-file and corrupt the CSV.
    encoding = "utf-8" if exists else "utf-8-sig"
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
            with open("failed_runs.log", "a", encoding="utf-8") as f:
                f.write(f"\n[{ts}] FAILED: mode=single model={args.model} chunk={cid}\n")
                f.write(traceback.format_exc())
                f.write("\n" + "-" * 70 + "\n")
            print(f"[single | {args.model} | {cid}] CRASHED -- logged to failed_runs.log, continuing...")

    print(f"\nDone. {failed} chunk(s) failed (see failed_runs.log if >0).")
