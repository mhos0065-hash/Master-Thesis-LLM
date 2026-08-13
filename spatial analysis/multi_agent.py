"""
multi_agent.py
================
MULTI-AGENT pipeline: Profiler -> Planner -> Coder -> Reporter.

Reporter's verdict format is chosen per-chunk via the manifest's
answer_format field, matching single_agent.py:
  - "json"   (default): Pydantic-validated JSON citing computed evidence
              (schemas.py), with a repair retry.
  - "casual" (used by the current spatial_correlation chunks): Reporter is
              asked for the SAME plain-text ending as the standalone
              insight_def prompts -- CLUSTERED/NOT CLUSTERED for Global
              chunks, or a HH/LL borough list for Local chunks -- parsed by
              casual_verdict.parse_casual_final_answer(), with its own
              repair retry.

Profiler runs ONCE PER DATASET FILE, not once per chunk (Section 4.2),
cached in-memory and on disk. Coder's code execution goes through the real
sandbox (sandbox.py).

Usage:
    python multi_agent.py
    python multi_agent.py --chunk moran_global_2023_crime
    python multi_agent.py --model qwen/qwen3-32b

Writes:
    runs/_profiler_cache/<dataset_file>/         (shared across chunks of that file)
    runs/<timestamp>_multi_<model>_<chunk_id>/
        planner.txt  coder_log.json  report.md
    and appends one row to results.csv (used by score_results.py)
"""

import argparse
import csv
import json
import os
import re
import traceback
from datetime import datetime, timezone

from agent_utils import agent_loop, chat, LLM_MODEL
from manifest_io import load_manifest
from schemas import parse_verdict, SCHEMA_INSTRUCTIONS
from casual_verdict import parse_casual_final_answer, parse_casual_with_citation

MANIFEST_PATH = "chunks_csv/manifest.csv"
RUNS_DIR = "runs"
PROFILE_CACHE_DIR = os.path.join(RUNS_DIR, "_profiler_cache")
RESULTS_CSV = "results.csv"

_profile_memory_cache = {}  # in-process cache: abs csv path -> profile summary

CASUAL_ENDING_GLOBAL = """Based on the Coder's computed results, state overall across the whole area
whether neighboring boroughs tend to have similar values to each other (CLUSTERED) or whether the
pattern appears randomly distributed with no relation to which boroughs are next to each other
(NOT CLUSTERED).

End your response with exactly one line:
FINAL ANSWER: CLUSTERED
or
FINAL ANSWER: NOT CLUSTERED"""

CASUAL_ENDING_LOCAL = """Based on the Coder's computed results, identify specific individual boroughs
(not an overall city-wide pattern) that qualify as:
- HOT SPOT: a high value that is ALSO surrounded by other high-value neighboring boroughs
- COLD SPOT: a low value that is ALSO surrounded by other low-value neighboring boroughs

End your response with exactly one line listing every borough you identified, using HH for hot spot
and LL for cold spot:
FINAL ANSWER: BoroughName1 (HH), BoroughName2 (LL), ...

If you find none, write: FINAL ANSWER: none"""

CITATION_ENDING_GLOBAL = """Based on the Coder's computed results, state overall across the whole area
whether neighboring boroughs tend to have similar values to each other (CLUSTERED) or whether the
pattern appears randomly distributed with no relation to which boroughs are next to each other
(NOT CLUSTERED).

End your response with exactly one line, including the computed number(s) you used:
FINAL ANSWER: CLUSTERED (Moran's I = <value>, p = <value>)
or
FINAL ANSWER: NOT CLUSTERED (Moran's I = <value>, p = <value>)"""

CITATION_ENDING_LOCAL = """Based on the Coder's computed results, identify specific individual boroughs
(not an overall city-wide pattern) that qualify as:
- HOT SPOT: a high value that is ALSO surrounded by other high-value neighboring boroughs
- COLD SPOT: a low value that is ALSO surrounded by other low-value neighboring boroughs

End your response with exactly one line listing every borough you identified, using HH for hot spot
and LL for cold spot, and including a computed value for each:
FINAL ANSWER: BoroughName1 (HH, I=<value>), BoroughName2 (LL, I=<value>), ...

If you find none, write: FINAL ANSWER: none"""


def _safe_dirname(path):
    return re.sub(r"[^A-Za-z0-9_-]", "_", os.path.abspath(path))


def profiler_agent(dataset_path, run_dir, model, adjacency_path=None):
    system = "You are the Profiler agent. Examine the dataset's structure only -- do not analyse insights."
    user = (f"Write ```python``` code to load the CSV at this path and print: columns, dtypes, "
            f"row count, and a few sample rows.\nWhen done, respond with FINAL ANSWER: followed by "
            f"a short summary of the dataset structure.")
    extra_files = {"ADJACENCY_PATH": adjacency_path} if adjacency_path else None
    _verdict, summary, log = agent_loop(system, user, dataset_path, run_dir, model=model,
                                         max_iterations=3, require_structured_verdict=False,
                                         extra_files=extra_files)
    return summary, log


def get_or_run_profiler(dataset_path, model, adjacency_path=None):
    """Runs the Profiler once per unique dataset file, caching in memory and
    on disk so subsequent chunks drawing on the same file skip re-profiling
    (Section 4.2 requirement)."""
    abs_path = os.path.abspath(dataset_path)
    if abs_path in _profile_memory_cache:
        return _profile_memory_cache[abs_path], True  # (summary, was_cached)

    cache_dir = os.path.join(PROFILE_CACHE_DIR, _safe_dirname(dataset_path))
    cache_file = os.path.join(cache_dir, "profile_summary.txt")
    os.makedirs(cache_dir, exist_ok=True)

    if os.path.exists(cache_file):
        summary = open(cache_file, encoding="utf-8").read()
        _profile_memory_cache[abs_path] = summary
        return summary, True

    summary, log = profiler_agent(dataset_path, cache_dir, model, adjacency_path=adjacency_path)
    with open(cache_file, "w", encoding="utf-8") as f:
        f.write(summary)
    with open(os.path.join(cache_dir, "profiler_log.json"), "w", encoding="utf-8") as f:
        json.dump(log, f, indent=2)
    _profile_memory_cache[abs_path] = summary
    return summary, False


def planner_agent(profile_summary, insight_type, insight_def, model):
    system = "You are the Planner agent. You do not write or run code. You only produce a short analysis plan."
    user = f"""Dataset profile:
{profile_summary}

Insight to investigate: {insight_type}
Definition: {insight_def}

Write a short, numbered plan (3-5 steps) describing exactly what computation the Coder
agent should perform, which columns to use, and what acceptance criteria to check.
Respond with FINAL ANSWER: followed by the plan."""
    reply = chat([{"role": "system", "content": system}, {"role": "user", "content": user}], model=model)
    plan = reply.split("FINAL ANSWER:")[-1].strip() if "FINAL ANSWER:" in reply else reply
    return plan


def coder_agent(plan, dataset_path, run_dir, model, adjacency_path=None):
    system = ("You are the Coder agent. Follow the given plan exactly, writing and "
              "running python code to compute the requested values. Always print() your results.")
    user = (f"Plan:\n{plan}\n\nExecute this plan on the dataset and report the computed numeric "
            f"results. Respond with FINAL ANSWER: followed by the computed results once done.")
    extra_files = {"ADJACENCY_PATH": adjacency_path} if adjacency_path else None
    _verdict, result, log = agent_loop(system, user, dataset_path, run_dir, model=model,
                                        max_iterations=5, require_structured_verdict=False,
                                        extra_files=extra_files)
    return result, log


def reporter_agent_json(profile_summary, plan, coder_result, insight_type, model, max_repair_attempts=2):
    """JSON verdict path (schemas.py) -- used when answer_format != 'casual'."""
    system = ("You are the Reporter agent. You summarise findings for a non-technical reader. "
              "You do not run code. Ground every claim in the Coder's computed results -- never guess.")
    user = f"""Dataset profile: {profile_summary}
Plan used: {plan}
Coder's computed results: {coder_result}

Insight: {insight_type}
Respond with FINAL ANSWER: followed by the JSON verdict described below.
{SCHEMA_INSTRUCTIONS}"""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    for attempt in range(max_repair_attempts + 1):
        reply = chat(messages, model=model)
        if "FINAL ANSWER:" not in reply:
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": "Respond with FINAL ANSWER: followed by the JSON verdict."})
            continue
        final_text = reply.split("FINAL ANSWER:")[-1].strip()
        verdict, err = parse_verdict(final_text)
        if verdict is not None:
            predicted = verdict.insight_present
            verdict_detail = verdict.verdict_detail
            evidence_text = "; ".join(f"{ev.description}: {ev.value}" for ev in verdict.evidence)
            return predicted, verdict_detail, evidence_text, True, len(verdict.evidence)
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": f"Your FINAL ANSWER could not be parsed: {err}\n"
                         f"Resend FINAL ANSWER: followed by ONLY the corrected JSON object."})
    return None, "(Reporter never produced a valid verdict after repair attempts)", "", False, 0


def reporter_agent_casual(profile_summary, plan, coder_result, insight_type, spatial_granularity,
                           model, max_repair_attempts=2):
    """Plain-text CLUSTERED/HH-LL verdict path (casual_verdict.py) -- used
    when the chunk's answer_format=='casual'. Asks for the SAME ending
    format as the standalone insight_def prompts, so a chunk scored this
    way is directly comparable to the same chunk run through single_agent.py."""
    system = ("You are the Reporter agent. You summarise findings for a non-technical reader. "
              "You do not run code. Ground every claim in the Coder's computed results -- never guess.")
    ending = CASUAL_ENDING_GLOBAL if spatial_granularity == "global" else CASUAL_ENDING_LOCAL
    user = f"""Dataset profile: {profile_summary}
Plan used: {plan}
Coder's computed results: {coder_result}

Insight: {insight_type}

{ending}"""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    for attempt in range(max_repair_attempts + 1):
        reply = chat(messages, model=model)
        if "FINAL ANSWER:" not in reply:
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": "Respond with FINAL ANSWER: followed by the required line."})
            continue
        predicted, verdict_detail, evidence_text = parse_casual_final_answer(reply, spatial_granularity)
        if predicted is not None:
            return predicted, verdict_detail, evidence_text, bool(evidence_text), (1 if evidence_text else 0)
        messages.append({"role": "assistant", "content": reply})
        messages.append({"role": "user", "content": "Your FINAL ANSWER line could not be parsed. "
                         "Resend FINAL ANSWER: followed by ONLY the required format."})
    return None, "(Reporter never produced a parseable answer after repair attempts)", "", False, 0


def reporter_agent_citation(profile_summary, plan, coder_result, insight_type, spatial_granularity,
                             model, max_repair_attempts=2):
    """HYBRID path (casual_verdict.parse_casual_with_citation) -- used when
    answer_format=='casual_with_citation'. Same casual wording as
    reporter_agent_casual, but the FINAL ANSWER line must cite at least one
    computed number (Section 4.3's evidence-citation requirement), retried
    if the model omits it."""
    system = ("You are the Reporter agent. You summarise findings for a non-technical reader. "
              "You do not run code. Ground every claim in the Coder's computed results -- never guess.")
    ending = CITATION_ENDING_GLOBAL if spatial_granularity == "global" else CITATION_ENDING_LOCAL
    user = f"""Dataset profile: {profile_summary}
Plan used: {plan}
Coder's computed results: {coder_result}

Insight: {insight_type}

{ending}"""
    messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]

    for attempt in range(max_repair_attempts + 1):
        reply = chat(messages, model=model)
        if "FINAL ANSWER:" not in reply:
            messages.append({"role": "assistant", "content": reply})
            messages.append({"role": "user", "content": "Respond with FINAL ANSWER: followed by the required line."})
            continue
        predicted, verdict_detail, evidence_text, has_citation, cited_numbers = parse_casual_with_citation(
            reply, spatial_granularity)
        if predicted is not None and has_citation:
            return predicted, verdict_detail, evidence_text, True, len(cited_numbers)
        messages.append({"role": "assistant", "content": reply})
        if predicted is None:
            msg = "Your FINAL ANSWER line could not be parsed. Resend FINAL ANSWER: followed by ONLY the required format."
        else:
            msg = "Your FINAL ANSWER must include at least one cited computed number (e.g. Moran's I value). Resend it with that included."
        messages.append({"role": "user", "content": msg})
    return None, "(Reporter never produced a valid cited verdict after repair attempts)", "", False, 0


def run_one(chunk_id, meta, model):
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = os.path.join(RUNS_DIR, f"{ts}_multi_{model.replace('/', '-')}_{chunk_id}")
    os.makedirs(run_dir, exist_ok=True)

    adjacency_path = meta.get("adjacency_path") or None
    answer_format = meta.get("answer_format", "json")

    profile_summary, was_cached = get_or_run_profiler(meta["csv_path"], model, adjacency_path=adjacency_path)
    open(os.path.join(run_dir, "profiler_summary_used.txt"), "w", encoding="utf-8").write(
        f"[from {'cache' if was_cached else 'fresh profiler run'}]\n\n{profile_summary}")

    plan = planner_agent(profile_summary, meta["insight_type"], meta["insight_def"], model)
    open(os.path.join(run_dir, "planner.txt"), "w", encoding="utf-8").write(plan)

    coder_result, coder_log = coder_agent(plan, meta["csv_path"], run_dir, model, adjacency_path=adjacency_path)
    json.dump(coder_log, open(os.path.join(run_dir, "coder_log.json"), "w", encoding="utf-8"), indent=2)

    if answer_format == "casual_with_citation":
        predicted, verdict_detail, evidence_text, has_evidence, n_evidence_items = reporter_agent_citation(
            profile_summary, plan, coder_result, meta["insight_type"],
            meta.get("spatial_granularity"), model)
    elif answer_format == "casual":
        predicted, verdict_detail, evidence_text, has_evidence, n_evidence_items = reporter_agent_casual(
            profile_summary, plan, coder_result, meta["insight_type"],
            meta.get("spatial_granularity"), model)
    else:
        predicted, verdict_detail, evidence_text, has_evidence, n_evidence_items = reporter_agent_json(
            profile_summary, plan, coder_result, meta["insight_type"], model)

    with open(os.path.join(run_dir, "report.md"), "w", encoding="utf-8") as f:
        f.write(f"# Multi-agent report: {chunk_id}\n\n")
        f.write(f"**Predicted:** {predicted}\n\n")
        f.write(f"**Detail:** {verdict_detail}\n\n")
        if evidence_text:
            f.write(f"**Answer line:** {evidence_text}\n")

    append_result(chunk_id, meta, model, "multi", predicted, verdict_detail,
                  evidence_text, has_evidence, n_evidence_items, run_dir)
    print(f"[multi  | {model} | {chunk_id}] predicted={predicted} "
          f"ground_truth={meta['ground_truth']['insight_present']} "
          f"(profiler {'cached' if was_cached else 'ran fresh'})"
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
                f.write(f"\n[{ts}] FAILED: mode=multi model={args.model} chunk={cid}\n")
                f.write(traceback.format_exc())
                f.write("\n" + "-" * 70 + "\n")
            print(f"[multi | {args.model} | {cid}] CRASHED -- logged to failed_runs.log, continuing...")

    print(f"\nDone. {failed} chunk(s) failed (see failed_runs.log if >0).")
