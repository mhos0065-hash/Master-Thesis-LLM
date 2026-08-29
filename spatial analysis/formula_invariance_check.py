"""
formula_invariance_check.py
==============================
Shared, format-agnostic shift-invariance re-check, usable from any
scoring script. Applies the SAME check to single-agent and multi-agent
results alike, since both log formats are handled here -- this lives
entirely in the scoring layer, not in the generation pipeline, so it
adds no complexity to multi_agent_utils.py or agent_utils.py.

Mathematical basis (same as used earlier this session): any statistic
built on deviations from the mean (Moran's I, Lee's L, Getis-Ord,
z-scores) is provably unaffected by adding a constant to the raw data,
since (x+c) - mean(x+c) = x - mean(x). NOT valid for tasks whose
statistic uses a nonlinear transform of raw values (e.g. distance-decay's
log-log regression) -- pass check_applicable=False for those.

SAFETY: never modifies the real dataset files -- runs entirely from an
isolated temp directory, confirmed by never writing inside the original
dataset's own directory. Uses mkdtemp() with explicit, error-tolerant
cleanup (not the TemporaryDirectory() context manager), since that
context manager's automatic cleanup can fail with a Windows file-lock
error and silently discard an already-successful result.
"""

import json
import os
import re
import shutil
import tempfile

import pandas as pd

from sandbox import run_sandboxed_code

SHIFT_CONSTANT = 1000.0
TOLERANCE = 1e-3

_FINAL_ANSWER_RE = re.compile(r"FINAL ANSWER:\s*(.*)", re.IGNORECASE)
_KV_PAIR_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_ \-()*]*?)\s*=\s*([\-0-9.eE]+|nan)", re.IGNORECASE)
_CODE_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


def _extract_statistic(text):
    m = _FINAL_ANSWER_RE.search(text or "")
    if not m:
        return None
    pairs = _KV_PAIR_RE.findall(m.group(1))
    return pairs[0][1] if pairs else None


def _is_finite(s):
    if s is None:
        return False
    try:
        f = float(s)
        return f == f and abs(f) != float("inf")
    except ValueError:
        return False


def extract_final_code(run_dir):
    """Returns the code that actually produced the FINAL ANSWER, from
    either log format. Returns None if no code can be found."""
    multi_agent_log = os.path.join(run_dir, "attempts_log.json")
    if os.path.exists(multi_agent_log):
        with open(multi_agent_log, "r", encoding="utf-8") as f:
            log = json.load(f)
        target = log.get("final_answer")
        for attempt in log.get("attempts", []):
            if attempt["coder_result"].get("final_answer") == target:
                return attempt["coder_result"].get("code")
        if log.get("attempts"):
            return log["attempts"][-1]["coder_result"].get("code")
        return None

    single_agent_log = os.path.join(run_dir, "conversation_log.json")
    if os.path.exists(single_agent_log):
        with open(single_agent_log, "r", encoding="utf-8") as f:
            log = json.load(f)
        # take the LAST assistant message's code block -- the one that
        # produced the final, reported answer (relevant for multi-turn
        # single-agent logs with more than one assistant turn)
        for msg in reversed(log):
            if msg.get("role") == "assistant":
                match = _CODE_BLOCK_RE.search(msg.get("content", ""))
                if match:
                    return match.group(1)
        return None

    return None


def _shift_and_rerun(code_text, dataset_path, extra_files):
    """Re-runs code_text against a copy of dataset_path shifted by a
    constant, from an isolated temp directory. Returns the resulting
    statistic (str) or None."""
    cwd_before = os.getcwd()
    tmp_root = tempfile.mkdtemp()
    output = None
    try:
        rel_dir = os.path.dirname(dataset_path) or "."
        shifted_dir = os.path.join(tmp_root, rel_dir)
        os.makedirs(shifted_dir, exist_ok=True)

        df = pd.read_csv(dataset_path)
        for col in df.columns:
            if pd.api.types.is_numeric_dtype(df[col]):
                df[col] = df[col] + SHIFT_CONSTANT
        shifted_csv = os.path.join(shifted_dir, os.path.basename(dataset_path))
        df.to_csv(shifted_csv, index=False)

        if extra_files:
            for _, path in extra_files.items():
                if path and os.path.exists(path):
                    dest_dir = os.path.join(tmp_root, os.path.dirname(path) or ".")
                    os.makedirs(dest_dir, exist_ok=True)
                    shutil.copy(path, os.path.join(tmp_root, path))

        run_dir = os.path.join(tmp_root, "run")
        os.makedirs(run_dir, exist_ok=True)

        os.chdir(tmp_root)
        output = run_sandboxed_code(code_text, run_dir, shifted_csv, extra_vars=extra_files)
    except Exception:
        output = None
    finally:
        os.chdir(cwd_before)
        shutil.rmtree(tmp_root, ignore_errors=True)

    if output is None:
        return None
    return _extract_statistic(output)


def check_shift_invariant(run_dir, dataset_path, adjacency_path, original_stat_str,
                           check_applicable=True):
    """Returns True/False/None.
    True  = re-run under a shift produced (approximately) the same statistic.
    False = statistic changed substantially, or became non-finite -- likely bug.
    None  = inconclusive (no code found, original statistic wasn't finite to
            begin with -- already caught by the fabrication check, or
            check_applicable=False for this task/method)."""
    if not check_applicable:
        return None
    if not _is_finite(original_stat_str):
        return None
    code_text = extract_final_code(run_dir)
    if not code_text:
        return None

    extra_files = {"ADJACENCY_PATH": adjacency_path} if adjacency_path and pd.notna(adjacency_path) else None
    shifted_stat = _shift_and_rerun(code_text, dataset_path, extra_files)
    if not _is_finite(shifted_stat):
        return False

    denom = max(abs(float(original_stat_str)), 1e-9)
    rel_diff = abs(float(original_stat_str) - float(shifted_stat)) / denom
    return rel_diff <= TOLERANCE
