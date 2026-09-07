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
_COLON_PAIR_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_ \-()*']*?)\s*:\s*([\-0-9.eE]+|nan)\b", re.IGNORECASE)
_CODE_BLOCK_RE = re.compile(r"```python\s*\n(.*?)```", re.DOTALL)


def _looks_like_bare_diagnostic_integer(value_str):
    """True for a plain integer with no decimal point or exponent (e.g.
    '32', '164', '500') -- almost always a row/permutation/neighbor
    COUNT printed alongside the real statistic, never the statistic
    itself (confirmed against real re-executed output this session:
    'Remaining rows after NaN removal: 32 | Lee's L: -0.138...' -- a
    naive first-match search would otherwise grab '32'). Real spatial
    statistics (Lee's L, correlations, z-scores) are essentially always
    printed with a decimal point in this project's actual outputs."""
    return "." not in value_str and "e" not in value_str.lower() and value_str.lower() != "nan"


def _pick_best_value(pairs):
    """Given a list of (label, value_str) matches, prefers the FIRST one
    that doesn't look like a bare diagnostic integer -- falls back to
    the literal first match only if EVERY candidate looks like a bare
    integer (better than returning nothing)."""
    if not pairs:
        return None
    for _, value in pairs:
        if not _looks_like_bare_diagnostic_integer(value):
            return value
    return pairs[0][1]


def _extract_statistic(text):
    """Primary path (UNCHANGED): looks for the statistic right after a
    literal 'FINAL ANSWER:' marker -- correct for single-agent/multi-
    agent code, which (via answer_from_code_output=True) always prints
    that marker itself as part of its own output.

    FALLBACK 1 (additive): if no 'FINAL ANSWER:' marker is found
    anywhere in the re-executed output, this is expected for ReAct code
    -- agent_loop_react() has no answer_from_code_output equivalent, and
    its normal path has the model print just the raw computed statistic
    from code (e.g. "Lee's L: 0.23, p = 0.04"), stating FINAL ANSWER:
    separately in a LATER prose-only message that is never re-executed
    here. Without this fallback, EVERY ReAct row would come back False
    (a confirmed "bug" flag) purely because the marker convention
    differs, not because anything is actually wrong -- confirmed by
    direct comparison against this project's own React harness
    (agent_utils_react.py / single_agent_bivariate_react.py) against
    single_agent_bivariate.py's answer_from_code_output=True convention.
    Searches the WHOLE output for the first plausible label=value pair
    instead -- the same generic, anchor-free pattern every scoring
    script's own _extract_raw_statistic() already uses.

    FALLBACK 2 (NEW, additive): if no '=' sign is found ANYWHERE either,
    tries a colon-separated 'Label: value' pattern instead. Confirmed
    necessary by direct diagnostic: deepseek-chat-v3-0324 and
    qwen3-32b's own ReAct code consistently prints results with a colon
    ("Lee's L: -0.0117", "Observed correlation: -0.0274...") and NEVER
    an '=' sign anywhere in their output -- without this, 100% of both
    models' chunks came back as a confirmed shift-invariance FAILURE,
    not because either model's code was wrong, but because nothing
    could be extracted from its differently-formatted output to compare
    against the original.

    Both fallback tiers prefer a value that isn't a bare diagnostic
    integer (see _pick_best_value) -- confirmed necessary since some
    real outputs print an unrelated row/permutation count using the
    exact same separator before the real statistic appears (e.g.
    "Remaining rows after NaN removal: 32 | Lee's L: -0.138...").

    Each fallback tier is only ever reached when the previous one
    failed, so this can only find MORE genuine cases, never fewer --
    single-agent/multiagent code (which always has the '=' marker) is
    completely unaffected."""
    m = _FINAL_ANSWER_RE.search(text or "")
    if m:
        pairs = _KV_PAIR_RE.findall(m.group(1))
        if pairs:
            return pairs[0][1]

    eq_pairs = _KV_PAIR_RE.findall(text or "")
    best = _pick_best_value(eq_pairs)
    if best is not None:
        return best

    colon_pairs = _COLON_PAIR_RE.findall(text or "")
    return _pick_best_value(colon_pairs)


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


def _shift_and_rerun(code_text, dataset_path, extra_files, shift_only_column=None):
    """Re-runs code_text against a copy of dataset_path shifted by a
    constant, from an isolated temp directory. Returns the resulting
    statistic (str) or None.

    shift_only_column: None (default, UNCHANGED behavior for bivariate/
    getisord/outlier-r/spatial) shifts EVERY numeric column by the same
    constant. If given a column name (or substring -- exact match tried
    first, then substring, same fallback order used by
    ground_truth_all_methods_changespillover.R's own compute_change()),
    ONLY that one column is shifted, every other numeric column is left
    untouched. This matters for changespillover specifically: its data
    has TWO year columns whose DIFFERENCE (year2 - year1) is the actual
    variable being tested. Shifting both by the same constant cancels
    exactly in that difference, so the re-run receives numerically
    IDENTICAL input to the original -- the check would trivially "pass"
    regardless of whether the code is correct, testing nothing. Shifting
    only the later year moves the derived change value by a uniform
    constant instead, which is the shift Moran's I/Geary's C/etc. are
    actually invariant to."""
    cwd_before = os.getcwd()
    tmp_root = tempfile.mkdtemp()
    output = None
    try:
        rel_dir = os.path.dirname(dataset_path) or "."
        shifted_dir = os.path.join(tmp_root, rel_dir)
        os.makedirs(shifted_dir, exist_ok=True)

        df = pd.read_csv(dataset_path)
        if shift_only_column is not None:
            target_col = shift_only_column if shift_only_column in df.columns else next(
                (c for c in df.columns if shift_only_column in c), None)
            if target_col is not None and pd.api.types.is_numeric_dtype(df[target_col]):
                df[target_col] = df[target_col] + SHIFT_CONSTANT
            else:
                # couldn't find/match the requested column -- fall back
                # to shifting every numeric column rather than silently
                # shifting nothing (which would make the check trivially
                # pass for a different, worse reason)
                for col in df.columns:
                    if pd.api.types.is_numeric_dtype(df[col]):
                        df[col] = df[col] + SHIFT_CONSTANT
        else:
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


def compute_shifted_statistic(run_dir, dataset_path, adjacency_path, shift_only_column=None):
    """Returns the recomputed statistic (str) after re-running the
    model's OWN saved code on shifted data, WITHOUT comparing it to any
    single original value -- unlike check_shift_invariant(), which
    assumes a strict 1:1 comparison against ONE original_stat_str.
    Needed for multi-value answer formats (e.g. getisord's list of
    boroughs), where the re-executed output's print order is not
    guaranteed to match the original evidence_text's order -- a strict
    positional match there would spuriously fail correct code just
    because two different (but both real) boroughs got compared to
    each other. Callers should instead check this returned value against
    the FULL set of originally-reported values, accepting a match with
    ANY of them. Returns None if code can't be found or execution
    failed -- same tri-state philosophy as check_shift_invariant()."""
    code_text = extract_final_code(run_dir)
    if not code_text:
        return None
    extra_files = {"ADJACENCY_PATH": adjacency_path} if adjacency_path and pd.notna(adjacency_path) else None
    return _shift_and_rerun(code_text, dataset_path, extra_files, shift_only_column=shift_only_column)


def check_shift_invariant(run_dir, dataset_path, adjacency_path, original_stat_str,
                           check_applicable=True, shift_only_column=None):
    """Returns True/False/None.
    True  = re-run under a shift produced (approximately) the same statistic.
    False = statistic changed substantially, or became non-finite -- likely bug.
    None  = inconclusive (no code found, original statistic wasn't finite to
            begin with -- already caught by the fabrication check, or
            check_applicable=False for this task/method).

    shift_only_column: passed straight through to _shift_and_rerun() --
    see that function's docstring. Leave as None (default) for every
    task except changespillover, whose two-year-column data needs only
    ONE column shifted for the check to be meaningful rather than
    trivially self-cancelling."""
    if not check_applicable:
        return None
    if not _is_finite(original_stat_str):
        return None
    code_text = extract_final_code(run_dir)
    if not code_text:
        return None

    extra_files = {"ADJACENCY_PATH": adjacency_path} if adjacency_path and pd.notna(adjacency_path) else None
    shifted_stat = _shift_and_rerun(code_text, dataset_path, extra_files, shift_only_column=shift_only_column)
    if not _is_finite(shifted_stat):
        return False

    denom = max(abs(float(original_stat_str)), 1e-9)
    rel_diff = abs(float(original_stat_str) - float(shifted_stat)) / denom
    return rel_diff <= TOLERANCE
