"""
multi_agent_utils.py
=======================
Multi-agent orchestration: Profiler -> (Planner -> Coder -> Validator)
loop, with two retry tiers (Coder-level bug fixes within a plan, then
Planner-level new plans across up to max_plan_count attempts).

Deliberately imports the reusable primitives from agent_utils_multi_turn.py
(chat(), _build_data_preview()) rather than duplicating them -- this file
adds NEW orchestration logic on top, since the existing agent_loop() is a
single-thread ReAct loop, not a 3-role pipeline with two retry tiers.

None of the 6 Validator checks reference the correct answer or the
correct method -- they check internal mathematical validity and
self-consistency of whatever the model produced, never "does this match
what Moran's I should give here."

Public entry point: multi_agent_loop(...)
"""

import re
import json
import os
import shutil
import tempfile

_TRACEBACK_LAST_LINE_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)):\s*(.*)$", re.MULTILINE)


def _extract_crash_message(output):
    """If output contains a real Python traceback, returns a short,
    honest description of the actual exception (e.g. 'TypeError:
    unsupported type: ...'), taken from the traceback's own last line
    -- not a generic validator message. Returns None if no traceback
    is present (i.e. the code ran to completion)."""
    if not output or "[STDERR]" not in output and "Traceback" not in output:
        return None
    matches = _TRACEBACK_LAST_LINE_RE.findall(output)
    if not matches:
        return None
    exc_type, exc_msg = matches[-1]  # last match = the actual raised exception
    return f"{exc_type}: {exc_msg}".strip()

from agent_utils_multi_turn import chat, _build_data_preview
from sandbox import run_sandboxed_code, extract_code_blocks


# ---------------------------------------------------------------------
# 1. Profiler agent -- runs ONCE per chunk, insight-agnostic
# ---------------------------------------------------------------------

def run_profiler(dataset_path, extra_files, model, run_dir):
    """Explores the real data via executed code (never imagined) and
    returns a structured summary: row count, per-column missing counts.
    This becomes the independent reference the Validator later checks
    the Coder's self-reported data_points_used against."""
    preview = _build_data_preview(dataset_path, "the main dataset")
    for var_name, filepath in (extra_files or {}).items():
        preview += _build_data_preview(filepath, var_name)

    prompt = (
        "You are the Profiler. Explore this dataset with real, executed "
        "pandas code -- do not guess or describe from the preview alone. "
        "For every numeric column, print its missing-value count using "
        "EXACTLY this format, one line per column:\n"
        "missing_count_<column_name> = <integer>\n"
        "Then print the total row count using EXACTLY this format:\n"
        "row_count = <integer>\n"
        "Do not analyze or answer any question -- only report these facts "
        "about the data itself, using the exact print formats above."
        f"\n\nDataset path: {dataset_path}\n"
        "Write ONE ```python``` block that loads the data and prints "
        "these facts. Re-import pandas/numpy; this runs in a fresh "
        "process.\n" + preview
    )
    reply = chat([{"role": "user", "content": prompt}], model=model)
    code_blocks = extract_code_blocks(reply)
    if not code_blocks:
        return {"row_count": None, "missing_by_column": {}, "raw_output": "(no code written)", "prompt": prompt}

    output = run_sandboxed_code(code_blocks[0], run_dir, dataset_path, extra_vars=extra_files)

    missing_by_column = {}
    for match in re.finditer(r"missing_count_([A-Za-z0-9_]+)\s*=\s*(\d+)", output):
        missing_by_column[match.group(1)] = int(match.group(2))
    row_count_match = re.search(r"row_count\s*=\s*(\d+)", output)
    row_count = int(row_count_match.group(1)) if row_count_match else None

    return {"row_count": row_count, "missing_by_column": missing_by_column, "raw_output": output, "prompt": prompt}


def _profile_summary_text(profile):
    """Renders the profiler's findings as plain text for downstream prompts."""
    lines = [f"Row count: {profile.get('row_count')}"]
    if profile.get("missing_by_column"):
        for col, n in profile["missing_by_column"].items():
            lines.append(f"  {col}: {n} missing values")
    else:
        lines.append("  (no missing values detected in profiler output)")
    return "\n".join(lines)


# ---------------------------------------------------------------------
# 2. Planner agent
# ---------------------------------------------------------------------

def run_planner(question, profile_text, prior_feedback, model):
    """Writes a plain-text plan. If prior_feedback is given (a retry),
    it contains ONLY the Validator's mechanical failure reason from the
    previous attempt -- never the correct answer or the correct method."""
    retry_block = f"\n\nYour previous plan led to this problem:\n{prior_feedback}\nWrite a NEW plan that avoids it.\n" if prior_feedback else ""
    prompt = (
        "You are the Planner. Write a short, numbered plan (steps only, "
        "no code) for answering this question, using the real data facts "
        "below. Your plan should explicitly state how missing values "
        "will be handled if any exist. Do not write a FINAL ANSWER line "
        "or invent any specific numbers -- describe the approach only, "
        "the actual values will come from real computation later.\n\n"
        f"Question: {question}\n\nData profile:\n{profile_text}"
        f"{retry_block}"
    )
    reply = chat([{"role": "user", "content": prompt}], model=model)
    return reply.strip(), prompt


# ---------------------------------------------------------------------
# 3. Coder agent
# ---------------------------------------------------------------------

_DATA_POINTS_USED_RE = re.compile(r"data_points_used\s*=\s*(-?\d+)")
_FINAL_ANSWER_RE = re.compile(r"FINAL ANSWER:\s*(.*)", re.IGNORECASE)
_KV_PAIR_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_ \-()*]*?)\s*=\s*([\-0-9.eE]+|nan)", re.IGNORECASE)


def run_coder(question, dataset_path, extra_files, profile_text, plan,
              prior_feedback, model, run_dir, answer_format_instructions=None):
    """Writes and executes code implementing the plan. Requires printing
    a FINAL ANSWER line AND a separate `data_points_used = <int>` line (the count
    of data points actually used in the final calculation), needed for
    the Validator's missing-value reconciliation check.
    answer_format_instructions: optional custom text describing the
    required FINAL ANSWER format, for tasks whose answer isn't a single
    verdict+statistic (e.g. Getis-Ord's multi-borough hot/cold list).
    Defaults to the standard single-statistic format if not given."""
    preview = _build_data_preview(dataset_path, "the main dataset")
    for var_name, filepath in (extra_files or {}).items():
        preview += _build_data_preview(filepath, var_name)

    default_format = (
        "`FINAL ANSWER: <verdict> (test statistic = <value>, "
        "significance measure = <value>)` -- using an f-string that "
        "references your actual computed variables, never a hand-typed "
        "number. 'test statistic' and 'significance measure' are just "
        "example labels -- if your method produces different names for "
        "these, use them instead, but the value after '=' must always "
        "be a real number. If your task has no natural secondary "
        "measure, omit it and print only the test statistic."
    )
    format_instructions = answer_format_instructions or default_format

    retry_block = f"\n\nYour previous attempt failed this check:\n{prior_feedback}\nFix this specific problem.\n" if prior_feedback else ""
    prompt = (
        "You are the Coder. Implement this plan with real, executed "
        "pandas/numpy code. Your code must print:\n"
        "1. `data_points_used = <integer>` -- the exact number of data "
        "points used in your FINAL calculation (after excluding any "
        "missing values)\n"
        f"2. {format_instructions}\n\n"
        f"Question: {question}\n\nData profile:\n{profile_text}\n\nPlan:\n{plan}"
        f"{retry_block}\n"
        "Write ONE ```python``` block. This runs in a fresh process; "
        "re-import pandas/numpy and reload the data.\n" + preview
    )
    reply = chat([{"role": "user", "content": prompt}], model=model)
    code_blocks = extract_code_blocks(reply)
    if not code_blocks:
        return {"code": None, "output": "(no code written)", "data_points_used": None,
                "statistic": None, "secondary_measures": [], "final_answer": None,
                "prompt": prompt}

    code = code_blocks[0]
    output = run_sandboxed_code(code, run_dir, dataset_path, extra_vars=extra_files)
    return {"code": code, "output": output, "prompt": prompt, **_parse_coder_output(output)}


def _parse_coder_output(output):
    dpu_match = _DATA_POINTS_USED_RE.search(output)
    final_match = _FINAL_ANSWER_RE.search(output)
    final_line = final_match.group(1).strip() if final_match else ""

    # Parse ALL "label = number" pairs found within the FINAL ANSWER
    # line itself -- fully generic, no assumption of specific labels
    # like "p". The first is treated as the primary statistic; any
    # further ones are secondary measures (whatever they're called).
    kv_pairs = _KV_PAIR_RE.findall(final_line)
    statistic = kv_pairs[0][1] if kv_pairs else None
    secondary_measures = [val for _, val in kv_pairs[1:]]

    return {
        "data_points_used": int(dpu_match.group(1)) if dpu_match else None,
        "statistic": statistic,
        "secondary_measures": secondary_measures,
        "final_answer": final_match.group(1).strip() if final_match else None,
    }


# ---------------------------------------------------------------------
# 4. Deterministic Validator -- NOT an LLM call. Every check here is a
#    plain code check: no reference to the correct answer or method.
# ---------------------------------------------------------------------

def _is_finite_number(s):
    if s is None:
        return False
    try:
        f = float(s)
        return f == f and abs(f) != float("inf")  # rejects NaN and inf
    except ValueError:
        return False


def run_validator(coder_result, code_text, profile, dataset_path, extra_files, run_dir,
                   check_shift_invariance=True):
    """Runs 6 checks. Returns (score, total, list_of_failed_check_messages).
    None of these checks reference the correct answer or the correct
    method -- they check internal mathematical validity and
    self-consistency of whatever the model produced.
    check_shift_invariance=False skips check #6 -- set this False for
    any task whose statistic is computed from a NONLINEAR transform of
    the raw values (e.g. distance-decay's log-log regression), since
    additive shift-invariance does not hold there and the check would
    produce false positives on genuinely correct code."""
    checks = []

    # A legitimate "no result" answer (e.g. Getis-Ord's "FINAL ANSWER: none",
    # meaning no borough was significant) has no number to validate at all --
    # this is a valid outcome, not a failure, so checks that require a
    # computed statistic are skipped for this case specifically.
    final_answer_text = (coder_result.get("final_answer") or "").strip().lower()
    is_no_result_answer = final_answer_text == "none"

    # 1. The primary statistic is a finite number
    checks.append(("finite_statistic", True if is_no_result_answer else _is_finite_number(coder_result.get("statistic"))))

    # 2. Any secondary measure (whatever it's called -- p-value,
    #    confidence, z-score's own significance, etc.) must also be
    #    finite if one was reported. No bound is assumed (e.g. no
    #    [0,1] requirement), since not every task's secondary measure
    #    is a probability. If no secondary measure was reported at
    #    all, this check trivially passes -- not every task needs one.
    secondary = coder_result.get("secondary_measures") or []
    secondary_ok = all(_is_finite_number(v) for v in secondary) if secondary else True
    checks.append(("finite_secondary_measures", secondary_ok))

    # 3. Final answer line uses a variable reference (f-string), not a
    #    hardcoded literal -- checked by finding the FINAL ANSWER print
    #    call in the code text and confirming it is an f-string with
    #    {...} placeholders rather than raw decimal literals. Not
    #    required for a "no result" answer -- there's no value to
    #    substitute when the answer is a fixed string like "none".
    final_line_ok = is_no_result_answer
    if code_text and not is_no_result_answer:
        has_any_real_fstring = re.search(r'f["\'][^"\']*\{[^{}]*\}', code_text) is not None
        for line in code_text.splitlines():
            if "FINAL ANSWER" in line and "print(" in line:
                is_fstring = re.search(r'print\(\s*f["\']', line) is not None
                has_placeholder = "{" in line and "}" in line
                if is_fstring and has_placeholder:
                    final_line_ok = True  # direct pattern: value substituted right on this line
                    break
                # list-building pattern: this line isn't itself an f-string,
                # but it's not a bare, fully-static literal either (it
                # references a variable via concatenation, .join(), etc.) --
                # accept it if the code genuinely used an f-string with a
                # real placeholder somewhere to build that variable's content
                is_pure_static_literal = re.fullmatch(
                    r'\s*print\(\s*["\'][^"\']*["\']\s*\)\s*', line) is not None
                if not is_pure_static_literal and has_any_real_fstring:
                    final_line_ok = True
                    break
    checks.append(("real_variable_reference", final_line_ok))

    # 4. data_points_used reconciliation against the Profiler's
    #    independent count -- a generic data-completeness check,
    #    applicable to any task involving possible exclusions.
    dpu = coder_result.get("data_points_used")
    missing_counts = list(profile.get("missing_by_column", {}).values())
    max_missing = max(missing_counts) if missing_counts else 0
    row_count = profile.get("row_count")
    dpu_ok = False
    if dpu is not None and row_count is not None:
        expected_min = row_count - max_missing
        dpu_ok = dpu <= row_count and (max_missing == 0 or dpu <= expected_min + 1)
    elif dpu is not None and row_count is None:
        dpu_ok = True  # profiler gave nothing to check against -- don't penalize the coder
    checks.append(("data_points_reconciliation", dpu_ok))

    # 5. Re-execution determinism -- re-run the SAME code, no new LLM
    #    call, compare the observed statistic (not any secondary
    #    measure, which may legitimately vary with a randomized method)
    determinism_ok = True
    if code_text:
        rerun_output = run_sandboxed_code(code_text, run_dir, dataset_path, extra_vars=extra_files)
        rerun_final = _FINAL_ANSWER_RE.search(rerun_output)
        rerun_line = rerun_final.group(1).strip() if rerun_final else ""
        rerun_pairs = _KV_PAIR_RE.findall(rerun_line)
        rerun_val = rerun_pairs[0][1] if rerun_pairs else None
        orig_val = coder_result.get("statistic")
        if _is_finite_number(orig_val) and _is_finite_number(rerun_val):
            determinism_ok = abs(float(orig_val) - float(rerun_val)) < 1e-6
        elif _is_finite_number(orig_val) != _is_finite_number(rerun_val):
            determinism_ok = False
    checks.append(("determinism", determinism_ok))

    # 6. Constant-shift invariance -- any statistic built on deviations
    #    from the mean (Moran's I, Lee's L, Getis-Ord, z-scores, etc.)
    #    is provably unaffected by adding a constant to the raw data,
    #    since (x+c) - mean(x+c) = x - mean(x). Re-runs the SAME code
    #    against a shifted copy of the data, from an isolated temp
    #    directory so the real dataset file is NEVER modified.
    invariance_ok = True
    if check_shift_invariance and code_text and _is_finite_number(coder_result.get("statistic")):
        shifted_stat = _run_with_shifted_data(code_text, dataset_path, extra_files)
        orig_val = coder_result.get("statistic")
        if _is_finite_number(shifted_stat):
            denom = max(abs(float(orig_val)), 1e-9)
            rel_diff = abs(float(orig_val) - float(shifted_stat)) / denom
            invariance_ok = rel_diff <= 1e-3
        else:
            invariance_ok = False  # became non-finite under a shift -- clear evidence of a raw-value bug
    checks.append(("shift_invariance", invariance_ok))

    score = sum(1 for _, ok in checks if ok)
    failed_messages = []
    for name, ok in checks:
        if not ok:
            failed_messages.append(_FAILURE_MESSAGES[name])
    return score, len(checks), failed_messages



_FAILURE_MESSAGES = {
    "finite_statistic": "Your printed test statistic is not a finite number.",
    "finite_secondary_measures": "A secondary measure you printed alongside your test statistic is not a finite number.",
    "real_variable_reference": "Your FINAL ANSWER line does not clearly reference a computed variable via an f-string.",
    "data_points_reconciliation": "The number of data points you reported using (data_points_used) does not match the data's actual completeness -- check your missing-value handling.",
    "determinism": "Re-running your exact code produced a different observed statistic the second time. The observed statistic should be identical every time on the same data -- check for unintended randomness or dependence on dictionary/set ordering.",
    "shift_invariance": "Your test statistic changed substantially when a constant was added to all values in the data. A correctly-computed statistic of this kind should be unaffected by such a shift -- re-examine your calculation for why this might happen.",
}

_SHIFT_CONSTANT = 1000.0


def _run_with_shifted_data(code_text, dataset_path, extra_files):
    """Re-runs code_text against a copy of dataset_path with a constant
    added to every numeric column, entirely from an isolated temp
    directory. Returns the resulting statistic (str) or None. The real
    dataset file is never touched -- confirmed by never writing inside
    the original dataset's own directory.

    Uses mkdtemp() with EXPLICIT, error-tolerant cleanup rather than the
    TemporaryDirectory() context manager: on Windows, that context
    manager's automatic cleanup can fail with a file-lock error if the
    sandbox subprocess hasn't fully released a handle yet -- and since
    that cleanup happens INSIDE the same try block, it would silently
    discard an already-successful result and report a false failure.
    Separating cleanup from result capture guarantees a real, obtained
    output is never thrown away because of an unrelated OS-level
    cleanup quirk."""
    import pandas as pd

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
                df[col] = df[col] + _SHIFT_CONSTANT
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
        shutil.rmtree(tmp_root, ignore_errors=True)  # never let cleanup failure discard a captured result

    if output is None:
        return None
    final = _FINAL_ANSWER_RE.search(output)
    if not final:
        return None
    pairs = _KV_PAIR_RE.findall(final.group(1))
    return pairs[0][1] if pairs else None



# ---------------------------------------------------------------------
# 5. Orchestration -- the loop shown in the flowchart
# ---------------------------------------------------------------------

def multi_agent_loop(question, dataset_path, run_dir, model, extra_files=None,
                      max_plan_count=3, max_coder_retries=2, check_shift_invariance=True,
                      answer_format_instructions=None):
    """Runs the full Profiler -> (Planner -> Coder -> Validator) loop.
    Returns a dict with the final result and full audit trail.
    check_shift_invariance: see run_validator()'s docstring -- set False
    for tasks whose statistic uses a nonlinear transform of raw values
    (e.g. distance-decay's log-log regression).
    answer_format_instructions: see run_coder()'s docstring -- customize
    for tasks whose answer isn't a single verdict+statistic."""
    profile = run_profiler(dataset_path, extra_files, model, run_dir)
    profile_text = _profile_summary_text(profile)

    attempts = []  # every (plan_number, coder_result, score, total, failed_msgs)
    prior_planner_feedback = None

    for plan_num in range(1, max_plan_count + 1):
        plan, planner_prompt = run_planner(question, profile_text, prior_planner_feedback, model)
        prior_coder_feedback = None

        for coder_try in range(1, max_coder_retries + 1):
            coder_result = run_coder(question, dataset_path, extra_files, profile_text,
                                      plan, prior_coder_feedback, model, run_dir,
                                      answer_format_instructions=answer_format_instructions)
            score, total, failed_msgs = run_validator(
                coder_result, coder_result.get("code"), profile, dataset_path, extra_files, run_dir,
                check_shift_invariance=check_shift_invariance)

            attempts.append({
                "plan_number": plan_num, "coder_try": coder_try,
                "plan": plan, "planner_prompt": planner_prompt, "coder_result": coder_result,
                "validation_score": score, "validation_total": total,
                "failed_checks": failed_msgs,
            })

            if score == total:
                return {
                    "status": "accepted", "final_answer": coder_result.get("final_answer"),
                    "validation_score": score, "validation_total": total,
                    "plan_attempts_used": plan_num, "coder_tries_used": coder_try,
                    "profiler_prompt": profile.get("prompt"), "attempts": attempts,
                }

            crash_msg = _extract_crash_message(coder_result.get("output"))
            if crash_msg:
                combined_feedback = f"Your code crashed with a real Python error: {crash_msg}; " + "; ".join(failed_msgs)
            else:
                combined_feedback = "; ".join(failed_msgs)
            prior_coder_feedback = combined_feedback

        # coder retries exhausted for this plan -- escalate to Planner
        prior_planner_feedback = combined_feedback

    # all plans exhausted -- return the best-scoring attempt, honestly tagged
    best = max(attempts, key=lambda a: a["validation_score"])
    return {
        "status": "best_effort", "final_answer": best["coder_result"].get("final_answer"),
        "validation_score": best["validation_score"], "validation_total": best["validation_total"],
        "plan_attempts_used": max_plan_count, "coder_tries_used": None,
        "profiler_prompt": profile.get("prompt"), "attempts": attempts,
    }
