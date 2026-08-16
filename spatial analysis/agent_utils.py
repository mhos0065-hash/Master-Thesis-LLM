"""
agent_utils.py
================
Shared infrastructure for single_agent.py and multi_agent.py:
  - chat()       : call an LLM (OpenAI-compatible endpoint)
  - agent_loop() : ReAct-style loop -- send prompt, extract ```python```
                   blocks, execute EACH ONE IN ITS OWN SANDBOXED SUBPROCESS
                   via sandbox.run_sandboxed_code(), feed results back,
                   repeat until the model emits a valid FINAL ANSWER: JSON
                   verdict (see schemas.py) or max_iterations is hit.

Code execution is delegated entirely to sandbox.py (subprocess isolation,
hard timeout, network blocked at the function level, Windows-safe UTF-8
handling). agent_utils.py does not implement its own code execution.

IMPORTANT -- statelessness: each code block the model writes runs in a
FRESH subprocess with no memory of variables from earlier blocks. The
system prompt below tells the model this explicitly, so it knows to
reload the dataset (pd.read_csv(DATASET_PATH)) in every block rather than
assuming a dataframe from a previous turn still exists.

Configure via environment variables (see .env.example):
    LLM_API_BASE   e.g. https://openrouter.ai/api/v1
    LLM_API_KEY    your key
    LLM_MODEL      e.g. openai/gpt-4o-mini, deepseek/deepseek-r1-0528, qwen/qwen3-32b
"""

import os
import time
import re
import requests
import pandas as pd
from dotenv import load_dotenv

from sandbox import run_sandboxed_code, extract_code_blocks
from schemas import parse_verdict, SCHEMA_INSTRUCTIONS

load_dotenv()  # reads .env in the current directory -- works the same on
                # Windows/PowerShell, macOS, and Linux

LLM_API_BASE = os.environ.get("LLM_API_BASE", "https://openrouter.ai/api/v1")
LLM_API_KEY = os.environ.get("LLM_API_KEY", "")
LLM_MODEL = os.environ.get("LLM_MODEL", "openai/gpt-4o-mini")

# Minimum seconds between consecutive chat() calls, to stay under free-tier
# rate limits proactively -- rather than only reacting after a 429 arrives.
# Override in .env if your provider allows more, e.g.
# LLM_MIN_INTERVAL_SECONDS=0.5 for a paid tier.
LLM_MIN_INTERVAL_SECONDS = float(os.environ.get("LLM_MIN_INTERVAL_SECONDS", "2.2"))
_last_call_time = [0.0]

# ---------------------------------------------------------------------
# 1. LLM call
# ---------------------------------------------------------------------

def chat(messages, model=None, temperature=0.0, max_retries=5, timeout=180):
    """Call an OpenAI-compatible chat completions endpoint. Returns the
    assistant's text reply (string). Proactively paces requests to avoid
    free-tier rate limits, and if a 429 does arrive anyway, backs off
    (honoring the provider's Retry-After header if present) and retries
    with a real, informative error if it still fails."""
    model = model or LLM_MODEL
    if not LLM_API_KEY:
        raise RuntimeError(
            "LLM_API_KEY not set. Put it in a .env file (see .env.example) "
            "before running."
        )
    url = f"{LLM_API_BASE}/chat/completions"
    headers = {"Authorization": f"Bearer {LLM_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature}

    last_err_msg = "no attempts were made"
    for attempt in range(max_retries):
        # proactive pacing -- wait out the minimum interval since the last call
        elapsed = time.time() - _last_call_time[0]
        if elapsed < LLM_MIN_INTERVAL_SECONDS:
            time.sleep(LLM_MIN_INTERVAL_SECONDS - elapsed)

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=timeout)
            _last_call_time[0] = time.time()

            if resp.status_code == 429:
                retry_after = resp.headers.get("Retry-After")
                wait = float(retry_after) if retry_after else min(60, 2 ** (attempt + 2))
                last_err_msg = f"HTTP 429 rate limited (attempt {attempt+1}/{max_retries}). Body: {resp.text[:300]}"
                print(f"  [rate limited] waiting {wait:.0f}s before retry "
                      f"({attempt+1}/{max_retries})...")
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            if "choices" not in data or not data["choices"]:
                # Some providers return HTTP 200 with an error-shaped body
                # (e.g. just {"error": ...}) instead of a proper 4xx/5xx --
                # especially likely right after rate-limit retries. Treat
                # this as retryable, same as null content below, instead of
                # letting a raw KeyError crash the caller.
                last_err_msg = (f"API response missing 'choices' (attempt {attempt+1}/{max_retries}). "
                                 f"Full response: {str(data)[:300]}")
                print(f"  [malformed response] retrying ({attempt+1}/{max_retries})...")
                time.sleep(min(30, 2 ** attempt))
                continue

            content = data["choices"][0]["message"]["content"]

            if content is None:
                # Some models (particularly reasoning models) can return a
                # response where message.content is literally null -- e.g.
                # if the model only produced internal reasoning tokens with
                # no visible final text, or was cut off before writing any
                # content. Returning None here would silently propagate into
                # agent_loop()'s "FINAL ANSWER:" in reply check and crash
                # with a confusing TypeError far from the real cause -- so
                # treat it as a retryable failure instead, same as a 429.
                last_err_msg = (f"API returned null content (attempt {attempt+1}/{max_retries}). "
                                 f"Full response: {str(data)[:300]}")
                print(f"  [null content] retrying ({attempt+1}/{max_retries})...")
                time.sleep(min(30, 2 ** attempt))
                continue

            return content

        except requests.exceptions.HTTPError as e:
            body = getattr(e.response, "text", "")[:300] if e.response is not None else ""
            last_err_msg = f"HTTP error: {e}. Body: {body}"
        except Exception as e:
            last_err_msg = f"{type(e).__name__}: {e}"

    raise RuntimeError(
        f"chat() failed after {max_retries} attempts. Last error: {last_err_msg}\n"
        f"If this was rate limiting, try increasing LLM_MIN_INTERVAL_SECONDS in your "
        f".env file (e.g. LLM_MIN_INTERVAL_SECONDS=4), or run fewer chunks/models at once."
    )

# ---------------------------------------------------------------------
# 2. ReAct-style agent loop, using the real sandbox for every code block
# ---------------------------------------------------------------------

STATELESSNESS_NOTICE = (
    "IMPORTANT -- two rules that are easy to get wrong, read carefully:\n"
    "1. STATELESS: each ```python``` block runs in its own fresh process with "
    "NO memory of variables, imports, or dataframes from earlier blocks. "
    "Every single block must start from scratch: re-import pandas/numpy if you "
    "use them by name, and reload the dataset (df = pd.read_csv(DATASET_PATH)) "
    "even if you already loaded it in a previous block. Referencing a variable "
    "from an earlier block will fail with NameError.\n"
    "2. ALWAYS print() what you want to see. This is NOT a notebook -- writing "
    "a bare variable name or expression at the end of a script (e.g. `Q1, Q3, IQR`) "
    "produces NO visible output here, unlike Jupyter. Every value you want to "
    "check or use as evidence must be wrapped in print(), e.g. print('Q1:', Q1). "
    "pandas and numpy are pre-imported as pd and np. If you want to save a "
    "chart or file, save it into ARTIFACTS_DIR (already defined), never "
    "elsewhere."
)

SINGLE_RUN_NOTICE = (
    "3. SINGLE REPLY ONLY: you have exactly ONE reply to solve this completely -- there is "
    "no follow-up turn, and you will NOT get to see your code's output before answering. "
    "Your single ```python``` block must load the data, run the FULL analysis, AND print "
    "your final answer AS THE LAST LINE OF YOUR CODE'S OWN OUTPUT, using an f-string that "
    "substitutes your actual computed variables directly -- for example, if your code "
    "computed a value in a variable called p_value, print it with "
    "print(f\"p = {p_value}\"), NEVER by typing a number you believe is plausible directly "
    "into the string, e.g. print(\"p = 0.031\") is WRONG even if 0.031 looks correct -- it "
    "is not connected to your actual computation and will not be trusted. Your print "
    "statement must reference the variable itself, not a number you typed by hand. "
    "Follow the exact FINAL ANSWER format given in the question above, but it must come "
    "from a real print() statement inside your code -- do NOT write 'FINAL ANSWER:' as "
    "prose text outside your code. Only text your code actually prints will be read as "
    "your answer."
)


def _build_data_preview(path, label):
    """Reads the REAL column names and a few real rows from a CSV, so a
    one-shot model doesn't have to guess file structure blind -- this is
    the dominant fix for one-shot failures: models routinely guess wrong
    column names on a first attempt, and with zero retry chances in
    one-shot mode, any such guess crashes the code before it ever reaches
    the final print() statement. Deliberately structural only -- shows
    real data, but does not coach on how to handle it (e.g. missing
    values), keeping single-agent minimally prescriptive. Built
    unconditionally in agent_loop() (both one-shot AND multi-turn modes
    get this preview), so multi-turn results aren't confounded by an
    information difference between the two conditions."""
    try:
        df = pd.read_csv(path)
        preview = df.head(3).to_string(index=False)
        return (f"\n\nReal preview of {label} (path: {path}) -- use these EXACT column "
                f"names, do not guess:\nColumns: {list(df.columns)}\nFirst 3 rows:\n{preview}")
    except Exception as e:
        return f"\n\n(Could not preview {label} at {path}: {e})"


def _extract_final_answer_from_output(output_text):
    """Looks for 'FINAL ANSWER:' in real sandbox output (used in
    answer_from_code_output mode) and returns the text after it. Case-
    insensitive, since a model might print slightly different
    capitalization inside code.

    Fallback (confirmed necessary from real data across multiple tasks):
    some models compute a completely real, correct answer but forget the
    literal 'FINAL ANSWER:' prefix -- e.g. printing just 'NOT AN OUTLIER
    (z = 0.51)' or 'DISTANCE_DECAY_PRESENT (slope = -4.9, p = 0.026)'
    instead of prefixing it, despite the explicit instruction. If no
    marker is found AND the code ran cleanly (no [STDERR] block, meaning
    no crash), the last non-empty printed line is used instead -- this is
    still exclusively real, executed sandbox output, never the model's
    own narrated prose, so it doesn't reopen the hallucination risk the
    marker requirement exists to prevent. If the code crashed, no
    fallback is attempted -- a traceback's last line is never a valid
    answer. If the marker IS found but nothing meaningful follows it
    (e.g. the print() call itself errored partway through), that also
    returns None cleanly rather than falling through to the last-line
    fallback (which would otherwise just return the marker text itself)."""
    match = re.search(r"FINAL ANSWER:", output_text, re.IGNORECASE)
    if match:
        remainder = output_text[match.end():].strip()
        if remainder:
            lines = remainder.splitlines()
            if lines:
                return lines[0].strip()
        return None  # marker present but nothing meaningful followed it -- never fall through

    if "[STDERR]" in output_text:
        return None  # a genuine crash -- never treat a traceback as an answer

    stripped = output_text.strip()
    if not stripped:
        return None
    lines = [l.strip() for l in stripped.splitlines() if l.strip()]
    return lines[-1] if lines else None


def agent_loop(system_prompt, user_prompt, dataset_path, run_dir, model=None,
                max_iterations=1, require_structured_verdict=False, extra_files=None,
                answer_from_code_output=False):
    """Runs one agent through an iterative code-and-reason loop, with every
    code block executed by the real sandbox (subprocess + timeout + network
    block). `run_dir` must already exist -- the sandbox writes its temp
    script and any ARTIFACTS_DIR contents there.

    extra_files: optional dict of {VAR_NAME: filepath} for insight types that
    need more than one input file (e.g. spatial correlation needing both a
    values CSV and a borough adjacency CSV). Passed straight through to
    sandbox.run_sandboxed_code()'s extra_vars, and mentioned in the prompt so
    the model knows the variable exists.

    If require_structured_verdict=True (use this for any agent making the
    actual insight_present/not-present call -- the single-agent baseline,
    or a multi-agent Reporter that itself runs code), the FINAL ANSWER must
    be valid JSON matching schemas.AgentVerdict, with a repair retry if it
    isn't. If False (use this for roles like Profiler/Coder that report
    findings but don't themselves decide insight_present), FINAL ANSWER is
    returned as plain text and verdict is always None.

    answer_from_code_output: set True for a genuine one-shot design (e.g.
    single-agent with max_iterations=1). When True, the model gets exactly
    one reply and must print its own 'FINAL ANSWER: ...' as part of its
    code's real output -- the answer is extracted from what the sandbox
    actually printed (with the safe fallback described in
    _extract_final_answer_from_output), never from the model's separate
    prose. Not compatible with require_structured_verdict=True -- code
    output is plain text, not JSON.

    Returns (verdict: AgentVerdict | None, raw_final_text: str, conversation_log: list[dict]).
    """
    if answer_from_code_output and require_structured_verdict:
        raise ValueError(
            "answer_from_code_output=True is not compatible with require_structured_verdict=True "
            "-- code output is parsed as plain text, not validated against the JSON schema."
        )

    if answer_from_code_output:
        schema_block = "\n\n" + SINGLE_RUN_NOTICE
    else:
        schema_block = (
            "\n\nWhen you have enough evidence, respond with 'FINAL ANSWER:' "
            "followed by the JSON verdict described below.\n" + SCHEMA_INSTRUCTIONS
            if require_structured_verdict else
            "\n\nWhen you are done, respond with 'FINAL ANSWER:' followed by your conclusion in plain text."
        )
    extra_files_notice = ""
    if extra_files:
        var_list = ", ".join(extra_files.keys())
        extra_files_notice = (
            f"\n\nAdditional input file(s) available as variables: {var_list} "
            f"(also read-only CSV paths, same rules as DATASET_PATH). "
            f"IMPORTANT: {var_list} is already a pre-loaded variable containing the correct file "
            f"path -- use it directly, e.g. pd.read_csv({list(extra_files.keys())[0]}). "
            f"Do NOT put it in quotes like pd.read_csv('{list(extra_files.keys())[0]}') -- that "
            f"looks for a file literally named '{list(extra_files.keys())[0]}', which does not "
            f"exist. Do NOT reassign it to a path you guess or invent -- the real value is already "
            f"provided; overwriting it will break your code."
        )

    # data preview built UNCONDITIONALLY -- both one-shot and multi-turn modes
    # get it, so results are never confounded by an information difference
    data_preview = _build_data_preview(dataset_path, "the main dataset")
    for var_name, filepath in (extra_files or {}).items():
        data_preview += _build_data_preview(filepath, var_name)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": (
            user_prompt
            + f"\n\nDataset path (read-only): {dataset_path}\n"
            + STATELESSNESS_NOTICE
            + extra_files_notice
            + data_preview
            + schema_block
        )},
    ]
    log = [{"role": "user", "content": messages[-1]["content"]}]

    for i in range(max_iterations):
        reply = chat(messages, model=model)
        log.append({"role": "assistant", "content": reply})
        messages.append({"role": "assistant", "content": reply})

        code_blocks = extract_code_blocks(reply)

        if answer_from_code_output:
            # one-shot mode: run whatever code was given (if any), then look
            # for FINAL ANSWER strictly in the REAL sandbox output -- never
            # in the model's own prose, since that's exactly how the
            # broken-code-fragment failure mode happens
            if not code_blocks:
                msg = "(Stopped: no code block was provided, so no analysis could run.)"
                log.append({"role": "system", "content": msg})
                return None, msg, log

            results = []
            for code in code_blocks:
                result = run_sandboxed_code(code, run_dir, dataset_path, extra_vars=extra_files)
                results.append(result)
            feedback = "\n\n".join(f"Output {j+1}:\n{r}" for j, r in enumerate(results))
            log.append({"role": "system", "content": f"(Sandbox output, not seen by the model in one-shot mode)\n{feedback}"})

            final_text = _extract_final_answer_from_output(feedback)
            if final_text is not None:
                return None, final_text, log
            # code ran but never printed a real FINAL ANSWER (crashed, or
            # the model forgot) -- a genuine, honest failure, not a retry
            msg = "(Stopped: code ran but never printed a FINAL ANSWER in its own output.)"
            return None, msg, log

        # ---- original multi-turn behavior, unchanged ----
        # CRITICAL: if this message contains code blocks, they must be
        # executed and their REAL output fed back -- even if the model also
        # wrote "FINAL ANSWER:" in the same message. Otherwise a model can
        # write code, never see it actually run, and just narrate/imagine
        # plausible-looking output ("suppose the output is...") before
        # declaring its answer -- a real observed failure mode, not a
        # hypothetical one. A FINAL ANSWER is only ever accepted from a
        # message that contains NO unexecuted code.
        if code_blocks:
            results = []
            for code in code_blocks:
                result = run_sandboxed_code(code, run_dir, dataset_path, extra_vars=extra_files)
                results.append(result)
            feedback = "\n\n".join(f"Output {j+1}:\n{r}" for j, r in enumerate(results))
            if "FINAL ANSWER:" in reply:
                msg = (f"Code executed. Here are the ACTUAL results (do not assume or invent "
                       f"output -- use exactly these numbers):\n{feedback}\n\n"
                       f"Your previous message included a FINAL ANSWER before you had seen "
                       f"real output, which is not valid. Now that you have the real results "
                       f"above, give your FINAL ANSWER based on THESE numbers.")
            else:
                msg = f"Code executed. Results:\n{feedback}\n\nContinue your analysis."
            messages.append({"role": "user", "content": msg})
            log.append({"role": "user", "content": msg})
            continue

        if "FINAL ANSWER:" in reply:
            final_text = reply.split("FINAL ANSWER:")[-1].strip()
            if not require_structured_verdict:
                return None, final_text, log
            verdict, err = parse_verdict(final_text)
            if verdict is not None:
                return verdict, final_text, log
            # invalid JSON/schema -- give the model ONE chance to fix it,
            # rather than silently discarding a real analytical result
            repair_msg = (f"Your FINAL ANSWER could not be parsed: {err}\n"
                          f"Please resend FINAL ANSWER: followed by ONLY the "
                          f"corrected JSON object, matching the schema exactly.")
            messages.append({"role": "user", "content": repair_msg})
            log.append({"role": "user", "content": repair_msg})
            continue

        nudge = ("No ```python``` code block found and no FINAL ANSWER given. "
                 "Please write a ```python``` block to continue, or give your FINAL ANSWER: now.")
        messages.append({"role": "user", "content": nudge})
        log.append({"role": "user", "content": nudge})

    return None, "(Stopped: reached iteration limit without a valid FINAL ANSWER.)", log
