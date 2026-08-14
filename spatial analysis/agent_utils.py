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

LLM_MIN_INTERVAL_SECONDS = float(os.environ.get("LLM_MIN_INTERVAL_SECONDS", "2.2"))
_last_call_time = [0.0]


def chat(messages, model=None, temperature=0.0, max_retries=5, timeout=180):
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
                time.sleep(wait)
                continue

            resp.raise_for_status()
            data = resp.json()

            if "choices" not in data or not data["choices"]:
                last_err_msg = (f"API response missing 'choices' (attempt {attempt+1}/{max_retries}). "
                                 f"Full response: {str(data)[:300]}")
                time.sleep(min(30, 2 ** attempt))
                continue

            content = data["choices"][0]["message"]["content"]

            if content is None:
                last_err_msg = (f"API returned null content (attempt {attempt+1}/{max_retries}). "
                                 f"Full response: {str(data)[:300]}")
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
    try:
        df = pd.read_csv(path)
        preview = df.head(3).to_string(index=False)
        return (f"\n\nReal preview of {label} (path: {path}) -- use these EXACT column "
                f"names, do not guess:\nColumns: {list(df.columns)}\nFirst 3 rows:\n{preview}")
    except Exception as e:
        return f"\n\n(Could not preview {label} at {path}: {e})"


def _extract_final_answer_from_output(output_text):
    match = re.search(r"FINAL ANSWER:", output_text, re.IGNORECASE)
    if not match:
        return None
    return output_text[match.end():].strip().splitlines()[0].strip()


def agent_loop(system_prompt, user_prompt, dataset_path, run_dir, model=None,
                max_iterations=1, require_structured_verdict=False, extra_files=None,
                answer_from_code_output=False):
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
            msg = "(Stopped: code ran but never printed a FINAL ANSWER in its own output.)"
            return None, msg, log

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
