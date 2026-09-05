"""
agent_utils_react.py
=======================
A genuine ReAct-style loop (Yao et al., 2022): the model must write an
explicit "Thought:" before every code block ("Action:"), and real
sandbox output is fed back labeled "Observation:". This is a NEW,
SEPARATE file -- agent_utils.py is never imported or modified. Reuses
the SAME sandbox execution (sandbox.run_sandboxed_code()), the SAME
LLM call wrapper (chat()), and the SAME dataset-preview logic as
agent_utils.py, so the ONLY thing that differs from your existing
multi-turn condition is the prompt structure and turn labeling below --
everything else (code execution, retries, rate limiting) is identical.

WHY THIS IS DIFFERENT FROM YOUR EXISTING multi-turn agent_loop():
Your existing multi-turn loop already does code -> real output -> revise,
which is close to ReAct in spirit, but it never REQUIRES the model to
verbalize reasoning before acting -- the model can just write code
directly. Classic ReAct requires that explicit "Thought" every turn.
This file adds that requirement, and (critically) MEASURES whether the
model actually followed it -- see check_react_fidelity() below -- so
you can report a real, auditable adherence number rather than just
claiming "we used ReAct" based on what the prompt asked for.

Configure via the same .env as agent_utils.py (LLM_API_BASE, LLM_API_KEY,
LLM_MODEL) -- imports LLM_MODEL and chat() directly from agent_utils.py
so both files always agree on model/config; nothing is duplicated there.
"""

import re

from agent_utils import chat, LLM_MODEL, _build_data_preview, STATELESSNESS_NOTICE
from sandbox import run_sandboxed_code, extract_code_blocks
from schemas import parse_verdict, SCHEMA_INSTRUCTIONS

# ---------------------------------------------------------------------
# ReAct-specific prompt scaffolding
# ---------------------------------------------------------------------

REACT_FORMAT_NOTICE = (
    "3. REACT FORMAT -- follow this exact pattern on every turn:\n"
    "Thought: <your reasoning about what to try next, one or two sentences>\n"
    "Action:\n"
    "```python\n"
    "<a single code block>\n"
    "```\n\n"
    "After your code runs, you will see:\n"
    "Observation: <the real output>\n\n"
    "Repeat Thought / Action / Observation as many times as you need. "
    "When you have enough evidence to answer, write a final Thought "
    "explaining why you are ready, THEN give your answer, e.g.:\n"
    "Thought: I now have enough information to answer.\n"
    "FINAL ANSWER: <your conclusion>\n\n"
    "Every single message you send MUST start with a line beginning "
    "'Thought:' -- a message with a ```python``` block but no preceding "
    "Thought line is not valid ReAct format and will be rejected."
)


def check_react_fidelity(reply_text):
    """Returns True if this single model reply actually followed the
    ReAct pattern: an explicit 'Thought:' line appearing before any
    ```python``` code block in the same message. This is checked against
    the REAL reply text, not assumed from the prompt -- the whole point
    is to measure whether the model actually complied, the same way the
    5-check validator never trusts a model's own claims without
    verifying them against real output."""
    thought_match = re.search(r"^\s*Thought:", reply_text, re.MULTILINE)
    code_match = re.search(r"```python", reply_text)
    if not code_match:
        # A final-answer-only turn (no code) still needs a Thought to
        # count as valid ReAct format, per the prompt's own instruction.
        return thought_match is not None
    if not thought_match:
        return False
    return thought_match.start() < code_match.start()


def _strip_code_blocks(text):
    """Removes ```python ... ``` blocks entirely, leaving only the
    model's own prose. Used to check for a prose FINAL ANSWER without
    false-matching a print() call inside the code itself -- e.g.
    print("FINAL ANSWER:", answer) is real code, not a prose claim."""
    return re.sub(r"```python.*?```", "", text, flags=re.DOTALL)


def _extract_final_answer(reply):
    """Same extraction rule as agent_utils.agent_loop()'s multi-turn
    branch: a FINAL ANSWER is only ever accepted from a message that
    also contains no unexecuted code (checked by the caller). Only
    checks the model's PROSE (code blocks stripped first) -- a
    print("FINAL ANSWER: ...") inside a code block is not a prose
    claim, it's an instruction for what the code will print once run."""
    prose = _strip_code_blocks(reply)
    if "FINAL ANSWER:" in prose:
        return prose.split("FINAL ANSWER:")[-1].strip()
    return None


def _extract_final_answer_from_observation(feedback):
    """If the model's code RAN and its real output itself printed a
    FINAL ANSWER line (rather than the model declaring one separately
    in prose), extract it directly from that real Observation -- the
    same principle as agent_utils.py's answer_from_code_output mode:
    only real, executed output is ever trusted, never the model's own
    narrated prose. This is what lets a model that prefers to print its
    conclusion from code (a pattern seen repeatedly across this
    project's other tasks) still succeed under ReAct, instead of being
    endlessly rebuked for a "premature" answer that was actually
    genuine, real, executed output."""
    match = re.search(r"FINAL ANSWER:", feedback)
    if not match:
        return None
    remainder = feedback[match.end():].strip()
    if not remainder:
        return None
    return remainder.splitlines()[0].strip()


def agent_loop_react(system_prompt, user_prompt, dataset_path, run_dir, model=None,
                      max_iterations=6, require_structured_verdict=False, extra_files=None):
    """Runs one agent through a genuine ReAct loop: Thought -> Action
    (```python``` code, executed in the real sandbox) -> Observation,
    repeated until the model gives a FINAL ANSWER or max_iterations is
    hit. Every code block is executed by sandbox.run_sandboxed_code() --
    the exact same execution path as agent_utils.agent_loop(), so
    results are comparable across conditions.

    Returns (verdict, raw_final_text, conversation_log, fidelity_record).
    fidelity_record is a list of bools, one per model reply, from
    check_react_fidelity() -- this is what lets you report a real
    "X% of turns followed ReAct format" number afterward, rather than
    just asserting the pattern was used.
    """
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
            f"Do NOT put it in quotes -- that looks for a file literally named "
            f"'{list(extra_files.keys())[0]}', which does not exist."
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
            + "\n" + REACT_FORMAT_NOTICE
            + extra_files_notice
            + data_preview
            + schema_block
        )},
    ]
    log = [{"role": "user", "content": messages[-1]["content"]}]
    fidelity_record = []

    for i in range(max_iterations):
        reply = chat(messages, model=model)
        log.append({"role": "assistant", "content": reply})
        messages.append({"role": "assistant", "content": reply})
        fidelity_record.append(check_react_fidelity(reply))

        code_blocks = extract_code_blocks(reply)

        # Same safety property as agent_utils.agent_loop(): a FINAL ANSWER
        # is only ever accepted from a message with NO unexecuted code --
        # otherwise a model could narrate imagined output instead of
        # waiting for the real Observation.
        if code_blocks:
            results = []
            for code in code_blocks:
                result = run_sandboxed_code(code, run_dir, dataset_path, extra_vars=extra_files)
                results.append(result)
            feedback = "\n\n".join(f"Observation {j+1}: {r}" for j, r in enumerate(results))
            prose_final_answer = "FINAL ANSWER:" in _strip_code_blocks(reply)
            observation_final_answer = _extract_final_answer_from_observation(feedback)

            if observation_final_answer is not None and not prose_final_answer:
                # The code RAN and genuinely printed a FINAL ANSWER as
                # part of its real output -- accept it directly, exactly
                # like answer_from_code_output mode does. This is the
                # fix for the case that caused a valid, correct result
                # to be discarded: a model that writes
                # print("FINAL ANSWER: ...") inside its code, then never
                # gets credit because the loop only ever checked its own
                # prose for the literal string.
                final_text = observation_final_answer
                if not require_structured_verdict:
                    return None, final_text, log, fidelity_record
                verdict, err = parse_verdict(final_text)
                if verdict is not None:
                    return verdict, final_text, log, fidelity_record
                # invalid JSON from a printed answer -- fall through to
                # the normal "continue" path below and let the model retry

            is_last_iteration = (i == max_iterations - 1)
            if (is_last_iteration and not prose_final_answer and observation_final_answer is None
                    and "[STDERR]" not in feedback):
                # Last chance, no next turn coming: the model computed
                # something real (code ran cleanly, no crash) but never
                # once wrote the literal FINAL ANSWER marker anywhere --
                # confirmed real failure mode (seen directly in this
                # project's own logs: a model can compute a fully
                # correct result and just print it plain, e.g.
                # print(final_output_str) instead of
                # print("FINAL ANSWER:", final_output_str), then loop
                # re-verifying the same result instead of adding the
                # label). Fall back to the last non-empty printed line,
                # same principle as agent_utils.py's
                # _extract_final_answer_from_output() -- still exclusively
                # real, executed output, never the model's own prose.
                cleaned = re.sub(r"^Observation \d+:\s*", "", feedback, flags=re.MULTILINE)
                NO_OUTPUT_PLACEHOLDER = "(code ran with no output"
                content_lines = [l.strip() for l in cleaned.splitlines()
                                  if l.strip() and NO_OUTPUT_PLACEHOLDER not in l]
                if content_lines:
                    # Use the FULL cleaned output, not just the last line --
                    # a model that prints a multi-row DataFrame (e.g. one row
                    # per hot/cold borough) has its result spread across
                    # several lines; taking only the last line silently drops
                    # every other finding. Confirmed necessary from a real
                    # run: print(hot_cold_spots[[...]]) produced several
                    # borough rows, and the old last-line-only fallback
                    # discarded all but one.
                    final_text = "\n".join(content_lines)
                    if not require_structured_verdict:
                        return None, final_text, log, fidelity_record
                    verdict, err = parse_verdict(final_text)
                    if verdict is not None:
                        return verdict, final_text, log, fidelity_record

            if prose_final_answer:
                msg = (f"{feedback}\n\nYour previous message included a FINAL ANSWER before you "
                       f"had seen the real Observation, which is not valid ReAct format. Now that "
                       f"you have the real result above, give a new Thought and either take another "
                       f"Action or give your FINAL ANSWER based on THIS Observation.")
            else:
                is_second_to_last_iteration = (i == max_iterations - 2)
                if is_second_to_last_iteration:
                    msg = (f"{feedback}\n\nYou have ONE turn remaining after this one. If you have "
                           f"enough evidence, give a final Thought and your FINAL ANSWER on your "
                           f"next reply -- do not continue exploring past that point. IMPORTANT: "
                           f"format your FINAL ANSWER exactly as specified in the ORIGINAL "
                           f"instructions above (the required output format given at the very "
                           f"start of this task) -- do not just print a raw table or DataFrame; "
                           f"write it out as the specific text format that was requested.")
                else:
                    msg = f"{feedback}\n\nContinue with your next Thought."
            messages.append({"role": "user", "content": msg})
            log.append({"role": "user", "content": msg})
            continue

        final_text = _extract_final_answer(reply)
        if final_text is not None:
            if not require_structured_verdict:
                return None, final_text, log, fidelity_record
            verdict, err = parse_verdict(final_text)
            if verdict is not None:
                return verdict, final_text, log, fidelity_record
            repair_msg = (f"Your FINAL ANSWER could not be parsed: {err}\n"
                          f"Please resend a Thought, then FINAL ANSWER: followed by ONLY the "
                          f"corrected JSON object, matching the schema exactly.")
            messages.append({"role": "user", "content": repair_msg})
            log.append({"role": "user", "content": repair_msg})
            continue

        nudge = ("No ```python``` code block found and no FINAL ANSWER given. "
                 "Please give a Thought followed by either a ```python``` Action, "
                 "or your FINAL ANSWER: now.")
        messages.append({"role": "user", "content": nudge})
        log.append({"role": "user", "content": nudge})

    return None, "(Stopped: reached iteration limit without a valid FINAL ANSWER.)", log, fidelity_record
